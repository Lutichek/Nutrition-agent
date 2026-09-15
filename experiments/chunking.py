"""
Размер чанка: последний параметр пайплайна, выбранный без замера качества.

Как его выбирали раньше (шаг 6 в EVOLUTION.md): сравнивали **статистику
длин** — сколько получится чанков, какая медиана, нет ли обрывков и выбросов.
По этим признакам 900/150 выглядел разумно, и на нём остановились. Но длина
чанка — это не косметика, а компромисс поиска: короткий чанк точнее по теме
и хуже по контексту, длинный наоборот. Как это сказывается на precision
и recall, не проверял никто.

Что здесь делается. Корпус перенарезается несколькими способами, для каждого
строится отдельный индекс, и на одних и тех же 210 вопросах считаются
retrieval-метрики.

## Главная сложность: разметку нельзя перенести напрямую

``qrels.csv`` хранит пары «вопрос — chunk_id». При другой нарезке chunk_id
означают совсем другие куски текста, и разметка превращается в мусор — молча,
без единой ошибки. Это ровно тот класс отказа, из-за которого ``doc_id``
в проекте сделан равным PMID.

Поэтому релевантность переносится **по тексту**: новый чанк считается
релевантным, если он из той же статьи и существенно перекрывается с чанком,
который судья признал релевантным. Мера перекрытия — коэффициент Отиаи
(overlap coefficient): доля общих слов от меньшего из двух фрагментов.
Она не штрафует за разницу в длине, а разница в длине здесь и есть предмет
эксперимента.

Заголовок статьи из сравнения исключается: он дописан к КАЖДОМУ чанку,
поэтому любые два фрагмента одной статьи из-за него выглядят похожими.

## Что сравнивать между строками

Число релевантных чанков на вопрос зависит от нарезки: один релевантный
кусок на 900 символов превращается в три на 300. Значит меняется и потолок
recall@5. Сравнивать голый recall между строками нельзя — сопоставимая
величина это ``of_max``, доля достижимого. Precision и MRR сравнимы как есть.

Поиск здесь **только плотный** (dense), без BM25 и реранкинга: меряется
влияние нарезки на эмбеддинги, а не сумма эффектов. Вызовов LLM нет вовсе,
платим только за эмбеддинги.

Запуск::

    python -m experiments.chunking
    python -m experiments.chunking --sizes 300,900     # подмножество
"""

from __future__ import annotations

import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from agent import Chunk, RAGAgentAnswer
from build_qrels import QRELS_PATH
from data_preporation import prepare_chunks
from embedder import Embedder
from evaluation import (
    _average_precision,
    _f1_at_k,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
    _recall_ceiling,
    _reciprocal_rank,
)
from measurement import (
    METRIC_COLUMNS,
    QUERIES_PATH,
    RESEARCH_DIR,
    paired_bootstrap,
)
from lance_db import build_vectorstore, open_table, read_all
from make_eval_dataset import DATASET_PATH
from providers import get_clients
from retriever import Retriever

# Корень проекта: файл лежит в пакете experiments/, поэтому на уровень выше.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = RESEARCH_DIR / "experiments_chunking.csv"

# Временные индексы. Каждый весит десятки мегабайт, поэтому живут отдельно
# от рабочего и удаляются после прогона.
SCRATCH_DB = PROJECT_ROOT / "lance_db" / "_chunking_scratch"

# Перекрытие держим пропорциональным размеру (примерно одна шестая),
# иначе вместе с размером менялся бы и второй параметр.
SIZES: list[tuple[int, int]] = [
    (300, 50),
    (600, 100),
    (900, 150),   # текущая рабочая нарезка
    (1200, 200),
]

BASELINE = "900/150"

TOP_K = 5

# Бюджет контекста в символах: столько текста уезжает в генерацию при
# текущей нарезке (5 чанков × 900).
#
# Зачем это нужно. Все retrieval-метрики считают ЧАНКИ, а не сведения.
# При k=5 мелкая нарезка отдаёт в генерацию 1500 символов, крупная — 4500,
# и метрика этой разницы не видит вовсе: единица измерения уменьшается
# вместе с измеряемым. Хуже того, один и тот же ответ, разрезанный на пять
# кусков, засчитывается пять раз — precision растёт, а сведений столько же.
#
# Поэтому сравнение идёт дважды: при равном k (как принято) и при равном
# бюджете символов (как честно).
CONTEXT_BUDGET = TOP_K * 900

# Порог переноса релевантности. 0.5 означает «половина слов меньшего
# фрагмента общая» — при нарезках, отличающихся в четыре раза, это
# разделяет «тот же кусок текста» и «соседний абзац той же статьи».
TRANSFER_THRESHOLD = 0.5


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.lower()))


def _overlap(first: set[str], second: set[str]) -> float:
    """Коэффициент Отиаи: пересечение, делённое на меньшее из множеств."""
    if not first or not second:
        return 0.0
    return len(first & second) / min(len(first), len(second))


def load_relevant_texts() -> dict[str, list[tuple[int, set[str]]]]:
    """Для каждого вопроса — слова релевантных чанков ТЕКУЩЕЙ нарезки.

    Заголовок статьи вычитается: он есть в каждом чанке и сам по себе
    создал бы перекрытие между любыми двумя фрагментами одной статьи.
    """
    qrels = pd.read_csv(QRELS_PATH)
    qrels = qrels[qrels["relevant"] == 1]

    corpus = read_all(open_table("lance_db/vectorstore", "chunks")).set_index("chunk_id")

    result: dict[str, list[tuple[int, set[str]]]] = {}
    for row in qrels.itertuples():
        chunk_id = int(row.chunk_id)
        if chunk_id not in corpus.index:
            continue
        record = corpus.loc[chunk_id]
        body = _tokens(str(record["text"])) - _tokens(str(record.get("title", "")))
        result.setdefault(str(row.id), []).append((int(record["doc_id"]), body))
    return result


def transfer_relevance(
    chunk_text: str,
    chunk_doc_id: int,
    chunk_title: str,
    relevant: list[tuple[int, set[str]]],
) -> bool:
    """Релевантен ли чанк НОВОЙ нарезки — по перекрытию со старыми."""
    body = _tokens(chunk_text) - _tokens(chunk_title)
    for doc_id, gold in relevant:
        if doc_id != chunk_doc_id:
            continue
        if _overlap(body, gold) >= TRANSFER_THRESHOLD:
            return True
    return False


def build_index(size: int, overlap: int, embedder: Embedder, table_name: str):
    """Перенарезать корпус и залить в отдельную таблицу."""
    chunks = prepare_chunks(chunk_size=size, chunk_overlap=overlap, verbose=False)
    print(f"  чанков: {len(chunks)}")
    return build_vectorstore(
        chunks, embedder, db_path=SCRATCH_DB, table_name=table_name,
        overwrite=True, verbose=False,
    )


def score_index(
    table,
    dataset: pd.DataFrame,
    queries: list[str],
    embedder: Embedder,
    relevant_by_question: dict[str, list[tuple[int, set[str]]]],
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Прогнать поиск и посчитать метрики с перенесённой разметкой."""
    corpus = read_all(table)
    titles = dict(zip(corpus["chunk_id"].astype(int), corpus["title"].astype(str), strict=True))

    # Сколько всего релевантных чанков у вопроса в ЭТОЙ нарезке. Считается
    # по всему корпусу, а не по выдаче: иначе recall был бы долей от
    # найденного, то есть всегда единицей.
    #
    # Перебирать весь корпус на каждый вопрос нельзя: 210 × 13000 сравнений
    # множеств — это минуты на конфигурацию. Совпасть может только чанк
    # из той же статьи, поэтому корпус сначала раскладывается по doc_id.
    by_doc: dict[int, list[tuple[str, str]]] = {}
    for record in corpus.itertuples():
        by_doc.setdefault(int(record.doc_id), []).append((str(record.text), str(record.title)))

    total_relevant: dict[str, int] = {}
    for row_id, relevant in relevant_by_question.items():
        docs = {doc_id for doc_id, _ in relevant}
        count = 0
        for doc_id in docs:
            for text, title in by_doc.get(doc_id, []):
                if transfer_relevance(text, doc_id, title, relevant):
                    count += 1
        total_relevant[row_id] = count

    retriever = Retriever(
        table, embedder, client=None, top_k=top_k, candidate_k=max(40, top_k * 4),
        use_hybrid=False, use_rerank=False,
    )

    def work(item: tuple[str, str]) -> RAGAgentAnswer:
        row_id, query = item
        hits = retriever.retrieve(query, k=top_k)
        chunks = [
            Chunk(text=h["text"], chunk_id=int(h["chunk_id"]), year=int(h["doc_id"]),
                  distance=float(h.get("distance", 0.0)))
            for h in hits
        ]
        return RAGAgentAnswer(dataset_row_id=row_id, answer="", retrieved_chunks=chunks or None)

    items = list(zip(dataset["id"].astype(str), queries, strict=True))
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(work, items))

    rows = []
    for answer in answers:
        row_id = str(answer.dataset_row_id)
        relevant = relevant_by_question.get(row_id, [])
        total = total_relevant.get(row_id, 0)

        vec = [
            int(transfer_relevance(c.text, c.year, titles.get(c.chunk_id, ""), relevant))
            for c in (answer.retrieved_chunks or [])
        ]
        rows.append({
            "id": row_id,
            "precision": _precision_at_k(vec),
            "recall": _recall_at_k(vec, total),
            "f1": _f1_at_k(vec, total),
            "mrr": _reciprocal_rank(vec),
            "map": _average_precision(vec, total),
            "ndcg": _ndcg_at_k(vec, total),
            "recall_ceiling": _recall_ceiling(len(vec), total),
            "n_relevant": total,
        })
    return pd.DataFrame(rows)


def main() -> None:
    if not DATASET_PATH.exists():
        print("Нет датасета. Выполните: python make_eval_dataset.py")
        return
    if not QRELS_PATH.exists():
        print("Нет разметки пула. Выполните: python build_qrels.py")
        return
    if not QUERIES_PATH.exists():
        print("Нет переводов запросов. Выполните: python -m experiments.retrieval")
        return

    sizes = SIZES
    if "--sizes" in sys.argv:
        wanted = {int(s) for s in sys.argv[sys.argv.index("--sizes") + 1].split(",")}
        sizes = [pair for pair in SIZES if pair[0] in wanted]

    dataset = pd.read_excel(DATASET_PATH)
    queries = pd.read_json(QUERIES_PATH, typ="series").tolist()
    _client, _model, embed_client = get_clients("polza")
    embedder = Embedder(embed_client)

    print("Перенос разметки: по перекрытию текста, порог "
          f"{TRANSFER_THRESHOLD} (коэффициент Отиаи)\n")
    relevant_by_question = load_relevant_texts()

    per_question: dict[str, pd.DataFrame] = {}
    budget_per_question: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    budget_rows: list[dict] = []

    for size, overlap in sizes:
        name = f"{size}/{overlap}"
        print("=" * 72)
        print(f"Нарезка {name}" + ("  (текущая рабочая)" if name == BASELINE else ""))
        print("=" * 72)

        table = build_index(size, overlap, embedder, table_name=f"chunks_{size}")
        metrics = score_index(table, dataset, queries, embedder, relevant_by_question)

        # Второй замер — при равном бюджете символов. Сколько чанков нужно,
        # чтобы в генерацию уехало столько же текста, сколько сейчас.
        budget_k = max(1, round(CONTEXT_BUDGET / size))
        budget_metrics = (
            metrics if budget_k == TOP_K
            else score_index(table, dataset, queries, embedder,
                             relevant_by_question, top_k=budget_k)
        )
        budget_rows.append({
            "config": name,
            "k": budget_k,
            "символов": budget_k * size,
            "precision": float(budget_metrics["precision"].mean()),
            "recall": float(budget_metrics["recall"].mean()),
            "mrr": float(budget_metrics["mrr"].mean()),
            "ndcg": float(budget_metrics["ndcg"].mean()),
            "of_max": float(budget_metrics["recall"].mean())
            / float(budget_metrics["recall_ceiling"].mean()),
        })
        budget_per_question[name] = budget_metrics

        row = {"chunk_size": size, "overlap": overlap, "config": name}
        row.update({m: float(metrics[m].mean()) for m in METRIC_COLUMNS})
        row["ceiling"] = float(metrics["recall_ceiling"].mean())
        row["of_max"] = row["recall"] / row["ceiling"] if row["ceiling"] else float("nan")
        row["relevant_per_q"] = float(metrics["n_relevant"].mean())
        rows.append(row)
        per_question[name] = metrics

        print("  " + "  ".join(f"{m}={row[m]:.3f}" for m in METRIC_COLUMNS))
        print(f"  релевантных на вопрос={row['relevant_per_q']:.1f}  "
              f"потолок={row['ceiling']:.3f}  взято от достижимого={row['of_max']:.1%}\n")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    # Контроль самого переноса. На текущей нарезке он обязан воспроизвести
    # исходную разметку: те же чанки, то же их число. Если на строке 900/150
    # релевантных заметно не 6.2 — сломан перенос, а не нарезка, и всей
    # таблице верить нельзя.
    baseline_row = next((r for r in rows if r["config"] == BASELINE), None)
    if baseline_row is not None:
        expected = sum(len(v) for v in relevant_by_question.values()) / len(relevant_by_question)
        got = baseline_row["relevant_per_q"]
        drift = abs(got - expected) / expected
        print()
        print(f"Контроль переноса на {BASELINE}: было {expected:.2f} релевантных "
              f"на вопрос, перенос дал {got:.2f} (расхождение {drift:.1%})")
        if drift > 0.15:
            print("  ⚠️ Расхождение больше 15% — порог переноса подобран неверно,")
            print("     сравнивать строки между собой нельзя.")

    print("=" * 72)
    print("СВОДКА")
    print("=" * 72)
    print(summary.drop(columns=["chunk_size", "overlap"]).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("⚠️ Голый recall между строками не сравнивать: чем мельче нарезка,")
    print("   тем больше релевантных кусков и тем ниже потолок. Сравнимы")
    print("   precision, mrr и of_max.")

    if BASELINE in per_question:
        print()
        print("=" * 72)
        print(f"ЗНАЧИМОСТЬ ОТНОСИТЕЛЬНО ТЕКУЩЕЙ НАРЕЗКИ ({BASELINE})")
        print("=" * 72)
        base = per_question[BASELINE].set_index("id")
        for name, frame in per_question.items():
            if name == BASELINE:
                continue
            variant = frame.set_index("id")
            common = base.index.intersection(variant.index)
            print(f"\n{name} против {BASELINE}")
            for metric in ("precision", "mrr", "ndcg"):
                mean, low, high = paired_bootstrap(
                    base.loc[common, metric], variant.loc[common, metric]
                )
                verdict = "значимо" if (low > 0 or high < 0) else "в пределах шума"
                print(f"    {metric:<10} {mean:+.3f}  [{low:+.3f}, {high:+.3f}]  {verdict}")

    print()
    print("=" * 72)
    print("ПРИ РАВНОМ БЮДЖЕТЕ КОНТЕКСТА")
    print("=" * 72)
    print("k подобран так, чтобы в генерацию уходило поровну текста.")
    print("Сравнение при равном k выше благоволит мелкой нарезке: тот же")
    print("ответ, разрезанный на пять кусков, засчитывается пять раз.")
    print()
    print(pd.DataFrame(budget_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if BASELINE in budget_per_question:
        print()
        base = budget_per_question[BASELINE].set_index("id")
        for name, frame in budget_per_question.items():
            if name == BASELINE:
                continue
            variant = frame.set_index("id")
            common = base.index.intersection(variant.index)
            print(f"{name} против {BASELINE}")
            for metric in ("precision", "mrr", "ndcg"):
                mean, low, high = paired_bootstrap(
                    base.loc[common, metric], variant.loc[common, metric]
                )
                verdict = "значимо" if (low > 0 or high < 0) else "в пределах шума"
                print(f"    {metric:<10} {mean:+.3f}  [{low:+.3f}, {high:+.3f}]  {verdict}")
            print()

    shutil.rmtree(SCRATCH_DB, ignore_errors=True)
    print(f"\nСохранено → {RESULTS_PATH}")
    print("Временные индексы удалены.")


if __name__ == "__main__":
    main()
