"""
Не меряет ли разметка сама себя: реранкер и судья пула — одна модель.

Проблема, из-за которой написан скрипт. Пул релевантности размечен
``gpt-4o-mini`` (``build_qrels.py``). LLM-реранкер в ``retriever.py`` —
тоже ``gpt-4o-mini``, и задача у него по сути та же: «отбери фрагменты,
которые реально помогают ответить на вопрос». Промпты почти дословно
совпадают.

Значит, когда реранкер выигрывает по метрикам, у этого есть два одинаково
правдоподобных объяснения:

1. он действительно поднимает наверх полезное;
2. он просто угадывает, что сочтёт полезным его же собственная копия,
   размечавшая эталон.

Различить их можно одним способом: переразметить тот же пул **независимой
моделью** и пересчитать. Если преимущество держится — оно настоящее.
Если исчезает — мы мерили согласие модели с самой собой.

Поиск при этом заново не запускается: выдачи всех конфигураций лежат
в ``experiments_retrieval_runs.json`` (их пишет ``experiments.py``).
Платим только за разметку.

Побочно скрипт отвечает на второй вопрос о валидности: **покрытие пула**.
Если какая-то конфигурация поднимает чанки, которых в разметке нет, они
по умолчанию считаются нерелевантными — и эта конфигурация штрафуется
ни за что. Доля неразмеченного печатается по каждой конфигурации.

Запуск::

    python -m experiments.qrels_bias             # разметить и сравнить
    python -m experiments.qrels_bias --coverage  # только покрытие, без вызовов
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from build_qrels import JUDGE_BATCH, JUDGE_PROMPT, QRELS_PATH, load_qrels
from evaluation_answers import _judge
from measurement import METRIC_COLUMNS, RESEARCH_DIR, RUNS_PATH, paired_bootstrap
from lance_db import open_table, read_all
from make_eval_dataset import DATASET_PATH
from providers import get_polza_client

# Независимый судья: другое семейство моделей, обучен отдельно.
# gpt-4o брать нельзя — он родня gpt-4o-mini и делит с ним слепые зоны.
INDEPENDENT_JUDGE = "meta-llama/llama-3.3-70b-instruct"
INDEPENDENT_QRELS_PATH = RESEARCH_DIR / "qrels_llama.csv"


def load_runs() -> dict[str, dict[str, list[int]]]:
    if not RUNS_PATH.exists():
        raise SystemExit(
            f"Нет сохранённых выдач {RUNS_PATH.name}. Выполните: python -m experiments.retrieval"
        )
    return json.loads(RUNS_PATH.read_text(encoding="utf-8"))


def report_coverage(runs: dict[str, dict[str, list[int]]], qrels_frame: pd.DataFrame) -> None:
    """Какая доля поднятых чанков вообще размечена.

    Неразмеченный чанк молча считается нерелевантным — это стандартная
    практика пулинга, но она бьёт по конфигурациям, которые находят
    что-то за пределами пула. Если покрытие у всех одинаковое, сравнение
    честное; если у кого-то ниже — его оценка занижена.
    """
    judged = {(str(row.id), int(row.chunk_id)) for row in qrels_frame.itertuples()}

    print("=" * 72)
    print("ПОКРЫТИЕ ПУЛА: доля поднятых чанков, у которых есть вердикт")
    print("=" * 72)
    for name, per_question in runs.items():
        pairs = [(qid, cid) for qid, ids in per_question.items() for cid in ids]
        covered = sum(1 for pair in pairs if pair in judged)
        share = covered / len(pairs) if pairs else float("nan")
        print(f"  {name:<34} {share:6.1%}   ({covered} из {len(pairs)})")
    print()


def build_pool_from_runs(
    runs: dict[str, dict[str, list[int]]],
    dataset: pd.DataFrame,
    qrels_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Собрать пул «вопрос-чанк» из сохранённых выдач и старой разметки.

    Старые пары добавляются, чтобы обе разметки покрывали одно и то же
    множество: иначе разница между судьями мешалась бы с разницей пулов.
    """
    table = open_table("lance_db/vectorstore", "chunks")
    corpus = read_all(table).set_index("chunk_id")
    questions = dict(zip(dataset["id"].astype(str), dataset["question"].astype(str), strict=True))

    pairs: set[tuple[str, int]] = set()
    for per_question in runs.values():
        for row_id, chunk_ids in per_question.items():
            pairs.update((row_id, int(cid)) for cid in chunk_ids)
    pairs.update((str(row.id), int(row.chunk_id)) for row in qrels_frame.itertuples())

    rows = []
    for row_id, chunk_id in sorted(pairs):
        if row_id not in questions or chunk_id not in corpus.index:
            continue
        record = corpus.loc[chunk_id]
        rows.append({
            "id": row_id,
            "question": questions[row_id],
            "chunk_id": chunk_id,
            "doc_id": int(record["doc_id"]),
            "text": str(record["text"]),
        })

    frame = pd.DataFrame(rows)
    per_question = frame.groupby("id").size()
    print(f"Пул для переразметки: {len(frame)} пар, "
          f"в среднем {per_question.mean():.1f} на вопрос\n")
    return frame


def judge_pool(pool: pd.DataFrame, client, model: str, max_workers: int = 6) -> pd.DataFrame:
    """Разметить пул указанной моделью. Батчами — как в build_qrels."""
    tasks: list[tuple[str, list[dict]]] = []
    for row_id, group in pool.groupby("id"):
        records = group.to_dict("records")
        for start in range(0, len(records), JUDGE_BATCH):
            tasks.append((str(row_id), records[start : start + JUDGE_BATCH]))

    def work(task: tuple[str, list[dict]]) -> list[dict]:
        _row_id, records = task
        context = "\n\n".join(
            f"[{i + 1}] {record['text'][:1200]}" for i, record in enumerate(records)
        )
        try:
            verdict = _judge(
                client, model,
                JUDGE_PROMPT.format(question=records[0]["question"], context=context),
            )
            relevant = verdict.get("relevant") or []
            indices = {int(n) - 1 for n in relevant if isinstance(n, (int, float))}
        except Exception:
            # Судья не ответил — вердикта нет. Помечаем, чтобы не выдать
            # молчание за «нерелевантно».
            indices = None

        return [
            {
                "id": record["id"],
                "chunk_id": record["chunk_id"],
                "doc_id": record["doc_id"],
                "relevant": np.nan if indices is None else int(position in indices),
            }
            for position, record in enumerate(records)
        ]

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool_exec:
        for batch in tqdm(pool_exec.map(work, tasks), total=len(tasks), desc="Разметка"):
            rows.extend(batch)

    return pd.DataFrame(rows).drop_duplicates(subset=["id", "chunk_id"])


def qrels_from_frame(frame: pd.DataFrame) -> dict[str, set[int]]:
    relevant = frame[frame["relevant"] == 1]
    return {
        str(row_id): set(group["chunk_id"].astype(int))
        for row_id, group in relevant.groupby("id")
    }


def score(runs: dict[str, dict[str, list[int]]], qrels: dict[str, set[int]]) -> dict[str, pd.DataFrame]:
    """Пересчитать метрики по сохранённым выдачам. Вызовов LLM нет."""
    from evaluation import (
        _average_precision,
        _f1_at_k,
        _ndcg_at_k,
        _precision_at_k,
        _recall_at_k,
        _recall_ceiling,
        _reciprocal_rank,
    )

    result: dict[str, pd.DataFrame] = {}
    for name, per_question in runs.items():
        rows = []
        for row_id, chunk_ids in per_question.items():
            relevant_ids = qrels.get(row_id, set())
            total = len(relevant_ids)
            vec = [int(cid in relevant_ids) for cid in chunk_ids]
            rows.append({
                "id": row_id,
                "precision": _precision_at_k(vec),
                "recall": _recall_at_k(vec, total),
                "f1": _f1_at_k(vec, total),
                "mrr": _reciprocal_rank(vec),
                "map": _average_precision(vec, total),
                "ndcg": _ndcg_at_k(vec, total),
                "recall_ceiling": _recall_ceiling(len(vec), total),
            })
        result[name] = pd.DataFrame(rows)
    return result


def report(label: str, scored: dict[str, pd.DataFrame], pairs: list[tuple[str, str]]) -> None:
    print("=" * 72)
    print(f"МЕТРИКИ ПО РАЗМЕТКЕ: {label}")
    print("=" * 72)

    summary = pd.DataFrame([
        {"config": name, **{m: float(frame[m].mean()) for m in METRIC_COLUMNS},
         "ceiling": float(frame["recall_ceiling"].mean())}
        for name, frame in scored.items()
    ])
    summary["of_max"] = summary["recall"] / summary["ceiling"]
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print()
    for baseline, variant in pairs:
        if baseline not in scored or variant not in scored:
            continue
        base = scored[baseline].set_index("id")
        var = scored[variant].set_index("id")
        common = base.index.intersection(var.index)
        print(f"{variant}\n  против: {baseline}")
        for metric in ("recall", "mrr", "ndcg"):
            mean, low, high = paired_bootstrap(base.loc[common, metric], var.loc[common, metric])
            verdict = "значимо" if (low > 0 or high < 0) else "в пределах шума"
            print(f"    {metric:<8} {mean:+.3f}  [{low:+.3f}, {high:+.3f}]  {verdict}")
        print()


def main() -> None:
    runs = load_runs()
    dataset = pd.read_excel(DATASET_PATH)
    qrels_frame = pd.read_csv(QRELS_PATH)

    report_coverage(runs, qrels_frame)

    if "--coverage" in sys.argv:
        return

    client = get_polza_client()

    if INDEPENDENT_QRELS_PATH.exists():
        print(f"Независимая разметка взята из кэша: {INDEPENDENT_QRELS_PATH.name}\n")
        independent = pd.read_csv(INDEPENDENT_QRELS_PATH)
    else:
        pool = build_pool_from_runs(runs, dataset, qrels_frame)
        print(f"Судья: {INDEPENDENT_JUDGE}\n")
        independent = judge_pool(pool, client, INDEPENDENT_JUDGE)
        independent.to_csv(INDEPENDENT_QRELS_PATH, index=False, encoding="utf-8")
        print(f"\nСохранено → {INDEPENDENT_QRELS_PATH}\n")

    missing = int(independent["relevant"].isna().sum())
    if missing:
        print(f"⚠️ Судья не ответил на {missing} пар из {len(independent)} — "
              "они исключены, а не засчитаны нерелевантными\n")

    # Насколько вообще согласны два судьи между собой.
    merged = pd.read_csv(QRELS_PATH).merge(
        independent, on=["id", "chunk_id"], suffixes=("_mini", "_llama")
    ).dropna(subset=["relevant_llama"])
    agreement = float((merged["relevant_mini"] == merged["relevant_llama"]).mean())
    print(f"Согласие судей по общим парам ({len(merged)}): {agreement:.1%}")
    print(f"  доля релевантных у mini:  {merged['relevant_mini'].mean():.1%}")
    print(f"  доля релевантных у llama: {merged['relevant_llama'].mean():.1%}\n")

    pairs = [
        ("1. Dense k=5", "2. Hybrid (BM25+RRF) k=5"),
        ("2. Hybrid (BM25+RRF) k=5", "5. Hybrid + LLM-реранкинг k=5"),
    ]

    report("gpt-4o-mini (родня реранкеру)", score(runs, load_qrels()), pairs)
    report(f"{INDEPENDENT_JUDGE} (независимый)", score(runs, qrels_from_frame(independent)), pairs)

    print("=" * 72)
    print("Как читать: если преимущество реранкинга держится под независимой")
    print("разметкой — оно настоящее. Если проседает вдвое и больше — мы мерили")
    print("согласие gpt-4o-mini с самим собой, а не качество поиска.")


if __name__ == "__main__":
    main()
