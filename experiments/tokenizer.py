"""
Токенизатор BM25: не обрезка ли виновата в том, что лексический поиск не даёт прироста.

Гипотеза, с которой начали. `tokenize` обрезает слова до шести символов,
и докстринг объясняет это русской морфологией: «гипертензия / гипертензии /
гипертензией» после обрезки дают одну основу. Но корпус англоязычный,
а вопрос переводится на английский ДО поиска — русского текста к моменту
токенизации нет вовсе. Обоснование осталось от прежнего корпуса.

Что обрезка делает с реальным словарём (13 825 слов, 65% длиннее шести
символов):

    склеивает несклеиваемое
      immuno  ← immunoglobulin, immunodeficiency, immunofluorescence
      contra  ← contraception, contractility, contraction
      crd420  ← семь разных регистрационных номеров обзоров

    и не склеивает очевидное
      obese → obese     obesity   → obesit
      child → child     children  → childr

Почему это могло убить именно BM25. Его ценность — точные термины,
где эмбеддинги плывут. У редкого слова высокий IDF, но после склейки
`immunoglobulin` → `immuno` оно сливается с десятком других, и вес
обрушивается. BM25 остаётся с общей лексикой, то есть ровно с тем,
что векторный поиск и так находит.

## Что сравнивается

Три токенизатора на одном корпусе и одних вопросах:

* **current** — как сейчас, обрезка до шести символов;
* **plain** — без обрезки вовсе, ноль зависимостей;
* **suffix** — снятие частых английских окончаний, тоже без зависимостей.

Для каждого — dense против гибрида, чтобы увидеть ВКЛАД BM25, а не
абсолютные числа. Плюс проверка на независимой разметке: у нас есть
`qrels_llama.csv`, и выводы о поиске положено проверять ею — см. шаг 24.

Вызовов LLM здесь нет ни одного: переводы вопросов взяты из кэша,
эмбеддинги запросов тоже кэшируются на диске.

Запуск::

    python -m experiments.tokenizer
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from agent import Chunk, RAGAgentAnswer
from build_qrels import QRELS_PATH, load_qrels
from embedder import Embedder
from evaluation import evaluate
from lance_db import open_table
from make_eval_dataset import DATASET_PATH
from measurement import CANDIDATE_K, METRIC_COLUMNS, QUERIES_PATH, RESEARCH_DIR, paired_bootstrap
from providers import get_clients
from retriever import STOPWORDS, Retriever

RESULTS_PATH = RESEARCH_DIR / "experiments_tokenizer.csv"
INDEPENDENT_QRELS_PATH = RESEARCH_DIR / "qrels_llama.csv"

TOP_K = 5


def tokenize_current(text: str) -> list[str]:
    """Как сейчас: обрезка до шести символов."""
    words = re.findall(r"[\w]+", text.lower())
    return [w[:6] for w in words if w not in STOPWORDS and len(w) > 2]


def tokenize_plain(text: str) -> list[str]:
    """Без обрезки вовсе. Самый дешёвый вариант: ничего не добавляем, убираем."""
    words = re.findall(r"[\w]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


# Окончания снимаются от длинных к коротким: иначе «-s» отрежется раньше,
# чем «-ies», и «studies» превратится в «studie» вместо «studi».
_SUFFIXES = ("ational", "ization", "iveness", "fulness", "ousness", "ations",
             "ities", "ively", "ement", "ation", "ingly", "edly", "ism",
             "ness", "ment", "ity", "ies", "ing", "ers", "ive", "al",
             "ic", "ed", "es", "s")

# Ниже этой длины основу не режем: от коротких слов ничего не остаётся,
# «bed» превратилось бы в «b».
_MIN_STEM = 4


def tokenize_suffix(text: str) -> list[str]:
    """Снятие частых английских окончаний. Грубее Портера, но без зависимости.

    Задача скромная: свести obesity/obese и children/child к общей основе,
    НЕ трогая длинные технические термины, которые обрезка уничтожала.
    """
    words = re.findall(r"[\w]+", text.lower())
    out: list[str] = []
    for word in words:
        if word in STOPWORDS or len(word) <= 2:
            continue
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
                word = word[: -len(suffix)]
                break
        # Хвостовая «e» снимается отдельно: без этого obese и obesity дают
        # «obese» и «obes» — то есть разные основы там, где нужна одна.
        if len(word) > _MIN_STEM and word.endswith("e"):
            word = word[:-1]
        out.append(word)
    return out


VARIANTS = [
    ("current (обрезка до 6)", tokenize_current),
    ("plain (без обрезки)", tokenize_plain),
    ("suffix (снятие окончаний)", tokenize_suffix),
]


def run_retrieval(retriever: Retriever, dataset: pd.DataFrame, queries: list[str]):
    def work(item: tuple[str, str]) -> RAGAgentAnswer:
        row_id, query = item
        hits = retriever.retrieve(query, k=TOP_K)
        chunks = [
            Chunk(text=h["text"], chunk_id=int(h["chunk_id"]),
                  year=int(h["doc_id"]), distance=float(h.get("distance", 0.0)))
            for h in hits
        ]
        return RAGAgentAnswer(dataset_row_id=row_id, answer="", retrieved_chunks=chunks or None)

    items = list(zip(dataset["id"].astype(str), queries, strict=True))
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(work, items))


def describe_vocabulary(table) -> None:
    """Во что каждый токенизатор превращает словарь корпуса."""
    from lance_db import read_all

    texts = read_all(table)["text"].tolist()
    print("=" * 72)
    print("ЧТО ТОКЕНИЗАТОР ДЕЛАЕТ СО СЛОВАРЁМ")
    print("=" * 72)
    for name, tokenizer in VARIANTS:
        vocabulary: set[str] = set()
        for text in texts:
            vocabulary.update(tokenizer(text))
        print(f"  {name:<28} уникальных основ: {len(vocabulary)}")
    print()
    print("  Меньше основ — не лучше: часть склеек ошибочна и рушит IDF")
    print("  у редких терминов, ради которых BM25 и добавляли.")
    print()


def where_bm25_helps(dense_frame, hybrid_frame, dataset, queries, bm25) -> None:
    """На каких вопросах гибрид выигрывает у плотного поиска, и чем они особенные.

    Среднее прячет неоднородность. Если BM25 помогает там, где в запросе
    есть точные термины и числа, а портит остальное — правильный ответ
    не «выключить», а «включать по признаку запроса».

    Признак, который проверяем: **максимальный IDF слова запроса**. BM25
    силён редкими словами; если он и выигрывает, то на них.
    """
    query_by_id = dict(zip(dataset["id"].astype(str), queries, strict=True))

    merged = dense_frame.set_index("id").join(
        hybrid_frame.set_index("id"), lsuffix="_dense", rsuffix="_hybrid")
    merged["delta"] = merged["ndcg_hybrid"] - merged["ndcg_dense"]

    wins = merged[merged["delta"] > 1e-9]
    losses = merged[merged["delta"] < -1e-9]
    ties = merged[merged["delta"].abs() <= 1e-9]

    print("=" * 72)
    print("ГДЕ ИМЕННО BM25 ВЫИГРЫВАЕТ (по NDCG, разметка-родня)")
    print("=" * 72)
    total = len(merged)
    print(f"  выигрывает:   {len(wins):3d} вопросов ({len(wins)/total:.0%}), "
          f"в среднем {wins['delta'].mean():+.3f}" if len(wins) else "  выигрывает:     0")
    print(f"  проигрывает:  {len(losses):3d} ({len(losses)/total:.0%}), "
          f"в среднем {losses['delta'].mean():+.3f}" if len(losses) else "  проигрывает:    0")
    print(f"  не меняет:    {len(ties):3d} ({len(ties)/total:.0%})")

    def max_idf(row_id: str) -> float:
        tokens = bm25.tokenize(query_by_id.get(str(row_id), ""))
        values = [bm25.idf.get(t, 0.0) for t in tokens]
        return max(values) if values else 0.0

    def has_digit(row_id: str) -> bool:
        return any(ch.isdigit() for ch in query_by_id.get(str(row_id), ""))

    print()
    print("  Чем выигрышные вопросы отличаются от проигрышных:")
    print(f"  {'':<22}{'выигрыш':>10}{'проигрыш':>11}")
    for label, fn in (("макс. IDF слова", max_idf), ):
        w = sum(fn(i) for i in wins.index) / len(wins) if len(wins) else 0
        loss = sum(fn(i) for i in losses.index) / len(losses) if len(losses) else 0
        print(f"  {label:<22}{w:>10.2f}{loss:>11.2f}")
    for label, fn in (("с числом в запросе", has_digit), ):
        w = sum(fn(i) for i in wins.index) / len(wins) if len(wins) else 0
        loss = sum(fn(i) for i in losses.index) / len(losses) if len(losses) else 0
        print(f"  {label:<22}{w:>9.0%}{loss:>11.0%}")

    if len(wins):
        print()
        print("  Примеры, где гибрид выиграл больше всего:")
        for row_id in wins["delta"].nlargest(4).index:
            print(f"    {wins.loc[row_id, 'delta']:+.2f}  "
                  f"{str(query_by_id.get(str(row_id), ''))[:66]}")
    print()


def main() -> None:
    if not DATASET_PATH.exists() or not QUERIES_PATH.exists():
        print("Нет датасета или переводов. Выполните: python -m experiments.retrieval")
        return

    dataset = pd.read_excel(DATASET_PATH)
    queries = pd.read_json(QUERIES_PATH, typ="series").tolist()
    _client, _model, embed_client = get_clients("polza")

    table = open_table("lance_db/vectorstore", "chunks")
    embedder = Embedder(embed_client)

    qrels_sets = [("gpt-4o-mini (родня)", load_qrels())]
    if INDEPENDENT_QRELS_PATH.exists():
        frame = pd.read_csv(INDEPENDENT_QRELS_PATH)
        relevant = frame[frame["relevant"] == 1]
        qrels_sets.append((
            "llama-3.3-70b (независимая)",
            {str(i): set(g["chunk_id"].astype(int)) for i, g in relevant.groupby("id")},
        ))

    describe_vocabulary(table)

    # Dense-выдача одна на всё: BM25 её не меняет, а гонять заново незачем.
    print("Плотный поиск (общая база для сравнения)...")
    dense = Retriever(table, embedder, top_k=TOP_K, candidate_k=CANDIDATE_K,
                      use_hybrid=False, use_rerank=False)
    answers = {"dense (без BM25)": run_retrieval(dense, dataset, queries)}

    bm25_for_idf = None
    for name, tokenizer in VARIANTS:
        print(f"Гибрид, токенизатор: {name}...")
        hybrid = Retriever(table, embedder, top_k=TOP_K, candidate_k=CANDIDATE_K,
                           use_hybrid=True, use_rerank=False, tokenizer=tokenizer)
        answers[f"hybrid / {name}"] = run_retrieval(hybrid, dataset, queries)
        # IDF действующего токенизатора нужен для разбора «где BM25 выигрывает».
        if bm25_for_idf is None:
            bm25_for_idf = hybrid._bm25

    rows: list[dict] = []
    for label, qrels in qrels_sets:
        print()
        print("=" * 72)
        print(f"РАЗМЕТКА: {label}")
        print("=" * 72)

        scored = {
            name: evaluate(answers=value, dataset=dataset, client=None,
                           compute_faithfulness=False, qrels=qrels)
            for name, value in answers.items()
        }

        summary = pd.DataFrame([
            {"разметка": label, "конфигурация": name,
             **{m: float(frame[m].mean()) for m in METRIC_COLUMNS}}
            for name, frame in scored.items()
        ])
        print(summary.drop(columns=["разметка"]).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
        rows.extend(summary.to_dict("records"))

        # Поразрядный разбор делаем один раз, на действующем токенизаторе:
        # вопрос «где BM25 выигрывает» не про варианты токенизации.
        if label.startswith("gpt-4o-mini"):
            where_bm25_helps(
                scored["dense (без BM25)"],
                scored["hybrid / current (обрезка до 6)"],
                dataset, queries, bm25_for_idf,
            )

        print()
        print("Вклад BM25 относительно чистого плотного поиска:")
        base = scored["dense (без BM25)"].set_index("id")
        for name in scored:
            if name == "dense (без BM25)":
                continue
            variant = scored[name].set_index("id")
            common = base.index.intersection(variant.index)
            print(f"  {name}")
            for metric in ("recall", "ndcg", "precision"):
                mean, low, high = paired_bootstrap(
                    base.loc[common, metric], variant.loc[common, metric])
                verdict = "значимо" if (low > 0 or high < 0) else "в пределах шума"
                print(f"    {metric:<10} {mean:+.3f}  [{low:+.3f}, {high:+.3f}]  {verdict}")

    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print()
    print("=" * 72)
    print("Как читать: если хоть один токенизатор даёт значимый прирост под")
    print("ОБЕИМИ разметками — дефект был в обрезке, и вывод шага 15 о BM25")
    print("(«ставка, а не выигрыш») надо пересматривать.")
    print(f"\nСохранено → {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
