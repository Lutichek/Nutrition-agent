"""
Вес голоса BM25 при слиянии: есть ли точка, где он помогает, но не мешает.

Откуда вопрос. Замер токенизаторов показал две вещи. Первая: чинить
токенизатор бесполезно — без обрезки становится хуже, а не лучше. Вторая,
важнее: BM25 при равном голосе **вредит** выдаче под обеими разметками.

Но разбор по вопросам показал, что «выключить» — вывод преждевременный:

    выигрывает:   64 вопроса (30%), в среднем +0.133 NDCG
    проигрывает:  88 вопросов (42%), в среднем −0.198
    не меняет:    58 (28%)

Выигрыши настоящие и приходятся на запросы с точными терминами —
«palmitic acid», «triglyceride and cholesterol», «vitamin E». Признака,
по которому их отличить заранее, найти не удалось: максимальный IDF слова
запроса у выигрышных 4.62 против 4.31 у проигрышных, а наличие числа
в запросе работает вообще в обратную сторону (3% против 8%).

Остаётся третья возможность. RRF складывает оба ранжирования **с равным
голосом**, и слабый сигнал перебивает сильный ровно там, где сам ошибается.
С меньшим весом BM25 перестанет перебивать, но сохранит способность
поднимать то, что нашёл только он.

Гипотеза проверяемая: если такая точка есть, на сетке весов будет виден
максимум между нулём (BM25 выключен) и единицей (как сейчас). Если кривая
монотонно падает — точки нет, и гибрид надо выключать.

Замер бесплатный: переводы вопросов и эмбеддинги берутся из кэша,
ни одного вызова модели.

Запуск::

    python -m experiments.bm25_weight
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from agent import Chunk, RAGAgentAnswer
from build_qrels import load_qrels
from embedder import Embedder
from evaluation import evaluate
from lance_db import open_table
from make_eval_dataset import DATASET_PATH
from measurement import CANDIDATE_K, QUERIES_PATH, RESEARCH_DIR, paired_bootstrap
from providers import get_clients
from retriever import Retriever

RESULTS_PATH = RESEARCH_DIR / "experiments_bm25_weight.csv"
INDEPENDENT_QRELS_PATH = RESEARCH_DIR / "qrels_llama.csv"

TOP_K = 5

# Ноль и единица — контрольные точки: ноль обязан совпасть с чистым плотным
# поиском, единица — с текущим поведением. Если не совпадут, сломан замер,
# а не гипотеза.
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


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


def main() -> None:
    if not DATASET_PATH.exists() or not QUERIES_PATH.exists():
        print("Нет датасета или переводов. Выполните: python -m experiments.retrieval")
        return

    dataset = pd.read_excel(DATASET_PATH)
    queries = pd.read_json(QUERIES_PATH, typ="series").tolist()
    _client, _model, embed_client = get_clients("polza")

    table = open_table("lance_db/vectorstore", "chunks")
    embedder = Embedder(embed_client)

    qrels_sets = [("родня", load_qrels())]
    if INDEPENDENT_QRELS_PATH.exists():
        frame = pd.read_csv(INDEPENDENT_QRELS_PATH)
        relevant = frame[frame["relevant"] == 1]
        qrels_sets.append((
            "независимая",
            {str(i): set(g["chunk_id"].astype(int)) for i, g in relevant.groupby("id")},
        ))

    # BM25-индекс строится один раз: он не зависит от веса слияния.
    print("Строю индекс и гоняю сетку весов...")
    shared = Retriever(table, embedder, top_k=TOP_K, candidate_k=CANDIDATE_K,
                       use_hybrid=True, use_rerank=False)

    answers: dict[float, list] = {}
    for weight in WEIGHTS:
        shared.bm25_weight = weight
        answers[weight] = run_retrieval(shared, dataset, queries)
        print(f"  вес {weight} — готово")

    rows: list[dict] = []
    for label, qrels in qrels_sets:
        scored = {
            weight: evaluate(answers=value, dataset=dataset, client=None,
                             compute_faithfulness=False, qrels=qrels)
            for weight, value in answers.items()
        }

        print()
        print("=" * 72)
        print(f"РАЗМЕТКА: {label}")
        print("=" * 72)
        print(f"  {'вес BM25':>9}{'precision':>11}{'recall':>9}{'ndcg':>8}"
              f"{'дельта ndcg к весу 0':>24}")

        zero = scored[0.0].set_index("id")
        for weight in WEIGHTS:
            frame = scored[weight]
            variant = frame.set_index("id")
            common = zero.index.intersection(variant.index)
            mean, low, high = paired_bootstrap(
                zero.loc[common, "ndcg"], variant.loc[common, "ndcg"])
            verdict = "" if weight == 0 else (
                "значимо" if (low > 0 or high < 0) else "шум")
            delta = "" if weight == 0 else f"{mean:+.3f} [{low:+.3f}, {high:+.3f}] {verdict}"
            print(f"  {weight:>9}{frame['precision'].mean():>11.3f}"
                  f"{frame['recall'].mean():>9.3f}{frame['ndcg'].mean():>8.3f}"
                  f"{delta:>24}")
            rows.append({"разметка": label, "вес": weight,
                         "precision": float(frame["precision"].mean()),
                         "recall": float(frame["recall"].mean()),
                         "ndcg": float(frame["ndcg"].mean())})

        best = max(WEIGHTS, key=lambda w: scored[w]["ndcg"].mean())
        print()
        print(f"  лучший вес по NDCG: {best}")

    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print()
    print("=" * 72)
    print("Как читать: если лучший вес больше нуля под ОБЕИМИ разметками —")
    print("точка есть, BM25 стоит оставить с этим весом. Если лучший везде")
    print("ноль — гибрид выключаем, и это обоснованное решение.")
    print(f"\nСохранено → {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
