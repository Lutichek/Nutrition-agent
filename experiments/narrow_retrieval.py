"""
Поиск на узких фактических вопросах: что из трёх приёмов поднимает
исходный фрагмент в пятёрку.

Откуда вопрос (шаг 64): на вопросах deepseek, более узких, чем вопросы
gpt-4o-mini, precision и NDCG ниже, а отказов втрое больше. Разбор этих
отказов (шаг 65) показал, что поиск слабее, чем кажется, лишь в средней
корзине: 13 из 19 отказов дал маршрутизатор, и судья согласен с ним 13/13,
а у вопросов с одним релевантным исходный фрагмент в пятёрке всегда.
Где поиск правда промахивается — исходный фрагмент дальше пятого места:
15% вопросов deepseek с 2–4 релевантными против 2% у gpt-4o-mini.

Замер детерминированный: запросы — из кэша переводов (тот же промпт
rewrite, температура 0), поиск без модели. Сравнение при РАВНОМ k=5.

Решающая метрика — попадание ИСХОДНОГО фрагмента (по нему писался вопрос)
в пятёрку: она не зависит от разметки пула. NDCG по разметке llama —
вторая, и рядом печатается доля неразмеченных фрагментов в выдаче:
неразмеченный считается нерелевантным, и конфигурация, поднимающая новое,
штрафуется ни за что (покрытие пула, CLAUDE.md).

⚠️ ``chunk_id`` в датасете устарел у 80 строк из 210: номера сдвинулись
при перенарезке корпуса, а текст лежит в другом чанке (с шага 67 столбец
пересчитывается ``make_eval_dataset.py --resync-chunk-ids``). Исходный
фрагмент ищется здесь ПО ТЕКСТУ (``make_eval_dataset.source_ids``), чтобы
замер не зависел от того, пересчитан ли столбец. Метрики оценки его
не читают — они идут по разметке пула.

    python -m experiments.narrow_retrieval

Результат (шаг 65): ни один приём не лучше текущего поиска, два снижают
NDCG на B на 0.025. Отдавать в grade 8–10 фрагментов отклонено по
потолку: первый релевантный вне пятёрки у 3 из 253 вопросов в области.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embedder import Embedder  # noqa: E402
from evaluation import _ndcg_at_k  # noqa: E402
from experiments.question_family import QRELS_PATH as QRELS_B_PATH  # noqa: E402
from experiments.question_family import QUERIES_PATH as QUERIES_B_PATH  # noqa: E402
from experiments.question_family import QUESTIONS_PATH as QUESTIONS_B_PATH  # noqa: E402
from lance_db import open_table  # noqa: E402
from make_eval_dataset import DATASET_PATH, source_ids  # noqa: E402
from measurement import CANDIDATE_K, RESEARCH_DIR, cached_queries, paired_bootstrap  # noqa: E402
from providers import get_clients  # noqa: E402
from retriever import Retriever, reciprocal_rank_fusion  # noqa: E402
from run_eval import INDEPENDENT_QRELS_PATH  # noqa: E402

TOP_K = 5
RESULTS_PATH = RESEARCH_DIR / "experiments_narrow_retrieval.csv"

# Вес второго голоса при слиянии. 0.5 — как у взвешенного BM25 в сетке
# experiments/bm25_weight.py: равный голос там тянул выдачу вниз.
SECOND_WEIGHT = 0.5


def configs(ret: Retriever, query_en: str, question_ru: str) -> dict[str, list[int]]:
    """Пятёрка каждой конфигурации по одному вопросу."""
    dense_en = [h["chunk_id"] for h in ret._dense_search(query_en, CANDIDATE_K)]
    dense_ru = [h["chunk_id"] for h in ret._dense_search(question_ru, CANDIDATE_K)]
    bm25_en = [h["chunk_id"] for h in ret._bm25_search(query_en, CANDIDATE_K)]

    def fuse(lists, weights):
        return [cid for cid, _ in reciprocal_rank_fusion(lists, weights=weights)][:TOP_K]

    return {
        "сейчас: dense(en)": dense_en[:TOP_K],
        "+ dense(ru) ×0.5": fuse([dense_en, dense_ru], [1.0, SECOND_WEIGHT]),
        "+ bm25(en) ×0.5": fuse([dense_en, bm25_en], [1.0, SECOND_WEIGHT]),
        "+ dense(ru) + bm25(en) ×0.5": fuse([dense_en, dense_ru, bm25_en],
                                            [1.0, SECOND_WEIGHT, SECOND_WEIGHT]),
    }


def _judged(path: Path) -> tuple[dict[str, set[int]], set[tuple[str, int]]]:
    frame = pd.read_csv(path, dtype={"id": str})
    relevant = frame[frame["relevant"] == 1]
    qrels = {i: set(g["chunk_id"].astype(int)) for i, g in relevant.groupby("id")}
    return qrels, set(zip(frame["id"], frame["chunk_id"].astype(int), strict=True))


def main() -> None:
    dataset = pd.read_excel(DATASET_PATH)
    dataset["id"] = dataset["id"].astype(str)
    written = pd.read_csv(QUESTIONS_B_PATH, dtype={"id": str})
    side_b = dataset.assign(question=dataset["id"].map(dict(zip(written["id"], written["question"],
                                                                strict=True))))

    _client, _model, embed_client = get_clients("polza")
    table = open_table(str(ROOT / "lance_db" / "vectorstore"), "chunks")
    ret = Retriever(table, Embedder(embed_client), top_k=TOP_K, candidate_k=CANDIDATE_K,
                    use_hybrid=True)
    corpus = ret._corpus_df

    rows = []
    for side, data, queries_path, qrels_path in (
            ("A", dataset, None, INDEPENDENT_QRELS_PATH),
            ("B", side_b, QUERIES_B_PATH, QRELS_B_PATH)):
        queries = cached_queries(data["question"].tolist(), queries_path)
        qrels, judged = _judged(qrels_path)
        for (_, row), query in zip(data.iterrows(), queries, strict=True):
            source = source_ids(json.loads(row["relevant_text"])[0], corpus)
            relevant = qrels.get(row["id"], set())
            for name, top in configs(ret, query, row["question"]).items():
                rel_vec = [int(c in relevant) for c in top]
                rows.append({
                    "side": side, "id": row["id"], "config": name, "n_rel": len(relevant),
                    "source_hit": int(bool(source & set(top))),
                    "ndcg": _ndcg_at_k(rel_vec, len(relevant)),
                    "unjudged": sum((row["id"], c) not in judged for c in top) / TOP_K,
                })
        print(f"{side}: готово")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    frame["корзина"] = pd.cut(frame["n_rel"], [-1, 1, 4, 10_000], labels=["≤1", "2-4", "5+"])

    print("\nСводка (среднее по вопросам):")
    print(frame.pivot_table(index=["side", "корзина"], columns="config", values="source_hit",
                            observed=True).round(3).to_string())
    print("\nNDCG по разметке llama / доля неразмеченных в пятёрке:")
    print(frame.groupby(["side", "config"])[["ndcg", "unjudged"]].mean().round(3).to_string())

    base = "сейчас: dense(en)"
    print("\nПарно против текущего, 95% ДИ:")
    for side in ("A", "B"):
        cur = frame[(frame.side == side) & (frame.config == base)].set_index("id")
        for name in frame["config"].unique():
            if name == base:
                continue
            alt = frame[(frame.side == side) & (frame.config == name)].set_index("id")
            for metric in ("source_hit", "ndcg"):
                d, lo, hi = paired_bootstrap(cur[metric], alt[metric])
                print(f"  {side} {name:<30} {metric:<10} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"\nСохранено → {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
