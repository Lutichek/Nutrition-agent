"""
Эксперименты с ретривером: честное сравнение стратегий поиска.

Скрипт прогоняет один и тот же тестовый датасет через разные конфигурации
поиска и считает retrieval-метрики модулем ``evaluation.py``. Генерация ответа
здесь НЕ выполняется: она не влияет на то, что нашёл ретривер, зато стоит
денег и времени. Результаты этого скрипта — основа таблиц в EVOLUTION.md.

⚠️ Про перевод запроса. Корпус англоязычный, вопросы русские. Если сравнивать
конфигурации на русских запросах, BM25 не найдёт ничего ни в одной из них,
и сравнение выйдет бессмысленным. Поэтому вопросы переводятся ОДИН раз,
результат кэшируется на диск, и все конфигурации работают с одинаковыми
запросами — иначе разница между ними мешалась бы с разницей переводов.

Сам перевод при этом тоже проверяется: конфигурация «0» намеренно ищет
по русскому запросу, чтобы было видно, чего стоит этот шаг.

Запуск::

    python -m experiments.retrieval             # все конфигурации
    python -m experiments.retrieval --quick     # без реранкинга (быстро и дёшево)
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from agent import Chunk, RAGAgentAnswer
from build_qrels import QRELS_PATH, load_qrels
from embedder import Embedder
from evaluation import evaluate
from lance_db import open_table
from measurement import (
    CANDIDATE_K,
    METRIC_COLUMNS,
    RESEARCH_DIR,
    RUNS_PATH,
    load_queries_en,
    report_significance,
)
from providers import get_clients
from retriever import Retriever

DATASET_PATH = RESEARCH_DIR / "test_dataframe.xlsx"
RESULTS_PATH = RESEARCH_DIR / "experiments_retrieval.csv"


# ────────────────────────────────────────────────────────────
# Прогон одной конфигурации
# ────────────────────────────────────────────────────────────


def run_retrieval_only(
    retriever: Retriever,
    dataset: pd.DataFrame,
    queries: list[str],
    k: int,
    max_workers: int = 4,
) -> list[RAGAgentAnswer]:
    """Прогнать только поиск (без генерации) по всем вопросам датасета."""

    def work(item: tuple[str, str]) -> RAGAgentAnswer:
        row_id, query = item
        hits = retriever.retrieve(query, k=k)
        chunks = [
            Chunk(
                text=hit["text"],
                chunk_id=int(hit["chunk_id"]),
                year=int(hit["doc_id"]),  # см. пояснение про doc_id в agent.py
                distance=float(hit.get("distance", 0.0)),
            )
            for hit in hits
        ]
        return RAGAgentAnswer(dataset_row_id=row_id, answer="", retrieved_chunks=chunks or None)

    items = list(zip(dataset["id"].astype(str), queries, strict=True))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(work, items))


def build_configs(quick: bool) -> list[tuple[str, dict]]:
    """Что именно сравниваем.

    Порядок не случаен: каждая следующая конфигурация добавляет ровно один
    приём к предыдущей, поэтому прирост можно отнести к конкретному изменению.
    """
    configs: list[tuple[str, dict]] = [
        # Нулевая — контрольная: показывает цену перевода запроса.
        ("0. Dense, русский запрос k=5", dict(translate=False, use_hybrid=False, use_rerank=False, top_k=5)),
        ("1. Dense k=5", dict(translate=True, use_hybrid=False, use_rerank=False, top_k=5)),
        ("2. Hybrid (BM25+RRF) k=5", dict(translate=True, use_hybrid=True, use_rerank=False, top_k=5)),
        ("3. Hybrid k=3", dict(translate=True, use_hybrid=True, use_rerank=False, top_k=3)),
        ("4. Hybrid k=10", dict(translate=True, use_hybrid=True, use_rerank=False, top_k=10)),
    ]

    if not quick:
        configs.append(
            ("5. Hybrid + LLM-реранкинг k=5",
             dict(translate=True, use_hybrid=True, use_rerank=True, top_k=5))
        )

    return configs


def main() -> None:
    quick = "--quick" in sys.argv

    if not DATASET_PATH.exists():
        print("Нет тестового датасета. Выполните: python make_eval_dataset.py")
        return

    client, model, embed_client = get_clients("polza")
    dataset = pd.read_excel(DATASET_PATH)
    print(f"Вопросов в датасете: {len(dataset)}\n")

    russian = dataset["question"].astype(str).tolist()
    english = load_queries_en(russian, client, model)
    print()

    table = open_table("lance_db/vectorstore", "chunks")
    embedder = Embedder(embed_client)

    # Разметка пула. Раньше этот скрипт считал релевантность по пересечению
    # токенов с ОДНИМ эталонным чанком — и под такой разметкой польза
    # реранкинга была невидима в принципе: эталон один, он либо в выдаче,
    # либо нет, переставлять внутри пятёрки нечего. Под пулингом релевантных
    # 6.2 на вопрос, и «поднять нужное с 12-й позиции в топ-5» наконец
    # становится измеримым действием.
    qrels = None
    if QRELS_PATH.exists() and "--single-gt" not in sys.argv:
        qrels = load_qrels()
        print(f"Разметка пула: {len(qrels)} вопросов, "
              f"{sum(len(v) for v in qrels.values()) / len(qrels):.1f} релевантных на вопрос\n")
    else:
        print("Разметки пула нет — релевантность по одному эталонному чанку\n")

    rows: list[dict] = []
    per_question: dict[str, pd.DataFrame] = {}
    runs: dict[str, dict[str, list[int]]] = {}

    for name, config in build_configs(quick):
        print("=" * 62)
        print(name)
        print("=" * 62)

        queries = english if config["translate"] else russian

        retriever = Retriever(
            table,
            embedder,
            client=client,
            model=model,
            candidate_k=CANDIDATE_K,
            top_k=config["top_k"],
            use_hybrid=config["use_hybrid"],
            use_rerank=config["use_rerank"],
        )

        answers = run_retrieval_only(retriever, dataset, queries, k=config["top_k"])
        runs[name] = {
            str(a.dataset_row_id): [int(c.chunk_id) for c in (a.retrieved_chunks or [])]
            for a in answers
        }
        metrics = evaluate(
            answers=answers,
            dataset=dataset,
            client=client,
            model=model,
            compute_faithfulness=False,  # генерации нет — считать нечего
            qrels=qrels,
        )

        row = {"config": name, "k": config["top_k"]}
        row.update({column: float(metrics[column].mean()) for column in METRIC_COLUMNS})
        row["ceiling"] = float(metrics["recall_ceiling"].mean())
        row["of_max"] = row["recall"] / row["ceiling"] if row["ceiling"] else float("nan")
        rows.append(row)
        per_question[name] = metrics[["id", *METRIC_COLUMNS]].copy()

        print("  " + "  ".join(f"{column}={row[column]:.3f}" for column in METRIC_COLUMNS))
        print(f"  потолок recall={row['ceiling']:.3f}, взято от достижимого={row['of_max']:.1%}")
        print()

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    RUNS_PATH.write_text(json.dumps(runs, ensure_ascii=False), encoding="utf-8")

    print("=" * 62)
    print("СВОДКА")
    print("=" * 62)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()
    if qrels is None:
        print(
            "⚠️ Precision и F1 между строками с разным k напрямую не сравнивать:\n"
            "   у каждого вопроса один эталонный чанк, поэтому precision физически\n"
            "   не превысит 1/k. Для сравнения разных k смотрите recall, MRR и NDCG."
        )
    else:
        print(
            "⚠️ Recall между строками с разным k не сравнивать напрямую: потолок\n"
            "   зависит от k (колонка ceiling). Сопоставимая величина — of_max,\n"
            "   доля достижимого. Реранкинг может вернуть меньше k фрагментов —\n"
            "   тогда у него и потолок ниже, и это видно в той же колонке."
        )
    print()

    # Каждая пара отличается ровно одним приёмом — тогда разницу можно
    # отнести к этому приёму, а не к сумме изменений.
    report_significance(
        per_question,
        [
            ("0. Dense, русский запрос k=5", "1. Dense k=5"),
            ("1. Dense k=5", "2. Hybrid (BM25+RRF) k=5"),
            ("2. Hybrid (BM25+RRF) k=5", "5. Hybrid + LLM-реранкинг k=5"),
        ],
    )

    print(f"\nСохранено → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
