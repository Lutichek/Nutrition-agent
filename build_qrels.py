"""
Пулинг релевантности: множественный эталон вместо одного «правильного» чанка.

Проблема, ради которой всё затевалось. В датасете у каждого вопроса ровно один
эталонный чанк — тот, из которого вопрос сгенерирован. Корпус тематический,
поэтому на вопрос про белок и ожирение отвечают ещё пять статей, и ретривер
их находит. Метрика считает это промахом. Замерено прямо: из 13 «промахов»
в 10 случаях агент дал содержательный ответ — просто из соседней статьи.

Расширение корпуса (682 → 1604 статьи) проблему усилило: чем больше статей
по теме, тем чаще верный ответ приходит «не из той».

Решение — методика TREC (pooled relevance judgments):

1. **Пул.** Для каждого вопроса собираем ОБЪЕДИНЕНИЕ того, что выдали разные
   конфигурации поиска. Смысл в том, что размечать весь корпус невозможно,
   а всё, что хоть одна конфигурация подняла наверх, — размечать обязательно:
   именно эти чанки и будут встречаться в сравнениях.
2. **Разметка.** Каждую пару «вопрос — чанк» судья оценивает один раз.
3. **Переиспользование.** Разметка сохраняется в ``_research/dataset/qrels.csv``
   и служит всем будущим экспериментам, не требуя новых вызовов LLM.

⚠️ Известное ограничение метода: конфигурация, появившаяся ПОСЛЕ разметки,
может найти релевантный чанк, которого нет в пуле, и он засчитается промахом.
В TREC с этим живут, добавляя новые прогоны в пул. Здесь так же: при
существенно новой стратегии поиска пул надо дополнить.

Запуск::

    python build_qrels.py              # собрать пул и разметить
    python build_qrels.py --pool-only  # только собрать пул, без разметки
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from embedder import Embedder
from evaluation_answers import _judge
from lance_db import open_table
from make_eval_dataset import DATASET_PATH
from measurement import RESEARCH_DIR, load_queries_en
from providers import get_clients
from retriever import Retriever

HERE = Path(__file__).parent

# Выгрузки замеров живут в _research/dataset/ — в репозиторий не идут,
# пересобираются скриптами. В репозитории только журнал EVOLUTION.md.
QRELS_PATH = RESEARCH_DIR / "qrels.csv"

# Конфигурации, чьи выдачи попадают в пул. Берём с запасом по k: чем шире пул,
# тем меньше шанс, что будущая конфигурация найдёт неразмеченный чанк.
POOL_CONFIGS = [
    {"use_hybrid": False, "use_rerank": False},
    {"use_hybrid": True, "use_rerank": False},
    {"use_hybrid": True, "use_rerank": True},
]
POOL_K = 10

# Сколько чанков отдаём судье за один вызов. Батчинг снижает число запросов
# с тысяч до сотен; больше десяти в одном промпте — модель начинает путать
# нумерацию.
JUDGE_BATCH = 10

JUDGE_PROMPT = """Оцени, какие из фрагментов содержат информацию, отвечающую на вопрос.

Вопрос: {question}

Фрагменты:
{context}

Фрагмент РЕЛЕВАНТЕН, если из него можно извлечь ответ на заданный вопрос
целиком или частично — конкретные сведения, а не общие рассуждения по теме.

Фрагмент НЕ релевантен, если он лишь о той же области (питание, здоровье),
но заданного вопроса не касается.

Верни ТОЛЬКО JSON: {{"relevant": [номера релевантных фрагментов]}}
Нумерация начинается с 1. Если релевантных нет — пустой список."""


def build_pool(dataset: pd.DataFrame, queries: list[str], client, model, embed_client) -> pd.DataFrame:
    """Собрать объединение выдач всех конфигураций поиска."""
    table = open_table("lance_db/vectorstore", "chunks")
    embedder = Embedder(embed_client)

    retrievers = [
        Retriever(table, embedder, client=client, model=model, top_k=POOL_K,
                  candidate_k=40, **config)
        for config in POOL_CONFIGS
    ]

    rows: list[dict] = []

    def work(item: tuple[str, str, str]) -> list[dict]:
        row_id, question, query = item
        seen: dict[int, dict] = {}
        for retriever in retrievers:
            for hit in retriever.retrieve(query, k=POOL_K):
                seen.setdefault(
                    int(hit["chunk_id"]),
                    {
                        "id": row_id,
                        "question": question,
                        "chunk_id": int(hit["chunk_id"]),
                        "doc_id": int(hit["doc_id"]),
                        "text": hit["text"],
                    },
                )
        return list(seen.values())

    items = list(zip(dataset["id"].astype(str), dataset["question"].astype(str), queries, strict=True))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in pool.map(work, items):
            rows.extend(batch)

    frame = pd.DataFrame(rows)
    per_question = frame.groupby("id").size()
    print(
        f"Пул собран: {len(frame)} пар «вопрос-чанк», "
        f"в среднем {per_question.mean():.1f} на вопрос "
        f"(мин {per_question.min()}, макс {per_question.max()})"
    )
    return frame


def judge_pool(pool: pd.DataFrame, client, model, max_workers: int = 6) -> pd.DataFrame:
    """Разметить пул: релевантен ли чанк вопросу."""
    tasks: list[tuple[str, list[dict]]] = []
    for row_id, group in pool.groupby("id"):
        records = group.to_dict("records")
        for start in range(0, len(records), JUDGE_BATCH):
            tasks.append((str(row_id), records[start : start + JUDGE_BATCH]))

    print(f"Разметка: {len(tasks)} вызовов судьи на {len(pool)} пар")
    started = time.perf_counter()
    done = 0

    def work(task: tuple[str, list[dict]]) -> list[dict]:
        _row_id, records = task
        context = "\n\n".join(
            f"Фрагмент {number}:\n{record['text'][:1200]}"
            for number, record in enumerate(records, start=1)
        )
        verdict = _judge(
            client, model,
            JUDGE_PROMPT.format(question=records[0]["question"], context=context),
        )
        relevant = verdict.get("relevant") or []
        indices = {
            int(number) - 1 for number in relevant if isinstance(number, (int, float))
        }
        return [
            {
                "id": record["id"],
                "chunk_id": record["chunk_id"],
                "doc_id": record["doc_id"],
                "relevant": int(position in indices),
            }
            for position, record in enumerate(records)
        ]

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool_exec:
        for batch in pool_exec.map(work, tasks):
            results.extend(batch)
            done += 1
            if done % 40 == 0:
                print(f"  размечено вызовов {done}/{len(tasks)}")

    print(f"Разметка заняла {time.perf_counter() - started:.0f} с")
    return pd.DataFrame(results)


def load_qrels() -> dict[str, set[int]]:
    """Прочитать разметку: id вопроса → множество релевантных chunk_id."""
    if not QRELS_PATH.exists():
        raise FileNotFoundError(
            f"Нет разметки {QRELS_PATH.name}. Выполните: python build_qrels.py"
        )
    frame = pd.read_csv(QRELS_PATH)
    relevant = frame[frame["relevant"] == 1]
    return {
        str(row_id): set(group["chunk_id"].astype(int))
        for row_id, group in relevant.groupby("id")
    }


def main() -> None:
    if not DATASET_PATH.exists():
        print("Нет датасета. Выполните: python make_eval_dataset.py")
        return

    dataset = pd.read_excel(DATASET_PATH)
    print(f"Вопросов в датасете: {len(dataset)}\n")

    client, model, embed_client = get_clients("polza")
    queries = load_queries_en(dataset["question"].astype(str).tolist(), client, model)

    pool = build_pool(dataset, queries, client, model, embed_client)

    if "--pool-only" in sys.argv:
        return

    print()
    qrels = judge_pool(pool, client, model)
    qrels = qrels.drop_duplicates(subset=["id", "chunk_id"])
    qrels.to_csv(QRELS_PATH, index=False, encoding="utf-8")

    per_question = qrels[qrels["relevant"] == 1].groupby("id").size()
    covered = qrels["id"].nunique()

    print()
    print("=" * 62)
    print("РАЗМЕТКА ГОТОВА")
    print("=" * 62)
    print(f"  пар размечено:                {len(qrels)}")
    print(f"  признано релевантными:        {int(qrels['relevant'].sum())} "
          f"({qrels['relevant'].mean():.1%})")
    print(f"  релевантных чанков на вопрос: {per_question.mean():.1f} "
          f"(медиана {per_question.median():.0f}, макс {per_question.max()})")
    print(f"  вопросов без единого релевантного чанка: {covered - len(per_question)}")
    print()
    print("Для сравнения: старая разметка давала ровно 1 эталонный чанк на вопрос.")
    print(f"\nСохранено → {QRELS_PATH}")


if __name__ == "__main__":
    main()
