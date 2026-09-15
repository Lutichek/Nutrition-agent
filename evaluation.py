"""Модуль оценки качества RAG-агента.

Retrieval-метрики (Precision@K, Recall@K, F1@K, MRR, MAP, NDCG@K):
    Стандартные IR-формулы на базе бинарного вектора релевантности.
    Релевантность определяется через token overlap (recall-oriented).

Generation-метрика (Faithfulness):
    Через библиотеку RAGAS (ragas.metrics.Faithfulness) — и только через неё.
    Запасной реализации нет намеренно: две реализации одной метрики дают
    разные числа, а в отчёте они выглядят одинаково. Нет RAGAS — нет оценки.

Использование::

    from evaluation import evaluate
    from agent import RAGAgentAnswer

    df_metrics = evaluate(
        answers=agent_answers,           # list[RAGAgentAnswer]
        dataset=df_test,                 # pd.DataFrame (id, question, ground_truth_chunks, document)
        client=openai_client,            # OpenAI client (для faithfulness)
        model="openai/gpt-4o",
    )
"""

from __future__ import annotations

import ast
import logging
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from agent import Chunk, RAGAgentAnswer

logger = logging.getLogger(__name__)

# ============================================================
# 1. Token Overlap Matching
# ============================================================


def _tokenize(text: str) -> set[str]:
    """Простая токенизация: lowercase + split по не-буквенным символам."""
    return set(re.findall(r"[\w]+", text.lower()))


def _token_overlap_recall(retrieved_text: str, ground_truth_text: str) -> float:
    """Доля токенов ground_truth, которые встречаются в retrieved_text."""
    gt_tokens = _tokenize(ground_truth_text)
    if not gt_tokens:
        return 0.0
    ret_tokens = _tokenize(retrieved_text)
    return len(gt_tokens & ret_tokens) / len(gt_tokens)


def _is_chunk_relevant(
    retrieved_text: str,
    ground_truth_chunks: list[str],
    threshold: float = 0.3,
) -> bool:
    """Проверяет, релевантен ли retrieved chunk хотя бы одному ground truth чанку."""
    return any(_token_overlap_recall(retrieved_text, gt) >= threshold for gt in ground_truth_chunks)


def _get_relevance_vector(
    retrieved_chunks: list[Chunk],
    ground_truth_chunks: list[str],
    threshold: float = 0.3,
    expected_year: int | None = None,
) -> list[int]:
    """Бинарный вектор релевантности: 1 если чанк релевантен, 0 иначе.

    Используется жадное one-to-one сопоставление: каждый ground-truth чанк
    может быть «использован» только одним retrieved чанком.  Это гарантирует,
    что recall, MAP, NDCG ≤ 1.
    """
    matched_gt: set[int] = set()  # индексы уже сопоставленных gt-чанков
    result = []
    for chunk in retrieved_chunks:
        if expected_year is not None and chunk.year != expected_year:
            result.append(0)
            continue
        found = False
        for gi, gt in enumerate(ground_truth_chunks):
            if gi in matched_gt:
                continue
            if _token_overlap_recall(chunk.text, gt) >= threshold:
                matched_gt.add(gi)
                found = True
                break
        result.append(int(found))
    return result


# ============================================================
# 2. Retrieval Metrics
# ============================================================


def _precision_at_k(relevance_vector: list[int], k: int | None = None) -> float:
    """Precision@K — доля релевантных среди top-K."""
    if not relevance_vector:
        return 0.0
    vec = relevance_vector[:k] if k else relevance_vector
    return sum(vec) / len(vec)


def _recall_at_k(
    relevance_vector: list[int],
    total_relevant: int,
    k: int | None = None,
) -> float:
    """Recall@K — доля найденных релевантных от общего числа релевантных."""
    if total_relevant == 0:
        return 0.0
    vec = relevance_vector[:k] if k else relevance_vector
    return sum(vec) / total_relevant


def _f1_at_k(
    relevance_vector: list[int],
    total_relevant: int,
    k: int | None = None,
) -> float:
    """F1@K — гармоническое среднее Precision@K и Recall@K."""
    p = _precision_at_k(relevance_vector, k)
    r = _recall_at_k(relevance_vector, total_relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _reciprocal_rank(relevance_vector: list[int]) -> float:
    """Reciprocal Rank — 1/позиция первого релевантного результата."""
    for i, rel in enumerate(relevance_vector):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0


def _average_precision(relevance_vector: list[int], total_relevant: int) -> float:
    """Average Precision для одного запроса."""
    if total_relevant == 0:
        return 0.0
    cumsum = 0
    ap = 0.0
    for i, rel in enumerate(relevance_vector):
        cumsum += rel
        if rel == 1:
            ap += cumsum / (i + 1)
    return ap / total_relevant


def _dcg(relevance_vector: list[int], k: int | None = None) -> float:
    """Discounted Cumulative Gain."""
    vec = relevance_vector[:k] if k else relevance_vector
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(vec))


def _ndcg_at_k(
    relevance_vector: list[int],
    total_relevant: int,
    k: int | None = None,
) -> float:
    """NDCG@K — нормализованный DCG.

    Идеальная выдача обрезается по ТОЙ ЖЕ длине, что и фактическая. Иначе
    метрика сравнивает пятёрку с двадцаткой и занижает всё подряд.

    Наблюдавшийся случай: при пулинговой разметке у вопроса бывает 20
    релевантных чанков, а выдача — пять. Идеальный вектор строился из
    20 единиц, DCG по нему считался по всем двадцати позициям, и выдача
    из пяти релевантных подряд — то есть безупречная — получала 0.419
    вместо 1.0.

    Дефект не проявлялся, пока эталон был один на вопрос: тогда
    total_relevant=1, идеальный вектор укладывался в длину выдачи,
    и нормировка выходила верной. Он появился вместе с пулингом
    и просидел незамеченным два шага — потому что цифры выглядели
    правдоподобно и падали в ту сторону, в которую ожидалось.

    Это та же арифметическая ловушка, что и потолок recall (см.
    ``_recall_ceiling``), только спрятанная внутри нормировки, где
    её не ждёшь.
    """
    length = len(relevance_vector[:k] if k else relevance_vector)
    ideal_vector = [1] * min(total_relevant, length) + [0] * max(0, length - total_relevant)
    ideal = _dcg(ideal_vector, k)
    if ideal == 0:
        return 0.0
    return _dcg(relevance_vector, k) / ideal


# ============================================================
# 3. Faithfulness
# ============================================================

_RAGAS_AVAILABLE: bool | None = None


def _check_ragas() -> bool:
    """Проверить доступность RAGAS."""
    global _RAGAS_AVAILABLE
    if _RAGAS_AVAILABLE is not None:
        return _RAGAS_AVAILABLE
    try:
        from langchain_openai import ChatOpenAI as _C  # noqa: F401
        from ragas.llms import LangchainLLMWrapper as _W  # noqa: F401
        from ragas.metrics import Faithfulness as _F  # noqa: F401

        _RAGAS_AVAILABLE = True
    except Exception:
        _RAGAS_AVAILABLE = False
    return _RAGAS_AVAILABLE


def _compute_faithfulness_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    client: Any,
    model: str,
    max_concurrent: int = 4,
) -> list[float]:
    """Вычислить faithfulness через RAGAS (async concurrent)."""
    import asyncio

    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass

    from langchain_openai import ChatOpenAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    base_url = str(client.base_url).rstrip("/")
    api_key = client.api_key

    langchain_llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    faithfulness_metric = Faithfulness(llm=ragas_llm)

    scores: list[float] = [np.nan] * len(questions)
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(questions), desc="Faithfulness (RAGAS)")

    async def _score_one(i: int) -> None:
        async with semaphore:
            sample = SingleTurnSample(
                user_input=questions[i],
                response=answers[i],
                retrieved_contexts=contexts[i],
            )
            try:
                score = await faithfulness_metric.single_turn_ascore(sample)
                scores[i] = float(score) if not np.isnan(score) else 1.0
            except Exception as exc:
                logger.warning("RAGAS faithfulness failed for sample %d: %s", i, exc)
                scores[i] = np.nan
            finally:
                pbar.update(1)

    async def _run_all() -> None:
        await asyncio.gather(*[_score_one(i) for i in range(len(questions))])

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(_run_all())
        else:
            asyncio.run(_run_all())
    except RuntimeError:
        asyncio.run(_run_all())
    finally:
        pbar.close()

    return scores


# ============================================================
# 4. Вспомогательные функции
# ============================================================


def _parse_ground_truth_chunks(value: Any) -> list[str]:
    """Распарсить ground_truth_chunks из DataFrame (может быть строкой-списком)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            return [value]
    return []


def _zero_row(row_id: str, question: str) -> dict:
    """Строка для вопроса, на котором поиск ничего не дал.

    Метрики поиска здесь честные нули: искали и не нашли.

    А вот faithfulness — NaN, а не ноль. Она проверяет, подтверждается ли
    каждое утверждение ответа найденным контекстом. Нет контекста — нечего
    проверять: агент отказался отвечать, утверждений нет. Ноль здесь читался
    бы как «выдумал», хотя агент сделал ровно то, что должен.

    Замер на сохранённом прогоне: пять таких вопросов из 210 занижали
    faithfulness с 0.941 до 0.919 — два пункта за счёт подсчёта,
    а не за счёт агента.
    """
    return {
        "id": row_id,
        "question": question,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mrr": 0.0,
        "map": 0.0,
        "ndcg": 0.0,
        "faithfulness": np.nan,
        # Потолок недостижим по другой причине: выдачи не было вовсе,
        # поэтому и делить нечего. См. _recall_ceiling.
        "recall_ceiling": np.nan,
    }


def _recall_ceiling(k: int, total_relevant: int) -> float:
    """Максимальный recall@k, достижимый при данной разметке.

    Под пулинговой разметкой у вопроса в среднем 6.2 релевантных чанка,
    а выдача ограничена k=5. Даже идеальный поиск, поставивший наверх
    пять релевантных из шести, даст recall 5/6 = 0.833, а не единицу.

    Без этой величины recall@5 = 0.436 читается как «нашли меньше половины
    нужного», хотя часть разрыва — арифметика, а не качество поиска.
    Осмысленная величина — отношение recall к потолку: какую долю
    достижимого система реально забрала.

    При одном эталоне на вопрос (старая разметка) потолок всегда 1.0,
    поэтому раньше вопрос и не вставал.
    """
    if total_relevant <= 0:
        return float("nan")
    return min(k, total_relevant) / total_relevant


# ============================================================
# 5. Главная функция оценки
# ============================================================


def evaluate(
    answers: list[RAGAgentAnswer],
    dataset: pd.DataFrame,
    client: Any = None,
    model: str = "openai/gpt-4o",
    overlap_threshold: float = 0.3,
    compute_faithfulness: bool = True,
    gt_column: str = "relevant_text",
    max_workers: int = 4,
    qrels: dict[str, set[int]] | None = None,
) -> pd.DataFrame:
    """Оценить качество RAG-агента.

    Parameters
    ----------
    answers:
        Список ответов агента (RAGAgentAnswer).
    dataset:
        DataFrame с колонками: ``id``, ``question``, ``<gt_column>``, ``document``.
    client:
        OpenAI-совместимый клиент (нужен для faithfulness).
    model:
        Модель для faithfulness-оценки.
    overlap_threshold:
        Порог token overlap для определения релевантности чанка (по умолчанию 0.3).
    compute_faithfulness:
        Считать ли faithfulness (требует LLM-вызовов).
    gt_column:
        Название колонки с ground truth чанками (по умолчанию ``relevant_text``).
    max_workers:
        Количество потоков для параллельного вычисления faithfulness.

    Returns
    -------
    pd.DataFrame
        Таблица с колонками: ``id``, ``question``, ``precision``, ``recall``,
        ``f1``, ``mrr``, ``map``, ``ndcg``, ``faithfulness``.
    """
    # Индекс: dataset_row_id -> RAGAgentAnswer
    answer_map: dict[str, RAGAgentAnswer] = {}
    for ans in answers:
        if ans.dataset_row_id is not None:
            answer_map[str(ans.dataset_row_id)] = ans

    rows: list[dict] = []

    # Для batch-вычисления faithfulness
    faith_indices: list[int] = []
    faith_questions: list[str] = []
    faith_answers: list[str] = []
    faith_contexts: list[list[str]] = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Retrieval metrics"):
        row_id = str(row["id"])
        question = str(row["question"])
        gt_chunks = _parse_ground_truth_chunks(row[gt_column])
        expected_year = int(row["document"]) if "document" in row.index else None
        total_relevant = len(gt_chunks)

        ans = answer_map.get(row_id)

        # Нет ответа или нет чанков → нули
        if ans is None or not ans.retrieved_chunks:
            rows.append(_zero_row(row_id, question))
            continue

        # Вектор релевантности
        if qrels is not None:
            # Множественный эталон (пулинг, методика TREC): релевантным считается
            # ЛЮБОЙ размеченный чанк, а не один-единственный «правильный».
            #
            # Зачем: при одном эталоне на вопрос корпус сам себе вредит — чем
            # больше статей по теме, тем чаще верный ответ приходит не из той
            # статьи, которую разметка считает единственно верной. Замерено:
            # из 13 «промахов» в 10 случаях агент дал содержательный ответ.
            relevant_ids = qrels.get(row_id, set())
            rel_vec = [int(chunk.chunk_id in relevant_ids) for chunk in ans.retrieved_chunks]
            total_relevant = len(relevant_ids)
        else:
            rel_vec = _get_relevance_vector(
                ans.retrieved_chunks, gt_chunks, overlap_threshold, expected_year
            )

        metrics_row = {
            "id": row_id,
            "question": question,
            "precision": _precision_at_k(rel_vec),
            "recall": _recall_at_k(rel_vec, total_relevant),
            "f1": _f1_at_k(rel_vec, total_relevant),
            "mrr": _reciprocal_rank(rel_vec),
            "map": _average_precision(rel_vec, total_relevant),
            "ndcg": _ndcg_at_k(rel_vec, total_relevant),
            "faithfulness": np.nan,
            "recall_ceiling": _recall_ceiling(len(rel_vec), total_relevant),
        }
        rows.append(metrics_row)

        # Собираем данные для faithfulness
        if compute_faithfulness and ans.retrieved_chunks:
            faith_indices.append(len(rows) - 1)
            faith_questions.append(question)
            faith_answers.append(ans.answer)
            faith_contexts.append([c.text for c in ans.retrieved_chunks])

    # --- Faithfulness ---
    #
    # Реализация ровно одна — RAGAS. Раньше рядом жила своя, «ручная»,
    # и код молча переключался на неё, если RAGAS не поставился или упал.
    # Это давало худший из возможных отказов: цифра в отчёте есть, выглядит
    # нормально, но посчитана другим методом — RAGAS раскладывает ответ
    # на утверждения и проверяет каждое, ручная версия просила модель сделать
    # то же самое одним запросом. Числа получались разные, а в отчёте
    # различить их было нельзя.
    #
    # Теперь нет оценки — значит нет: NaN и громкое предупреждение.
    if compute_faithfulness and faith_indices and client is not None:
        if not _check_ragas():
            print(
                "⚠️ RAGAS недоступен — faithfulness НЕ посчитана.\n"
                "   Установите: uv sync --extra dev. Подменять её другой метрикой нельзя:\n"
                "   цифры двух методов не сравнимы между собой."
            )
        else:
            try:
                scores = _compute_faithfulness_ragas(
                    faith_questions, faith_answers, faith_contexts, client, model, max_workers
                )
                for idx, score in zip(faith_indices, scores, strict=True):
                    # NaN оставляем NaN: это сбой оценки, а не оценка «ноль».
                    rows[idx]["faithfulness"] = score
            except Exception as exc:
                logger.warning("RAGAS faithfulness failed: %s", exc)
                print(f"⚠️ Faithfulness не посчитана: {exc}")

    df_result = pd.DataFrame(rows)

    # NaN НЕ заполняется нулями — это принципиально.
    #
    # Раньше здесь стоял fillna(0.0), и сбой провайдера выглядел как обвал
    # качества агента. Ровно на этом уже обожглись с correctness: при недоступном
    # провайдере метрика показала 0.498 вместо 0.869, хотя агент не менялся.
    # NaN исключается из среднего, а число пропусков печатается — так сбой
    # инфраструктуры видно как сбой, а не как регрессию.
    missing = int(df_result["faithfulness"].isna().sum())
    if missing and compute_faithfulness:
        share = missing / len(df_result)
        print(
            f"⚠️ Faithfulness не посчитана для {missing} из {len(df_result)} "
            f"({share:.1%}) — нет контекста либо сбой оценки; исключены из средних"
        )
        if share > 0.2:
            print("   Больше пятой части оценок потеряно: прогон надо повторить")

    return df_result
