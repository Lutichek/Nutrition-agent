"""
Метрики ответа: прав ли агент и честно ли молчит.

Разделение с ``evaluation.py`` проходит по тому, ЧТО меряется, а не по тому,
что было раньше написано. Там — метрики ПОИСКА: нашёл ли ретривер нужный
фрагмент и не противоречит ли ответ найденному. Здесь — метрики ОТВЕТА,
и они требуют судьи-модели, потому что проверяются не совпадением строк.

Два вопроса, на которые отвечает этот модуль:

    ПРАВ ЛИ агент?   и   ЧЕСТНО ЛИ он молчит, когда не знает?

Здесь добавлены:

* ``answer_correctness`` — совпадает ли ответ с эталонным по фактам.
  Отличается от faithfulness: можно добросовестно процитировать
  нерелевантный фрагмент — faithfulness будет высоким, а ответ неверным.
* ``answer_relevancy``   — отвечает ли ответ на заданный вопрос
  (а не на соседний).
* ``refusal_accuracy``   — на вопросах, ответа на которые в базе нет,
  агент должен отказаться, а не выдумывать.

Использование::

    from evaluation_answers import evaluate_answers, evaluate_refusals

    df = evaluate_answers(answers, dataset, client, model)
    ref = evaluate_refusals(negative_answers, client, model)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

# ────────────────────────────────────────────────────────────
# Промпты судьи
# ────────────────────────────────────────────────────────────

CORRECTNESS_PROMPT = """Ты — строгий эксперт по нутрициологии, проверяющий ответ справочной системы.

Вопрос: {question}

ЭТАЛОННЫЙ ответ (считается верным):
{reference}

ОТВЕТ СИСТЕМЫ:
{answer}

Оцени фактическое совпадение ответа системы с эталонным.

Шкала:
1.0 — все ключевые факты эталона присутствуют, противоречий нет
0.5 — часть ключевых фактов есть, часть упущена, противоречий нет
0.0 — ключевые факты отсутствуют ИЛИ есть фактические противоречия эталону

Формулировки могут отличаться — важны факты, а не слова.

⚠️ Оценивай ТОЛЬКО полноту и верность относительно эталона. Подробности,
которых в эталоне нет, — это НЕ повод снижать оценку: эталон построен
по одному источнику, а система могла найти другой, не менее верный.
Снижай оценку за пропуск фактов эталона или за противоречие им,
но никогда — за лишние сведения.

Верни ТОЛЬКО JSON: {{"score": 1.0 / 0.5 / 0.0, "reason": "краткое обоснование"}}"""

RELEVANCY_PROMPT = """Оцени, отвечает ли текст на заданный вопрос.

Вопрос: {question}

Ответ системы:
{answer}

Шкала:
1.0 — прямо и по существу отвечает на вопрос
0.5 — отвечает частично или с большим количеством лишнего
0.0 — не отвечает на вопрос (говорит о другом либо уходит от ответа)

Оценивай только соответствие вопросу, а не фактическую верность.

Верни ТОЛЬКО JSON: {{"score": 1.0 / 0.5 / 0.0, "reason": "краткое обоснование"}}"""

REFUSAL_PROMPT = """Твоя задача — определить, ОТКАЗАЛАСЬ ли система отвечать.

⚠️ Ты оцениваешь ТЕКСТ ОТВЕТА, а не правдивость вопроса. Не рассуждай о том,
есть ли в науке данные по теме, — это уже известно, их нет. Смотри только
на то, что система написала.

Вопрос: {question}

Ответ системы:
{answer}

refused = true — если система сообщила, что таких сведений нет / не может
    ответить / направила к специалисту И НЕ дала содержательного ответа
    по существу вопроса.

refused = false — если система изложила содержательный ответ: перечислила
    продукты, механизмы, эффекты, привела данные исследований. Это остаётся
    false, даже если ответ выглядит разумным и даже если система при этом
    подменила вопрос соседней темой, о которой данные действительно есть.

Проверь себя: если из ответа можно выписать хотя бы одно утверждение
по существу заданного вопроса — значит refused = false.

Верни ТОЛЬКО JSON: {{"refused": true/false, "reason": "что именно сделала система"}}"""


# ────────────────────────────────────────────────────────────
# Вызов судьи
# ────────────────────────────────────────────────────────────


def _judge(client: Any, model: str, prompt: str, max_retries: int = 4) -> dict:
    """Один запрос к модели-судье со строгим JSON и повторами при сбоях."""
    from agent import parse_json_response

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return parse_json_response(response.choices[0].message.content or "{}")
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)

    raise RuntimeError(f"судья не ответил: {last_error}")


def _run_parallel(tasks: list, worker, desc: str, max_workers: int = 4) -> list:
    """Выполнить задачи параллельно с прогресс-баром, не падая на отдельных сбоях."""
    results: list = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, i, task): i for i, task in enumerate(tasks)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception:
                results[index] = None

    return results


# ────────────────────────────────────────────────────────────
# Метрики по отвечаемым вопросам
# ────────────────────────────────────────────────────────────


def evaluate_answers(
    answers: list,
    dataset: pd.DataFrame,
    client: Any,
    model: str,
    max_workers: int = 4,
    with_relevancy: bool = True,
) -> pd.DataFrame:
    """Посчитать answer_correctness и answer_relevancy.

    Args:
        answers: список RAGAgentAnswer.
        dataset: датасет с колонками id, question, reference_answer.
        client: клиент LLM-судьи.
        model: модель судьи.

    Returns:
        DataFrame: id, question, correctness, relevancy, correctness_reason.
    """
    answer_by_id = {str(a.dataset_row_id): a for a in answers if a.dataset_row_id is not None}

    tasks = []
    for _, row in dataset.iterrows():
        row_id = str(row["id"])
        agent_answer = answer_by_id.get(row_id)
        tasks.append(
            {
                "id": row_id,
                "question": str(row["question"]),
                "reference": str(row.get("reference_answer", "") or ""),
                "answer": agent_answer.answer if agent_answer else "",
            }
        )

    def worker(_index: int, task: dict) -> dict:
        # Нет ответа или нет эталона — считать нечего.
        if not task["answer"] or not task["reference"]:
            return {**task, "correctness": 0.0, "relevancy": 0.0, "correctness_reason": "нет ответа/эталона"}

        correctness = _judge(
            client, model,
            CORRECTNESS_PROMPT.format(
                question=task["question"], reference=task["reference"], answer=task["answer"][:4000]
            ),
        )
        # Relevancy насыщена (0.95-0.97) и совпадает у трёх независимых
        # судей — на итерациях её можно не считать вовсе. Не «считать на
        # выборке поменьше»: метрика, посчитанная на другом числе вопросов,
        # чем соседние, рано или поздно будет сравнена с ними как равная.
        # Либо весь прогон, либо ничего.
        relevancy = (
            _judge(
                client, model,
                RELEVANCY_PROMPT.format(question=task["question"], answer=task["answer"][:4000]),
            )
            if with_relevancy
            else {}
        )

        # Пустой ответ судьи (все ретраи не прошли) — это НЕ оценка «ноль»,
        # это отсутствие оценки. Разница принципиальная: см. комментарий ниже.
        return {
            **task,
            "correctness": float(correctness["score"]) if "score" in correctness else float("nan"),
            "relevancy": float(relevancy["score"]) if "score" in relevancy else float("nan"),
            "correctness_reason": str(correctness.get("reason", "судья не ответил"))[:200],
        }

    results = _run_parallel(tasks, worker, "Correctness / Relevancy", max_workers)

    rows = []
    for task, result in zip(tasks, results, strict=True):
        if result is None:
            # Сбой судьи — это ПРОПУСК, а не оценка «ноль».
            #
            # Раньше здесь стоял 0.0, и это оказалось опасно: при недоступности
            # провайдера (503) все вызовы падали, метрика показывала 0.498
            # вместо 0.869, и выглядело это как обвал качества агента,
            # хотя агент вообще не менялся. Тот же класс тихого отказа,
            # что и ловушка с nutrient_nbr: цифра есть, а смысла у неё нет.
            #
            # NaN исключается из среднего, а число пропусков печатается ниже —
            # так сбой инфраструктуры видно как сбой, а не как регрессию.
            rows.append({"id": task["id"], "question": task["question"],
                         "correctness": float("nan"), "relevancy": float("nan"),
                         "correctness_reason": "сбой судьи"})
        else:
            rows.append({k: result[k] for k in ("id", "question", "correctness", "relevancy", "correctness_reason")})

    frame = pd.DataFrame(rows)

    failed = int(frame["correctness"].isna().sum())
    if failed:
        share = failed / len(frame)
        print(f"⚠️ Судья не ответил на {failed} из {len(frame)} ({share:.1%}) — исключены из средних")
        if share > 0.2:
            print("   Больше пятой части оценок потеряно: цифрам верить нельзя, прогон надо повторить")

    return frame


# ────────────────────────────────────────────────────────────
# Метрика по вопросам без ответа в базе
# ────────────────────────────────────────────────────────────


def evaluate_refusals(
    answers: list,
    dataset: pd.DataFrame,
    client: Any,
    model: str,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Проверить, честно ли агент отказывается отвечать, когда в базе нет ответа.

    Returns:
        DataFrame: id, question, refused (bool), answer, reason.
    """
    answer_by_id = {str(a.dataset_row_id): a for a in answers if a.dataset_row_id is not None}

    tasks = []
    for _, row in dataset.iterrows():
        row_id = str(row["id"])
        agent_answer = answer_by_id.get(row_id)
        tasks.append(
            {
                "id": row_id,
                "question": str(row["question"]),
                "answer": agent_answer.answer if agent_answer else "",
            }
        )

    def worker(_index: int, task: dict) -> dict:
        if not task["answer"]:
            return {**task, "refused": False, "reason": "нет ответа"}

        verdict = _judge(
            client, model,
            REFUSAL_PROMPT.format(question=task["question"], answer=task["answer"][:4000]),
        )
        return {
            **task,
            "refused": bool(verdict.get("refused", False)),
            "reason": str(verdict.get("reason", ""))[:200],
        }

    results = _run_parallel(tasks, worker, "Refusal", max_workers)

    rows = []
    for task, result in zip(tasks, results, strict=True):
        base = {"id": task["id"], "question": task["question"], "answer": task["answer"][:300]}
        if result is None:
            # None, а НЕ False. Разница принципиальная: False означает
            # «агент не отказался», то есть провал агента. Сбой судьи —
            # это отсутствие вердикта, и решать за него нельзя.
            #
            # Раньше здесь стоял False, и из-за этого запасной путь в run_eval
            # (взять оценку по маркерам, если судья молчит) не срабатывал
            # никогда: fillna не видит False. Недоступный провайдер выглядел
            # бы как агент, разучившийся отказываться.
            rows.append({**base, "refused": None, "reason": "сбой судьи"})
        else:
            rows.append({**base, "refused": result["refused"], "reason": result["reason"]})

    frame = pd.DataFrame(rows)

    failed = int(frame["refused"].isna().sum())
    if failed:
        print(f"⚠️ Судья не ответил на {failed} из {len(frame)} — оценка по маркерам")

    return frame


# ────────────────────────────────────────────────────────────
# Доверительный интервал
# ────────────────────────────────────────────────────────────


def confidence_interval(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Доверительный интервал среднего (нормальное приближение).

    Нужен, чтобы отличать реальное улучшение от случайного разброса:
    на 59 вопросах один вопрос — это уже ~1.7% метрики.
    """
    import math

    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))

    mean = float(values.mean())
    std_error = float(values.std(ddof=1)) / math.sqrt(n)
    z = 1.96 if confidence == 0.95 else 2.576
    margin = z * std_error

    return (max(0.0, mean - margin), min(1.0, mean + margin))
