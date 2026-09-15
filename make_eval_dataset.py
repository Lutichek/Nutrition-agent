"""
Генерация тестового датасета для оценки поиска (synthetic eval dataset).

Идея: берём реальные чанки из индекса и просим модель придумать к каждому
вопрос, ответ на который содержится ИМЕННО в этом чанке. Тогда чанк-источник
и есть ground truth — с ним сравнивается то, что нашёл ретривер.

⚠️ Ключевое решение: чанки английские, а **вопросы генерируются русские**.
Так и приходит настоящий пользователь, и так меряется весь путь целиком,
включая перевод запроса в узле ``rewrite``. Если генерировать вопросы
по-английски, метрики получатся красивее, но измерять будут не тот сценарий,
который работает в проде.

Негативные вопросы (``--negative``) — отдельный файл: правдоподобные вопросы
о питании, ответа на которые в корпусе нет. На них агент обязан отказаться,
а не сочинить. Без этой части метрики поощряют болтливость.

Результат — ``_research/dataset/test_dataframe.xlsx`` с колонками, которые ждёт
``evaluation.py``:

    id            — идентификатор вопроса
    question      — вопрос на русском
    relevant_text    — эталонный фрагмент (ground truth для поиска)
    document         — doc_id статьи-источника
    reference_answer — эталонный ответ по этому фрагменту (для correctness)

Запуск::

    python make_eval_dataset.py                # 80 вопросов + эталонные ответы
    python make_eval_dataset.py --n 40
    python make_eval_dataset.py --references   # дописать эталоны в готовый датасет
    python make_eval_dataset.py --negative     # вопросы без ответа в корпусе
"""

from __future__ import annotations

import json
import random
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from lance_db import open_table, read_all
from measurement import RESEARCH_DIR
from providers import get_clients

HERE = Path(__file__).parent

# Выгрузки замеров живут в _research/dataset/ — в репозиторий не идут,
# пересобираются скриптами. В репозитории только журнал EVOLUTION.md.
DATASET_DIR = RESEARCH_DIR
DATASET_PATH = DATASET_DIR / "test_dataframe.xlsx"
NEGATIVE_PATH = DATASET_DIR / "negative_questions.xlsx"

DEFAULT_N = 80

# Чанк короче этого не содержит самостоятельного факта — вопрос по нему
# получится либо тривиальным, либо неотвечаемым.
MIN_CHUNK_CHARS = 400

QUESTION_PROMPT = """Ниже — фрагмент научного абстракта по нутрициологии на английском языке.

Придумай ОДИН вопрос НА РУССКОМ ЯЗЫКЕ, ответ на который содержится
именно в этом фрагменте.

Требования к вопросу:
- на русском языке, как его задал бы обычный человек, а не учёный
- конкретный: на него нельзя ответить общими словами
- самодостаточный: понятен без чтения фрагмента
- НЕ используй слова «в исследовании», «в статье», «согласно фрагменту»
- одно предложение

Фрагмент:
{chunk}

Верни ТОЛЬКО текст вопроса."""

REFERENCE_PROMPT = """Ответь на вопрос, опираясь ТОЛЬКО на приведённый фрагмент
научного абстракта.

Вопрос: {question}

Фрагмент:
{chunk}

Требования:
- отвечай НА РУССКОМ, 2-4 предложения
- только факты из фрагмента, ничего от себя
- если во фрагменте есть числа, приведи их
- без вводных оборотов вроде «согласно фрагменту»

Верни ТОЛЬКО текст ответа."""

NEGATIVE_PROMPT = """Придумай {n} правдоподобных вопросов о питании НА РУССКОМ ЯЗЫКЕ,
ответа на которые заведомо НЕТ в научной литературе по нутрициологии.

Это должны быть вопросы, на которые добросовестный ассистент обязан ответить
«не знаю», а не сочинить ответ. Например: псевдонаучные утверждения, вопросы
о несуществующих веществах, вопросы не о питании вовсе.

Верни ТОЛЬКО JSON: {{"questions": ["вопрос 1", "вопрос 2", ...]}}"""


def _complete(client, model: str, prompt: str, max_retries: int = 3) -> str | None:
    """Вызов модели с повторами."""
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            continue
    return None


def generate_questions(
    n: int, seed: int = 0, exclude_documents: set[int] | None = None
) -> pd.DataFrame:
    """Сгенерировать вопросы по случайным чанкам индекса.

    Args:
        exclude_documents: статьи, по которым вопросы уже есть. Нужно, чтобы
            дописывать вопросы в существующий датасет, а не начинать заново:
            иначе теряются эталонные ответы и разметка пула.
    """
    client, model, _ = get_clients("polza")

    table = open_table("lance_db/vectorstore", "chunks")
    corpus = read_all(table)
    usable = corpus[corpus["text"].str.len() >= MIN_CHUNK_CHARS]

    if exclude_documents:
        usable = usable[~usable["doc_id"].isin(exclude_documents)]
        print(f"Исключено статей, по которым вопросы уже есть: {len(exclude_documents)}")

    print(f"Чанков в индексе: {len(corpus)}, годных для вопроса: {len(usable)}")

    # Не больше одного вопроса на статью: иначе датасет перекосится
    # в сторону длинных абстрактов, у которых чанков больше.
    rng = random.Random(seed)
    by_document = usable.groupby("doc_id").apply(
        lambda group: group.iloc[rng.randrange(len(group))], include_groups=False
    )
    sample = by_document.sample(n=min(n, len(by_document)), random_state=seed)

    rows: list[dict] = []

    def work(record: dict) -> dict | None:
        question = _complete(client, model, QUESTION_PROMPT.format(chunk=record["text"]))
        if not question or len(question) < 15:
            return None
        return {
            "id": str(uuid.uuid4())[:8],
            "question": question.strip().strip('"'),
            "relevant_text": json.dumps([record["text"]], ensure_ascii=False),
            "document": int(record["doc_id"]),
            "chunk_id": int(record["chunk_id"]),
        }

    records = sample.reset_index().to_dict("records")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(work, record) for record in records]
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result:
                rows.append(result)
            if number % 20 == 0:
                print(f"  сгенерировано {number}/{len(records)}")

    return pd.DataFrame(rows)


def add_reference_answers(dataset: pd.DataFrame) -> pd.DataFrame:
    """Дописать в датасет эталонные ответы (колонка ``reference_answer``).

    Эталон строится по тому же чанку, из которого сгенерирован вопрос. Это
    даёт метрике ``answer_correctness`` то, с чем сравнивать: ответ агента
    против ответа, заведомо выведенного из нужного фрагмента.

    Важно, чем этот эталон НЕ является: это не «единственно верный ответ»,
    а верный ответ по одному конкретному источнику. Агент мог найти другую
    статью и ответить не хуже — судья в ``evaluation_answers`` поэтому и сверяет
    факты, а не формулировки.
    """
    client, model, _ = get_clients("polza")

    def work(row: dict) -> str:
        chunks = json.loads(row["relevant_text"])
        chunk = chunks[0] if chunks else ""
        answer = _complete(client, model, REFERENCE_PROMPT.format(
            question=row["question"], chunk=chunk
        ))
        return (answer or "").strip()

    records = dataset.to_dict("records")
    with ThreadPoolExecutor(max_workers=6) as pool:
        references = list(pool.map(work, records))

    dataset = dataset.copy()
    dataset["reference_answer"] = references

    empty = sum(1 for reference in references if not reference)
    if empty:
        print(f"  ⚠️ не удалось получить эталон для {empty} вопросов")

    return dataset


def generate_negative(n: int = 60, batches: int = 3) -> pd.DataFrame:
    """Сгенерировать вопросы, ответа на которые в корпусе нет.

    Генерируем несколькими заходами и складываем: за один вызов модель выдаёт
    вопросы в одном стиле, и набор получается однообразным. Дубликаты убираем.

    Размер набора важнее, чем кажется: на 25 вопросах разница между
    конфигурациями в 12% — это три вопроса, то есть почти шум. Чтобы
    сравнивать варианты ``grade``, нужен набор побольше.
    """
    client, model, _ = get_clients("polza")
    per_batch = max(n // batches, 1)

    questions: list[str] = []
    for _ in range(batches):
        raw = _complete(client, model, NEGATIVE_PROMPT.format(n=per_batch))
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            questions.extend(json.loads(text).get("questions", []))
        except json.JSONDecodeError:
            continue

    unique: list[str] = []
    seen: set[str] = set()
    for question in questions:
        key = question.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(question.strip())

    return pd.DataFrame(
        [{"id": str(uuid.uuid4())[:8], "question": question} for question in unique]
    )


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if "--negative" in sys.argv:
        frame = generate_negative()
        frame.to_excel(NEGATIVE_PATH, index=False)
        print(f"Негативных вопросов: {len(frame)} → {NEGATIVE_PATH}")
        return

    if "--add" in sys.argv:
        # Дописать вопросов к существующему датасету, не трогая старые:
        # у них уже есть эталонные ответы, а скоро будет и разметка пула.
        count = int(sys.argv[sys.argv.index("--add") + 1])
        existing = pd.read_excel(DATASET_PATH)
        print(f"В датасете уже {len(existing)} вопросов, добавляю {count}")

        fresh = generate_questions(
            count, seed=42, exclude_documents=set(existing["document"].astype(int))
        )
        print(f"\nГенерирую эталонные ответы для {len(fresh)} новых вопросов...")
        fresh = add_reference_answers(fresh)

        merged = pd.concat([existing, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset="question").reset_index(drop=True)
        merged.to_excel(DATASET_PATH, index=False)

        print(f"\nСтало вопросов: {len(merged)}, статей: {merged['document'].nunique()}")
        print(f"Сохранено → {DATASET_PATH}")
        return

    if "--references" in sys.argv:
        # Дописать эталонные ответы в уже готовый датасет — чтобы не
        # перегенерировать вопросы ради одной новой колонки.
        frame = pd.read_excel(DATASET_PATH)
        print(f"Генерирую эталонные ответы для {len(frame)} вопросов...")
        frame = add_reference_answers(frame)
        frame.to_excel(DATASET_PATH, index=False)
        print(f"Готово → {DATASET_PATH}\n")
        print(frame[["question", "reference_answer"]].head(3).to_string(index=False))
        return

    n = DEFAULT_N
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    frame = generate_questions(n)
    print(f"\nГенерирую эталонные ответы для {len(frame)} вопросов...")
    frame = add_reference_answers(frame)
    frame.to_excel(DATASET_PATH, index=False)

    print(f"\nВопросов: {len(frame)}, статей-источников: {frame['document'].nunique()}")
    print(f"Сохранено → {DATASET_PATH}\n")
    print(frame[["id", "question"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
