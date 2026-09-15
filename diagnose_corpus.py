"""
Диагностика покрытия корпуса: на какие живые вопросы агент вообще может ответить.

Зачем нужен отдельный инструмент. Тестовый датасет (`make_eval_dataset.py`)
собран из чанков самого корпуса, поэтому ответ на любой его вопрос там есть
**по построению**. Такой датасет меряет поиск и генерацию, но принципиально
не способен обнаружить, что в корпусе чего-то нет. Метрика `recall=0` там
означает «поиск промахнулся», а не «темы нет».

Здесь наоборот: вопросы придумываются **независимо от корпуса** — так, как их
задаёт живой человек. Что в корпусе лежит, при генерации не показывается.
Эталонов нет и не нужно: мерим не правильность ответа, а саму способность
ответить.

Что считается:

* **coverage** — доля вопросов, на которые агент дал содержательный ответ;
* разбивка по категориям — где именно провал;
* для каждого неотвеченного вопроса — расстояние до ближайшего чанка корпуса.

Последнее и есть главный диагностический признак. Он разделяет две разные
беды, которые снаружи выглядят одинаково:

    ближайший чанк далеко  → в корпусе нет темы, надо расширять данные
    ближайший чанк близко  → тема есть, но поиск или grade её не взяли,
                             надо чинить пайплайн, а не докачивать статьи

Запуск::

    python diagnose_corpus.py            # сгенерировать вопросы и прогнать
    python diagnose_corpus.py --reuse    # прогнать по уже сохранённым вопросам
"""

from __future__ import annotations

import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from agent import REWRITE_PROMPT, build_agent
from embedder import Embedder
from lance_db import open_table, vector_search
from make_eval_dataset import _complete
from measurement import RESEARCH_DIR
from providers import get_clients, get_polza_client
from run_eval import looks_like_refusal

HERE = Path(__file__).parent

# Выгрузки замеров живут в _research/dataset/ — в репозиторий не идут,
# пересобираются скриптами. В репозитории только журнал EVOLUTION.md.
QUESTIONS_PATH = RESEARCH_DIR / "coverage_questions.xlsx"
RESULTS_PATH = RESEARCH_DIR / "coverage_report.csv"

# Категории пользовательских намерений. Список составлен от лица человека,
# а не от содержания корпуса: сюда намеренно попали и темы, которых там
# может не быть, — иначе диагностика ничего не найдёт.
CATEGORIES = [
    "похудение, калории и дефицит",
    "белок, мышцы и состав тела",
    "конкретные продукты (яйца, кофе, молоко, орехи)",
    "популярные диеты (кето, интервальное голодание, средиземноморская)",
    "витамины и добавки",
    "пищеварение, клетчатка и микробиом",
    "непереносимости и аллергии (глютен, лактоза)",
    "питание при состояниях (давление, холестерин, сахар крови)",
    "напитки, вода, алкоголь, кофеин",
    "распространённые мифы о питании",
    "возрастные группы и жизненные периоды",
    "спорт, выносливость и восстановление",
]

PER_CATEGORY = 5

QUESTIONS_PROMPT = """Придумай {n} вопросов НА РУССКОМ ЯЗЫКЕ, которые обычный
человек задал бы ассистенту по питанию.

Тема: {category}

Требования:
- так, как спрашивает живой человек, а не учёный: простым языком
- конкретные, а не «расскажи всё о питании»
- вопросы о том, что известно науке, а НЕ просьбы составить рацион
- разные по смыслу, не переформулировки одного и того же
- одно предложение каждый

Верни ТОЛЬКО JSON: {{"questions": ["вопрос 1", "вопрос 2", ...]}}"""

# Ближе этого — тема в корпусе точно есть. Порог взят по зондированию:
# уверенно покрытые темы давали расстояние 0.54-0.78, явные пробелы — выше 1.0.
NEAR_THRESHOLD = 0.85


def generate_questions() -> pd.DataFrame:
    """Придумать вопросы по категориям, не заглядывая в корпус."""
    client, model, _ = get_clients("polza")

    def work(category: str) -> list[dict]:
        raw = _complete(client, model, QUESTIONS_PROMPT.format(n=PER_CATEGORY, category=category))
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            questions = json.loads(text).get("questions", [])
        except json.JSONDecodeError:
            return []
        return [
            {"id": str(uuid.uuid4())[:8], "category": category, "question": q.strip()}
            for q in questions
            if q and q.strip()
        ]

    with ThreadPoolExecutor(max_workers=6) as pool:
        batches = list(pool.map(work, CATEGORIES))

    rows = [row for batch in batches for row in batch]
    frame = pd.DataFrame(rows).drop_duplicates(subset="question").reset_index(drop=True)

    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(QUESTIONS_PATH, index=False)
    print(f"Сгенерировано вопросов: {len(frame)} → {QUESTIONS_PATH.name}")
    return frame


def nearest_chunk(question: str, embedder: Embedder, table, client, model) -> tuple[float, str]:
    """Найти ближайший к вопросу фрагмент корпуса.

    Запрос переводится тем же промптом, что и в агенте: корпус англоязычный,
    и без перевода расстояние будет говорить о языке, а не о теме.
    """
    query = _complete(client, model, REWRITE_PROMPT.format(question=question)) or question
    hits = vector_search(table, embedder.embed_query(query.strip().strip('"')), k=1)
    if not hits:
        return 99.0, ""
    return float(hits[0].get("_distance", 99.0)), str(hits[0].get("title", ""))


def classify(answer: str, has_chunks: bool) -> str:
    """К чему свёлся ответ агента."""
    if "только с питанием" in answer.lower():
        return "не по теме"
    if "не могу составить рацион" in answer.lower():
        return "красный флаг"
    if not has_chunks or looks_like_refusal(answer):
        return "отказ"
    return "ответ"


def main() -> None:
    if "--reuse" in sys.argv and QUESTIONS_PATH.exists():
        questions = pd.read_excel(QUESTIONS_PATH)
        print(f"Взято из файла: {len(questions)} вопросов")
    else:
        questions = generate_questions()

    client, model, embed_client = get_clients("polza")
    agent = build_agent(client, model=model, embed_client=embed_client)
    embedder = Embedder(get_polza_client())
    table = open_table("lance_db/vectorstore", "chunks")

    print(f"\nПрогоняю агента по {len(questions)} вопросам...")

    def work(row: dict) -> dict:
        answer = agent.run(str(row["question"]), remember=False)
        distance, title = nearest_chunk(str(row["question"]), embedder, table, client, model)
        return {
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "outcome": classify(answer.answer, bool(answer.retrieved_chunks)),
            "nearest_distance": round(distance, 3),
            "nearest_title": title[:70],
            "answer": answer.answer[:400],
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, questions.to_dict("records")))

    report = pd.DataFrame(results)
    report.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    # ── Сводка ───────────────────────────────────────────────
    answered = report["outcome"] == "ответ"

    print()
    print("=" * 66)
    print(f"COVERAGE: {answered.mean():.1%} ({answered.sum()} из {len(report)})")
    print("=" * 66)
    print()
    print(report["outcome"].value_counts().to_string())

    print()
    print("По категориям:")
    by_category = (
        report.assign(answered=answered)
        .groupby("category")["answered"]
        .agg(["mean", "sum", "count"])
        .sort_values("mean")
    )
    for category, row in by_category.iterrows():
        print(f"  {row['mean']:>6.0%}  ({int(row['sum'])}/{int(row['count'])})  {category}")

    # ── Диагноз по неотвеченным ──────────────────────────────
    failed = report[~answered & (report["outcome"] == "отказ")]
    if not failed.empty:
        gaps = failed[failed["nearest_distance"] > NEAR_THRESHOLD]
        pipeline = failed[failed["nearest_distance"] <= NEAR_THRESHOLD]

        print()
        print("=" * 66)
        print("ДИАГНОЗ ПО ОТКАЗАМ")
        print("=" * 66)
        print(f"  нет темы в корпусе (далеко):     {len(gaps)}")
        print(f"  тема есть, не взяли (близко):    {len(pipeline)}")

        if not gaps.empty:
            print("\n  Пробелы корпуса:")
            for _, row in gaps.head(12).iterrows():
                print(f"    [{row['nearest_distance']:.2f}] {row['question'][:72]}")

        if not pipeline.empty:
            print("\n  Тема есть, но агент не ответил — чинить пайплайн, не данные:")
            for _, row in pipeline.head(12).iterrows():
                print(f"    [{row['nearest_distance']:.2f}] {row['question'][:60]}")
                print(f"           ближайшее: {row['nearest_title']}")

    print(f"\nПодробности → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
