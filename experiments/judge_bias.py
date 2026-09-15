"""
Проверка судьи: не завышает ли модель оценку собственным ответам.

Проблема. Во всех прогонах метрик генератор и судья — одна и та же модель
(``openai/gpt-4o-mini``). Это классическое self-preference bias: модель
склонна выше оценивать текст, который сама и написала. Плюс корреляция ошибок —
если модель неверно поняла вопрос, она и ответит мимо, и свой ответ одобрит.

Что здесь делается. Одни и те же сохранённые ответы агента переоцениваются
несколькими судьями, в том числе моделью ДРУГОГО семейства. Если оценки
расходятся систематически — смещение есть, и его размер видно в цифрах.

Почему важна именно другая СЕМЬЯ моделей: gpt-4o и gpt-4o-mini обучены
похоже и разделяют слепые зоны, поэтому их согласие ничего не доказывает.
Llama обучена независимо — расхождение с ней информативнее.

Запуск::

    python -m experiments.judge_bias
    python -m experiments.judge_bias --n 80     # дешевле
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from evaluation_answers import evaluate_answers
from make_eval_dataset import DATASET_PATH
from measurement import RESEARCH_DIR
from providers import get_polza_client
from run_eval import load_answers, strip_boilerplate

# Выгрузки замеров живут в _research/dataset/ — в репозиторий не идут,
# пересобираются скриптами. В репозитории только журнал EVOLUTION.md.
RESULTS_PATH = RESEARCH_DIR / "experiments_judge.csv"
PER_QUESTION_PATH = RESEARCH_DIR / "experiments_judge_per_question.csv"

# Судьи для сравнения. Первый — тот, которым считались все метрики проекта.
JUDGES = [
    ("gpt-4o-mini (он же генератор)", "openai/gpt-4o-mini"),
    ("gpt-4o (то же семейство, сильнее)", "openai/gpt-4o"),
    ("llama-3.3-70b (другое семейство)", "meta-llama/llama-3.3-70b-instruct"),
]


def main() -> None:
    if not DATASET_PATH.exists():
        print("Нет датасета. Выполните: python make_eval_dataset.py")
        return

    dataset = pd.read_excel(DATASET_PATH)
    answers = load_answers()

    if "--n" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--n") + 1])
        dataset = dataset.head(limit)
        keep = set(dataset["id"].astype(str))
        answers = [a for a in answers if str(a.dataset_row_id) in keep]

    # Оцениваем те же тексты, что и в основном прогоне: без служебных фраз.
    scored = [a.model_copy(update={"answer": strip_boilerplate(a.answer)}) for a in answers]
    print(f"Ответов на оценку: {len(scored)}, вопросов: {len(dataset)}\n")

    client = get_polza_client()
    rows: list[dict] = []
    per_question: pd.DataFrame | None = None

    for label, model in JUDGES:
        print("=" * 62)
        print(f"{label}  [{model}]")
        print("=" * 62)

        judged = evaluate_answers(scored, dataset, client, model)
        rows.append(
            {
                "judge": label,
                "model": model,
                "correctness": float(judged["correctness"].mean()),
                "relevancy": float(judged["relevancy"].mean()),
            }
        )
        print(f"  correctness={rows[-1]['correctness']:.3f}  "
              f"relevancy={rows[-1]['relevancy']:.3f}\n")

        column = model.split("/")[-1]
        slice_ = judged[["id", "correctness", "relevancy"]].rename(
            columns={"correctness": f"correctness_{column}", "relevancy": f"relevancy_{column}"}
        )
        per_question = slice_ if per_question is None else per_question.merge(slice_, on="id")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    if per_question is not None:
        per_question.to_csv(PER_QUESTION_PATH, index=False, encoding="utf-8")

    print("=" * 62)
    print("СРАВНЕНИЕ СУДЕЙ")
    print("=" * 62)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    # Смещение относительно судьи-генератора.
    base = summary.iloc[0]
    print()
    print("Отклонение от судьи-генератора:")
    for _, row in summary.iloc[1:].iterrows():
        print(f"  {row['judge']:<38} correctness {row['correctness'] - base['correctness']:+.3f}"
              f"   relevancy {row['relevancy'] - base['relevancy']:+.3f}")

    # Согласие по каждому вопросу: доля совпавших оценок.
    if per_question is not None and len(JUDGES) > 1:
        print()
        print("Согласие с судьёй-генератором (доля вопросов с той же оценкой):")
        base_col = f"correctness_{JUDGES[0][1].split('/')[-1]}"
        for _label, model in JUDGES[1:]:
            column = f"correctness_{model.split('/')[-1]}"
            agree = (per_question[base_col] == per_question[column]).mean()
            print(f"  {model:<40} {agree:.1%}")

    print(f"\nСохранено → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
