"""
Эксперимент с промптом генерации: полнота ответа против краткости.

Что чиним. Разбор метрик показал, что главный источник неточных ответов —
не поиск: у 17 вопросов из 80 нужный фрагмент был найден, а ответ всё равно
получил 0.5 («упоминает X, но не отражает Y»). Подозрение на потолок
«3-6 предложений» в промпте: он заставляет модель выбирать между фактами.

Как проверяем. Два варианта промпта, один датасет, одни и те же судьи.
Отличие ровно в двух строках — требование перечислить все относящиеся
к вопросу сведения и снятие жёсткого лимита длины. Минимальная правка выбрана
намеренно: попытка переписать промпт генерации целиком ухудшала все метрики
сразу.

За чем следим, кроме correctness:

* **faithfulness** — требование «скажи больше» легко превращается
  в «додумай». Если она просядет, правка вредна независимо от correctness.
* **длина ответа** — чтобы видеть цену в токенах.

Запуск::

    python -m experiments.answer_prompt
    python -m experiments.answer_prompt --n 40     # быстрее и дешевле
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from agent import RAGAgentAnswer, build_agent
from build_qrels import QRELS_PATH, load_qrels
from evaluation import evaluate
from evaluation_answers import evaluate_answers
from measurement import RESEARCH_DIR, paired_bootstrap
from make_eval_dataset import DATASET_PATH
from providers import get_clients
from run_eval import strip_boilerplate

# Выгрузки замеров живут в _research/dataset/ — в репозиторий не идут,
# пересобираются скриптами. В репозитории только журнал EVOLUTION.md.
RESULTS_PATH = RESEARCH_DIR / "experiments_answer.csv"
PER_QUESTION_PATH = RESEARCH_DIR / "experiments_answer_per_question.csv"

# Порог приёмки, заданный ДО прогона: собственный шум судьи, замеренный
# в experiments/judge_noise.py. Разницу меньше этой прибор не различает,
# и «значимо» по бутстрэпу тут ничего не значит — бутстрэп меряет шум
# выборки, а не шум самого измерителя.
JUDGE_NOISE_FLOOR = 0.010


def run_variant(name: str, answer_style: str, dataset: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Прогнать агента с одним из вариантов промпта и посчитать метрики."""
    print("=" * 62)
    print(name)
    print("=" * 62)

    client, model, embed_client = get_clients("polza")
    agent = build_agent(client, model=model, embed_client=embed_client, answer_style=answer_style)

    def work(row: tuple[str, str]) -> RAGAgentAnswer:
        row_id, question = row
        return agent.run(question, dataset_row_id=row_id, remember=False)

    started = time.perf_counter()
    items = list(zip(dataset["id"].astype(str), dataset["question"].astype(str), strict=True))
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(work, items))

    # Служебные предложения из оценки исключаем — см. run_eval.strip_boilerplate.
    scored = [a.model_copy(update={"answer": strip_boilerplate(a.answer)}) for a in answers]

    qrels = load_qrels() if QRELS_PATH.exists() else None
    metrics = evaluate(answers=scored, dataset=dataset, client=client, model=model,
                       compute_faithfulness=True, qrels=qrels)
    extra = evaluate_answers(scored, dataset, client, model)
    metrics = metrics.merge(extra[["id", "correctness", "relevancy"]], on="id", how="left")

    lengths = pd.Series([len(a.answer) for a in scored])

    row = {
        "config": name,
        "correctness": float(metrics["correctness"].mean()),
        "faithfulness": float(metrics["faithfulness"].mean()),
        "relevancy": float(metrics["relevancy"].mean()),
        "recall": float(metrics["recall"].mean()),
        "answer_chars": float(lengths.mean()),
        "elapsed_s": round(time.perf_counter() - started),
    }

    print(
        f"  correctness={row['correctness']:.3f}  faithfulness={row['faithfulness']:.3f}  "
        f"relevancy={row['relevancy']:.3f}  длина={row['answer_chars']:.0f} симв."
    )
    print()

    return row, metrics[["id", "correctness", "faithfulness", "relevancy"]].copy()


def main() -> None:
    if not DATASET_PATH.exists():
        print("Нет датасета. Выполните: python make_eval_dataset.py")
        return

    dataset = pd.read_excel(DATASET_PATH)
    if "--n" in sys.argv:
        dataset = dataset.head(int(sys.argv[sys.argv.index("--n") + 1]))

    if "reference_answer" not in dataset.columns:
        print("Нет эталонных ответов. Выполните: python make_eval_dataset.py --references")
        return

    print(f"Вопросов: {len(dataset)}\n")

    # Что с чем сравниваем. По умолчанию — боевой промпт против кандидата
    # «сохраняй состав доказательств». Старое сравнение с коротким промптом
    # доступно флагом --vs-short: оно уже решено на шаге 22, повторять его
    # на каждом прогоне незачем.
    if "--vs-short" in sys.argv:
        baseline_name, baseline_style = "Базовый промпт (3-6 предложений)", "short"
        variant_name, variant_style = "Промпт с требованием полноты", "full"
    else:
        baseline_name, baseline_style = "Боевой промпт (требование полноты)", "full"
        variant_name, variant_style = "Кандидат: + состав доказательств", "evidence"

    base_row, base_metrics = run_variant(baseline_name, baseline_style, dataset)
    full_row, full_metrics = run_variant(variant_name, variant_style, dataset)

    summary = pd.DataFrame([base_row, full_row])
    summary.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    # Подушевые оценки сохраняем отдельно: если разница окажется пограничной
    # (а она оказалась), доразобрать её можно будет по этому файлу,
    # не гоняя оба варианта заново.
    per_question = base_metrics.merge(
        full_metrics, on="id", suffixes=("_base", "_full")
    )
    per_question.to_csv(PER_QUESTION_PATH, index=False, encoding="utf-8")

    print("=" * 62)
    print("СРАВНЕНИЕ")
    print("=" * 62)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print()
    print("Значимость разницы (парный бутстрэп, 95% ДИ):")
    base_indexed = base_metrics.set_index("id")
    full_indexed = full_metrics.set_index("id")
    common = base_indexed.index.intersection(full_indexed.index)

    deltas = {}
    for metric in ("correctness", "faithfulness", "relevancy"):
        mean, low, high = paired_bootstrap(
            base_indexed.loc[common, metric], full_indexed.loc[common, metric]
        )
        deltas[metric] = (mean, low, high)
        verdict = "значимо" if (low > 0 or high < 0) else "в пределах шума"
        print(f"  {metric:<13} {mean:+.3f}  [{low:+.3f}, {high:+.3f}]  {verdict}")

    # Вердикт печатается автоматически по критерию, заданному ДО прогона:
    # подгонять его, увидев цифры, нечем.
    c_mean, c_low, c_high = deltas["correctness"]
    f_mean, f_low, f_high = deltas["faithfulness"]
    correctness_grew = c_low > JUDGE_NOISE_FLOOR
    faithfulness_held = f_high > 0 or f_low > 0

    print()
    print("=" * 62)
    print("ВЕРДИКТ")
    print("=" * 62)
    print("Критерий, заданный до прогона:")
    print(f"  1) нижняя граница correctness выше {JUDGE_NOISE_FLOOR:.3f} — "
          "собственного шума судьи")
    print("     (experiments/judge_noise.py; интервал уже этого порога "
          "ничего не доказывает)")
    print("  2) faithfulness не просела — «скажи больше» не должно "
          "означать «додумай»")
    print()
    print(f"  correctness:  нижняя граница {c_low:+.3f}  "
          f"{'проходит' if correctness_grew else 'НЕ проходит'}")
    print(f"  faithfulness: {f_mean:+.3f} [{f_low:+.3f}, {f_high:+.3f}]  "
          f"{'не просела' if faithfulness_held else 'ПРОСЕЛА'}")
    print()
    if correctness_grew and faithfulness_held:
        print("ПРИНЯТЬ: сделать 'evidence' промптом по умолчанию в build_agent.")
    else:
        print("ОТКЛОНИТЬ: оставить боевой промпт как есть.")
    print(f"\nСохранено → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
