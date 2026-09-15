"""
Собственный шум судьи: какую разницу он вообще способен различить.

Зачем. Каждое решение в проекте принимается по парному бутстрэпу: если
95% доверительный интервал разницы не накрывает ноль — правку принимаем.
Но бутстрэп ресэмплит **вопросы** и потому меряет только шум выборки.
Судья при этом сам по себе нестабилен: та же модель на том же тексте
может поставить 1.0, а через минуту 0.5. Этот источник разброса в интервал
не входит вовсе, и значит все интервалы в проекте систематически УЖЕ,
чем должны быть.

Что здесь делается. Один и тот же судья прогоняется по одним и тем же
сохранённым ответам несколько раз. Между прогонами не меняется ничего —
ни агент, ни ответы, ни промпт, ни модель. Настоящая разница равна нулю
по построению, поэтому всё, что покажет замер, — и есть шум.

Результат — **эмпирический ноль**: интервал, который получается при
сравнении системы с самой собой. Любая правка, чей интервал не шире
этого, померена в пределах собственной погрешности прибора.

Отдельно стоит помнить: судья вызывается с ``temperature=0``. Если
разброс окажется нулевым — это тоже результат, и он означает, что порог
значимости целиком определяется размером выборки.

Запуск::

    python -m experiments.judge_noise                # 3 прогона, весь датасет
    python -m experiments.judge_noise --runs 2 --n 80
"""

from __future__ import annotations

import sys

import pandas as pd

from evaluation_answers import evaluate_answers
from make_eval_dataset import DATASET_PATH
from measurement import RESEARCH_DIR, paired_bootstrap
from providers import get_polza_client
from run_eval import load_answers, strip_boilerplate

RESULTS_PATH = RESEARCH_DIR / "experiments_judge_noise.csv"

# Судья, которым посчитаны все метрики проекта. Здесь важно брать именно его:
# меряется погрешность того прибора, которым пользуемся, а не лучшего из возможных.
JUDGE_MODEL = "openai/gpt-4o-mini"

METRICS = ("correctness", "relevancy")

# Решения, которые в проекте уже приняты или отклонены по оценкам судьи.
# Границы интервалов — из EVOLUTION.md. Смысл в том, чтобы сверить их
# с измеренным здесь порогом: решение, чей интервал уже собственной
# погрешности прибора, держится на удаче, а не на замере.
PAST_DECISIONS: dict[str, dict[str, tuple[float, float]]] = {
    "correctness": {
        "полнота ответа, n=210 (принято)": (+0.038, +0.100),
        "полнота ответа, n=80 (отклонено)": (+0.000, +0.094),
    },
}


def main() -> None:
    if not DATASET_PATH.exists():
        print("Нет датасета. Выполните: python make_eval_dataset.py")
        return

    runs = 3
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])

    dataset = pd.read_excel(DATASET_PATH)
    answers = load_answers()

    if "--n" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--n") + 1])
        dataset = dataset.head(limit)
        keep = set(dataset["id"].astype(str))
        answers = [a for a in answers if str(a.dataset_row_id) in keep]

    scored = [a.model_copy(update={"answer": strip_boilerplate(a.answer)}) for a in answers]
    print(f"Судья: {JUDGE_MODEL}")
    print(f"Ответов: {len(scored)}, прогонов: {runs}")
    print("Между прогонами не меняется ничего — настоящая разница равна нулю.\n")

    client = get_polza_client()
    per_run: list[pd.DataFrame] = []

    for run in range(runs):
        print("=" * 62)
        print(f"Прогон {run + 1} из {runs}")
        print("=" * 62)

        judged = evaluate_answers(scored, dataset, client, JUDGE_MODEL)
        slice_ = judged[["id", *METRICS]].rename(
            columns={metric: f"{metric}_{run}" for metric in METRICS}
        )
        per_run.append(slice_)

        means = "  ".join(f"{m}={judged[m].mean():.3f}" for m in METRICS)
        print(f"  {means}\n")

    merged = per_run[0]
    for slice_ in per_run[1:]:
        merged = merged.merge(slice_, on="id")
    merged.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    print("=" * 62)
    print("РАЗБРОС МЕЖДУ ПРОГОНАМИ")
    print("=" * 62)

    for metric in METRICS:
        columns = [f"{metric}_{run}" for run in range(runs)]
        values = merged[columns]

        run_means = values.mean()
        spread = float(run_means.max() - run_means.min())

        # Доля вопросов, где судья сам с собой не согласился.
        unstable = float((values.nunique(axis=1) > 1).mean())

        print(f"\n{metric}")
        print("  средние по прогонам: " + ", ".join(f"{v:.3f}" for v in run_means))
        print(f"  размах средних:      {spread:.3f}")
        print(f"  вопросов с разными оценками между прогонами: {unstable:.1%}")

        print("  эмпирический ноль (сравнение системы с самой собой):")
        widths = []
        for i in range(runs):
            for j in range(i + 1, runs):
                mean, low, high = paired_bootstrap(values[columns[i]], values[columns[j]])
                widths.append(high - low)
                print(f"    прогон {i + 1} → {j + 1}:  {mean:+.3f}  [{low:+.3f}, {high:+.3f}]")

        floor = max(widths) / 2
        print(f"  ПОРОГ: разницу меньше {floor:.3f} прибор не различает")

        if metric in PAST_DECISIONS:
            print("  Проверка уже принятых решений этой метрикой:")
            for name, (low, high) in PAST_DECISIONS[metric].items():
                narrowest = min(abs(low), abs(high))
                mark = "держится" if narrowest > floor else "В ПРЕДЕЛАХ ПОГРЕШНОСТИ"
                print(f"    {name:<38} [{low:+.3f}, {high:+.3f}]  {mark}")

    print(f"\nСохранено → {RESULTS_PATH}")
    print()
    print("Как этим пользоваться: если у принятой правки интервал уже этого")
    print("порога, значит она померена в пределах собственной погрешности")
    print("судьи, и вывод о ней держится на удаче, а не на замере.")


if __name__ == "__main__":
    main()
