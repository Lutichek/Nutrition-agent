"""
Отделить эффект конфигурации от шума прогона по контрольной группе.

Зачем. Сквозные прогоны делаются в разное время, а генерация стохастична:
та же продовая конфигурация в двух прогонах дала correctness 0.860 и 0.876.
Значит наблюдаемая дельта между конфигурациями — это

    эффект конфигурации + шум прогона + дрейф остального кода

и парный бутстрэп по вопросам их не разделяет: он ресэмплит вопросы,
а не прогоны. Порог различимости судьи (0.010) тоже не помогает — он
измерен на ФИКСИРОВАННЫХ ответах и про стохастичность генерации
не знает вовсе.

Приём. Контрольная группа ищется в самих данных: берём вопросы, где обе
конфигурации отдали в генерацию **один и тот же контекст**. Повлиять там
было не на что, значит вся разница — шум и дрейф. Остальные вопросы —
«лечение»: шум плюс эффект. Разность между группами и есть эффект,
не объяснимый шумом.

⚠️ Тонкость, на которой уже ошиблись: «один и тот же контекст» — это
состав **И ПОРЯДОК** чанков. Порядок фрагментов идёт в промпт генерации,
поэтому перестановка — это уже эффект проверяемой конфигурации. Считая
по одному составу, в контроль пускаешь переставленные и завышаешь оценку
шума: на выключении BM25 таких было 15 из 74, и оценка шума выходила
вдвое больше настоящей (+0.014 против +0.008).

⚠️ Пары надо брать ОДНОРОДНЫЕ — отличающиеся ровно одним приёмом.
Сохранённый прогон реранкинга делался поверх гибрида, поэтому его база —
``hybrid``, а не нынешняя продовая ``base``: иначе замер посчитает сразу
две правки (добавили реранкинг И убрали BM25) и назовёт это одной.

⚠️ Чего этот приём НЕ умеет: чистить метрики ПОИСКА. Контрольная группа
определена через совпадение выдачи, поэтому дельта поиска в ней равна
нулю по построению. Если перед поиском стоит вызов модели (здесь — узел
``rewrite``), сквозные precision и recall наследуют шум прогона так же,
как correctness, и вычесть его нечем. Решать по поиску в таком случае —
только детерминированными замерами, где запрос берётся из кэша один
на все конфигурации (образец — ``experiments/bm25_weight.py``).

⚠️ И приём НЕ работает между разными индексами. Он опознаёт контекст
по ``chunk_id``, а при другой нарезке идентификаторы означают другое.
Контрольная группа выйдет пустой — скрипт это ловит и отказывается
считать, вместо того чтобы напечатать правдоподобную ерунду.

Замер бесплатный: ни одного вызова модели, всё считается по сохранённым
ответам и таблицам метрик.

Запуск::

    python -m experiments.run_noise                    # base против hybrid
    python -m experiments.run_noise hybrid rerank      # реранкинг поверх гибрида
    python -m experiments.run_noise --list             # какие прогоны есть
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from measurement import BOOTSTRAP_SAMPLES, RESEARCH_DIR, paired_bootstrap
from run_eval import load_answers

GENERATION_METRICS = ["faithfulness", "correctness", "relevancy"]

# Реестр сохранённых прогонов: имя → (ответы, метрики, чем отличается).
#
# Заводится явно, а не выводится из флагов run_eval, потому что смысл
# прогона не всегда совпадает со смыслом флага: `rerank` здесь исторический,
# он делался поверх гибрида, когда гибрид ещё был продовой конфигурацией.
RUNS: dict[str, tuple[str, str, str]] = {
    "base": ("agent_answers.json", "metrics.csv", "плотный поиск (продовая)"),
    "hybrid": ("agent_answers_hybrid.json", "metrics_hybrid.csv", "плотный + BM25/RRF"),
    "rerank": (
        "agent_answers_rerank_over_hybrid.json",
        "metrics_rerank_over_hybrid.csv",
        "гибрид + LLM-реранкинг",
    ),
}

# С чем осмысленно сравнивать каждый прогон, если пара не задана руками.
DEFAULT_BASE = {"hybrid": "base", "rerank": "hybrid", "base": "hybrid"}


def contexts(answers) -> dict[str, tuple[int, ...]]:
    """Контекст каждого вопроса: чанки В ТОМ ПОРЯДКЕ, в каком ушли в промпт.

    Кортеж, а не множество — см. предупреждение в докстроке модуля.
    """
    return {
        str(answer.dataset_row_id): tuple(
            chunk.chunk_id for chunk in (answer.retrieved_chunks or [])
        )
        for answer in answers
    }


def _load_metrics(name: str) -> pd.DataFrame:
    frame = pd.read_csv(RESEARCH_DIR / RUNS[name][1])
    return frame.assign(id=frame["id"].astype(str)).set_index("id")


def between_groups(
    deltas: pd.Series,
    control: list[str],
    treated: list[str],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Непарный бутстрэп разности средних дельт двух групп.

    Группы независимы (вопросы разные), поэтому ресэмплим каждую отдельно.
    """
    rng = np.random.default_rng(seed)
    a = deltas.reindex(control).dropna().to_numpy()
    b = deltas.reindex(treated).dropna().to_numpy()
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan"), float("nan")

    draws = np.empty(samples)
    for i in range(samples):
        draws[i] = (
            rng.choice(b, size=len(b), replace=True).mean()
            - rng.choice(a, size=len(a), replace=True).mean()
        )
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(b.mean() - a.mean()), float(low), float(high)


def compare(base_name: str, variant_name: str) -> None:
    missing = [
        path
        for name in (base_name, variant_name)
        for path in (RESEARCH_DIR / RUNS[name][0], RESEARCH_DIR / RUNS[name][1])
        if not path.exists()
    ]
    if missing:
        print("Нет файлов: " + ", ".join(p.name for p in missing))
        return

    base_ctx = contexts(load_answers(RESEARCH_DIR / RUNS[base_name][0]))
    var_ctx = contexts(load_answers(RESEARCH_DIR / RUNS[variant_name][0]))
    base = _load_metrics(base_name)
    var = _load_metrics(variant_name)

    common = sorted(set(base.index) & set(var.index) & set(base_ctx) & set(var_ctx))
    control = [i for i in common if base_ctx[i] == var_ctx[i]]
    treated = [i for i in common if base_ctx[i] != var_ctx[i]]
    reordered = [i for i in treated if set(base_ctx[i]) == set(var_ctx[i])]

    print("=" * 78)
    print(f"«{RUNS[base_name][2]}»  ПРОТИВ  «{RUNS[variant_name][2]}»")
    print("=" * 78)
    print(f"вопросов: {len(common)}")
    print(f"  контекст совпал (состав и порядок): {len(control):>3}  — КОНТРОЛЬ")
    print(f"  контекст различается:               {len(treated):>3}")
    print(f"     из них только перестановкой:     {len(reordered):>3}")
    print()

    if len(control) < 10:
        print("Контрольная группа пуста или почти пуста — приём неприменим.")
        print("Обычно это значит, что прогоны сделаны на РАЗНЫХ индексах:")
        print("chunk_id там означают разное, и совпасть не могут в принципе.")
        print("Шум прогона в этом случае надо оценивать повторным прогоном")
        print("той же конфигурации, а не контрольной группой.")
        return

    header = f"{'метрика':<14}{'контроль':>20}{'изменился':>20}{'разность':>28}"
    print(header)
    print("-" * len(header))

    for metric in GENERATION_METRICS:
        if metric not in base.columns or metric not in var.columns:
            continue
        deltas = (var[metric] - base[metric]).dropna()

        control_mean, *_ = paired_bootstrap(
            base.loc[base.index.intersection(control), metric].dropna(),
            var.loc[var.index.intersection(control), metric].dropna(),
        )
        treated_delta = deltas.reindex(treated).dropna()
        mean, low, high = between_groups(deltas, control, treated)
        verdict = "ЗНАЧИМО" if (low > 0 or high < 0) else "шум"
        n_control = len(deltas.reindex(control).dropna())

        print(
            f"{metric:<14}"
            f"{f'{control_mean:+.3f} (n={n_control})':>20}"
            f"{f'{treated_delta.mean():+.3f} (n={len(treated_delta)})':>20}"
            f"{f'{mean:+.3f} [{low:+.3f}, {high:+.3f}] {verdict}':>28}"
        )

    print()
    print("Как читать. «Контроль» — оценка шума прогона и дрейфа кода: эффекта")
    print("там нет по построению. «Разность» — то, что шумом не объясняется.")
    print("Значимой разности нет — значит влияние на ОТВЕТ не доказано, каким")
    print("бы убедительным ни выглядело сырое сравнение средних.")
    print()
    print("По метрикам ПОИСКА этот приём вывода не даёт — см. докстроку модуля.")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--list" in sys.argv[1:]:
        print("Доступные прогоны:")
        for name, (answers, metrics, what) in RUNS.items():
            mark = "" if (RESEARCH_DIR / metrics).exists() else "   (файла нет)"
            print(f"  {name:<10} {what:<28} {metrics}{mark}")
        return

    if len(args) == 2:
        base_name, variant_name = args
    elif len(args) == 1:
        variant_name = args[0]
        base_name = DEFAULT_BASE.get(variant_name, "base")
    else:
        base_name, variant_name = "base", "hybrid"

    unknown = [n for n in (base_name, variant_name) if n not in RUNS]
    if unknown:
        print(f"Неизвестный прогон: {', '.join(unknown)}. См. --list")
        return

    compare(base_name, variant_name)


if __name__ == "__main__":
    main()
