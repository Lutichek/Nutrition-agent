"""
Замена одного блюда: насколько день после замены хуже исходного.

``measure_solver`` замены не меряет. Тесты стерегут только «прошёл
проверки», а проверки широкие (жиры и углеводы ±35%). Живой прогон:
«замени завтрак» вместо греческого йогурта с гранолой дало швейцарский
сыр — жиры +32%, углеводы −19%. Проверки пройдены, а карточка показала
три отклонения. Сыр был четвёртым из пяти лучших, а среди этих пяти
выбор случаен (SWAP_TOP): невязка у первых двух около 1.0, у сыров 1.4–1.6.

Для каждого профиля из measure_solver, каждой позиции дня и трёх замен
подряд («замени ещё раз»: прежние блюда отвергнуты) считается:

* прошла ли замена все проверки — обязано быть 100%;
* в полосе — калории, белок, жиры и углеводы дня после замены в ±10%,
  то есть карточка не покажет отклонений;
* рост невязки — во сколько раз день хуже исходного (медиана);
* разных блюд за три замены — разнообразие «замени ещё раз»;
* отказов — замена не нашлась (PlanNotFeasible).

Запуск::

    python -m experiments.swap_quality
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import solver  # noqa: E402
from experiments.measure_solver import PROFILES  # noqa: E402
from foods import load_catalog  # noqa: E402
from targets import compute_targets  # noqa: E402

# (порог близости к лучшей замене, когда подстраивать порции остальных)
VARIANTS = {
    "было": (None, None),
    "порог ×1.2": (1.2, None),
    "порог ×1.2, порции при ×1.5": (1.2, 1.5),
    "порог ×1.2, порции при ×1.2": (1.2, 1.2),
}
SEEDS = (0, 1)
CHAIN = 3
BAND = 0.10


def in_band(plan: solver.DayPlan) -> bool:
    return all(abs(plan.deviation.get(k, 0.0)) <= BAND
               for k in ("kcal", "protein_g", "fat_g", "carb_g"))


def main() -> None:
    catalog = load_catalog()
    days = []
    for name, profile in PROFILES.items():
        targets = compute_targets(profile)
        for seed in SEEDS:
            days.append((name, profile, solver.build_day(
                catalog, targets, exclude=profile.exclude, seed=seed), seed))
    print(f"дней: {len(days)}, замен на вариант: {len(days) * 7 * CHAIN}\n")

    for label, (margin, tune) in VARIANTS.items():
        solver.SWAP_LOSS_MARGIN, solver.SWAP_TUNE_ABOVE = margin, tune
        passed = band = failed = total = 0
        growth, variety, seconds = [], [], []
        for _, profile, plan, seed in days:
            for index in range(len(plan.items)):
                rejected: list[int] = []
                for step in range(CHAIN):
                    started = time.perf_counter()
                    try:
                        new = solver.swap_item(catalog, plan, index, exclude=profile.exclude,
                                               seed=seed + len(rejected), avoid_ids=set(rejected))
                    except solver.PlanNotFeasible:
                        failed += 1
                        break
                    finally:
                        seconds.append(time.perf_counter() - started)
                    total += 1
                    passed += all(solver.check_plan(new).values())
                    band += in_band(new)
                    growth.append(new.loss / max(plan.loss, 1e-9))
                    rejected.append(new.items[index].fdc_id)
                    del step
                variety.append(len(set(rejected)))
        print(f"{label:<28} проверки {passed}/{total}   в полосе ±10% {band / total:.0%}   "
              f"рост невязки ×{statistics.median(growth):.2f} (90% ×{sorted(growth)[int(.9 * len(growth))]:.2f})   "
              f"разных за {CHAIN}: {statistics.mean(variety):.2f}   отказов {failed}   "
              f"{statistics.median(seconds):.2f} с", flush=True)
    solver.SWAP_LOSS_MARGIN = solver.SWAP_TUNE_ABOVE = None


if __name__ == "__main__":
    main()
