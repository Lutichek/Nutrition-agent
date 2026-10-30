"""
«Съел не по плану»: оставлять ли приёмы пищи ДО съеденного как были.

Живой прогон: «на обед съел пиццу» пересобрал и завтрак — йогурт сменился
вафлей, хотя к обеду завтрак уже съеден. Оставить его — честно, но у
солвера меньше свободы добрать норму. Замер — во что это обходится:

* доля дней, прошедших ВСЕ проверки check_plan;
* доля дней, где каждый нутриент в ±10% нормы;
* отказы (остаток меньше MIN_REST_KCAL);
* время пересборки.

Варианты: «всё заново» (было) и «до съеденного как было». Сценарии:
пицца на обед, бургер на ужин, сырники на завтрак (оставлять нечего —
контроль: варианты обязаны совпасть).

    python -m experiments.eaten_keep

Результат (6 профилей measure_solver × 6 сидов, соль остатка уже учтена):

    сценарий               вариант             проверки   ±10%   отказы
    пицца на обед          всё заново             100%     92%      0
                           до съеденного как было  67%     42%      1
    бургер на ужин         всё заново             100%     92%      0
                           до съеденного как было  22%     17%     13
    сырники на завтрак     оба (контроль)         100%     89%      0

«Всё заново» проходит проверки, потому что ужимает уже съеденный завтрак, —
число красивое, а план нечестный. Провалы честного варианта почти все
неустранимы: съеденное вместе с оставленным само даёт 100-140% нормы жиров
или больше суточного предела натрия, остатку дня исправлять нечего. Такой
день агент называет прямо («из-за съеденного день в норму не уложился»),
а при остатке меньше MIN_REST_KCAL — отказывается добавлять приёмы пищи.

До учёта соли съеденного (rebuild_rest, sodium_limit) контроль проходил
проверки в 3% дней: остаток собирался под весь суточный предел натрия.
"""

from __future__ import annotations

import time

from experiments.measure_solver import PROFILES
from foods import load_catalog
from nodes_plan import PlanNodes
from profile_rules import _MEAL_SLOTS
from solver import (
    DAY_TEMPLATE,
    PlanNotFeasible,
    build_day,
    check_plan,
    eaten_item,
    rebuild_rest,
)
from targets import compute_targets

SEEDS = range(6)
# (название, шаблон в name, граммы, приём пищи)
SCENARIOS = [
    ("пицца на обед", "Pizza, cheese", 300, "lunch"),
    ("бургер на ужин", "Hamburger (McDonalds)", 250, "dinner"),
    ("пицца на обед, большая", "Pizza, cheese",
     600, "lunch"),
    ("сырники на завтрак (контроль)", "Pancakes, plain", 200, "breakfast"),
]


class _Host(PlanNodes):
    def __init__(self, catalog):
        self.catalog = catalog


def main() -> None:
    catalog = load_catalog()
    host = _Host(catalog)
    print(f"{'сценарий':32} {'вариант':22} {'проверки':>9} {'±10%':>6} {'отказы':>7} {'с/день':>7}")
    for title, pattern, grams, meal in SCENARIOS:
        row = catalog[catalog["name"].str.startswith(pattern)]
        if row.empty:
            print(f"{title}: нет блюда «{pattern}»")
            continue
        fdc_id = int(row.iloc[0]["fdc_id"])
        for variant in ("всё заново", "до съеденного как было"):
            passed = band = failed = total = 0
            spent = 0.0
            for profile in PROFILES.values():
                targets = compute_targets(profile)
                for seed in SEEDS:
                    try:
                        before = build_day(catalog, targets, exclude=profile.exclude, seed=seed)
                    except PlanNotFeasible:
                        continue
                    covered = set(_MEAL_SLOTS[meal])
                    slot = DAY_TEMPLATE[_MEAL_SLOTS[meal][0]][0]
                    eaten = [eaten_item(catalog, fdc_id, grams, slot)]
                    kept = {}
                    if variant != "всё заново":
                        state = {"last_plan": before.model_dump(mode="json")}
                        kept = host._kept_before(state, profile, [{"meal": meal}], covered)
                    total += 1
                    start = time.perf_counter()
                    try:
                        plan = rebuild_rest(catalog, targets, eaten, covered,
                                            exclude=profile.exclude, seed=seed, kept=kept)
                    except PlanNotFeasible:
                        failed += 1
                        continue
                    finally:
                        spent += time.perf_counter() - start
                    passed += all(check_plan(plan).values())
                    band += all(abs(v) <= 0.10 for v in plan.deviation.values())
            print(f"{title:32} {variant:22} {passed / total:>8.0%} {band / total:>6.0%} "
                  f"{failed:>7} {spent / total:>7.2f}")


if __name__ == "__main__":
    main()
