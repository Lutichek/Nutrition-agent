"""
Замер качества солвера: профили × сиды, одна таблица, один вердикт.

Зачем это отдельным скриптом. Все числа в ``solver.py`` — веса невязки,
допуск на приём пищи, число итераций, размеры порций — подбирались замером.
Каждый раз приходилось заново писать один и тот же прогон, и каждый раз
чуть иначе: то другой набор профилей, то другое число сидов. Сравнивать
такие замеры между собой нельзя.

Что меряется:

``провалов``
    Сколько прогонов не прошли ``check_plan``. Главный показатель:
    любое значение больше нуля — отказ.

``откл. ккал``
    Среднее отклонение от нормы калорийности. Допуск в проверках 6%,
    так что 4% — это уже опасная близость к границе.

``худший слот``
    Во сколько раз самый крупный приём пищи превысил свою долю дня.
    Показатель появился после случая, когда план идеально сходился
    по суточной сумме, а состоял из 745 ккал фисташек «на перекус».

``время``
    Секунд на один день. Растёт линейно по числу дней в меню, поэтому
    на месяц умножай на 30.

Запуск::

    python -m experiments.measure_solver              # текущее состояние
    python -m experiments.measure_solver --seeds 12   # надёжнее, но дольше
    python -m experiments.measure_solver --save до    # сохранить как точку отсчёта
    python -m experiments.measure_solver --against до # сравнить с ней

Порядок работы: сохранить точку отсчёта ДО правки, внести правку,
сравнить. Изменение принимается, только если ни один показатель
не ухудшился.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from foods import load_catalog
from solver import DAY_TEMPLATE, SLOT_TOLERANCE, build_day, check_plan
from targets import Profile, compute_targets

# Профили намеренно разные по требованиям к каталогу: дефицит и профицит,
# крайние значения массы, все уровни активности. Дефекты солвера почти
# всегда проявляются на краях, а не в середине.
PROFILES: dict[str, Profile] = {
    "похудение, женщина": Profile(
        sex="female", age=31, height_cm=168, weight_kg=74,
        activity="light", goal="lose"),
    "похудение, мужчина": Profile(
        sex="male", age=40, height_cm=182, weight_kg=95,
        activity="moderate", goal="lose"),
    "набор массы": Profile(
        sex="male", age=25, height_cm=178, weight_kg=63,
        activity="high", goal="gain"),
    "удержание веса": Profile(
        sex="female", age=55, height_cm=160, weight_kg=58,
        activity="sedentary"),
    "с ограничениями": Profile(
        sex="male", age=24, height_cm=168, weight_kg=65,
        activity="moderate", goal="gain",
        exclude=["pork", "sweets", "flour", "lactose"]),
}

BASELINE_DIR = Path(__file__).parent / "_research" / "baselines"


def measure(seeds: int) -> dict:
    """Прогнать все профили по всем сидам и собрать показатели."""
    catalog = load_catalog()
    shares = [share for _slot, _role, share in DAY_TEMPLATE]

    failures = 0
    runs = 0
    kcal_deviations: list[float] = []
    slot_ratios: list[float] = []
    durations: list[float] = []
    broken: list[str] = []

    for label, profile in PROFILES.items():
        targets = compute_targets(profile)

        for seed in range(seeds):
            runs += 1
            started = time.perf_counter()
            try:
                plan = build_day(catalog, targets, exclude=profile.exclude, seed=seed)
            except Exception as error:
                failures += 1
                broken.append(f"{label}, seed={seed}: {type(error).__name__}: {error}")
                continue
            durations.append(time.perf_counter() - started)

            checks = check_plan(plan)
            if not all(checks.values()):
                failures += 1
                not_passed = [name for name, ok in checks.items() if not ok]
                broken.append(f"{label}, seed={seed}: не прошли {', '.join(not_passed)}")

            kcal_deviations.append(
                abs(plan.totals["kcal"] - targets.kcal) / targets.kcal)

            # Во сколько раз приём пищи превысил свою долю дня.
            for item, share in zip(plan.items, shares, strict=True):
                slot_ratios.append(item.kcal / (share * targets.kcal))

    return {
        "runs": runs,
        "failures": failures,
        "kcal_deviation": statistics.mean(kcal_deviations) if kcal_deviations else 0.0,
        "worst_slot": max(slot_ratios) if slot_ratios else 0.0,
        "seconds_per_day": statistics.mean(durations) if durations else 0.0,
        "broken": broken,
        "slot_tolerance": SLOT_TOLERANCE,
    }


def show(result: dict, against: dict | None = None) -> None:
    rows = [
        ("провалов проверок", result["failures"], against and against["failures"],
         "{:.0f}", "меньше"),
        ("отклонение ккал", result["kcal_deviation"],
         against and against["kcal_deviation"], "{:.1%}", "меньше"),
        ("худший слот", result["worst_slot"], against and against["worst_slot"],
         "{:.1f}x", "меньше"),
        ("секунд на день", result["seconds_per_day"],
         against and against["seconds_per_day"], "{:.2f}", "меньше"),
    ]

    print(f"\nПрогонов: {result['runs']} "
          f"({len(PROFILES)} профилей × {result['runs'] // len(PROFILES)} сидов)\n")

    header = f"  {'показатель':<22} {'сейчас':>10}"
    if against:
        header += f" {'было':>10}  {'':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    worse = False
    for name, now, before, fmt, _direction in rows:
        line = f"  {name:<22} {fmt.format(now):>10}"
        if before is not None:
            # Все показатели «чем меньше, тем лучше». Порог в 1% отсекает
            # дрожание от порядка обхода словарей и округлений.
            delta = now - before
            if abs(delta) < abs(before) * 0.01 or (before == 0 and now == 0):
                mark = "="
            elif delta > 0:
                mark, worse = "хуже", True
            else:
                mark = "лучше"
            line += f" {fmt.format(before):>10}  {mark:>8}"
        print(line)

    if result["broken"]:
        print("\n  Что именно не сошлось:")
        for item in result["broken"][:10]:
            print(f"    {item}")
        if len(result["broken"]) > 10:
            print(f"    ... и ещё {len(result['broken']) - 10}")

    print()
    if result["failures"]:
        print("  ВЕРДИКТ: отказ — есть непройденные проверки.")
    elif against and worse:
        print("  ВЕРДИКТ: отказ — показатели ухудшились.")
    elif against:
        print("  ВЕРДИКТ: можно принимать.")
    else:
        print("  Точки отсчёта нет. Сохрани: --save <имя>")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6,
                        help="сидов на профиль (по умолчанию 6)")
    parser.add_argument("--save", metavar="ИМЯ",
                        help="сохранить результат как точку отсчёта")
    parser.add_argument("--against", metavar="ИМЯ",
                        help="сравнить с сохранённой точкой отсчёта")
    args = parser.parse_args()

    result = measure(args.seeds)

    baseline = None
    if args.against:
        path = BASELINE_DIR / f"{args.against}.json"
        if not path.exists():
            parser.error(f"нет такой точки отсчёта: {path}")
        baseline = json.loads(path.read_text(encoding="utf-8"))

    show(result, baseline)

    if args.save:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        path = BASELINE_DIR / f"{args.save}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"  Точка отсчёта сохранена → {path}\n")


if __name__ == "__main__":
    main()
