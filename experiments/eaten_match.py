"""
Сопоставление съеденного с каталогом: «съел два куска пиццы» → блюдо и граммы.

Фраза идёт в настоящий узел ``match_eaten``: модель выписывает съеденное
и выбирает блюдо из кандидатов, кандидатов ищет код (food_search), граммы
считает код (portions.py). Верно — если найдено нужное число блюд, каждое
того вида, что назван, ни одно не из «ловушек» (кожа вместо курицы,
«картофель со сметаной» вместо сметаны) и граммы в разумных пределах.

Наблюдавшийся случай: на «две куски пиццы» поиск дал 12 видов пиццы,
а модель ответила «ничего не подходит» — вид не назван, а подсказка
требовала выбрать то, что И ЕСТЬ названный продукт. Человек получил
«не нашёл в справочнике». Набор ``plain`` — именно такие фразы: вид
не назван, нужен самый обычный вариант.

Итог (2 прогона; цифры и выбор — у GENERIC_FIRST в nodes_plan.py):
переписанная подсказка + «X, NFS» первым — plain 4/16 → 14/16, named
12/16 без изменений. Остаётся: «выпил сок» без количества — 512 г
(справочная порция FNDDS), сметана и курица иногда «не найдены».

Запуск::

    python -m experiments.eaten_match                  # оба набора, 2 прогона
    python -m experiments.eaten_match --sets plain --repeats 3

Стоит денег: два вызова модели и эмбеддинг на фразу × варианты × прогоны.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nodes_plan  # noqa: E402
import prompts  # noqa: E402
from agent import build_agent  # noqa: E402
from providers import get_clients  # noqa: E402

# (фраза, [варианты слова в английском названии — по одному списку на блюдо],
#  (граммы всего: от, до))
NAMED = [
    ("На обед съел два куска пиццы с пепперони", [["pizza"]], (150, 400)),
    ("На завтрак были блины со сметаной",
     [["pancake", "crepe", "blini"], ["sour cream", "cream, sour"]], (80, 400)),
    ("На ужин съела гречку с курицей", [["buckwheat"], ["chicken"]], (100, 600)),
    ("Перекусила шоколадным батончиком", [["chocolate", "candy", "bar"]], (20, 120)),
    ("На завтрак съел 2 яйца и бутерброд с сыром",
     [["egg"], ["sandwich", "cheese", "bread"]], (60, 400)),
    ("На обед был борщ", [["borscht", "beet", "soup"]], (150, 600)),
    ("Перекусил яблоком", [["apple"]], (80, 300)),
    ("На ужин съел 200 г макарон", [["pasta", "macaroni", "spaghetti", "noodle"]], (190, 210)),
]
PLAIN = [
    ("На обед съел две куски пиццы и выпил колу", [["pizza"], ["cola"]], (300, 900)),
    ("На обед съел 2 куска пиццы, примерно 250 гр", [["pizza"]], (240, 260)),
    ("На обед был суп", [["soup"]], (150, 600)),
    ("На ужин съел котлету с пюре",
     [["patty", "meatball", "cutlet", "burger", "meat loaf"], ["potato"]], (150, 700)),
    ("Перекусил печеньем", [["cookie", "biscuit"]], (10, 150)),
    ("На завтрак выпил сок", [["juice"]], (150, 400)),
    ("На обед съел бутерброд", [["sandwich"]], (60, 400)),
    ("На ужин был салат", [["salad"]], (50, 500)),
]
SETS = {"named": NAMED, "plain": PLAIN}
BAD = ("kiev", "skin", "with sour cream", "dessert pizza", "diet")

# Подсказка выбора до исправления — для сравнения с нынешней.
OLD_PICK = """Сопоставь съеденное с блюдами справочника. Для каждой позиции
выбери номер самого похожего блюда из ЕЁ списка или 0, если подходящего нет.

Выбирай то, что И ЕСТЬ названный продукт или блюдо. Если в варианте он лишь
ингредиент («сметана» → «картофель со сметаной») или это его часть
(«курица» → «куриная кожа»), это не подходит — ставь 0.

{blocks}

Верни ТОЛЬКО JSON: {{"choices": [номер для позиции 1, номер для позиции 2, ...]}}"""
_ANCHOR = "Выбирай то, что И ЕСТЬ названный продукт или блюдо."
PICK = {
    "старая": OLD_PICK,
    "старая + строка": OLD_PICK.replace(
        _ANCHOR, _ANCHOR + "\nЕсли вид не уточнён («пицца», «суп», «сок»), выбери самый обычный,\n"
        "распространённый вариант из списка: 0 — только когда названного\nпродукта в списке нет вовсе."),
    "нынешняя": prompts.EATEN_PICK_PROMPT,
}
# Обобщённая запись каталога («Soup, NFS») первой в кандидатах — или нет.
GENERIC = {"": False, " + NFS": True}


def judge(eaten: list[dict], catalog, expected: list[list[str]], grams: tuple) -> str | None:
    names = [str(catalog.loc[catalog["fdc_id"] == e["fdc_id"], "name"].iloc[0]).lower()
             for e in eaten]
    if len(names) != len(expected):
        return f"блюд {len(names)} вместо {len(expected)}: {names}"
    for name, words in zip(names, expected, strict=True):
        if not any(w in name for w in words):
            return f"не то блюдо: {name}"
        if any(b in name for b in BAD):
            return f"ловушка: {name}"
    total = sum(e["grams"] for e in eaten)
    if not grams[0] <= total <= grams[1]:
        return f"граммы {total:.0f} вне {grams}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sets", nargs="*", default=list(SETS), choices=list(SETS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--pick", nargs="*", default=list(PICK), choices=list(PICK))
    args = parser.parse_args()

    client, model, embed_client = get_clients("polza")
    agent = build_agent(client, model=model, embed_client=embed_client)
    variants = {f"{p}{g}": (PICK[p], flag) for p in args.pick for g, flag in GENERIC.items()}

    for set_name in args.sets:
        stats = {name: [0, 0] for name in variants}
        for repeat in range(args.repeats):
            for phrase, expected, grams in SETS[set_name]:
                for name, (prompt, generic) in variants.items():
                    nodes_plan.EATEN_PICK_PROMPT = prompt
                    nodes_plan.GENERIC_FIRST = generic
                    update = agent._node_match_eaten({"question": phrase})
                    eaten = update.get("eaten_new") or []
                    problem = (judge(eaten, agent.catalog, expected, grams) if eaten
                               else f"ответ: {str(update.get('answer'))[:60]}")
                    stats[name][0] += problem is None
                    stats[name][1] += 1
                    if problem:
                        print(f"  [{set_name} {repeat + 1}] {name:<24} ✗ «{phrase[:40]}»: {problem}",
                              flush=True)
        print(f"\n{set_name} ({len(SETS[set_name])} фраз × {args.repeats}):")
        for name, (ok, n) in stats.items():
            print(f"  {name:<24} верно {ok}/{n}")
        print()


if __name__ == "__main__":
    main()
