"""Аудит ограничения: что проходит фильтр, хотя не должно.

Запуск из корня проекта:

    python <skill>/scripts/leak_scan.py gluten                  # подозрительные ингредиенты
    python <skill>/scripts/leak_scan.py gluten --hidden "teriyaki|malt"   # посчитать утечку
    python <skill>/scripts/leak_scan.py gluten --dish "Croutons"          # полный состав блюда
    python <skill>/scripts/leak_scan.py gluten --removed                  # что отсечено зря?

Смотрит в ПОЛНЫЙ состав (ingredient_leaves.parquet), а не в показываемый:
там раскрыты вложенные рецептуры и не отброшены доли меньше 1%.
Совпадения — только через restrictions.matches_words (RE2 в pandas 3
не понимает \\b для кириллицы, и наивный .str.contains врёт молча).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from foods import load_catalog  # noqa: E402
from ingredients import LEAVES_PATH  # noqa: E402
from restrictions import (  # noqa: E402
    ALIASES,
    CATEGORIES,
    expand,
    ingredient_terms,
    matches_words,
    to_category_pattern,
    to_pattern,
)
from solver import _dishes_with_ingredient, _filter_catalog  # noqa: E402

# Ингредиенты, за которыми чаще всего прячется чужой состав: соусы, смеси,
# напитки, «NFS». Именно там нашлись свинина в «Breakfast meat», пшеница
# в терияки, солод в «Ovaltine».
COMPOSITE = re.compile(
    r"(?i)sauce|mix|seasoning|dressing|gravy|\bnfs\b|as ingredient|beverage|"
    r"cereal|powder|spread|marinade|broth|stock|bouillon|topping|filling|crumb|"
    r"batter|breading|soup|syrup|flavor|crouton|malt"
)
# Известные скрытые источники: продукт есть, а слова категории в названии
# ингредиента нет. Общее знание о продуктах, а не о нашем каталоге —
# поэтому список, а не догадка по одному блюду. Пополнять по находкам.
KNOWN_HIDDEN = {
    # Первый аудит глютена: соусы и подливы SR без рецептуры — главный
    # источник (gravy 34–45 блюд, cream of mushroom ~20, cheese sauce ~25).
    "gluten": r"soy sauce|sauce, soy|teriyaki|hoisin|\bmalt|barley|\bbeer\b|crouton|seitan|"
              r"bulgur|semolina|farina|spelt|graham|panko|breadcrumb|bread, crumbs|"
              r"gravy|cream of mushroom|cheese sauce|lo mein|chow mein|fried rice|"
              r"pretzel|breading|nugget|ice cream cone|vermicelli mix|surimi|"
              r"cereal grain beverage",
    "gluten?": r"\boat",  # чистый овёс без глютена, но при целиакии — только сертифицированный
    "eggs": r"mayonnaise|custard|meringue|hollandaise|aioli|eclair|cream puff|"
            r"tartar sauce|caesar|egg noodle",
    "seafood": r"worcestershire|fish sauce|oyster sauce|anchov|caesar|surimi|nuoc",
    "nuts": r"pesto|marzipan|praline|nougat|nutella|gianduja|baklava",
    "lactose": r"whey|casein|ghee|eclair|cream puff|ranch|alfredo|caesar",
    "pork": r"gelatin|lard|bacon grease|drippings|pepperoni|salami",
}

# Шум, в котором прятаться нечему: вода, витамины, жир на выбор,
# отдельный овощ «как ингредиент».
NOISE = re.compile(
    r"(?i)\bwater\b|vitamin|calcium|iron as|salt, table|"
    r"cooked, as ingredient|^vegetable oil|table fat|^oil\b"
)


def show_removed(base: pd.DataFrame, allowed_ids: set, leaves: pd.DataFrame,
                 exclude: list[str], top: int) -> None:
    """Отсечённое — по проходу фильтра и сработавшему слову.

    Ложное отсечение видно именно по слову: подстрока «noodles» в разделе
    «Pasta, noodles, cooked grains» уносила гречку, а не лапшу.
    """
    removed = base[~base["fdc_id"].isin(allowed_ids)]
    category_re = to_category_pattern(exclude)
    composition = _dishes_with_ingredient(tuple(sorted(ingredient_terms(exclude))))
    words = expand(exclude)

    rows = []
    for row in removed.itertuples():
        name = pd.Series([row.name], dtype=object)
        word = next((w for w in words if matches_words(name, to_pattern([w])).iloc[0]), None)
        if word:
            rows.append(("название", word, row.name))
            continue
        match = re.search(category_re, str(row.category).lower()) if category_re else None
        if match:
            rows.append(("раздел", match.group(0), row.name))
            continue
        if row.fdc_id in composition:
            own = leaves.loc[leaves["fdc_id"] == row.fdc_id, "name_en"].astype(object)
            hit = own[matches_words(own, to_pattern(exclude), skip_absent=True).to_numpy()]
            rows.append(("состав", hit.iloc[0] if len(hit) else "?", row.name))
    frame = pd.DataFrame(rows, columns=["проход", "сработало", "блюдо"])
    print(f"Отсечено {len(removed)} блюд. По проходу и слову — просмотри, нет ли лишнего")
    groups = sorted(frame.groupby(["проход", "сработало"]), key=lambda kv: -len(kv[1]))
    print(f"(групп {len(groups)}, показаны первые {top}; редкие группы — внизу, смотри и их)\n")
    for (stage, trigger), group in groups[:top]:
        examples = "; ".join(group["блюдо"].head(4))
        print(f"  [{stage}] «{trigger}»: {len(group)} — {examples}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exclude", nargs="+", help="ограничения как в профиле: gluten, pork…")
    parser.add_argument("--hidden", help="регулярка: что считать скрытым нарушением")
    parser.add_argument("--dish", help="показать полный состав блюда (подстрока названия)")
    parser.add_argument("--removed", action="store_true",
                        help="что фильтр отсёк и каким проходом — искать ложные отсечения")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    # Незнакомое слово ищется буквально — и молча. «gluten-free» так уносит
    # именно безглютеновое, «celiac» не отсекает ничего.
    for term in args.exclude:
        key = term.strip().lower()
        if key not in CATEGORIES and key not in ALIASES:
            print(f"⚠ «{term}» — не категория и не синоним (ALIASES): ищется буквально. "
                  f"Если это формулировка категории — добавь в ALIASES.")

    catalog = load_catalog()
    names = dict(zip(catalog["fdc_id"], catalog["name"], strict=True))
    leaves = pd.read_parquet(LEAVES_PATH)

    base = _filter_catalog(catalog, None)
    allowed = _filter_catalog(catalog, args.exclude)
    allowed_ids = set(allowed["fdc_id"])
    print(f"Ограничение {args.exclude} → слова: {expand(args.exclude)[:12]}…")
    print(f"По составу проверяются: {ingredient_terms(args.exclude) or 'ничего (тип блюда)'}")
    print(f"Подбор: {len(base)} → {len(allowed)} блюд")
    by_role = pd.concat([base["role"].value_counts().rename("было"),
                         allowed["role"].value_counts().rename("стало")], axis=1).fillna(0)
    by_role["осталось"] = (by_role["стало"] / by_role["было"]).round(2)
    print(by_role.to_string(), "\n")

    in_pool = leaves[leaves["fdc_id"].isin(allowed_ids)]

    if args.removed:
        show_removed(base, allowed_ids, leaves, args.exclude, args.top)
        return

    if args.dish:
        for fdc_id in [f for f, n in names.items() if args.dish.lower() in n.lower()][:5]:
            status = "В ПОДБОРЕ" if fdc_id in allowed_ids else "отсечено"
            print(f"{fdc_id} {names[fdc_id]} [{status}]")
            for leaf in leaves.loc[leaves["fdc_id"] == fdc_id, "name_en"]:
                print("   ", leaf)
        return

    if args.hidden:
        unique = pd.Series(in_pool["name_en"].unique(), dtype=object)
        hit = set(unique[unique.str.contains(args.hidden, case=False, regex=True)])
        found = in_pool[in_pool["name_en"].isin(hit)]
        print(f"Проходят фильтр со скрытым «{args.hidden}»: {found['fdc_id'].nunique()} блюд")
        print(found["name_en"].value_counts().to_string())
        print("\nПримеры блюд:")
        for fdc_id in sorted(found["fdc_id"].unique())[: args.top]:
            print("   ", fdc_id, names[fdc_id])
        return

    unique = pd.Series(in_pool["name_en"].unique(), dtype=object)

    # 1. Известные скрытые источники для названных ограничений.
    print("Известные скрытые источники (KNOWN_HIDDEN):")
    for key, regex in KNOWN_HIDDEN.items():
        if key.rstrip("?") not in [term.lower() for term in args.exclude]:
            continue
        hit = set(unique[unique.str.contains(regex, case=False, regex=True)])
        found = in_pool[in_pool["name_en"].isin(hit)]
        mark = "  (СПОРНО — решает человек)" if key.endswith("?") else ""
        print(f"  {key}: {found['fdc_id'].nunique()} блюд{mark}")
        for leaf, count in found.groupby("name_en")["fdc_id"].nunique().sort_values(
                ascending=False).items():
            print(f"      {count:4} {leaf}")
    print()

    # 2. Без гипотезы: составные ингредиенты в оставшихся блюдах. Редкие
    # утечки (1-2 блюда) внизу списка — смотри его целиком, не только топ.
    pattern = to_pattern(args.exclude)
    composite = unique[unique.str.contains(COMPOSITE) & ~unique.str.contains(NOISE)]
    flagged = composite[~matches_words(composite, pattern, skip_absent=True).to_numpy()]
    counts = (in_pool[in_pool["name_en"].isin(set(flagged))]
              .groupby("name_en")["fdc_id"].nunique().sort_values(ascending=False))
    print(f"Составные ингредиенты в оставшихся блюдах (просмотри глазами, топ {args.top}):")
    print(counts.head(args.top).to_string())


if __name__ == "__main__":
    main()
