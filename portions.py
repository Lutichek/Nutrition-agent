"""
Сколько граммов съедено, если человек граммов не назвал (этап 2).

Наблюдалось на живой модели: «два куска пиццы» превращались в 1079 ккал,
«шоколадный батончик» — в 160 г конфет. Брался справочный размер порции
каталога, а он в FNDDS — «1 cup» или целая пицца, а не кусок и не батончик.
Тихая ошибка: план собирается, числа уверенные и неверные в разы.

Правильные числа в данных USDA уже есть — таблица порций FNDDS
(food_portion.csv): граммы на «1 piece», «1 slice», «1 medium pancake»
и особая строка «Quantity not specified» — типичная порция, которую
USDA берёт, когда количество не названо. Модель граммы НЕ оценивает:

* граммы названы — берутся как есть;
* иначе — количество (не названо — одна) × вес единицы из таблицы:
  той, что назвал человек («батончик» → bar), а без неё — средней штуки;
* единицы в таблице нет — «Quantity not specified», затем порция каталога.

Сырая таблица в образ не попадает (data/raw исключён), поэтому нужные
строки собираются в data/processed/unit_portions.parquet::

    python portions.py --rebuild
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PORTIONS_PATH = ROOT / "data" / "processed" / "unit_portions.parquet"
RAW_PATH = ROOT / "data" / "raw" / "usda" / "survey" / "food_portion.csv"

UNSPECIFIED = "Quantity not specified"
# Единица по умолчанию — «средняя штука». Фиксированный порядок «piece,
# slice, …» на живом прогоне давал батончик = 1 конфета (8 г) и яблоко =
# 1 ломтик (25 г): какую единицу назвал человек, решает его фраза (unit),
# а без неё — средний размер.
DEFAULT_UNITS = ("medium", "regular", "piece", "item", "sandwich", "bun", "slice",
                 "small", "large")
# Названная единица → какие описания USDA ей соответствуют, по порядку.
UNIT_ALIASES = {
    "bar": ("bar", "regular size", "fun/snack size"),
    "slice": ("slice", "piece"),
    # «Штука» — целая вещь: ломтик тут не замена (яблоко «штукой» на живом
    # прогоне становилось ломтиком в 25 г).
    "piece": ("piece", "item"),
    "cup": ("cup",),
    "small": ("small",), "medium": ("medium", "regular"), "large": ("large",),
}
# Не единицы вовсе: «1 surface inch» — это квадратный дюйм пиццы.
NOT_UNITS = ("surface inch", "cubic inch", UNSPECIFIED.lower())

# ── Напитки: количество спрашивается, а не угадывается ────────────────
#
# Живой прогон: «выпил колу» без количества — 512 г, «1 medium drink» из
# таблицы порций. Средний стакан фастфуда — не то, что человек имел
# в виду, а ошибка в 200 ккал на сладком напитке — уже заметная часть дня.
# Поэтому у калорийного напитка без количества агент СПРАШИВАЕТ, сколько
# выпито. Напиток — по разделу справочника, а не по слову модели.
DRINK_CATEGORIES = frozenset({
    "Apple juice", "Citrus juice", "Other fruit juice", "Vegetable juice", "Fruit drinks",
    "Soft drinks", "Diet soft drinks", "Sport and energy drinks",
    "Diet sport and energy drinks", "Other diet drinks", "Coffee", "Tea",
    "Bottled water", "Tap water", "Enhanced water", "Flavored or carbonated water",
    "Flavored milk, lowfat", "Flavored milk, nonfat", "Flavored milk, reduced fat",
    "Flavored milk, whole", "Milk, lowfat", "Milk, nonfat", "Milk, reduced fat",
    "Milk, whole", "Plant-based milk", "Milk shakes and other dairy drinks",
    "Smoothies and grain drinks", "Nutritional beverages",
})
# Ниже этого спрашивать незачем: вода, чёрный чай и кофе, диетическая
# газировка — 0-4 ккал на 100 г, даже литр — меньше 40 ккал. Кола 42,
# сок 45, молоко 43-69.
DRINK_ASK_KCAL_100G = 10.0

# Объём посуды, которую называют в ответ на «сколько выпили?». Стакан —
# 250 мл, мера русских рецептов (USDA «cup» — 240 мл, почти то же);
# банка газировки — 0,33 л, бутылка — 0,5 л: самые частые объёмы.
VESSEL_ML = {"стакан": 250.0, "чашк": 250.0, "кружк": 300.0, "банк": 330.0,
             "бутылк": 500.0}
ML_PER_FL_OZ = 29.5735


def is_drink(category: str | None, kcal_per_100g: float | None) -> bool:
    """Напиток, количество которого стоит спросить."""
    return (category in DRINK_CATEGORIES
            and float(kcal_per_100g or 0.0) >= DRINK_ASK_KCAL_100G)


def grams_per_ml(fdc_id: int) -> float:
    """Плотность напитка по таблице USDA: граммы жидкой унции / 29.57 мл.

    Кола — 31 г на унцию, 1.05 г/мл; без строки «fl oz» — как у воды.
    «with ice» не берём: там часть объёма занимает лёд.
    """
    for description, weight in unit_portions().get(int(fdc_id), []):
        lowered = description.lower()
        if lowered.startswith("1 fl oz") and "with ice" not in lowered:
            return weight / ML_PER_FL_OZ
    return 1.0


def unspecified_grams(fdc_id: int, catalog_portion_g: float) -> float:
    """«Не помню сколько» — типичная порция USDA («Quantity not specified»)."""
    for description, weight in unit_portions().get(int(fdc_id), []):
        if description == UNSPECIFIED:
            return weight
    return catalog_portion_g


def build(catalog_ids: set[int]) -> pd.DataFrame:
    """Строки таблицы порций для блюд каталога: fdc_id, описание, граммы."""
    raw = pd.read_csv(RAW_PATH, usecols=["fdc_id", "seq_num", "portion_description",
                                         "gram_weight"])
    raw = raw[raw["fdc_id"].isin(catalog_ids) & (raw["gram_weight"] > 0)]
    raw = raw.rename(columns={"portion_description": "description", "gram_weight": "grams"})
    raw["description"] = raw["description"].fillna("").str.strip()
    return raw.sort_values(["fdc_id", "seq_num"])[["fdc_id", "description", "grams"]]


@lru_cache(maxsize=1)
def unit_portions() -> dict[int, list[tuple[str, float]]]:
    """fdc_id → [(описание, граммы)]; без артефакта — пусто (запасной путь)."""
    if not PORTIONS_PATH.exists():
        return {}
    table = pd.read_parquet(PORTIONS_PATH)
    result: dict[int, list[tuple[str, float]]] = {}
    for fdc_id, description, grams in table.itertuples(index=False):
        result.setdefault(int(fdc_id), []).append((str(description), float(grams)))
    return result


def eaten_grams(fdc_id: int, grams: float | None, count: float | None,
                catalog_portion_g: float, unit: str | None = None) -> float:
    """Граммы съеденного — по словам человека и данным USDA, не по модели.

    unit — единица, которую НАЗВАЛ человек («батончик» → bar, «кусок» →
    slice); модель лишь сопоставляет слово, число берётся из таблицы.
    """
    if isinstance(grams, (int, float)) and grams > 0:
        return float(grams)
    rows = unit_portions().get(int(fdc_id), [])
    named = isinstance(count, (int, float)) and 0 < count <= 10
    number = float(count) if named else 1.0
    order = [*UNIT_ALIASES.get(str(unit or "").lower(), ()), *DEFAULT_UNITS]
    for word in order:
        for description, weight in rows:
            lowered = description.lower()
            if word in lowered and not any(n in lowered for n in NOT_UNITS):
                return number * weight
    for description, weight in rows:
        if description == UNSPECIFIED:
            return number * weight
    return number * catalog_portion_g


if __name__ == "__main__":
    if "--rebuild" not in sys.argv:
        print(__doc__)
        sys.exit(0)
    from foods import load_catalog

    table = build(set(load_catalog()["fdc_id"].astype(int)))
    table.to_parquet(PORTIONS_PATH, index=False)
    print(f"{PORTIONS_PATH}: {len(table)} строк, {table['fdc_id'].nunique()} блюд")
