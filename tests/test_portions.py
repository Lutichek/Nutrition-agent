"""
Граммы съеденного, когда человек их не назвал (portions.py).

Живой прогон: «два куска пиццы» = 1079 ккал и «шоколадный батончик» =
160 г конфет — бралась справочная порция каталога («1 cup», целая пицца).
Потом фиксированный порядок единиц: батончик = 1 конфета (8 г), яблоко
«штукой» = ломтик (25 г). Числа берутся из таблицы порций USDA по той
единице, что назвал человек; модель их не оценивает.
"""

from __future__ import annotations

import pytest

import portions
from portions import eaten_grams, unit_portions


def fdc(catalog, text: str) -> int:
    return int(catalog[catalog["name"].str.contains(text, regex=False)].iloc[0]["fdc_id"])


def test_artifact_covers_the_catalog(catalog):
    assert set(unit_portions()) >= set(catalog["fdc_id"].astype(int))


def test_named_grams_win(catalog):
    assert eaten_grams(fdc(catalog, "Apple, raw"), 150, 3, 999, unit="large") == 150


def test_pieces_of_pizza(catalog):
    pizza = fdc(catalog, "Pizza with pepperoni, from frozen, medium crust")
    piece = dict(unit_portions()[pizza])["1 piece, NFS"]
    assert eaten_grams(pizza, None, 2, 999, unit="slice") == pytest.approx(2 * piece)


def test_chocolate_bar_is_not_one_candy(catalog):
    candy = fdc(catalog, "Chocolate candy")
    grams = eaten_grams(candy, None, None, 999, unit="bar")
    assert grams > 8, "не одна конфета"
    assert grams < 160, "и не «1 cup»"


def test_whole_apple_is_not_a_slice(catalog):
    apple = fdc(catalog, "Apple, raw")
    assert eaten_grams(apple, None, None, 999, unit="piece") > 100
    assert eaten_grams(apple, None, None, 999) == pytest.approx(dict(unit_portions()[apple])["1 medium"])


def test_without_artifact_falls_back_to_catalog_portion(monkeypatch):
    monkeypatch.setattr(portions, "unit_portions", lambda: {})
    assert eaten_grams(1, None, 2, 120.0) == 240.0


def test_name_vectors_match_the_catalog(catalog):
    """Пересобрали каталог, а векторы названий — нет: новые блюда молча
    никогда не найдутся, удалённые — найдутся и упадут. Как и для разметки
    каталога (test_shipped_catalog_matches_the_rule), расхождение — провал."""
    import food_search

    if not food_search.available():
        pytest.skip("векторы названий не собраны: python food_search.py --rebuild")
    ids, vectors = food_search._load()
    assert sorted(int(i) for i in ids) == sorted(catalog["fdc_id"].astype(int)), \
        "выполните python food_search.py --rebuild"
    assert len(ids) == len(vectors)
