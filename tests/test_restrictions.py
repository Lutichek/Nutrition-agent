"""
Тесты пищевых ограничений.

Оба дефекта, ради которых написан модуль, были тихими: рацион собирался,
проверки проходили, и только в тарелке оказывалось не то. Поэтому проверки
здесь идут не по коду, а по наблюдавшимся случаям.
"""

from __future__ import annotations

import re

import pytest

from restrictions import (
    CATEGORIES,
    describe,
    expand,
    to_category_pattern,
    to_pattern,
)
from solver import _filter_catalog


def matches(term: str, name: str) -> bool:
    return bool(re.search(to_pattern([term]), name.lower()))


class TestCategories:
    def test_category_expands_to_many_words(self):
        """«Сладости» — это не одно слово, а полсотни конкретных блюд."""
        assert len(expand(["sweets"])) > 30

    def test_plain_term_passes_through(self):
        assert expand(["pork"]) == ["pork"]

    def test_unknown_term_is_kept_not_dropped(self):
        """Незнакомое ограничение применяем буквально, а не выбрасываем."""
        assert expand(["unobtainium"]) == ["unobtainium"]

    def test_aliases_reach_the_same_category(self):
        assert expand(["dairy"]) == expand(["lactose"])

    def test_vegan_covers_vegetarian(self):
        assert set(expand(["vegetarian"])) <= set(expand(["vegan"]))

    def test_no_category_is_empty(self):
        assert all(words for words in CATEGORIES.values())


class TestWordBoundaries:
    """Наивная подстрока давала тихие ложные срабатывания.

    «oil» находился внутри «broiled» и выбрасывал 82 блюда, включая обычную
    запечённую курицу; «ham» — внутри «graham cracker». Человек этого не видел:
    рацион просто становился беднее без объяснения.
    """

    @pytest.mark.parametrize("term,name", [
        ("oil", "Chicken breast, baked or broiled"),
        ("ham", "Cookie, graham cracker"),
        ("egg", "Eggplant, cooked"),
        ("rice", "Licorice candy"),
    ])
    def test_does_not_match_inside_another_word(self, term, name):
        assert not matches(term, name)

    @pytest.mark.parametrize("term,name", [
        ("oil", "Vegetable oil"),
        ("ham", "Ham, sliced"),
        ("egg", "Egg, fried"),
        ("pork", "Pork chop, fried"),
    ])
    def test_matches_the_whole_word(self, term, name):
        assert matches(term, name)

    def test_matches_english_plural(self):
        assert matches("doughnut", "Doughnuts, glazed")

    def test_special_characters_do_not_break_the_pattern(self):
        """Слова приходят от модели, а через неё — от человека."""
        assert to_pattern(["a(b"]) is not None
        assert not matches("a(b", "Chicken, roasted")


class TestAgainstTheCatalog:
    """Наблюдавшийся случай целиком.

    Запрос: «не ем свинину, сладости, мучное и аллергия на лактозу».
    В плане оказались «Пончик, бисквитный, простой» и «Пончик с сахаром»:
    модель вернула ["pork","milk","cheese","yogurt","cream"], а обе категории
    потеряла молча — и ничто в списке слову «doughnut» не соответствовало.
    """

    EXCLUDE = ["pork", "sweets", "flour", "lactose"]

    def test_doughnuts_are_gone(self, catalog):
        left = _filter_catalog(catalog, self.EXCLUDE)
        names = left["name"].str.lower()
        assert not names.str.contains("doughnut|donut", regex=True, na=False).any()

    def test_sweets_and_pastry_are_gone(self, catalog):
        left = _filter_catalog(catalog, self.EXCLUDE)
        names = left["name"].str.lower()
        for word in ("cookie", "cake", "candy", "bread", "muffin", "breaded"):
            pattern = to_pattern([word])
            assert not names.str.contains(pattern, regex=True, na=False).any(), word

    def test_breadfruit_is_not_bread(self, catalog):
        """Совпадение по целому слову: «breadfruit» — это фрукт, а не хлеб."""
        assert matches("bread", "White bread")
        assert not matches("bread", "Breadfruit, cooked")

    def test_enough_dishes_remain_to_build_a_day(self, catalog):
        """Ограничения не должны выкашивать каталог до непригодности."""
        left = _filter_catalog(catalog, self.EXCLUDE)
        assert len(left) > 1000
        assert set(left["role"]) >= {"main", "side", "snack", "breakfast"}

    def test_baked_chicken_survives_an_oil_restriction(self, catalog):
        """Раньше «не ем масло» уносило и запечённую курицу тоже."""
        left = _filter_catalog(catalog, ["oil"])
        assert left["name"].str.lower().str.contains("broiled", na=False).any()


class TestFilteringByCatalogSection:
    """Раздел справочника знает то, чего не знает название.

    В «Gyro sandwich» нет слова «bread», в «Italian Ice» — ни «candy»,
    ни «ice cream». По названию через фильтр «мучное» проходил 211 сэндвич,
    и оба этих блюда попали в реальный план человека, который просил
    без мучного и без сладкого.
    """

    @pytest.mark.parametrize("dish", ["gyro", "sandwich", "burger", "pizza"])
    def test_flour_restriction_removes_bread_shaped_dishes(self, catalog, dish):
        left = _filter_catalog(catalog, ["flour"])
        # Овощ «для сэндвича» — не сэндвич, поэтому смотрим на раздел.
        assert not left["category"].str.lower().str.contains(dish, na=False).any()

    def test_sweets_restriction_removes_italian_ice(self, catalog):
        left = _filter_catalog(catalog, ["sweets"])
        assert not left["name"].str.lower().str.contains("italian ice", na=False).any()

    def test_both_mechanisms_are_needed(self, catalog):
        """Раздел знает тип блюда, название — состав. Порознь дырявы оба."""
        # Свинина прячется внутри раздела «Meat mixed dishes» — её видно
        # только по названию.
        by_name = _filter_catalog(catalog, ["pork"])
        assert not by_name["name"].str.lower().str.contains(r"\bpork\b", regex=True, na=False).any()

        # А лепёшка в гиросе — только по разделу.
        assert to_category_pattern(["flour"]) != ""

    def test_unknown_term_has_no_category_rule(self):
        assert to_category_pattern(["quinoa"]) == ""


class TestDescription:
    def test_categories_are_shown_in_russian(self):
        assert describe(["pork", "sweets", "flour", "lactose"]) == \
            "свинина, сладости, мучное, лактоза"

    def test_unknown_term_shown_as_is(self):
        assert describe(["quinoa"]) == "quinoa"

    def test_empty_is_empty(self):
        assert describe(None) == "" and describe([]) == ""
