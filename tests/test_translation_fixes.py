"""
Тесты исправлений перевода.

Модуль появился не от любви к порядку. Часть правок жила разовыми скриптами
и применялась прямо к parquet: в коде их не было нигде. Такая правка держится
до первой пересборки каталога, после чего «Омлет или scrambled eggs»
и «Горчица, вареная» возвращаются молча — а заметить это можно только
случайно, увидев название в своём же плане питания.

Поэтому здесь два рода проверок: что исправления работают на сломанных
данных и что каталог в репозитории уже им соответствует.
"""

from __future__ import annotations

import pandas as pd
import pytest

from translation_fixes import NAME_FIXES, SCOPED_FIXES, apply


def make(name: str, name_ru: str) -> pd.DataFrame:
    return pd.DataFrame({"name": [name], "name_ru": [name_ru]})


class TestPlainReplacements:
    """Обороты, однозначно неверные в любом блюде."""

    @pytest.mark.parametrize("broken,expected", [
        ("Куриное бедро, гриль без соуса, шкура съедена",
         "Куриное бедро, гриль без соуса, с кожей"),
        ("Куриное бедро, гриль, шкура не съедена",
         "Куриное бедро, гриль, без кожи"),
        ("Куриное бедро, панированное, шкура/панировка съедена",
         "Куриное бедро, панированное, с кожей и панировкой"),
        ("Говядина, Т-бон, без учета съеденного жира",
         "Говядина, Т-бон, жирность не уточнена"),
        ("Омлет или scrambled eggs с сыром",
         "Омлет или яичница-болтунья с сыром"),
        ("Кофе, кафе кон leche", "Кофе, кофе с молоком"),
    ])
    def test_broken_name_is_repaired(self, broken, expected):
        assert apply(make("Chicken thigh", broken))["name_ru"].iloc[0] == expected

    def test_longer_phrase_wins_over_shorter(self):
        """«не съедена» должно разбираться раньше «съедена»."""
        fixed = apply(make("Chicken thigh", "бедро, шкура не съедена"))
        assert fixed["name_ru"].iloc[0] == "бедро, без кожи"

    def test_order_is_longest_first(self):
        """Порядок в списке — часть контракта, а не случайность."""
        phrases = [wrong for wrong, _ in NAME_FIXES]
        assert phrases.index("шкура не съедена") < phrases.index("шкура съедена")
        assert phrases.index("панировка не съедена") < phrases.index("панировка съедена")


class TestScopedReplacements:
    """Замены, верные только для конкретных блюд оригинала."""

    def test_rotisserie_chicken_is_cooked_on_a_spit(self):
        fixed = apply(make("Chicken thigh, rotisserie, skin eaten",
                           "Куриное бедро, роти, с кожей"))
        assert fixed["name_ru"].iloc[0] == "Куриное бедро, на вертеле, с кожей"

    def test_indian_flatbread_is_left_alone(self):
        """«Роти» — лепёшка. Она не должна попасть под замену rotisserie."""
        fixed = apply(make("Bread, chapatti or roti", "Хлеб, чапатти или роти"))
        assert fixed["name_ru"].iloc[0] == "Хлеб, чапатти или роти"

    def test_mustard_greens_are_a_vegetable(self):
        fixed = apply(make("Mustard greens, cooked", "Горчица, вареная"))
        assert fixed["name_ru"].iloc[0] == "Листовая горчица, вареная"

    def test_real_mustard_condiment_is_left_alone(self):
        """Для приправы «Горчица» — правильный перевод."""
        fixed = apply(make("Mustard, prepared, yellow", "Горчица столовая"))
        assert fixed["name_ru"].iloc[0] == "Горчица столовая"

    def test_every_scoped_fix_names_an_english_pattern(self):
        assert all(pattern and wrong and right for pattern, wrong, right in SCOPED_FIXES)


class TestIdempotence:
    def test_applying_twice_changes_nothing(self, catalog):
        once = apply(catalog.copy())
        twice = apply(once.copy())
        assert (once["name_ru"] == twice["name_ru"]).all()

    def test_survives_a_catalog_without_translations(self):
        """Модуль зовут и до перевода — падать он не должен."""
        frame = pd.DataFrame({"name": ["Chicken"]})
        assert "name_ru" not in apply(frame).columns


class TestShippedCatalogIsClean:
    """Каталог в репозитории обязан уже соответствовать исправлениям.

    Если кто-то пересоберёт каталог и забудет применить модуль, этот тест
    упадёт — раньше, чем пользователь увидит «шкуру» в своём меню.
    """

    def test_nothing_left_to_fix(self, catalog):
        before = catalog["name_ru"].copy()
        after = apply(catalog.copy())["name_ru"]
        broken = before[before != after]
        assert broken.empty, f"неисправленных названий: {len(broken)}"
