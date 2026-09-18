"""
Тесты состава блюд и расчёта цены по рецептуре.

Модуль появился из опровергнутого утверждения. В `shopping.py` было
записано, что разложить блюдо на продукты нечем — «состава рецептуры
в данных нет». Утверждение оказалось неверным: файл ``input_food.csv``
приезжает в той же выгрузке FNDDS и покрывает каталог целиком.

Цена из-за этого считалась по одному «главному» ингредиенту и ошибалась
в разы в ОБЕ стороны::

    Креветки с овощами, 200 г   199 ₽ → 94 ₽   креветок 69 г, не 200
    Говядина тушёная, 200 г     219 ₽ → 92 ₽   мяса 69 г, воды 72 г
    Салат с макаронами и сыром   10 ₽ → 41 ₽   занижено: 26 г чеддера

Поэтому главная проверка здесь — что доли считаются правильно, а всё
остальное сторожит частные ловушки.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ingredients import MIN_SHARE, Ingredient, build_shares, describe, load_composition
from prices import cost_by_composition, price_key_for


@pytest.fixture
def raw_recipes(tmp_path):
    """Выгрузка-заглушка: две рецептуры с разными суммами граммовок.

    Первая уже даёт сотню, вторая — производственная закладка на партию.
    В настоящих данных встречается и то и другое.
    """
    frame = pd.DataFrame({
        "fdc_id": [1, 1, 2, 2, 2, 3],
        "sr_description": ["Chicken", "Rice, cooked",
                           "Flour", "Water, tap", "Salt, table",
                           "Milk"],
        # Соль 30 г от 7294 — это 0.4%, ниже порога отсева. Число подобрано
        # так, чтобы после её выброса остаток давал круглые 4540/7264.
        "gram_weight": [60.0, 40.0, 4540.0, 2724.0, 30.0, 100.0],
    })
    path = tmp_path / "input_food.csv"
    frame.to_csv(path, index=False)
    return path


class TestShares:
    """Рецептурные граммовки приводятся к долям."""

    def test_shares_sum_to_one(self, raw_recipes) -> None:
        """Главная проверка модуля.

        Граммовки в FNDDS рецептурные, а не на 100 г: у половины записей
        сумма не сотня, а у хлеба — семь килограммов на партию. Считать
        их как «на 100 г» значит ошибиться в разы.
        """
        shares = build_shares(raw_recipes)
        for _, group in shares.groupby("fdc_id"):
            assert group["share"].sum() == pytest.approx(1.0)

    def test_proportions_are_preserved(self, raw_recipes) -> None:
        shares = build_shares(raw_recipes).set_index(["fdc_id", "name_en"])
        assert shares.loc[(1, "Chicken"), "share"] == pytest.approx(0.6)
        assert shares.loc[(1, "Rice, cooked"), "share"] == pytest.approx(0.4)

    def test_bulk_recipe_normalised_the_same_way(self, raw_recipes) -> None:
        """Закладка на партию даёт те же доли, что и рецепт на порцию."""
        shares = build_shares(raw_recipes)
        bread = shares[shares["fdc_id"] == 2].set_index("name_en")["share"]
        assert bread["Flour"] == pytest.approx(4540 / 7264, rel=0.01)

    def test_tiny_ingredients_are_dropped(self, raw_recipes) -> None:
        """Соль в 1.2% массы в состав не идёт.

        На цену она не влияет, а список превращает в простыню: рецептуры
        доходят до двадцати позиций.
        """
        shares = build_shares(raw_recipes)
        bread = set(shares[shares["fdc_id"] == 2]["name_en"])
        assert "Salt, table" not in bread
        # Отсекли именно по порогу, а не случайно: доля соли ниже него,
        # а доля муки — заведомо выше.
        assert 30 / 7294 < MIN_SHARE < 4540 / 7294

    def test_renormalised_after_dropping(self, raw_recipes) -> None:
        """После отсева мелочи доли снова дают единицу.

        Без повторной нормировки стоимость блюда систематически
        занижалась бы ровно на вес отброшенного.
        """
        shares = build_shares(raw_recipes)
        assert shares[shares["fdc_id"] == 2]["share"].sum() == pytest.approx(1.0)

    def test_filter_by_catalog(self, raw_recipes) -> None:
        shares = build_shares(raw_recipes, keep_ids={1})
        assert set(shares["fdc_id"]) == {1}

    def test_zero_weight_recipe_does_not_divide_by_zero(self, tmp_path) -> None:
        """Испорченная строка выгрузки не должна давать inf."""
        path = tmp_path / "bad.csv"
        pd.DataFrame({"fdc_id": [9, 9], "sr_description": ["A", "B"],
                      "gram_weight": [0.0, 0.0]}).to_csv(path, index=False)
        shares = build_shares(path)
        assert shares.empty


class TestCostByComposition:
    """Цена блюда складывается из цен ингредиентов."""

    def test_composite_dish_is_not_priced_as_its_most_expensive_part(self) -> None:
        """Наблюдавшийся случай: «креветки с овощами».

        По названию блюдо считалось креветками целиком — 199 ₽ за 200 г.
        Креветок там треть массы.
        """
        parts = [
            Ingredient(name="Crustaceans, shrimp, cooked", name_ru="креветки", share=0.35),
            Ingredient(name="Corn, sweet, frozen", name_ru="кукуруза", share=0.45),
            Ingredient(name="Beverages, water, tap, drinking", name_ru="вода", share=0.20),
        ]
        by_parts, _ = cost_by_composition(200, parts)
        from prices import cost_of

        as_shrimp = cost_of(200, "Shrimp and vegetables")
        assert by_parts < as_shrimp

    def test_hidden_expensive_ingredient_raises_the_price(self) -> None:
        """Обратная ошибка: «салат с макаронами и сыром».

        По названию — макароны, 10 ₽. В составе четверть массы чеддер,
        которого в названии нет вовсе.
        """
        parts = [
            Ingredient(name="Pasta, cooked", name_ru="паста", share=0.75),
            Ingredient(name="Cheese, cheddar", name_ru="сыр", share=0.25),
        ]
        cost, _ = cost_by_composition(200, parts)
        from prices import cost_of

        assert cost > cost_of(200, "Macaroni salad")

    def test_tap_water_is_nearly_free(self) -> None:
        """Водопроводная вода — не бутилированная.

        В рецептурах вода это 6% всей массы. По цене питьевой (60 ₽/кг)
        тушёная говядина набирала лишние рубли просто за бульон.
        """
        assert price_key_for("Beverages, water, tap, drinking") == "вода из-под крана"
        parts = [Ingredient(name="Beverages, water, tap, drinking",
                            name_ru="вода", share=1.0)]
        cost, _ = cost_by_composition(1000, parts)
        assert cost < 1.0

    def test_sauce_is_not_priced_as_ground_spice(self) -> None:
        """Соус и специи — разные продукты.

        По цене приправ (2405 ₽/кг) банка маринары выходила дороже мяса,
        а порция пасты с соусом — 592 ₽ вместо 90.
        """
        assert price_key_for("Sauce, pasta, spaghetti/marinara") == "соус"
        assert price_key_for("Spices, cinnamon, ground") == "специи"

    def test_returns_none_when_half_the_recipe_is_unknown(self) -> None:
        """Растягивать цену известной трети на всё блюдо — домысел.

        Лучше честное «не знаю» и откат к расчёту по названию.
        """
        parts = [
            Ingredient(name="Zzzqqq", name_ru="?", share=0.7),
            Ingredient(name="Cheese, cheddar", name_ru="сыр", share=0.3),
        ]
        cost, share = cost_by_composition(100, parts)
        assert cost is None
        assert share == pytest.approx(0.3)

    def test_extrapolates_over_small_gaps(self) -> None:
        """Небольшой пробел досчитывается, а не считается бесплатным."""
        parts = [
            Ingredient(name="Cheese, cheddar", name_ru="сыр", share=0.8),
            Ingredient(name="Zzzqqq", name_ru="?", share=0.2),
        ]
        cost, share = cost_by_composition(100, parts)
        from prices import PRICES

        assert share == pytest.approx(0.8)
        # 100 г по цене сыра, а не 80 г.
        assert cost == pytest.approx(PRICES["сыр"].rub_per_kg * 0.1, rel=0.01)

    def test_empty_composition(self) -> None:
        assert cost_by_composition(100, []) == (None, 0.0)


class TestDescribe:
    """Состав словами."""

    def test_shows_grams_not_shares(self) -> None:
        parts = [Ingredient(name="Chicken", name_ru="Курица", share=0.6),
                 Ingredient(name="Rice", name_ru="Рис", share=0.4)]
        assert describe(parts, 200) == "курица 120 г, рис 80 г"

    def test_long_recipes_are_cut(self) -> None:
        """Рецептуры доходят до двадцати позиций, читают первые три."""
        parts = [Ingredient(name=f"X{i}", name_ru=f"ингредиент{i}", share=0.1)
                 for i in range(10)]
        text = describe(parts, 100, limit=3)
        assert text.count(",") == 2
        assert "и ещё 7" in text

    def test_empty(self) -> None:
        assert describe([], 100) == ""


class TestShippedComposition:
    """Проверки собранного артефакта — того, что реально поедет в образ."""

    @pytest.fixture(scope="class")
    def composition(self):
        data = load_composition()
        if not data:
            pytest.skip("Состав не собран: python ingredients.py --rebuild")
        return data

    def test_covers_the_catalog(self, composition, catalog) -> None:
        missing = set(catalog["fdc_id"].astype(int)) - set(composition)
        assert not missing, f"без состава осталось {len(missing)} блюд"

    def test_every_dish_sums_to_one(self, composition) -> None:
        for fdc_id, items in composition.items():
            total = sum(item.share for item in items)
            assert total == pytest.approx(1.0), f"блюдо {fdc_id}: сумма долей {total}"

    def test_names_are_translated(self, composition) -> None:
        """Человеку состав показывается по-русски."""
        untranslated = [
            item.name for items in composition.values() for item in items
            if item.name_ru == item.name
        ]
        assert not untranslated, f"без перевода: {untranslated[:5]}"

    def test_names_are_short(self, composition) -> None:
        """Длинные названия превращают состав в простыню.

        Первый прогон переводил ингредиенты промптом для блюд и давал
        «Молоко с пониженным содержанием жира, 2% жира, с добавлением
        витаминов A и D» — 78 символов в строке, где их шесть подряд.
        """
        names = {item.name_ru for items in composition.values() for item in items}
        average = sum(len(name) for name in names) / len(names)
        assert average < 40, f"средняя длина названия {average:.0f} символов"
