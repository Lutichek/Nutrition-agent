"""
Тесты оценки стоимости рациона.

Почти все проверки здесь — наблюдавшиеся случаи, и почти все одного класса:
**слово из названия блюда попало в чужую ценовую категорию**. Ошибка тихая
и дорогая одновременно — рацион продолжает считаться, просто сумма
становится неверной в разы, и увидеть это можно только глазами.

Найдены они были сканированием каталога по цене за 100 г: настоящие
ошибки всплывают на верхушке списка, потому что ошибочная категория
почти всегда дороже правильной.
"""

from __future__ import annotations

import pytest

from prices import PRICES, cost_of, price_key_for
from shopping import ShoppingItem, ShoppingList, build_shopping_list


class TestMatchingTraps:
    """Наблюдавшиеся ошибки сопоставления блюда с ценовой категорией."""

    @pytest.mark.parametrize(("name", "expected"), [
        # «Mustard greens» — листовая зелень, а не горчичный порошок.
        # Цена отличается в семь раз: 1033 против 2405 ₽/кг. Та же ловушка,
        # что в переводах каталога, где «Горчица» верна для приправы
        # и неверна для mustard greens.
        ("Mustard greens, raw", "зелень"),
        ("Mustard, yellow", "специи"),
        # «Cheesecake» пишется слитно, границы слова внутри нет — чизкейк
        # считался по цене сыра (982 ₽/кг).
        ("Cheesecake, plain", "торты"),
        ("Cheese, cheddar", "сыр"),
        # Главное слово стоит вторым: позиция в названии обманывает.
        ("Chicken liver, cooked", "печень"),
        ("Peanut butter, smooth", "орехи"),
        ("Apple pie, fresh", "торты"),
        # ...но «fish cake» — это рыба, а не выпечка.
        ("Crab cake", "рыбное филе"),
        ("Fish cake, fried", "рыбное филе"),
        # Сироп ушёл из специй в сахар: политая сиропом выпечка выходила
        # дороже мяса.
        ("Maple syrup", "сахар"),
        # Дичи в правилах не было вовсе, и блюдо ловилось словом «gravy».
        ("Venison or deer with gravy", "баранина"),
        # Форм макарон не хватало: тортеллини с сыром считались по сыру.
        ("Tortellini, cheese-filled, with tomato sauce", "макароны"),
    ])
    def test_known_trap(self, name: str, expected: str) -> None:
        assert price_key_for(name) == expected


class TestFirstIngredientWins:
    """В описаниях USDA главный продукт называется первым."""

    def test_earliest_match_wins(self) -> None:
        """«Макароны с тунцом» — это макароны.

        Первая версия брала первое подходящее ПРАВИЛО, а рыба стояла
        в списке выше макарон. Из-за этого дешёвый гарнир считался
        по цене рыбы, и цена завышалась втрое.
        """
        assert price_key_for("Macaroni or pasta salad with tuna") == "макароны"

    def test_uncertain_pairs_take_the_cheaper_first_name(self) -> None:
        """«Chicken or turkey, rice, and vegetables» — курица.

        Название прямо говорит «или», и выбирать надо по порядку слов,
        а не по тому, какое правило стоит выше в списке.
        """
        assert price_key_for("Chicken or turkey, rice, and vegetables") == "курица"

    def test_priority_rules_beat_position(self) -> None:
        """Устойчивые словосочетания позиции не подчиняются.

        Иначе «chicken liver» стало бы курицей: слово «chicken» стоит
        первым, но блюдо — печень.
        """
        assert price_key_for("Chicken liver") == "печень"
        assert price_key_for("Chicken breast") == "курица"


class TestCookingYield:
    """Вес при готовке меняется, и без поправки ошибка кратная."""

    def test_cooked_grain_priced_by_dry_weight(self) -> None:
        """300 г варёного риса — это примерно 100 г сухого.

        Без множителя варёный рис считался бы по цене сухого и выходил
        втрое дороже правды.
        """
        cooked = cost_of(300, "Rice, white, cooked")
        dry_price = PRICES["рис"].rub_per_kg * 0.1  # 100 г сухого
        assert cooked == pytest.approx(dry_price, rel=0.01)

    def test_raw_dish_ignores_the_multiplier(self) -> None:
        """«Raw» в названии означает, что готовки не было.

        Продукт уже лежит в магазине в этом весе, и множитель применять
        нельзя.
        """
        assert cost_of(100, "Rice, white, raw") == pytest.approx(
            PRICES["рис"].rub_per_kg * 0.1, rel=0.01
        )

    def test_cooked_meat_is_more_expensive_per_gram(self) -> None:
        """Мясо при готовке ТЕРЯЕТ вес — значит грамм готового дороже."""
        cooked = cost_of(100, "Beef, roasted")
        raw = cost_of(100, "Beef, raw")
        assert cooked > raw

    def test_brewed_coffee_is_not_priced_as_beans(self) -> None:
        """Чашка кофе не должна стоить дороже стейка.

        Наблюдавшийся случай: 100 г кофе считались по 2056 ₽/кг зерна
        и давали 206 ₽ — в напитке, который почти целиком вода.
        """
        assert cost_of(200, "Coffee, brewed") < cost_of(200, "Beef, roasted")


class TestMissingPriceIsNotZero:
    """Отсутствие цены — не ноль. Главное правило всего модуля."""

    def test_unknown_dish_costs_none(self) -> None:
        assert cost_of(100, "Zzzqqq unknown foodstuff", "Zzzqqq") is None

    def test_none_is_not_counted_as_free(self) -> None:
        """Неоценённая позиция не попадает в сумму и видна в покрытии."""
        items = [
            ShoppingItem(fdc_id=1, name="A", department="X", grams=500,
                         times=1, kcal=100, cost=100.0),
            ShoppingItem(fdc_id=2, name="B", department="X", grams=500,
                         times=1, kcal=100, cost=None),
        ]
        shopping = ShoppingList(days=1, items=items)
        assert shopping.total_cost == 100.0
        assert shopping.priced_share == pytest.approx(0.5)

    def test_share_counts_weight_not_positions(self) -> None:
        """Доля считается по ВЕСУ.

        По позициям вышло бы «оценено 50%» и там, где без цены остался
        грамм специй, и там, где килограмм мяса.
        """
        items = [
            ShoppingItem(fdc_id=1, name="мясо", department="X", grams=1000,
                         times=1, kcal=100, cost=None),
            ShoppingItem(fdc_id=2, name="специи", department="X", grams=10,
                         times=1, kcal=1, cost=5.0),
        ]
        shopping = ShoppingList(days=1, items=items)
        assert shopping.priced_share < 0.02

    def test_low_coverage_refuses_to_name_a_sum(self) -> None:
        """При покрытии ниже половины сумма не называется вовсе.

        Число, под которым оценена треть корзины, выглядит как цена всей
        корзины и вводит в заблуждение сильнее, чем отказ.
        """
        items = [
            ShoppingItem(fdc_id=1, name="A", department="X", grams=900,
                         times=1, kcal=100, cost=None),
            ShoppingItem(fdc_id=2, name="B", department="X", grams=100,
                         times=1, kcal=100, cost=50.0),
        ]
        text = ShoppingList(days=1, items=items).cost_estimate()
        assert "не удалось" in text
        assert "50" not in text

    def test_empty_list_does_not_divide_by_zero(self) -> None:
        assert ShoppingList(days=1, items=[]).priced_share == 0.0


class TestEstimateIsHonestAboutPrecision:
    """Сумма подаётся как оценка, а не как чек."""

    @pytest.fixture
    def shopping(self, catalog):
        from solver import build_menu
        from targets import Profile, compute_targets

        targets = compute_targets(Profile(sex="male", age=30, height_cm=180, weight_kg=80))
        return build_shopping_list(build_menu(catalog, targets, days=7, seed=0), catalog)

    def test_sum_is_rounded_to_tens(self, shopping: ShoppingList) -> None:
        """Точность, которой нет, обещать нельзя.

        Источник — средние цены по стране: разброс по регионам до трети,
        овощи ходят в разы. «1 847 ₽» здесь было бы враньём.
        """
        text = shopping.cost_estimate()
        digits = text.split("около ")[1].split(" ₽")[0].replace(" ", "")
        assert int(digits) % 10 == 0

    def test_names_its_source(self, shopping: ShoppingList) -> None:
        text = shopping.cost_estimate()
        assert "Росстат" in text
        assert "оценка" in text

    def test_multi_day_shows_per_day(self, shopping: ShoppingList) -> None:
        assert "в день" in shopping.cost_estimate()


class TestCatalogCoverage:
    """Покрытие каталога ценами — замер, а не утверждение."""

    def test_almost_every_dish_has_a_price(self, catalog) -> None:
        """Ниже 95% оценка теряет смысл: слишком многое выпадает из суммы."""
        keys = [price_key_for(row["name"], row["category"])
                for _, row in catalog.iterrows()]
        covered = sum(1 for key in keys if key)
        share = covered / len(catalog)
        assert share >= 0.95, f"покрыто только {share:.1%}"

    def test_derived_prices_stay_a_minority(self, catalog) -> None:
        """Выведенных цен должно быть мало.

        Они не выдумка, но и не данные: посчитаны из соседних позиций,
        потому что в источнике таких строк нет. Если их доля поползёт
        вверх, таблица перестанет быть источником и станет догадкой.
        """
        keys = [price_key_for(row["name"], row["category"])
                for _, row in catalog.iterrows()]
        derived = sum(1 for key in keys if key and PRICES[key].derived)
        assert derived / len(catalog) <= 0.15


class TestDayCostIsPlausible:
    """Здравый смысл: сумма должна быть похожа на правду."""

    def test_daily_cost_in_a_sane_range(self, catalog) -> None:
        """День еды в России — это сотни рублей, не десятки и не десятки тысяч.

        Тест грубый намеренно. Он не проверяет точность — её здесь нет, —
        а ловит поломку разрядности: потерянный множитель готовки или
        деление на тысячу не в ту сторону сдвигают сумму на порядок,
        и это единственная проверка, которая такое увидит.
        """
        from solver import build_menu
        from targets import Profile, compute_targets

        targets = compute_targets(Profile(sex="male", age=30, height_cm=180, weight_kg=80))
        for seed in range(3):
            shopping = build_shopping_list(
                build_menu(catalog, targets, days=7, seed=seed), catalog)
            per_day = shopping.total_cost / 7
            assert 150 < per_day < 3000, f"seed {seed}: {per_day:.0f} ₽ в день"
