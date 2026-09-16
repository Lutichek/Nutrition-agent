"""
Тесты списка покупок.

Список — это меню, свёрнутое по блюдам и разложенное по отделам магазина.
Считать тут нечего, вся сложность в раскладке, и она делается сопоставлением
по названию категории USDA. А это ровно тот приём, который в проекте уже
дважды давал тихие ложные срабатывания: «oil» внутри «broiled», «ham»
внутри «graham».

Здесь он дал три, и все три проверяются ниже:

* «Soy and meat-alternative products» содержит **meat** — тофу уезжал в мясо;
* «Other fruits and fruit salads» содержит **salad** — киви уезжал в овощи;
* «Pasta mixed dishes, excludes macaroni and cheese» содержит **cheese** —
  паста уезжала в молочное.

Третий случай порядком правил не лечится: слово стоит внутри фразы
«excludes …», то есть означает прямо противоположное. Такие хвосты
отрезаются до сопоставления, и на это есть отдельный тест.
"""

from __future__ import annotations

import pytest

from shopping import (
    DEFAULT_DEPARTMENT,
    DEPARTMENTS,
    build_shopping_list,
    department_for,
)
from solver import DayPlan, MealItem, Menu


def item(fdc_id: int, name: str, grams: float, kcal: float = 100.0) -> MealItem:
    return MealItem(
        slot="Обед", fdc_id=fdc_id, name=name, name_ru=name, role="main",
        portion="1 cup", servings=1.0, grams=grams,
        kcal=kcal, protein_g=10.0, fat_g=5.0, carb_g=20.0,
        fiber_g=2.0, sugar_g=1.0, sodium_mg=100.0,
    )


def day(*items: MealItem) -> DayPlan:
    return DayPlan(items=list(items), totals={}, targets={})


class TestTrapsFromSubstringMatching:
    """Три ловушки, на которых первая версия ошиблась."""

    def test_meat_alternative_is_not_meat(self):
        """«meat-alternative» — это соя, а не мясо."""
        assert department_for("Soy and meat-alternative products") == "Соя и растительный белок"

    def test_fruit_salad_is_not_a_vegetable(self):
        """«fruit salads» ловилось словом salad из овощного шаблона."""
        assert department_for("Other fruits and fruit salads") == "Фрукты и ягоды"

    def test_exclusion_clause_is_not_a_match(self):
        """Ключевой случай: слово стоит внутри «excludes …».

        Порядком правил это не лечится — «cheese» здесь означает, чего
        в категории НЕТ. Хвост отрезается до сопоставления.
        """
        assert department_for(
            "Pasta mixed dishes, excludes macaroni and cheese") == "Крупы, хлеб, макароны"

    def test_other_exclusion_wordings(self):
        assert department_for("Crackers, excludes saltines") == "Крупы, хлеб, макароны"
        assert department_for("Beef, excludes ground") == "Мясо и птица"

    def test_corn_chips_are_not_vegetables(self):
        """«Tortilla, corn, other chips» ловилось и словом corn, и tortilla."""
        assert department_for("Tortilla, corn, other chips") == "Снеки и сладкое"


class TestDepartmentCoverage:
    def test_every_catalog_category_has_a_department(self, catalog):
        """Ни одно блюдо не должно попасть в «Прочее».

        Это не эстетика: «Прочее» в конце списка — то, мимо чего человек
        пройдёт, не поняв, где это искать.
        """
        selected = catalog[catalog.mainstream & catalog.available_ru & catalog.familiar_ru]
        unplaced = {
            category for category in selected["category"].dropna().unique()
            if department_for(category) == DEFAULT_DEPARTMENT
        }
        assert not unplaced, f"без отдела остались категории: {sorted(unplaced)}"

    def test_unknown_category_falls_back(self):
        """Незнакомое — в «Прочее», а не исключение."""
        assert department_for("Совершенно новая категория") == DEFAULT_DEPARTMENT
        assert department_for("") == DEFAULT_DEPARTMENT
        assert department_for(None) == DEFAULT_DEPARTMENT


class TestAggregation:
    def test_same_dish_across_days_becomes_one_line(self):
        """Главное, ради чего список и нужен."""
        menu = Menu(days=[day(item(1, "Курица", 200.0)),
                          day(item(1, "Курица", 150.0))], targets={})
        result = build_shopping_list(menu)

        assert len(result.items) == 1
        assert result.items[0].grams == pytest.approx(350.0)
        assert result.items[0].times == 2

    def test_different_dishes_stay_separate(self):
        menu = Menu(days=[day(item(1, "Курица", 200.0), item(2, "Рис", 100.0))], targets={})
        assert len(build_shopping_list(menu).items) == 2

    def test_single_day_plan_also_works(self):
        """На один день список тоже осмыслен — идти в магазин всё равно надо."""
        result = build_shopping_list(day(item(1, "Курица", 200.0)))
        assert result.days == 1
        assert result.items[0].grams == pytest.approx(200.0)

    def test_totals_match_the_menu(self):
        menu = Menu(days=[day(item(1, "Курица", 200.0, kcal=300.0),
                              item(2, "Рис", 100.0, kcal=130.0))], targets={})
        result = build_shopping_list(menu)
        assert result.total_grams == pytest.approx(300.0)
        assert result.total_kcal == pytest.approx(430.0)

    def test_heaviest_first_within_department(self):
        """В отделе крупное ищут первым, мелочь набирают заодно."""
        menu = Menu(days=[day(item(1, "Мало", 50.0), item(2, "Много", 500.0))], targets={})
        grams = [i.grams for i in build_shopping_list(menu).items]
        assert grams == sorted(grams, reverse=True)


class TestDisplay:
    @pytest.mark.parametrize("grams,expected", [
        (72.0, "72 г"), (950.0, "950 г"), (1000.0, "1 кг"), (1240.0, "1.2 кг"),
    ])
    def test_amount_is_shown_the_way_it_is_read(self, grams, expected):
        menu = Menu(days=[day(item(1, "Блюдо", grams))], targets={})
        assert build_shopping_list(menu).items[0].display_amount == expected


class TestOnTheRealCatalog:
    def test_week_menu_folds_and_places_everything(self, catalog):
        from solver import build_menu
        from targets import Profile, compute_targets

        targets = compute_targets(Profile(
            sex="male", age=24, height_cm=167, weight_kg=64,
            activity="moderate", goal="lose"))
        menu = build_menu(catalog, targets, days=7, seed=0)
        result = build_shopping_list(menu, catalog)

        total_lines = sum(len(d.items) for d in menu.days)
        assert len(result.items) < total_lines, "свёртка не уменьшила список"
        assert all(i.department != DEFAULT_DEPARTMENT for i in result.items)

        known = {name for name, _ in DEPARTMENTS} | {DEFAULT_DEPARTMENT}
        assert {name for name, _ in result.by_department()} <= known

    def test_render_mentions_every_department_present(self, catalog):
        from solver import build_menu
        from targets import Profile, compute_targets

        targets = compute_targets(Profile(
            sex="female", age=31, height_cm=168, weight_kg=74,
            activity="sedentary", goal="lose"))
        text = build_shopping_list(build_menu(catalog, targets, days=3, seed=1),
                                   catalog).render()
        assert "СПИСОК ПОКУПОК НА 3" in text
