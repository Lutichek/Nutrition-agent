"""
Тесты подбора рациона под бюджет.

Бюджет — ОГРАНИЧЕНИЕ, а не цель, и почти все проверки здесь об этом.
Пока план укладывается в сумму, солвер про деньги не думает; минимизировать
цену нельзя, потому что дешевле всего каждый день одно и то же.

Отдельный пласт — ловушки, которые вскрылись ровно в тот момент, когда
цена из отчёта стала целью оптимизации. Пока она была отчётом, ошибки
в ценах гасили друг друга на корзине из полусотни позиций. Как только
солвер начал искать дешёвое, он пошёл ИМЕННО за недооценёнными блюдами:

    Раки, жареные          цены нет → считались за 0 ₽, солвер брал 169 г
    Батончик мюсли         30 ₽/кг вместо ~700: множитель каши на сухом
    Киноа                  20 ₽/кг вместо 150: считалась пшеном
    Тофу                   46 ₽/кг вместо 350: считался сухим горохом
    Утка                   332 ₽/кг вместо 1000: считалась курицей

Отсюда общий вывод, который стоит за половиной этих тестов: **оптимизатор
находит ошибки в данных быстрее человека**, и после его появления цены
надо пересматривать заново — сканируя ДЕШЁВЫЙ конец, а не дорогой.
"""

from __future__ import annotations

import pytest

from agent import NutritionAgent, _budget_from_text
from prices import cost_per_gram, fallback_cost_per_gram, price_key_for
from solver import build_day, build_menu, check_plan, day_cost
from targets import Profile, compute_targets


@pytest.fixture(scope="module")
def targets():
    return compute_targets(Profile(sex="male", age=30, height_cm=180, weight_kg=80))


class TestBudgetIsAConstraintNotAGoal:
    """Пока план укладывается — солвер денег не замечает."""

    def test_generous_budget_changes_nothing(self, catalog, targets) -> None:
        """Заведомо большой бюджет обязан дать тот же план до грамма.

        Иначе это была бы минимизация цены, а не выполнение просьбы:
        человек, назвавший потолок, не просил экономить сверх него.
        """
        free = build_day(catalog, targets, seed=0)
        generous = build_day(catalog, targets, seed=0, budget=free.cost * 3)

        assert [i.fdc_id for i in free.items] == [i.fdc_id for i in generous.items]
        assert free.cost == generous.cost

    def test_no_budget_leaves_plans_untouched(self, catalog, targets) -> None:
        """Планы тех, кто про деньги не говорил, не должны сдвинуться."""
        plan = build_day(catalog, targets, seed=3)
        assert plan.budget is None
        assert plan.within_budget is None

    def test_tighter_budget_gives_cheaper_plan(self, catalog, targets) -> None:
        expensive = build_day(catalog, targets, seed=0)
        cheap = build_day(catalog, targets, seed=0, budget=250)
        assert cheap.cost < expensive.cost


class TestNutritionAlwaysWins:
    """Деньги — пожелание, норма — то, ради чего человек пришёл."""

    @pytest.mark.parametrize("budget", [60, 150, 300, 500])
    def test_no_check_fails_at_any_budget(self, catalog, targets, budget) -> None:
        """Наблюдавшийся случай, и он опасный.

        На норме 3755 ккал с бюджетом 60 ₽ солвер честно уложился
        в деньги и выдал план, проваливший проверки И по калориям,
        И по белку. Формально просьба выполнена, по существу человек
        получил недоедание.
        """
        for seed in range(4):
            plan = build_day(catalog, targets, seed=seed, budget=budget)
            failed = [name for name, ok in check_plan(plan).items() if not ok]
            assert not failed, f"бюджет {budget}, seed {seed}: {failed}"

    def test_impossible_budget_falls_back_to_a_sound_plan(self, catalog) -> None:
        """Если в деньги не влезть — возвращается нормальный план.

        Ранний выход по калориям однажды уже пропускал такой день мимо
        проверки: 2023 ккал укладывались в допуск, а белок 92 г против
        112 — нет, и план уезжал человеку.
        """
        targets = compute_targets(Profile(sex="male", age=25, height_cm=190,
                                          weight_kg=95, activity="high", goal="gain"))
        plan = build_day(catalog, targets, seed=0, budget=60)

        assert not [n for n, ok in check_plan(plan).items() if not ok]
        # Бюджет сохранён, хотя и не выполнен: агент обязан сказать об этом.
        assert plan.budget == 60
        assert plan.within_budget is False


class TestUnpricedIsNotFree:
    """Отсутствие цены — не ноль. Главное правило измерительного слоя."""

    def test_fallback_is_the_catalog_median(self) -> None:
        """Неопознанное блюдо считается по медиане, а не за ноль.

        Наблюдавшийся случай: в подборе под бюджет цена бралась как
        ``costs.get(fdc_id, 0.0)``. Солвер немедленно нашёл «Раки,
        жареные» — цены нет, белка 24 г — и поставил 169 г за ноль рублей.
        Отсутствие оценки превратилось в самую выгодную оценку.
        """
        fallback = fallback_cost_per_gram()
        assert fallback > 0

        values = sorted(cost_per_gram().values())
        assert values[0] < fallback < values[-1]

    def test_day_cost_does_not_treat_unknown_as_free(self, catalog, targets) -> None:
        plan = build_day(catalog, targets, seed=0)
        assert day_cost(plan.items) == pytest.approx(plan.cost, abs=0.5)
        assert plan.cost > 0


class TestPricesTheOptimiserExploited:
    """Ошибки цен, которые нашёл именно оптимизатор.

    Все пять лежали в чужих категориях и были СИЛЬНО дешевле правды.
    Пока цена была отчётом, ошибки гасились соседями по корзине.
    """

    @pytest.mark.parametrize(("name", "expected"), [
        # Батончика мюсли здесь нет намеренно: он остаётся в «злаковых
        # хлопьях» (483 ₽/кг из источника), и это ВЕРНЕЕ, чем увести его
        # в «батончики» (1200 ₽/кг — грубая догадка для спортпита).
        # Проверяется он ниже по цене, а не по названию категории:
        # важно число, а не то, в какую полку оно записано.
        ("Quinoa, cooked", "киноа"),
        ("Couscous, plain, cooked", "кускус и булгур"),
        ("Duck, roasted, without skin", "утка"),
        ("Tofu, firm", "тофу"),
        ("Soy nuts", "тофу"),
        ("Crayfish, fried", "рыба мороженая"),
    ])
    def test_category_is_no_longer_borrowed(self, name: str, expected: str) -> None:
        assert price_key_for(name) == expected

    def test_dry_products_do_not_get_the_water_multiplier(self) -> None:
        """Батончик мюсли не варят, и воды он не набирает.

        Слово granola уводило его в «овсяные хлопья» с множителем 4.5 —
        тем, что задан для каши на воде. Цена делилась на 4.5: тридцать
        рублей за килограмм вместо примерно семисот.
        """
        from prices import cost_of

        assert 300 < cost_of(1000, "Cereal or Granola bar, NFS") < 1500
        # Каша при этом по-прежнему дешёвая — множитель для неё верен.
        assert cost_of(1000, "Oatmeal, cooked") < 100

    def test_neighbours_kept_their_prices(self) -> None:
        """Разведённые категории не должны утянуть за собой соседей."""
        assert price_key_for("Chicken breast, roasted") == "курица"
        assert price_key_for("Millet, cooked") == "пшено"
        assert price_key_for("Barley, cooked") == "пшено"


class TestVarietySurvives:
    """Экономия и разнообразие тянут в разные стороны."""

    def test_week_does_not_collapse_into_repeats(self, catalog, targets) -> None:
        """Самый дешёвый рацион — это одно и то же каждый день.

        Ограничение (а не минимизация) должно этого избегать: пока
        просьба выполнена, солверу незачем экономить дальше.
        """
        free = build_menu(catalog, targets, days=7, seed=0)
        tight = build_menu(catalog, targets, days=7, seed=0, budget=250)

        assert tight.unique_dishes >= free.unique_dishes - 3
        assert tight.unique_dishes >= 35  # из 49 слотов


class TestBudgetFromText:
    """Разбор суммы из реплики — кодом, а не моделью.

    Здесь нужен ПЕРЕСЧЁТ периода, а на арифметике модель врёт молча:
    «15 000 в месяц» возвращались то как 15000, то как 500, и оба раза
    уверенно.
    """

    @pytest.mark.parametrize(("text", "expected"), [
        ("хочу уложиться в 500 рублей в день", 500),
        ("бюджет 3500 в неделю", 500),
        ("готов тратить 20 тысяч в месяц", 666.67),
        ("15 000 рублей в месяц на еду", 500),
        ("не больше 600 ₽ в день", 600),
    ])
    def test_period_is_converted_to_a_day(self, text: str, expected: float) -> None:
        assert _budget_from_text(text) == pytest.approx(expected, abs=0.01)

    def test_picks_the_right_number_when_there_are_two(self) -> None:
        """«Меню на 30 дней, бюджет 18000 в месяц» — это 600 ₽, а не 30."""
        assert _budget_from_text("меню на 30 дней, бюджет 18000 в месяц") == 600

    @pytest.mark.parametrize("text", [
        "рацион на 7 дней",
        "составь меню на месяц",
        "у меня 2000 калорий в день",
    ])
    def test_ignores_text_without_money(self, text: str) -> None:
        """Без признака денег «500 в день» — это с тем же успехом калории."""
        assert _budget_from_text(text) is None

    def test_absurd_amounts_are_rejected(self) -> None:
        """Такие числа означают, что разбор зацепил чужую цифру."""
        assert _budget_from_text("бюджет 25 рублей в день") is None
        assert _budget_from_text("бюджет 900000 рублей в день") is None


class TestBudgetNote:
    """Агент обязан сказать, уложился он или нет."""

    def test_silent_without_budget(self, catalog, targets) -> None:
        plan = build_day(catalog, targets, seed=0)
        assert NutritionAgent._budget_note(plan, 1) == ""

    def test_says_it_fits(self, catalog, targets) -> None:
        plan = build_day(catalog, targets, seed=0, budget=900)
        note = NutritionAgent._budget_note(plan, 1)
        assert "уложил" in note
        assert "900" in note

    def test_says_it_does_not_fit(self, catalog) -> None:
        """Молчать нельзя: человек назвал сумму и должен узнать исход."""
        targets = compute_targets(Profile(sex="male", age=25, height_cm=190,
                                          weight_kg=95, activity="high", goal="gain"))
        plan = build_day(catalog, targets, seed=0, budget=60)
        note = NutritionAgent._budget_note(plan, 7)

        assert "не удалось" in note
        assert "перерасход" in note
        # Причина, а не только факт.
        assert "минимум" in note
