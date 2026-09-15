"""
Тесты солвера рациона.

Здесь проверяется главное обещание проекта: собранный день действительно
сходится с нормой, и это можно доказать арифметикой, а не мнением LLM.
"""

from __future__ import annotations

import pytest

from solver import (
    DAY_TEMPLATE,
    MAX_PORTION_G,
    MIN_PORTION_G,
    SLOT_TOLERANCE_RELAXED,
    SODIUM_LIMIT_MG,
    PlanNotFeasible,
    build_day,
    build_menu,
    check_plan,
)
from targets import Profile, compute_targets

PROFILES = {
    "похудение, женщина": Profile(
        sex="female", age=31, height_cm=168, weight_kg=74, activity="light", goal="lose",
    ),
    "похудение, мужчина": Profile(
        sex="male", age=40, height_cm=182, weight_kg=95, activity="moderate", goal="lose",
    ),
    "набор массы": Profile(
        sex="male", age=25, height_cm=178, weight_kg=63, activity="high", goal="gain",
    ),
    "удержание веса": Profile(
        sex="female", age=55, height_cm=160, weight_kg=58, activity="sedentary",
    ),
}


@pytest.fixture(params=list(PROFILES), ids=list(PROFILES))
def profile(request):
    return PROFILES[request.param]


class TestPlanHitsTarget:
    def test_all_acceptance_checks_pass(self, catalog, profile):
        targets = compute_targets(profile)
        plan = build_day(catalog, targets, seed=7)
        failed = [name for name, passed in check_plan(plan).items() if not passed]
        assert not failed, f"не прошли проверки: {failed}"

    def test_calories_within_six_percent(self, catalog, profile):
        targets = compute_targets(profile)
        plan = build_day(catalog, targets, seed=7)
        assert abs(plan.deviation["kcal"]) <= 0.06

    def test_protein_not_undershot(self, catalog, profile):
        """Недобор белка на дефиците — это потеря мышц, а не жира."""
        targets = compute_targets(profile)
        plan = build_day(catalog, targets, seed=7)
        assert plan.deviation["protein_g"] >= -0.12

    @pytest.mark.parametrize("seed", range(5))
    def test_stable_across_seeds(self, catalog, seed):
        """Качество не должно зависеть от везения: проверяем пять разных дней."""
        targets = compute_targets(PROFILES["похудение, женщина"])
        plan = build_day(catalog, targets, seed=seed)
        failed = [name for name, passed in check_plan(plan).items() if not passed]
        assert not failed, f"seed={seed}: {failed}"


class TestReproducibility:
    def test_same_seed_gives_same_plan(self, catalog):
        targets = compute_targets(PROFILES["похудение, мужчина"])
        first = build_day(catalog, targets, seed=3)
        second = build_day(catalog, targets, seed=3)
        assert [item.fdc_id for item in first.items] == [item.fdc_id for item in second.items]
        assert first.totals == second.totals

    def test_different_seeds_give_different_plans(self, catalog):
        """Иначе человек каждый день получал бы один и тот же завтрак."""
        targets = compute_targets(PROFILES["похудение, мужчина"])
        plans = [
            [item.fdc_id for item in build_day(catalog, targets, seed=seed).items]
            for seed in range(3)
        ]
        assert len({tuple(plan) for plan in plans}) > 1


class TestPlanStructure:
    def test_slots_match_template(self, catalog):
        targets = compute_targets(PROFILES["удержание веса"])
        plan = build_day(catalog, targets, seed=1)
        assert len(plan.items) == len(DAY_TEMPLATE)
        assert [item.role for item in plan.items] == [role for _slot, role, _share in DAY_TEMPLATE]

    def test_no_repeated_dishes(self, catalog):
        targets = compute_targets(PROFILES["удержание веса"])
        plan = build_day(catalog, targets, seed=1)
        assert len({item.fdc_id for item in plan.items}) == len(plan.items)

    def test_portions_are_edible(self, catalog):
        targets = compute_targets(PROFILES["набор массы"])
        plan = build_day(catalog, targets, seed=1)
        assert all(MIN_PORTION_G <= item.grams <= MAX_PORTION_G for item in plan.items)

    def test_totals_equal_sum_of_items(self, catalog):
        """Итог обязан быть суммой позиций — иначе показанное не сходится."""
        targets = compute_targets(PROFILES["похудение, женщина"])
        plan = build_day(catalog, targets, seed=2)
        assert plan.totals["kcal"] == pytest.approx(sum(i.kcal for i in plan.items), abs=0.2)
        assert plan.totals["protein_g"] == pytest.approx(
            sum(i.protein_g for i in plan.items), abs=0.2
        )

    def test_sodium_within_guideline(self, catalog, profile):
        targets = compute_targets(profile)
        plan = build_day(catalog, targets, seed=7)
        assert plan.totals["sodium_mg"] <= SODIUM_LIMIT_MG


class TestDietaryRestrictions:
    def test_excluded_terms_absent_from_plan(self, catalog):
        excluded = ["pork", "bacon", "ham"]
        targets = compute_targets(PROFILES["похудение, женщина"])
        plan = build_day(catalog, targets, exclude=excluded, seed=5)
        names = " ".join(item.name.lower() for item in plan.items)
        assert not any(term in names for term in excluded)

    def test_plan_still_hits_target_with_restrictions(self, catalog):
        targets = compute_targets(PROFILES["похудение, женщина"])
        plan = build_day(
            catalog, targets, exclude=["pork", "milk", "cheese", "beef"], seed=5,
        )
        failed = [name for name, passed in check_plan(plan).items() if not passed]
        assert not failed

    def test_exotic_dishes_never_offered(self, catalog):
        """Первый прогон ставил в обед Opossum."""
        targets = compute_targets(PROFILES["похудение, женщина"])
        offered = set()
        for seed in range(8):
            offered.update(item.name for item in build_day(catalog, targets, seed=seed).items)
        exotic = set(catalog.loc[~catalog["mainstream"], "name"])
        assert not (offered & exotic)


class TestInfeasiblePlans:
    """Каталог, в котором день собрать не из чего.

    Раньше эту ситуацию изображали исключениями ["a", "e", "o"]: при поиске
    по голой подстроке три буквы выкашивали почти весь справочник. После
    перехода на совпадение по границам слова так больше не сделать — и это
    правильно, но проверять ветку всё равно нужно. Поэтому каталог урезается
    прямо: остаются блюда одной роли, а шаблон дня требует всех.
    """

    @pytest.fixture
    def one_role_catalog(self, catalog):
        return catalog[catalog["role"] == "main"]

    def test_too_narrow_catalog_raises_typed_error(self, one_role_catalog):
        """Не голый ValueError из недр, а понятный тип с указанием роли."""
        targets = compute_targets(PROFILES["похудение, женщина"])
        with pytest.raises(PlanNotFeasible) as info:
            build_day(one_role_catalog, targets, seed=0)
        assert info.value.role is not None

    def test_error_message_is_actionable(self, one_role_catalog):
        targets = compute_targets(PROFILES["похудение, женщина"])
        with pytest.raises(PlanNotFeasible, match="исключений"):
            build_day(one_role_catalog, targets, seed=0)


class TestMealBalance:
    """Ни один приём пищи не должен съедать день.

    Целевая функция считает только суточные итоги, поэтому локальный поиск
    охотно ставил 745 ккал фисташек в перекус, запланированный на 7%:
    по сумме за день всё сходилось. Отсюда жёсткий потолок на долю приёма.
    """

    def test_no_meal_exceeds_its_share(self, catalog, profile):
        targets = compute_targets(profile)
        plan = build_day(catalog, targets, seed=11)

        # Потолок берётся ослабленный: у профилей с широкими исключениями
        # строгий проход может не сойтись по калориям, и тогда включается
        # запасной. Он всё равно ограничен — просто чуть шире.
        shares = [share for _slot, _role, share in DAY_TEMPLATE]
        for item, share in zip(plan.items, shares, strict=True):
            limit = share * targets.kcal * SLOT_TOLERANCE_RELAXED
            assert item.kcal <= limit, (
                f"{item.slot}: {item.kcal:.0f} ккал при потолке {limit:.0f} "
                f"(доля {item.kcal / targets.kcal:.0%} вместо {share:.0%})"
            )

    def test_snack_stays_a_snack(self, catalog):
        """Перекус на 12% нормы не должен превращаться в третий обед."""
        profile = Profile(sex="male", age=24, height_cm=168, weight_kg=65,
                          activity="moderate", goal="gain")
        targets = compute_targets(profile)

        for seed in range(6):
            plan = build_day(catalog, targets, seed=seed)
            snacks = sum(item.kcal for item in plan.items if item.slot == "Перекус")
            assert snacks < targets.kcal * 0.30, f"seed={seed}: перекус {snacks:.0f} ккал"


class TestRelaxedFallback:
    """Запасной проход при недостижимой доле слота.

    У человека с набором массы и исключениями «свинина, сладости, мучное,
    лактоза» на завтрак остаётся 123 блюда, и лишь 8 дотягивают до нужных
    683 ккал. Строгий потолок не даёт другим приёмам добрать недостачу,
    и день недобирает 6-10% калорий. Наращивание итераций не помогает:
    300, 600, 1200 и 2400 дают ровно те же провалы.
    """

    CONSTRAINED = Profile(
        sex="male", age=24, height_cm=168, weight_kg=65, activity="moderate",
        goal="gain", exclude=["pork", "sweets", "flour", "lactose"],
    )

    def test_constrained_profile_still_hits_calories(self, catalog):
        targets = compute_targets(self.CONSTRAINED)
        for seed in range(6):
            plan = build_day(catalog, targets, exclude=self.CONSTRAINED.exclude, seed=seed)
            assert check_plan(plan)["kcal_in_range"], f"seed={seed}"

    def test_relaxed_pass_stays_bounded(self, catalog):
        """Ослабление — не отмена: потолок всё равно есть."""
        targets = compute_targets(self.CONSTRAINED)
        shares = [share for _slot, _role, share in DAY_TEMPLATE]

        for seed in range(6):
            plan = build_day(catalog, targets, exclude=self.CONSTRAINED.exclude, seed=seed)
            for item, share in zip(plan.items, shares, strict=True):
                limit = share * targets.kcal * SLOT_TOLERANCE_RELAXED
                assert item.kcal <= limit, f"seed={seed}, {item.slot}"

    def test_ordinary_profile_does_not_use_the_fallback(self, catalog):
        """Обычный день должен остаться таким же сбалансированным, как был."""
        profile = Profile(sex="female", age=31, height_cm=168, weight_kg=74,
                          activity="light", goal="lose")
        targets = compute_targets(profile)
        shares = [share for _slot, _role, share in DAY_TEMPLATE]

        for seed in range(4):
            plan = build_day(catalog, targets, seed=seed)
            worst = max(item.kcal / (share * targets.kcal)
                        for item, share in zip(plan.items, shares, strict=True))
            assert worst <= 2.0 + 1e-9, f"seed={seed}: перекос {worst:.1f}x"


class TestCycledMenu:
    """Меню, где у дней разные нормы.

    До циклирования build_menu собирал все дни по одной норме. Теперь
    тренировочные и выходные дни различаются, и солвер обязан целиться
    в норму КАЖДОГО дня, а не в общую.
    """

    def cycled(self, activity="moderate"):
        from targets import cycle_targets

        return cycle_targets(Profile(
            sex="male", age=24, height_cm=167, weight_kg=65,
            activity=activity, goal="recomp"))

    def test_each_day_hits_its_own_target(self, catalog):
        cycled = self.cycled()
        per_day = [cycled.for_day(i) for i in range(7)]
        menu = build_menu(catalog, per_day, days=7, seed=0)

        for index, (plan, target) in enumerate(zip(menu.days, per_day, strict=True)):
            off = abs(plan.totals["kcal"] - target.kcal) / target.kcal
            assert off <= 0.08, f"день {index + 1}: отклонение {off:.1%}"

    def test_training_days_really_differ_from_rest(self, catalog):
        cycled = self.cycled()
        per_day = [cycled.for_day(i) for i in range(7)]
        menu = build_menu(catalog, per_day, days=7, seed=0)

        train = [d.totals["kcal"] for d in menu.days[:cycled.training_days]]
        rest = [d.totals["kcal"] for d in menu.days[cycled.training_days:]]
        assert min(train) > max(rest)

    def test_single_target_still_works(self, catalog):
        """Старый вызов с одной нормой ломать нельзя."""
        targets = compute_targets(PROFILES["похудение, женщина"])
        menu = build_menu(catalog, targets, days=3, seed=0)
        assert len(menu.days) == 3

    def test_wrong_number_of_targets_is_refused(self, catalog):
        cycled = self.cycled()
        with pytest.raises(ValueError, match="должно совпадать"):
            build_menu(catalog, [cycled.training] * 3, days=7, seed=0)

    def test_menu_target_is_the_weekly_mean(self, catalog):
        """Итог меню — средняя норма за период, а не норма первого дня."""
        cycled = self.cycled()
        per_day = [cycled.for_day(i) for i in range(7)]
        menu = build_menu(catalog, per_day, days=7, seed=0)
        assert menu.targets["kcal"] == pytest.approx(cycled.base.kcal, rel=0.01)
