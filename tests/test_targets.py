"""
Тесты расчёта норм.

Формулы проверяются числами, посчитанными на бумаге. Это главный смысл
детерминированного слоя: если кто-то поправит коэффициент, тест упадёт,
а не «ответ станет чуть другим».
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from targets import (
    ACTIVITY_FACTORS,
    KCAL_PER_KG_FAT,
    MAX_PROTEIN_KCAL_SHARE,
    MIN_KCAL,
    Profile,
    compute_targets,
    cycle_targets,
    explain,
    mifflin_st_jeor,
    total_expenditure,
)


class TestMifflinStJeor:
    """Уравнение Миффлина–Сан Жеора."""

    def test_female_matches_hand_calculation(self):
        # 10*74 + 6.25*168 - 5*31 - 161 = 740 + 1050 - 155 - 161 = 1474
        profile = Profile(sex="female", age=31, height_cm=168, weight_kg=74)
        assert mifflin_st_jeor(profile) == pytest.approx(1474.0)

    def test_male_matches_hand_calculation(self):
        # 10*95 + 6.25*182 - 5*40 + 5 = 950 + 1137.5 - 200 + 5 = 1892.5
        profile = Profile(sex="male", age=40, height_cm=182, weight_kg=95)
        assert mifflin_st_jeor(profile) == pytest.approx(1892.5)

    def test_male_higher_than_female_all_else_equal(self):
        """Разница в 166 ккал — это константа уравнения (+5 против −161)."""
        common = {"age": 30, "height_cm": 170, "weight_kg": 70}
        male = mifflin_st_jeor(Profile(sex="male", **common))
        female = mifflin_st_jeor(Profile(sex="female", **common))
        assert male - female == pytest.approx(166.0)


class TestActivity:
    def test_tdee_is_bmr_times_factor(self):
        profile = Profile(sex="female", age=31, height_cm=168, weight_kg=74, activity="moderate")
        expected = mifflin_st_jeor(profile) * ACTIVITY_FACTORS["moderate"]
        assert total_expenditure(profile) == pytest.approx(expected)

    def test_factors_increase_monotonically(self):
        order = ["sedentary", "light", "moderate", "high", "athlete"]
        values = [ACTIVITY_FACTORS[name] for name in order]
        assert values == sorted(values)


class TestGoalShift:
    def test_deficit_matches_requested_rate(self):
        """0.5 кг/нед — это 7700*0.5/7 = 550 ккал в день."""
        profile = Profile(
            sex="male", age=30, height_cm=180, weight_kg=85,
            activity="moderate", goal="lose", rate_kg_per_week=0.5,
        )
        targets = compute_targets(profile)
        expected_shift = 0.5 * KCAL_PER_KG_FAT / 7
        assert targets.tdee - targets.kcal == pytest.approx(expected_shift, abs=1.0)

    def test_maintain_equals_tdee(self):
        profile = Profile(sex="female", age=45, height_cm=165, weight_kg=62, activity="light")
        targets = compute_targets(profile)
        assert targets.kcal == pytest.approx(targets.tdee, abs=1.0)

    def test_gain_adds_calories(self):
        profile = Profile(
            sex="male", age=25, height_cm=178, weight_kg=63, activity="high", goal="gain",
        )
        targets = compute_targets(profile)
        assert targets.kcal > targets.tdee


class TestSafetyGuards:
    """Предохранители — то, что агент не имеет права обойти."""

    def test_weekly_loss_capped_at_one_percent_of_body_mass(self):
        profile = Profile(
            sex="male", age=40, height_cm=182, weight_kg=95,
            activity="moderate", goal="lose", rate_kg_per_week=1.5,
        )
        targets = compute_targets(profile)
        assert targets.rate_kg_per_week == pytest.approx(0.95)
        assert any("Темп снижен" in note for note in targets.adjustments)

    def test_calories_never_below_physiological_floor(self):
        """Маленькая женщина с агрессивным дефицитом ушла бы в 477 ккал."""
        profile = Profile(
            sex="female", age=70, height_cm=140, weight_kg=40,
            activity="sedentary", goal="lose", rate_kg_per_week=1.0,
        )
        targets = compute_targets(profile)
        assert targets.kcal >= MIN_KCAL["female"]
        assert any("Калорийность поднята" in note for note in targets.adjustments)

    @pytest.mark.parametrize(
        "profile",
        [
            Profile(sex="female", age=31, height_cm=168, weight_kg=74,
                    activity="sedentary", goal="lose"),
            Profile(sex="male", age=25, height_cm=200, weight_kg=200,
                    activity="sedentary", goal="lose", rate_kg_per_week=2.0),
        ],
        ids=["женщина 74 кг на дефиците", "мужчина 200 кг на дефиците"],
    )
    def test_protein_never_exceeds_share_ceiling(self, profile):
        """Белок от массы тела на низкой калорийности давал 44-54% — не съесть."""
        targets = compute_targets(profile)
        protein_share = targets.protein_g * 4 / targets.kcal
        assert protein_share <= MAX_PROTEIN_KCAL_SHARE + 0.01

    def test_adjustments_are_reported_not_silent(self):
        """Сработавший предохранитель обязан оставить след — иначе агент
        выдаст урезанную цифру молча."""
        profile = Profile(
            sex="female", age=25, height_cm=160, weight_kg=50,
            activity="sedentary", goal="lose", rate_kg_per_week=1.5,
        )
        targets = compute_targets(profile)
        assert targets.adjustments


class TestMacroConsistency:
    @pytest.mark.parametrize("goal", ["lose", "maintain", "gain"])
    def test_macros_sum_to_calories(self, goal):
        """Б×4 + Ж×9 + У×4 должно сходиться с заявленной калорийностью."""
        profile = Profile(
            sex="female", age=35, height_cm=170, weight_kg=68, activity="light", goal=goal,
        )
        targets = compute_targets(profile)
        from_macros = targets.protein_g * 4 + targets.fat_g * 9 + targets.carb_g * 4
        # Допуск — на округление каждого макронутриента до целых граммов.
        assert from_macros == pytest.approx(targets.kcal, rel=0.02)

    def test_all_macros_non_negative(self):
        profile = Profile(
            sex="male", age=30, height_cm=175, weight_kg=120,
            activity="sedentary", goal="lose", rate_kg_per_week=1.0,
        )
        targets = compute_targets(profile)
        assert min(targets.protein_g, targets.fat_g, targets.carb_g, targets.fiber_g) >= 0


class TestProfileValidation:
    @pytest.mark.parametrize(
        "field,value",
        [("age", 5), ("age", 150), ("height_cm", 50), ("weight_kg", 500)],
    )
    def test_out_of_range_values_rejected(self, field, value):
        data = {"sex": "female", "age": 30, "height_cm": 165, "weight_kg": 60}
        data[field] = value
        with pytest.raises(ValidationError):
            Profile(**data)

    def test_rate_contradicting_goal_rejected(self):
        """Удерживать вес и худеть на 0.5 кг/нед одновременно нельзя."""
        with pytest.raises(ValidationError):
            Profile(
                sex="female", age=30, height_cm=165, weight_kg=60,
                goal="maintain", rate_kg_per_week=0.5,
            )


class TestRecomposition:
    """Рекомпозиция: калорийность поддерживающая, белок как при наборе.

    Раньше «рекомпозиция» отображалась в maintain и получала 1.4 г/кг белка.
    На нулевом балансе калорий мышцы растут именно за счёт белка, и 1.4 —
    это норма обычного поддержания, а не смены состава тела.
    """

    PROFILE = dict(sex="male", age=24, height_cm=167, weight_kg=65,
                   activity="moderate")

    def test_calories_match_maintenance(self):
        recomp = compute_targets(Profile(**self.PROFILE, goal="recomp"))
        maintain = compute_targets(Profile(**self.PROFILE, goal="maintain"))
        assert recomp.kcal == maintain.kcal

    def test_protein_is_higher_than_maintenance(self):
        recomp = compute_targets(Profile(**self.PROFILE, goal="recomp"))
        maintain = compute_targets(Profile(**self.PROFILE, goal="maintain"))
        assert recomp.protein_g > maintain.protein_g

    def test_protein_reaches_two_grams_per_kg(self):
        targets = compute_targets(Profile(**self.PROFILE, goal="recomp"))
        assert targets.protein_g / 65 == pytest.approx(2.0, abs=0.05)

    def test_weight_is_not_meant_to_change(self):
        targets = compute_targets(Profile(**self.PROFILE, goal="recomp"))
        assert targets.rate_kg_per_week == 0.0

    def test_explanation_says_what_recomposition_is(self):
        profile = Profile(**self.PROFILE, goal="recomp")
        text = explain(profile, compute_targets(profile))
        assert "рекомпозиц" in text.lower()
        assert "состав тела" in text.lower()


class TestActivityLabels:
    def test_every_level_has_a_human_description(self):
        from targets import ACTIVITY_FACTORS, ACTIVITY_LABELS

        assert set(ACTIVITY_LABELS) == set(ACTIVITY_FACTORS)

    def test_explanation_uses_words_not_codes(self):
        """В объяснении не должно быть «moderate» — человек его не поймёт."""
        profile = Profile(sex="male", age=24, height_cm=167, weight_kg=65,
                          activity="moderate", goal="recomp")
        text = explain(profile, compute_targets(profile))
        assert "moderate" not in text
        assert "2-3 раза в неделю" in text


class TestLabelsCoverEveryValue:
    """Русские названия должны покрывать ВСЕ значения перечислений.

    Наблюдалось: добавили цель «recomp», обновили один словарь из трёх —
    и ответ упал с KeyError уже в проде, на второй реплике разговора.
    Копии словаря жили в agent.py, export.py и targets.py.
    """

    def test_every_goal_has_a_label(self):
        from typing import get_args

        from targets import GOAL_LABELS, Goal

        assert set(GOAL_LABELS) == set(get_args(Goal))

    def test_every_activity_has_a_label(self):
        from targets import ACTIVITY_FACTORS, ACTIVITY_LABELS

        assert set(ACTIVITY_LABELS) == set(ACTIVITY_FACTORS)

    def test_labels_are_defined_once(self):
        """Словарь должен быть один: копии неизбежно разъезжаются."""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        copies = [
            path.name
            for path in root.glob("*.py")
            if '"lose": "снижение веса"' in path.read_text(encoding="utf-8")
        ]
        assert copies == ["targets.py"], f"словарь целей продублирован в {copies}"


class TestDayCycling:
    """Циклирование по дням недели.

    Одна калорийность на все дни — упрощение: в дни тренировок нужно
    больше углеводов, в дни отдыха меньше. Разводим все цели, не только
    рекомпозицию, — тренирующемуся человеку это нужно одинаково.

    Главное свойство: недельное среднее РАВНО обычной норме. Дефицит или
    профицит уже заложен в цель, и удваивать его циклированием нельзя —
    человек просил не этого.
    """

    def profile(self, activity="moderate", goal="recomp"):
        return Profile(sex="male", age=24, height_cm=167, weight_kg=65,
                       activity=activity, goal=goal)

    @pytest.mark.parametrize("goal", ["lose", "maintain", "gain", "recomp"])
    def test_weekly_mean_equals_the_plain_target(self, goal):
        cycled = cycle_targets(self.profile(goal=goal))
        assert cycled.weekly_mean_kcal == pytest.approx(cycled.base.kcal, rel=0.005)

    @pytest.mark.parametrize("activity", ["light", "moderate", "high", "athlete"])
    def test_training_day_is_richer_than_rest(self, activity):
        cycled = cycle_targets(self.profile(activity=activity))
        assert cycled.training.kcal > cycled.rest.kcal

    def test_sedentary_is_not_cycled(self):
        """Не тренируется — разводить нечего, две одинаковые карточки лгут."""
        assert cycle_targets(self.profile(activity="sedentary")) is None

    @pytest.mark.parametrize("activity", ["light", "moderate", "high", "athlete"])
    def test_protein_does_not_change(self, activity):
        """Белок считается от массы тела, а она в выходной та же."""
        cycled = cycle_targets(self.profile(activity=activity))
        assert cycled.training.protein_g == cycled.base.protein_g
        assert cycled.rest.protein_g == cycled.base.protein_g

    @pytest.mark.parametrize("activity", ["light", "moderate", "high", "athlete"])
    def test_carbs_absorb_the_difference(self, activity):
        cycled = cycle_targets(self.profile(activity=activity))
        assert cycled.training.carb_g > cycled.rest.carb_g

    @pytest.mark.parametrize("activity", ["light", "moderate", "high", "athlete"])
    def test_rest_day_never_collapses(self, activity):
        """У спортсмена шесть тренировок компенсировались одним выходным.

        Выходной выходил на 40% нормы — 1200 ккал и 45 г углеводов вместо
        3000. Среднее сходилось, день был несъедобным.
        """
        from targets import MIN_REST_DAY_FACTOR

        cycled = cycle_targets(self.profile(activity=activity))
        assert cycled.rest.kcal >= cycled.base.kcal * MIN_REST_DAY_FACTOR * 0.99

    def test_days_add_up_to_a_week(self):
        for activity in ("light", "moderate", "high", "athlete"):
            cycled = cycle_targets(self.profile(activity=activity))
            assert cycled.training_days + cycled.rest_days == 7

    def test_schedule_puts_training_days_first(self):
        cycled = cycle_targets(self.profile(activity="moderate"))
        kinds = [cycled.for_day(i).kcal for i in range(7)]
        assert kinds[:3] == [cycled.training.kcal] * 3
        assert kinds[3:] == [cycled.rest.kcal] * 4

    def test_schedule_repeats_every_week(self):
        cycled = cycle_targets(self.profile(activity="moderate"))
        assert cycled.for_day(0).kcal == cycled.for_day(7).kcal
        assert cycled.for_day(6).kcal == cycled.for_day(13).kcal
