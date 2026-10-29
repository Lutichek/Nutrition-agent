"""
Адаптивная норма по динамике веса (ревью R9, adaptive.py).

Формула ошибается у конкретного человека, и это видно по весу. Но оценка
по весу — ещё и способ уверенно ошибиться: вес за пару дней скачет водой,
пропуски, молчаливые отступления от плана. Проверяются крайние случаи
из задания — мало данных, скачки, пропуски — и что предохранители нормы
действуют и для адаптивной нормы.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest
from conftest import FakeLLMClient, FakeRetriever, consented
from fastapi.testclient import TestClient

import api
from adaptive import STEP, adapted_tdee, estimate
from agent import NutritionAgent
from api import app
from storage import MemoryStorage
from targets import MIN_KCAL, Profile, compute_targets, total_expenditure

TODAY = date(2026, 11, 1)


def day(n: int) -> str:
    return (TODAY - timedelta(days=n)).isoformat()


def losing(per_week: float, days=(21, 14, 7, 0), start=80.0) -> dict[str, float]:
    return {day(n): start - (days[0] - n) / 7 * per_week for n in days}


class TestEstimate:
    def test_stable_weight_means_expenditure_equals_intake(self):
        found = estimate({day(n): 80.0 for n in (21, 14, 7, 0)}, {day(30): 2000}, 1800, TODAY)
        assert found.tdee == pytest.approx(2000)

    def test_losing_half_kilo_a_week(self):
        """−0.5 кг/нед ≈ 550 ккал/день дефицита: расход = 1800 + 550."""
        found = estimate(losing(0.5), {day(30): 1800}, 1800, TODAY)
        assert found.tdee == pytest.approx(2350, abs=1)

    @pytest.mark.parametrize("weighins", [
        {day(3): 80.0, day(0): 79.5},                           # мало взвешиваний
        {day(9): 80.0, day(6): 79.8, day(3): 79.6, day(0): 79.5},  # период < 14 дней
        {day(60): 80.0, day(45): 79.0, day(40): 78.5, day(0): 78.0},  # старые — вне окна
    ])
    def test_little_data_means_formula(self, weighins):
        assert estimate(weighins, {}, 1800, TODAY) is None

    def test_water_spike_barely_moves_it(self):
        """Солёный ужин: +2.5 кг воды в одном взвешивании. МНК-наклон он
        сдвинул бы заметно; медиана наклонов — нет."""
        clean = estimate(losing(0.5), {day(30): 1800}, 1800, TODAY)
        spiky = losing(0.5)
        spiky[day(10)] = 82.5
        assert estimate(spiky, {day(30): 1800}, 1800, TODAY).tdee == pytest.approx(clean.tdee, abs=30)

    def test_gaps_use_the_last_known_intake(self):
        """Пропуск в записях — день по последней известной калорийности."""
        intake = {day(30): 1800, day(10): 2200}
        found = estimate({day(n): 80.0 for n in (21, 14, 7, 0)}, intake, 1500, TODAY)
        assert found.intake == pytest.approx((11 * 1800 + 10 * 2200) / 21)


class TestStep:
    def test_step_is_limited(self):
        result = adapted_tdee(losing(0.5), {day(30): 1800}, 2100, 1800, None, TODAY)
        assert result["tdee"] == round(2100 * (1 + STEP))

    def test_not_recalculated_within_a_week(self):
        previous = {"tdee": 2200, "at": day(3)}
        assert adapted_tdee(losing(1.0), {day(30): 1800}, 2100, 1800, previous, TODAY) == previous

    def test_no_data_keeps_previous_not_jumps_back(self):
        previous = {"tdee": 2200, "at": day(10)}
        assert adapted_tdee({}, {}, 2100, 1800, previous, TODAY) == previous


class TestSafetiesStillHold:
    PROFILE = Profile(sex="female", age=31, height_cm=168, weight_kg=74, goal="lose")

    def test_calorie_floor(self):
        """Очень низкий фактический расход не опускает норму ниже минимума."""
        targets = compute_targets(self.PROFILE, tdee_override=1100)
        assert targets.kcal >= MIN_KCAL["female"]
        assert targets.adjustments, "срабатывание предохранителя проговаривается"

    def test_formula_is_named_in_explanation(self):
        from targets import explain

        targets = compute_targets(self.PROFILE, tdee_override=1900)
        assert targets.tdee == 1900 and targets.tdee_formula == round(total_expenditure(self.PROFILE))
        assert "по динамике вашего веса" in explain(self.PROFILE, targets)


def make_agent(catalog) -> NutritionAgent:
    return NutritionAgent(
        client=FakeLLMClient(script={
            "маршрутизатор": json.dumps({"intent": "plan", "red_flag": ""}),
            "Извлеки параметры": json.dumps({"sex": "female", "age": 31, "height_cm": 168,
                                             "weight_kg": 74, "activity": "light",
                                             "goal": "lose"}),
            "Объясни человеку": "План готов.",
        }),
        model="fake", retriever=FakeRetriever([]), catalog=catalog,
    )


class TestInConversation:
    def test_weight_said_is_recorded_and_plan_day_too(self, catalog):
        agent = make_agent(catalog)
        answer = agent.run("Мне 31, женщина, 168 см, вешу 74 кг, хочу похудеть")
        state = agent._own_state
        assert list(state.weighins.values()) == [74.0]
        assert list(state.intake.values()) == [pytest.approx(answer.plan.totals["kcal"])]

    def test_weight_of_someone_else_is_not_recorded(self, catalog):
        agent = make_agent(catalog)
        agent.run("Сколько белка нужно спортсмену весом 90 кг?")
        assert agent._own_state.weighins == {}

    def test_enough_history_switches_to_actual_expenditure(self, catalog):
        agent = make_agent(catalog)
        # Сутки — по UTC, как у агента: с местной датой тест падал бы ночью.
        today = datetime.now(UTC).date()
        agent._own_state.weighins = {
            (today - timedelta(days=n)).isoformat(): 76 - (21 - n) / 7 * 0.2
            for n in (21, 14, 7, 0)}
        agent._own_state.intake = {(today - timedelta(days=30)).isoformat(): 1500.0}
        answer = agent.run("Составь рацион")
        assert answer.targets.tdee_formula is not None, "норма — от фактического расхода"
        assert agent._own_state.adaptive["tdee"] == answer.targets.tdee


class TestWeightEndpoint:
    @pytest.fixture
    def http(self, catalog):
        api._state.update(catalog=catalog, storage=MemoryStorage())
        yield TestClient(app)
        api._state.clear()

    def test_needs_consent(self, http):
        assert http.post("/weight", json={"weight_kg": 80}).status_code == 403

    def test_records_and_reports_readiness(self, http):
        consented(http)
        assert http.post("/weight", json={"weight_kg": 80}).json() == {
            "weighins": 1, "adaptive_ready": False}

    def test_future_date_rejected(self, http):
        consented(http)
        response = http.post("/weight", json={"weight_kg": 80, "day": "2999-01-01"})
        assert response.status_code == 422
