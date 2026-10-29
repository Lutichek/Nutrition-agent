"""
Замена одного блюда в плане (ревью R8): «замени творог на что-нибудь».

Раньше на такую просьбу собирался новый день целиком — человек терял
весь план из-за одного блюда. Проверяется: новый день проходит все восемь
проверок, меняется только одна позиция, результат один при одном seed,
исключения соблюдаются, числа берутся из каталога, а не из присланного
плана, а «замени ещё раз» не возвращает отвергнутое.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from conftest import FakeLLMClient, FakeRetriever
from fastapi.testclient import TestClient

import api
from agent import NutritionAgent, _swap_index
from api import app
from restrictions import matches_terms
from solver import PlanNotFeasible, build_day, check_plan, swap_item
from targets import Profile, compute_targets

PROFILE = {"sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74,
           "activity": "light", "goal": "lose"}


@pytest.fixture(scope="module")
def plan(catalog):
    return build_day(catalog, compute_targets(Profile(**PROFILE)), seed=0)


class TestSolverSwap:
    @pytest.mark.parametrize("index", range(7))
    def test_every_slot_swaps_and_passes_checks(self, catalog, plan, index):
        new = swap_item(catalog, plan, index, seed=0)
        assert all(check_plan(new).values())
        assert new.items[index].fdc_id != plan.items[index].fdc_id
        assert new.items[index].role == plan.items[index].role
        others = [it.fdc_id for j, it in enumerate(new.items) if j != index]
        assert others == [it.fdc_id for j, it in enumerate(plan.items) if j != index]

    def test_same_seed_same_swap(self, catalog, plan):
        assert (swap_item(catalog, plan, 0, seed=3).items[0].fdc_id
                == swap_item(catalog, plan, 0, seed=3).items[0].fdc_id)

    def test_exclusions_hold(self, catalog, plan):
        new = swap_item(catalog, plan, 1, exclude=["chicken", "pork"], seed=0)
        name = pd.Series([new.items[1].name], dtype=object)
        assert not matches_terms(name, ["chicken", "pork"]).any()

    def test_rejected_dishes_not_offered_again(self, catalog, plan):
        first = swap_item(catalog, plan, 0, seed=0).items[0].fdc_id
        second = swap_item(catalog, plan, 0, seed=0, avoid_ids={first}).items[0].fdc_id
        assert second not in (first, plan.items[0].fdc_id)

    def test_numbers_come_from_catalog_not_from_request(self, catalog, plan):
        """План присылает клиент API — подделанные цифры не должны пройти."""
        forged = plan.model_copy(deep=True)
        forged.items[2].kcal = 1.0
        new = swap_item(catalog, forged, 0, seed=0)
        assert new.items[2].kcal == pytest.approx(plan.items[2].kcal)

    def test_impossible_swap_is_said_not_hidden(self, catalog, plan):
        role = plan.items[5].role
        everything = set(catalog.loc[catalog["role"] == role, "fdc_id"].astype(int))
        with pytest.raises(PlanNotFeasible):
            swap_item(catalog, plan, 5, avoid_ids=everything)


class TestApi:
    @pytest.fixture
    def http(self, catalog):
        api._state["catalog"] = catalog
        yield TestClient(app)
        api._state.clear()

    def test_swap_endpoint(self, http):
        plan = http.post("/plan", json={"profile": PROFILE, "seed": 0}).json()["plan"]
        body = {"profile": PROFILE, "plan": plan, "index": 0, "seed": 0}
        first = http.post("/plan/swap", json=body).json()
        assert first["checks_passed"] is True
        assert first["replaced"]["from"]["fdc_id"] != first["replaced"]["to"]["fdc_id"]
        assert http.post("/plan/swap", json=body).json() == first, "детерминированно"

    def test_bad_index_is_422(self, http):
        plan = http.post("/plan", json={"profile": PROFILE, "seed": 0}).json()["plan"]
        response = http.post("/plan/swap", json={"profile": PROFILE, "plan": plan, "index": 99})
        assert response.status_code == 422


def test_finds_dish_by_name_then_by_meal(plan):
    word = plan.items[3].title.split(",")[0].split()[0]
    assert _swap_index(f"Замени {word.lower()} на что-нибудь", plan) == 3
    assert _swap_index("Поменяй завтрак", plan) == 0
    assert _swap_index("Сколько белка?", plan) is None


class TestInConversation:
    def _agent(self, catalog, extracted=None):
        return NutritionAgent(
            client=FakeLLMClient(script={
                "маршрутизатор": json.dumps({"intent": "plan", "red_flag": ""}),
                "Извлеки параметры": json.dumps(extracted or PROFILE),
                "Объясни человеку": "План готов.",
            }),
            model="fake", retriever=FakeRetriever([]), catalog=catalog,
        )

    def test_swap_keeps_the_rest_and_skips_the_model(self, catalog):
        agent = self._agent(catalog)
        first = agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        word = first.plan.items[1].title.split(",")[0].split()[0].lower()
        explained = sum("Объясни человеку" in c for c in agent.client.calls)

        second = agent.run(f"Замени {word} на что-нибудь")

        assert second.answer.startswith("Заменил «")
        assert second.plan.items[1].fdc_id != first.plan.items[1].fdc_id
        assert ([it.fdc_id for j, it in enumerate(second.plan.items) if j != 1]
                == [it.fdc_id for j, it in enumerate(first.plan.items) if j != 1])
        assert sum("Объясни человеку" in c for c in agent.client.calls) == explained, \
            "замена отвечает шаблоном, без модели"

        third = agent.run("Замени ещё раз, тоже не хочу")
        assert third.answer.startswith("Заменил «"), "«ещё раз» — тот же слот"
        assert third.plan.items[1].fdc_id not in (
            first.plan.items[1].fdc_id, second.plan.items[1].fdc_id), "отвергнутое не вернулось"

    def test_new_restriction_rebuilds_the_day(self, catalog):
        """«Замени X, и я не ем молочное»: другие блюда плана могли стать
        запретными — одной заменой тут не обойтись."""
        agent = self._agent(catalog)
        first = agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        word = first.plan.items[0].title.split(",")[0].split()[0].lower()
        # Исключаем то, что заведомо есть в ДРУГОМ блюде плана.
        other = first.plan.items[2].name.split(",")[0].split()[0].lower()
        agent.client.script["Извлеки параметры"] = json.dumps({**PROFILE, "exclude": [other]})
        second = agent.run(f"Замени {word}, и я не ем {other}")
        assert not second.answer.startswith("Заменил")
        names = pd.Series([it.name for it in second.plan.items], dtype=object)
        assert not matches_terms(names, [other]).any()
