"""
«Съел не по плану — пересчитай день» (этап 2).

Съеденное фиксируется, остальные приёмы пищи собираются под остаток
нормы. Проверяется то, что легко сломать молча: граммы и калории
съеденного — из каталога, а не от модели (она их только выписывает,
и то лишь названные человеком); день проверяется ЦЕЛИКОМ; при нехватке
остатка — честный отказ с числами, а не крошечные порции; съеденное
копится в пределах суток и не тянется во вчерашний день.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLMClient, FakeRetriever

from agent import NutritionAgent
from solver import DAY_TEMPLATE, PlanNotFeasible, check_plan, eaten_item, rebuild_rest
from targets import Profile, compute_targets

PROFILE = {"sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74,
           "activity": "light", "goal": "lose"}


@pytest.fixture(scope="module")
def targets():
    return compute_targets(Profile(**PROFILE))


def first_id(catalog, text: str) -> int:
    return int(catalog[catalog["name"].str.contains(text, regex=False)].iloc[0]["fdc_id"])


class TestSolver:
    def test_numbers_come_from_catalog(self, catalog):
        fdc_id = first_id(catalog, "Pizza, cheese")
        row = catalog[catalog["fdc_id"] == fdc_id].iloc[0]
        item = eaten_item(catalog, fdc_id, 300, "Обед — основное")
        assert item.grams == pytest.approx(300)
        assert item.kcal == pytest.approx(row["portion_kcal"] * 300 / row["portion_g"])
        assert item.slot == "Обед — основное (съедено)"

    def test_rest_is_built_around_what_was_eaten(self, catalog, targets):
        pizza = eaten_item(catalog, first_id(catalog, "Pizza, cheese"), 300, "Обед — основное")
        plan = rebuild_rest(catalog, targets, [pizza], {1, 2}, seed=0)

        assert plan.items[1] == pizza, "съеденное — на месте обеда"
        slots = [it.slot for it in plan.items]
        assert "Обед — гарнир" not in slots, "весь обед закрыт съеденным"
        assert len(plan.items) == len(DAY_TEMPLATE) - 1
        assert pizza.fdc_id not in [it.fdc_id for j, it in enumerate(plan.items) if j != 1]
        assert check_plan(plan)["kcal_in_range"], "остаток добирает норму дня"
        assert plan.targets == targets.as_dict(), "проверки — по всему дню"

    def test_deterministic(self, catalog, targets):
        pizza = eaten_item(catalog, first_id(catalog, "Pizza, cheese"), 300, "Обед — основное")
        a = rebuild_rest(catalog, targets, [pizza], {1, 2}, seed=4)
        b = rebuild_rest(catalog, targets, [pizza], {1, 2}, seed=4)
        assert [it.fdc_id for it in a.items] == [it.fdc_id for it in b.items]

    def test_too_little_left_is_said_with_numbers(self, catalog, targets):
        """Не натягивать ужин и перекусы на остаток в 50 ккал."""
        pizza = eaten_item(catalog, first_id(catalog, "Pizza, cheese"), 600, "Обед — основное")
        with pytest.raises(PlanNotFeasible, match="ккал"):
            rebuild_rest(catalog, targets, [pizza], {0, 1, 2}, seed=0)


def make_agent(catalog, eaten_items: list[dict], choices: list[int]) -> NutritionAgent:
    return NutritionAgent(
        client=FakeLLMClient(script={
            "маршрутизатор": json.dumps({"intent": "plan", "red_flag": ""}),
            "Извлеки параметры": json.dumps(PROFILE),
            "Объясни человеку": "План готов.",
            "Человек рассказывает, что съел": json.dumps({"items": eaten_items}),
            "Сопоставь съеденное": json.dumps({"choices": choices}),
        }),
        model="fake", retriever=FakeRetriever([]), catalog=catalog,
    )


PIZZA = {"phrase_ru": "пиццу", "query_en": "pizza", "grams": 300, "count": None, "meal": "lunch"}


class TestInConversation:
    def test_lunch_replaced_and_rest_rebuilt(self, catalog):
        agent = make_agent(catalog, [PIZZA], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        explained = sum("Объясни человеку" in c for c in agent.client.calls)

        answer = agent.run("На обед съел пиццу, 300 г")

        assert answer.answer.startswith("Учёл съеденное")
        eaten = [it for it in answer.plan.items if it.slot.endswith("(съедено)")]
        assert len(eaten) == 1 and eaten[0].grams == pytest.approx(300)
        assert "pizza" in eaten[0].name.lower()
        assert sum("Объясни человеку" in c for c in agent.client.calls) == explained, \
            "ответ — шаблон кода, числа не пересказывает модель"

    def test_grams_not_said_come_from_usda_units(self, catalog):
        """Модель граммы не оценивает: «два куска» — два веса единицы
        из таблицы порций USDA (portions.py), а не справочной порции
        каталога: та у пиццы — целая пицца."""
        from portions import eaten_grams

        two = {**PIZZA, "grams": None, "count": 2, "unit": "slice"}
        agent = make_agent(catalog, [two], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        answer = agent.run("На обед съел два куска пиццы")
        eaten = next(it for it in answer.plan.items if it.slot.endswith("(съедено)"))
        portion = catalog.loc[catalog["fdc_id"] == eaten.fdc_id, "portion_g"].iloc[0]
        assert eaten.grams == pytest.approx(eaten_grams(eaten.fdc_id, None, 2, portion, "slice"))
        assert eaten.grams < 2 * portion or portion < 150, "не две целых пиццы"

    def test_unknown_meal_asks_instead_of_guessing(self, catalog):
        agent = make_agent(catalog, [{**PIZZA, "meal": "unknown"}], [1])
        first = agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        answer = agent.run("Съел пиццу")
        assert "на какой приём пищи" in answer.answer
        assert agent._own_state.last_plan == first.plan.model_dump(mode="json")

    def test_not_found_in_catalog_is_said(self, catalog):
        agent = make_agent(catalog, [PIZZA], [0])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        assert "Не нашёл в справочнике" in agent.run("На обед съел пиццу").answer

    def test_eaten_accumulates_within_the_day(self, catalog):
        agent = make_agent(catalog, [PIZZA], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.run("На обед съел пиццу, 300 г")
        agent.client.script["Человек рассказывает, что съел"] = json.dumps({"items": [
            {"phrase_ru": "банан", "query_en": "banana", "grams": 120, "count": None,
             "meal": "snack"}]})
        answer = agent.run("Ещё перекусил бананом")
        eaten = [it for it in answer.plan.items if it.slot.endswith("(съедено)")]
        assert len(eaten) == 2, "обед никуда не делся"

    def test_yesterdays_food_does_not_count_today(self, catalog):
        agent = make_agent(catalog, [PIZZA], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.run("На обед съел пиццу, 300 г")
        agent._own_state.eaten_date = "2000-01-01"
        agent.client.script["Человек рассказывает, что съел"] = json.dumps({"items": [
            {"phrase_ru": "банан", "query_en": "banana", "grams": 120, "count": None,
             "meal": "snack"}]})
        answer = agent.run("Перекусил бананом")
        eaten = [it for it in answer.plan.items if it.slot.endswith("(съедено)")]
        assert len(eaten) == 1

    def test_without_a_plan_it_is_a_normal_turn(self, catalog):
        """«Съел» без показанного плана пересчитывать нечего."""
        agent = make_agent(catalog, [PIZZA], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг. На обед съел пиццу")
        assert not any("Человек рассказывает, что съел" in c for c in agent.client.calls)
