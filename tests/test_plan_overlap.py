"""
Меню на несколько дней собирается, пока модель объясняет первый день.

Замер (experiments/latency.py): меню на месяц собиралось 16 с, и только
после этого модель начинала объяснение, хотя читает она один первый день.
Теперь дни со второго идут в фоне. Ускорение не должно стоить ничего,
кроме времени, — это и проверяется: меню то же до блюда, модель вызвана
раньше, чем дособраны дни, отказ позднего дня не теряется.
"""

from __future__ import annotations

import json
import threading

from conftest import FakeLLMClient, FakeRetriever

import agent as agent_module
from agent import NutritionAgent
from chat_log import replay_plan
from solver import PlanNotFeasible

PROFILE = {
    "sex": "male", "age": 35, "height_cm": 180, "weight_kg": 85,
    "activity": "moderate", "goal": "lose", "days": 7,
}


def make_agent(catalog) -> NutritionAgent:
    return NutritionAgent(
        client=FakeLLMClient(script={
            "маршрутизатор": json.dumps({"intent": "plan", "red_flag": ""}),
            "Извлеки параметры": json.dumps(PROFILE),
            "Объясни человеку": "Норма посчитана.",
        }),
        model="fake",
        retriever=FakeRetriever([]),
        catalog=catalog,
    )


def ids(menu) -> list[list[int]]:
    return [[item.fdc_id for item in day.items] for day in menu.days]


def test_menu_same_as_built_at_once(catalog):
    """Фоновая сборка обязана дать то же меню, что сборка целиком.

    Окно разнообразия помнит блюда прошлых дней: потеряй генератор своё
    состояние — дни разойдутся молча, а тесты на «меню собралось» пройдут.
    """
    agent = make_agent(catalog)
    answer = agent.run("Мне 35, мужчина, 180 см, 85 кг, 3 тренировки, похудеть, на неделю")

    assert answer.menu is not None and len(answer.menu.days) == 7
    assert answer.plan.items == answer.menu.days[0].items
    assert ids(answer.menu) == replay_plan(agent, answer.trace)


def test_model_called_before_rest_of_menu(catalog, monkeypatch):
    """Ради этого всё и делалось: объяснение не ждёт дней со второго."""
    explained = threading.Event()
    seen: list[bool] = []
    real = agent_module.menu_days

    def slow_rest(*args, **kwargs):
        days = real(*args, **kwargs)
        yield next(days)
        # Второй день начнётся, только когда модель уже позвали.
        # Последовательный код здесь упёрся бы в таймаут и записал False.
        seen.append(explained.wait(timeout=10))
        yield from days

    monkeypatch.setattr(agent_module, "menu_days", slow_rest)
    agent = make_agent(catalog)
    create = agent.client.chat.completions.create

    def watched(**kwargs):
        if "Объясни человеку" in kwargs["messages"][0]["content"]:
            explained.set()
        return create(**kwargs)

    monkeypatch.setattr(agent.client.chat.completions, "create", watched)
    answer = agent.run("Мне 35, мужчина, 180 см, 85 кг, 3 тренировки, похудеть, на неделю")

    assert seen == [True]
    assert len(answer.menu.days) == 7


def test_late_day_failure_is_still_reported(catalog, monkeypatch):
    """Раньше несобравшийся день любого номера давал «не получилось собрать».

    В фоне ошибка всплывает уже после объяснения первого дня, и без
    отдельной ветки человек получил бы план недели, которой нет.
    """
    real = agent_module.menu_days

    def fails_on_day_two(*args, **kwargs):
        days = real(*args, **kwargs)
        yield next(days)
        raise PlanNotFeasible("Подходящих блюд не хватило.")

    monkeypatch.setattr(agent_module, "menu_days", fails_on_day_two)
    answer = make_agent(catalog).run("Мне 35, мужчина, 180 см, 85 кг, 3 тренировки, на неделю")

    assert "Не получилось собрать рацион" in answer.answer
    assert answer.plan is None, "карточка первого дня рядом с отказом сбивала бы с толку"
    assert answer.menu is None
