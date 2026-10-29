"""
Извлечение профиля по схеме с разбором упоминаний (SGR).

Зачем: на фразах со смешанным знаком («молочку не ем, мясо ем, но не
свинину») свободный JSON терял ограничение 3 прогона из 3, а с разбором
упоминаний — ни разу (experiments/extract_profile.py). Здесь проверяется,
что переход на схему ничего не сломал вокруг: ответ приводится к прежнему
виду, «не названо» не затирает известное, и наперёд модель спрашивается
в том же режиме, что и сам шаг.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLMClient, FakeRetriever

from agent import (
    EXTRACT_PROFILE_PROMPT,
    EXTRACT_PROFILE_PROMPT_SGR,
    EXTRACT_PROFILE_SCHEMA,
    NutritionAgent,
    _from_structured,
)

SCHEMA_ANSWER = {
    "mentions": [
        {"phrase_ru": "молочку не ем", "kind": "food_excluded", "english": "lactose"},
        {"phrase_ru": "не свинину", "kind": "food_excluded", "english": "pork"},
        {"phrase_ru": "рыбу люблю", "kind": "food_wanted", "english": "fish"},
    ],
    "sex": None, "age": None, "height_cm": None, "weight_kg": None, "activity": None,
    "goal": None, "rate_kg_per_week": None, "days": None,
    "exclude": ["lactose", "pork"], "include": ["fish"],
    "include_ru": [{"en": "fish", "ru": "рыба"}],
}


def test_schema_answer_becomes_old_shape():
    assert _from_structured(SCHEMA_ANSWER) == {
        "exclude": ["lactose", "pork"], "include": ["fish"], "include_ru": {"fish": "рыба"},
    }


def test_free_json_passes_unchanged():
    old = {"age": 31, "exclude": ["pork"], "include_ru": {"buckwheat": "гречка"}}
    assert _from_structured(old) == old


def make_agent(catalog, answer: dict, speculate=False) -> NutritionAgent:
    return NutritionAgent(
        client=FakeLLMClient(script={
            "маршрутизатор": json.dumps({"intent": "plan", "red_flag": ""}),
            "Извлеки параметры": json.dumps(answer),
            "не хватает данных": "Уточните рост и вес.",
        }),
        model="fake", retriever=FakeRetriever([]), catalog=catalog, speculate=speculate,
    )


def test_not_named_does_not_erase_known_restriction(catalog):
    """По схеме модель ОБЯЗАНА вернуть exclude — и на реплике без
    ограничений возвращает пустой список. Наложи его поверх профиля —
    названная раньше свинина молча пропала бы из исключений."""
    agent = make_agent(catalog, {**SCHEMA_ANSWER, "mentions": [], "exclude": [],
                                 "include": [], "include_ru": [], "age": 31})
    agent._own_state.profile = {"exclude": ["pork"]}
    agent.run("Мне 31")

    assert agent.profile["exclude"] == ["pork"]
    assert agent.profile["age"] == 31


@pytest.mark.parametrize("speculate", [False, True])
def test_schema_used_for_step_and_ahead(catalog, speculate):
    """Наперёд и в самом шаге — один режим: ответ, полученный наперёд,
    подставляется по совпадению промпта, и режим он не сверяет."""
    agent = make_agent(catalog, SCHEMA_ANSWER, speculate=speculate)
    create = agent.client.chat.completions.create
    seen: list[dict] = []

    def watched(**kwargs):
        if "Извлеки параметры" in kwargs["messages"][0]["content"]:
            seen.append(kwargs)
        return create(**kwargs)

    agent.client.chat.completions.create = watched
    agent.run("Молочку не ем, мясо ем, но не свинину, а рыбу люблю")

    assert len(seen) == 1
    assert seen[0]["response_format"] == {"type": "json_schema",
                                          "json_schema": EXTRACT_PROFILE_SCHEMA}
    assert seen[0]["messages"][0]["content"].startswith(EXTRACT_PROFILE_PROMPT_SGR[:40])
    assert agent.profile["exclude"] == ["lactose", "pork"]
    assert agent.profile["include_ru"] == {"fish": "рыба"}


def test_free_json_mode_keeps_old_prompt(catalog):
    """Режим для провайдера без строгих схем — прежний промпт и json_object."""
    agent = make_agent(catalog, {"exclude": ["pork"]})
    agent.structured_extract = False
    create = agent.client.chat.completions.create
    seen: list[dict] = []

    def watched(**kwargs):
        if "Извлеки параметры" in kwargs["messages"][0]["content"]:
            seen.append(kwargs)
        return create(**kwargs)

    agent.client.chat.completions.create = watched
    agent.run("Не ем свинину")

    assert seen[0]["response_format"] == {"type": "json_object"}
    assert "Верни ТОЛЬКО JSON" in seen[0]["messages"][0]["content"]
    assert EXTRACT_PROFILE_PROMPT.split("{known}")[0] in seen[0]["messages"][0]["content"]


def test_new_exclusion_taken_from_mentions(catalog):
    """«Ещё не ем грибы» после «Не ем свинину» — живая модель, 3/3:
    разбор верный (mushroom, food_excluded), а в exclude модель переписала
    только известное ["pork"]. Грибы молча оставались в каталоге."""
    answer = {**SCHEMA_ANSWER, "include": [], "include_ru": [],
              "mentions": [{"phrase_ru": "не ем грибы", "kind": "food_excluded",
                            "english": "mushroom"}],
              "exclude": ["pork"]}
    agent = make_agent(catalog, answer)
    agent._own_state.profile = {"exclude": ["pork"]}
    agent.run("Ещё не ем грибы")

    assert agent.profile["exclude"] == ["pork", "mushroom"]


def test_exclusions_accumulate_across_turns(catalog):
    """Свободный JSON на «Люблю курицу, но не ем грибы» после «Молочку не ем»
    возвращал exclude ["mushroom"] (3/3), и замена списка снимала молочное."""
    agent = make_agent(catalog, {"exclude": ["mushroom"], "include": ["chicken"]})
    agent.structured_extract = False
    agent._own_state.profile = {"exclude": ["lactose"]}
    agent.run("Люблю курицу, но не ем грибы")

    assert agent.profile["exclude"] == ["lactose", "mushroom"]


@pytest.mark.parametrize("phrase, word, expected", [
    ("без сладкого", "sugar", "sweets"),
    ("не ем сладости", "sugar", "sweets"),
    ("без сахара", "sugar", "sugar"),
    ("не пью сладкие напитки", "sugar", "sugar"),
    ("без шоколада", "chocolate", "chocolate"),
])
def test_sweets_not_sugar(phrase, word, expected):
    """«Ещё без сладкого» — живая модель, 3/3 в обоих режимах: sugar вместо
    sweets, и выпечка оставалась в меню. Поправка — только у фразы, где
    сказано «сладк…», а не «сахар» и не «напитки»."""
    from agent import _category_fix
    assert _category_fix(word, phrase) == expected


def test_sweets_fixed_through_mentions(catalog):
    answer = {**SCHEMA_ANSWER, "include": [], "include_ru": [],
              "mentions": [{"phrase_ru": "без сладкого", "kind": "food_excluded",
                            "english": "sugar"}],
              "exclude": ["pork", "sugar"]}
    agent = make_agent(catalog, answer)
    agent._own_state.profile = {"exclude": ["pork"]}
    agent.run("Ещё без сладкого")
    assert "sweets" in agent.profile["exclude"]
