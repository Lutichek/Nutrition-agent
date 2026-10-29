"""
Аллергия — не то же, что «не ем» (ревью R5).

Фильтр для обоих один и тот же: аллерген отсекает exclude по названию,
разделу и составу. Но различие терялось ещё до фильтра — «аллергия на
арахис» превращалась в просто "peanut", — и агент не мог сказать
главного: фильтр по справочнику не гарантирует отсутствия следов.
Проверяется, что аллергия узнаётся (моделью, формулировкой или самим
текстом), не теряется между репликами, остаётся в exclude и что
предупреждение — шаблон кода, а не текст модели.
"""

from __future__ import annotations

import json

import pandas as pd
from conftest import FakeLLMClient, FakeRetriever

from agent import ALLERGY_NOTE, NutritionAgent, _merge_allergies, allergy_note
from restrictions import matches_terms

FULL = {"sex": "male", "age": 30, "height_cm": 180, "weight_kg": 80,
        "activity": "moderate", "goal": "maintain"}


def make_agent(catalog, extracted: dict, structured: bool = False) -> NutritionAgent:
    agent = NutritionAgent(
        client=FakeLLMClient(script={
            "маршрутизатор": json.dumps({"intent": "plan", "red_flag": ""}),
            "Извлеки параметры": json.dumps(extracted),
            "Объясни человеку": "План готов.",
        }),
        model="fake", retriever=FakeRetriever([]), catalog=catalog,
    )
    agent.structured_extract = structured
    return agent


def test_allergy_gets_warning_and_allergen_stays_out(catalog):
    agent = make_agent(catalog, {**FULL, "exclude": ["peanut"], "allergies": ["peanut"]})
    answer = agent.run("Мне 30, мужчина, 180 см, 80 кг. Аллергия на арахис")

    assert agent.profile["allergies"] == ["peanut"]
    assert "⚠️ Про аллергию (арахис)" in answer.answer
    assert "этикетк" in answer.answer and "следов" in answer.answer
    names = pd.Series([item.name for item in answer.plan.items], dtype=object)
    assert not matches_terms(names, ["peanut"]).any()


def test_preference_has_no_allergy_warning(catalog):
    """«Не ем свинину» — убеждение или вкус: про этикетку говорить незачем."""
    agent = make_agent(catalog, {**FULL, "exclude": ["pork"]})
    answer = agent.run("Мне 30, мужчина, 180 см, 80 кг. Не ем свинину")
    assert "Про аллергию" not in answer.answer
    assert not agent.profile.get("allergies")


def test_intolerance_is_not_allergy(catalog):
    agent = make_agent(catalog, {**FULL, "exclude": ["lactose"]})
    answer = agent.run("Мне 30, мужчина, 180 см, 80 кг. Непереносимость лактозы")
    assert "Про аллергию" not in answer.answer


def test_text_mentions_allergy_but_model_missed_it(catalog):
    """Страховка: «аллергия» в тексте, а модель ничего не отметила —
    предупреждение всё равно есть, с просьбой назвать аллерген."""
    agent = make_agent(catalog, {**FULL})
    answer = agent.run("Мне 30, мужчина, 180 см, 80 кг. У меня пищевая аллергия")
    assert "Про аллергию:" in answer.answer
    assert "Напишите, на что именно аллергия" in answer.answer


def test_allergy_phrase_in_exclude_counts(catalog):
    """«nut allergy» в exclude — это уже аллергия, даже без поля allergies."""
    agent = make_agent(catalog, {**FULL, "exclude": ["nut allergy"]})
    answer = agent.run("Мне 30, мужчина, 180 см, 80 кг. Аллергия на орехи")
    assert "⚠️ Про аллергию (орехи)" in answer.answer


def test_allergy_from_structured_mentions(catalog):
    """Схема: вид упоминания allergy пополняет и allergies, и exclude,
    даже если итоговые поля модель не заполнила."""
    schema_answer = {
        "mentions": [{"phrase_ru": "аллергия на кунжут", "kind": "allergy",
                      "english": "sesame"}],
        **{k: None for k in ("sex", "age", "height_cm", "weight_kg", "activity",
                             "goal", "rate_kg_per_week", "days")},
        "exclude": [], "allergies": [], "include": [], "include_ru": [],
    }
    agent = make_agent(catalog, schema_answer, structured=True)
    agent.run("Аллергия на кунжут")
    assert agent.profile["allergies"] == ["sesame"]
    assert agent.profile["exclude"] == ["sesame"]


def test_allergy_survives_next_turn():
    """Следующая реплика без упоминания аллергии не может её снять."""
    allergies, exclude = _merge_allergies("Ещё не ем грибы", ["peanut"], [], ["mushroom"])
    assert allergies == ["peanut"]
    assert "peanut" in exclude and "mushroom" in exclude


def test_note_is_code_template():
    """Текст — шаблон кода: модель его не пишет и не может смягчить."""
    note = allergy_note(["eggs", "eggs", "lactose"])
    assert note == ALLERGY_NOTE.format(names=" (яйца, молочное)", ask="")
    assert allergy_note([]) == ""


def test_model_allergy_needs_the_word():
    """«Хочу творог, но у меня непереносимость лактозы» — живая модель
    отмечала аллергией 3/3. Без слова «аллергия» её отметку не берём."""
    allergies, exclude = _merge_allergies(
        "Хочу творог, но у меня непереносимость лактозы", [], ["lactose"], ["lactose"])
    assert allergies == []
    assert exclude == ["lactose"], "исключение остаётся — меняется только предупреждение"


def test_allergy_word_said_earlier_counts():
    """«И на кунжут тоже» после «аллергия на арахис» — слово было раньше."""
    allergies, _ = _merge_allergies("И на кунжут тоже", ["peanut"], ["sesame"],
                                    ["peanut", "sesame"], said_before=True)
    assert allergies == ["peanut", "sesame"]
