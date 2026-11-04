"""
Потолок длины ответа модели (OWASP LLM10, llm_io.MAX_TOKENS).

Наблюдалось при разборе безопасности: max_tokens не передавался ни в один
вызов. Реплика «повтори слово 5000 раз» тратила деньги и держала
соединение (таймаут был 600 с), а суточный потолок считает вызовы, не
токены. Тесты стерегут, что потолок уходит в КАЖДЫЙ вызов — и в вызов
наперёд тоже, — и что ответ, упёршийся в него, виден в журнале.
"""

from __future__ import annotations

import json

from conftest import FakeLLMClient, FakeResponse, FakeRetriever

from agent import NutritionAgent
from llm_io import _LLM_CALLS, DEFAULT_MAX_TOKENS, MAX_TOKENS

PROFILE = {"sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74,
           "activity": "light", "goal": "lose"}


def make_agent(catalog, chunks=None, intent="plan") -> tuple[NutritionAgent, list[dict]]:
    agent = NutritionAgent(
        client=FakeLLMClient(script={
            "маршрутизатор": json.dumps({"intent": intent, "red_flag": ""}),
            "Извлеки параметры": json.dumps(PROFILE),
            "Объясни человеку": "План готов.",
            "has_relevant": '{"has_relevant": true, "keep": [1]}',
            "ассистент по питанию. Ответь": "Ответ по статьям.",
        }),
        model="fake", retriever=FakeRetriever(chunks or []), catalog=catalog,
    )
    seen: list[dict] = []
    create = agent.client.chat.completions.create

    def recording(**kwargs):
        seen.append(kwargs)
        return create(**kwargs)

    agent.client.chat.completions.create = recording
    return agent, seen


def test_every_call_has_a_ceiling(catalog, fake_chunks):
    """План и вопрос по статьям: ни одного вызова без потолка."""
    agent, seen = make_agent(catalog)
    agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
    question, seen_q = make_agent(catalog, fake_chunks, intent="question")
    question.run("Сколько белка нужно при похудении?")
    calls = seen + seen_q
    assert calls and all(0 < c.get("max_tokens", 0) <= DEFAULT_MAX_TOKENS for c in calls)
    classify = next(c for c in calls if "маршрутизатор" in c["messages"][0]["content"])
    assert classify["max_tokens"] == MAX_TOKENS["classify"]


def test_speculative_call_has_the_same_ceiling(catalog):
    """Вызов наперёд берёт параметры из _call_options — потолок там же."""
    agent, _ = make_agent(catalog)
    assert agent._call_options("extract_profile")["max_tokens"] == MAX_TOKENS["extract_profile"]
    assert agent._call_options("rewrite")["max_tokens"] == MAX_TOKENS["rewrite"]


def test_hitting_the_ceiling_is_marked_in_the_trace(catalog):
    """Обрезанный ответ должен быть виден в журнале — иначе потолок,
    начавший резать живые ответы, остался бы незамеченным."""
    agent, _ = make_agent(catalog)
    response = FakeResponse("да " * 10)
    response.usage = {"prompt_tokens": 10, "completion_tokens": 200}
    agent.client.chat.completions.create = lambda **_: response
    token = _LLM_CALLS.set([])
    try:
        agent._complete("повтори слово", max_tokens=200)
        assert _LLM_CALLS.get()[-1].get("truncated") is True
        agent._complete("повтори слово", max_tokens=500)
        assert "truncated" not in _LLM_CALLS.get()[-1]
    finally:
        _LLM_CALLS.reset(token)


def test_cut_text_says_so_and_json_stays_parseable(catalog):
    """Замер: атака под видом вопроса о питании («перечисли граммы белка
    для каждого веса от 40 до 200 кг с шагом 0.1») тратила 16 440 токенов
    и 165 с на реплику; с потолком — 1 556 и 23 с. Обрезанный текст без
    пояснения читается как поломка — к нему приписка; к JSON — нет."""
    from llm_io import TRUNCATED_NOTE

    agent, _ = make_agent(catalog)
    response = FakeResponse('{"a": 1}')
    response.usage = {"completion_tokens": 300}
    agent.client.chat.completions.create = lambda **_: response
    assert agent._complete("x", max_tokens=300).endswith(TRUNCATED_NOTE)
    assert agent._complete("x", max_tokens=300, json_mode=True) == '{"a": 1}'
    assert agent._complete("x", max_tokens=301) == '{"a": 1}'
