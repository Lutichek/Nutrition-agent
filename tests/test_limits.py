"""
Ограничения частоты и суточный потолок вызовов модели (ревью R1).

Ручки были открыты без единого ограничения: реплика — 4-7 платных вызовов
модели, меню на месяц — 15 с процессора. Проверяется то, ради чего
лимиты заведены: 429 с понятным текстом и Retry-After; скрипт без cookie
не обходит лимит; потолок считает КАЖДЫЙ вызов, включая вызовы наперёд;
исчерпанный потолок не тратит больше ни одного вызова.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLMClient, FakeRetriever
from fastapi.testclient import TestClient

import api
from agent import NutritionAgent
from api import BUDGET_TEXT, app
from config import Settings
from limits import Limits, LLMBudget, _today
from storage import MemoryStorage

PROFILE = {"sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74,
           "activity": "sedentary", "goal": "lose"}
SCRIPT = {
    "маршрутизатор": json.dumps({"intent": "question", "red_flag": ""}),
    "has_relevant": '{"has_relevant": true, "keep": [1]}',
    "ассистент по питанию. Ответь": "Белка нужно больше.",
}


@pytest.fixture
def setup(catalog, fake_chunks):
    def make(speculate=False, per_day=None, **limits):
        agent = NutritionAgent(client=FakeLLMClient(script=SCRIPT), model="fake",
                               retriever=FakeRetriever(fake_chunks), catalog=catalog,
                               speculate=speculate)
        storage = MemoryStorage()
        if per_day is not None:
            agent.budget = LLMBudget(storage, per_day)
        api._state.update(agent=agent, catalog=catalog, storage=storage,
                          limits=Limits(**limits))
        api._state.pop("limiter", None)
        return agent, storage
    yield make
    api._state.clear()


def ask(http: TestClient, path="/chat"):
    return http.post(path, json={"message": "Сколько белка?"})


class TestRate:
    def test_session_limit_is_429_with_text(self, setup):
        setup(chat_per_minute_session=2)
        http = TestClient(app)
        assert [ask(http).status_code for _ in range(2)] == [200, 200]

        response = ask(http)
        assert response.status_code == 429
        assert "подождите" in response.json()["detail"]
        assert int(response.headers["retry-after"]) >= 1

    def test_script_without_cookies_hits_ip_limit(self, setup):
        """Без cookie каждый запрос — НОВАЯ сессия: держит только лимит по адресу."""
        setup(chat_per_minute_ip=3)
        codes = [ask(TestClient(app)).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]

    def test_calc_and_menu_limited_separately(self, setup):
        """Меню — строже: 15 с процессора на месяц. Расчёт нормы — мягче."""
        setup(calc_per_minute_ip=3, menu_per_minute_ip=1)
        http = TestClient(app)
        assert [http.post("/targets", json=PROFILE).status_code for _ in range(4)] == \
            [200, 200, 200, 429]
        body = {"profile": PROFILE, "days": 1}
        assert http.post("/menu", json=body).status_code == 200
        assert http.post("/menu", json=body).status_code == 429

    def test_concurrent_streams_per_ip(self, setup):
        setup(streams_per_ip=1)
        api._limiter().acquire("stream:testclient", 1)  # один ответ уже идёт
        response = ask(TestClient(app), "/chat/stream")
        assert response.status_code == 429

        api._limiter().release("stream:testclient")
        assert ask(TestClient(app), "/chat/stream").status_code == 200
        assert api._limiter()._active["stream:testclient"] == 0, "место освобождено"


class TestDailyBudget:
    def test_exhausted_mid_reply_is_429_and_stops(self, setup):
        agent, storage = setup(per_day=2)  # вопрос стоит 4 вызова
        response = ask(TestClient(app))

        assert response.status_code == 429
        assert response.json()["detail"] == BUDGET_TEXT
        assert len(agent.client.calls) == 2, "сверх потолка — ни одного вызова"

    def test_exhausted_before_reply_spends_nothing(self, setup):
        agent, storage = setup(per_day=2)
        storage.add_usage(_today(), 2)
        for path in ("/chat", "/chat/stream"):
            response = ask(TestClient(app), path)
            assert response.status_code == 429, path
        assert agent.client.calls == []

    def test_counts_every_call_including_ahead(self, setup):
        """Вызовы наперёд (speculate) — тоже деньги: невостребованный
        extract_profile на вопросе обязан попасть в счётчик."""
        agent, storage = setup(speculate=True, per_day=1000)
        assert ask(TestClient(app)).status_code == 200
        used = storage.get_usage(_today())
        assert used == len(agent.client.calls) == 5


def test_limits_from_environment(monkeypatch):
    monkeypatch.setenv("CHAT_PER_MINUTE_SESSION", "5")
    monkeypatch.setenv("LLM_CALLS_PER_DAY", "77")
    limits = Limits.from_settings(Settings())
    assert limits.chat_per_minute_session == 5
    assert limits.llm_calls_per_day == 77
