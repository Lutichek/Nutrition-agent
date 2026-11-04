"""
Персональные данные (ревью R4): согласие, журнал, «удалить мои данные».

В разговоре — вес, возраст и диагнозы словами человека: специальная
категория по 152-ФЗ. Раньше согласия не было вовсе, журнал писал всё
подряд на 30 дней, а удаление сессии не трогало ни журнал, ни меню.
Проверяется, что каждая из этих дыр закрыта СЕРВЕРОМ, а не только окном
на странице: клиент API окно не видит.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLMClient, FakeRetriever, consented
from fastapi.testclient import TestClient

import api
from agent import NutritionAgent
from api import SESSION_COOKIE, app
from chat_log import ChatLog
from storage import MemoryStorage

SCRIPT = {
    "маршрутизатор": json.dumps({"intent": "question", "red_flag": ""}),
    "has_relevant": '{"has_relevant": true, "keep": [1]}',
    "ассистент по питанию. Ответь": "Белка нужно больше.",
}


@pytest.fixture
def journal(tmp_path):
    return ChatLog(tmp_path)


@pytest.fixture
def agent(catalog, fake_chunks, journal):
    agent = NutritionAgent(client=FakeLLMClient(script=SCRIPT), model="fake",
                           retriever=FakeRetriever(fake_chunks), catalog=catalog)
    api._state.update(agent=agent, catalog=catalog, storage=MemoryStorage(), chat_log=journal)
    yield agent
    api._state.clear()


def records(directory) -> list[dict]:
    return [json.loads(line) for path in sorted(directory.glob("chat-*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestConsent:
    @pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
    def test_no_consent_no_conversation(self, agent, path):
        """Без согласия разговор не начинается — и сессия не заводится."""
        response = TestClient(app).post(path, json={"message": "Мне 31, вешу 74 кг"})
        assert response.status_code == 403
        assert "согласие" in response.json()["detail"]
        assert api._state["storage"].count_sessions() == 0
        assert agent.client.calls == [], "текст о здоровье не ушёл модели"

    def test_refusal_is_explained(self, agent):
        response = TestClient(app).post("/consent", json={"health_data": False})
        assert response.status_code == 400

    def test_status_before_and_after(self, agent):
        http = TestClient(app)
        assert http.get("/session").json()["consented"] is False
        consented(http)
        assert http.get("/session").json() == {"consented": True, "improve_service": False}


class TestJournalNeedsSeparateConsent:
    def test_without_it_nothing_is_written(self, agent, tmp_path):
        http = consented(TestClient(app), improve_service=False)
        assert http.post("/chat", json={"message": "У меня диабет"}).status_code == 200
        assert records(tmp_path) == []

    def test_with_it_turns_are_written(self, agent, tmp_path):
        http = consented(TestClient(app), improve_service=True)
        http.post("/chat", json={"message": "Сколько белка?"})
        (record,) = records(tmp_path)
        assert record["message"] == "Сколько белка?"
        token = http.cookies[SESSION_COOKIE]
        assert token not in json.dumps(record), "в журнале — не пропуск, а public_id"


class TestDeleteMyData:
    def test_everything_goes(self, agent, tmp_path):
        """Раньше удаление не трогало журнал и меню — данные оставались на диске."""
        mine = consented(TestClient(app), improve_service=True)
        other = consented(TestClient(app), improve_service=True)
        mine.post("/chat", json={"message": "Моё"})
        other.post("/chat", json={"message": "Чужое"})
        token = mine.cookies[SESSION_COOKIE]
        owner = api._state["storage"].load_session(token)["public_id"]
        menu_id = api._state["storage"].put_menu(owner, "{}", {})
        day = next(tmp_path.glob("chat-*.jsonl"))
        day.with_suffix(".md").write_text("# копия переписки", encoding="utf-8")

        response = mine.delete("/session")

        assert response.status_code == 200
        assert response.json()["journal_deleted"] == 1
        assert [r["message"] for r in records(tmp_path)] == ["Чужое"]
        assert not day.with_suffix(".md").exists(), "читаемая копия тоже стёрта"
        assert api._state["storage"].get_menu(menu_id) is None
        assert api._state["storage"].load_session(token) is None
        assert mine.get("/session").json()["consented"] is False
        assert mine.post("/chat", json={"message": "Ещё"}).status_code == 403


def test_forget_keeps_other_sessions(tmp_path):
    log = ChatLog(tmp_path)
    for session in ("a", "b", "a"):
        log.write({"session_id": session, "message": session})
    assert log.forget("a") == 2
    assert [r["session_id"] for r in records(tmp_path)] == ["b"]
    assert ChatLog(None).forget("a") == 0


def test_privacy_page_is_served_as_draft(agent):
    response = TestClient(app).get("/privacy.html")
    assert response.status_code == 200
    assert "ЧЕРНОВИК" in response.text
