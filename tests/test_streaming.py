"""
Ответ по частям: /chat/stream и события агента.

Стриминг обязан менять только то, КОГДА человек видит текст. Поэтому
проверяется: итог тот же, что без потока; этапы — настоящие шаги графа,
а не угаданные таймером; номер PMID не мелькает ни в одном промежуточном
событии (ссылка приходит кусками); поток дочитывается до конца — иначе
соединение с провайдером выбрасывается (замер в agent._stream); обрыв
попытки не склеивает обрывок с повтором; сбой не теряется.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLMClient, FakeRetriever, FakeStreamedResponse, consented
from fastapi.testclient import TestClient

import agent as agent_module
import api
from agent import NutritionAgent, for_reader_partial
from api import app
from chat_log import ChatLog
from storage import MemoryStorage

PROFILE = {
    "sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74,
    "activity": "sedentary", "goal": "lose",
}
ANSWER = "Белка нужно больше, чем обычно, в [PMID 111]. Это сохраняет мышцы (PMID 222)."
QUESTION_SCRIPT = {
    "маршрутизатор": json.dumps({"intent": "question", "red_flag": ""}),
    "has_relevant": '{"has_relevant": true, "keep": [1, 2]}',
    "ассистент по питанию. Ответь": ANSWER,
}
PLAN_SCRIPT = {
    "маршрутизатор": json.dumps({"intent": "plan", "red_flag": ""}),
    "Извлеки параметры": json.dumps(PROFILE),
    "Объясни человеку": "Норма посчитана по формуле.",
}


def make_agent(catalog, script, chunks=None, **kwargs) -> NutritionAgent:
    return NutritionAgent(client=FakeLLMClient(script=dict(script), **kwargs), model="fake",
                          retriever=FakeRetriever(chunks or []), catalog=catalog)


class TestAgentEvents:
    def test_stages_follow_the_graph(self, catalog, fake_chunks):
        events: list[dict] = []
        make_agent(catalog, QUESTION_SCRIPT, fake_chunks).run("Сколько белка?",
                                                               on_event=events.append)
        stages = [e["node"] for e in events if e["type"] == "stage"]
        assert stages == ["classify", "rewrite", "retrieve", "grade", "generate"]

    def test_same_answer_with_and_without_stream(self, catalog, fake_chunks):
        plain = make_agent(catalog, QUESTION_SCRIPT, fake_chunks).run("Сколько белка?")
        events: list[dict] = []
        streamed = make_agent(catalog, QUESTION_SCRIPT, fake_chunks).run(
            "Сколько белка?", on_event=events.append)

        assert streamed.answer == plain.answer
        texts = [e["raw"] for e in events if e["type"] == "text"]
        assert len(texts) > 5, "ответ должен приходить кусками"
        assert texts[-1] == ANSWER

    def test_plan_text_then_notes_in_final(self, catalog):
        """К объяснению плана код дописывает стоимость — её в потоке нет,
        поэтому итог обязан прийти отдельно и заменить показанное."""
        events: list[dict] = []
        answer = make_agent(catalog, PLAN_SCRIPT).run("Мне 31, женщина, 168, 74",
                                                      on_event=events.append)
        streamed = [e["raw"] for e in events if e["type"] == "text"][-1]
        assert streamed == "Норма посчитана по формуле."
        assert answer.answer.startswith(streamed) and len(answer.answer) > len(streamed)

    def test_stream_read_to_the_end(self, catalog, fake_chunks, monkeypatch):
        """SDK бросал чтение на [DONE], и соединение выбрасывалось — с
        зависающим рукопожатием это секунды на каждом ответе."""
        opened: list[FakeStreamedResponse] = []

        class Tracked(FakeStreamedResponse):
            def iter_lines(self):
                self.finished = False
                yield from super().iter_lines()
                yield ""  # хвост после [DONE], как у настоящего потока
                self.finished = True

        agent = make_agent(catalog, QUESTION_SCRIPT, fake_chunks)
        streaming = agent.client.chat.completions.with_streaming_response
        original = streaming.create

        def create(**kwargs):
            response = Tracked(original(**kwargs)._content)
            opened.append(response)
            return response

        monkeypatch.setattr(streaming, "create", create)
        agent.run("Сколько белка?", on_event=lambda e: None)
        assert opened and all(r.finished for r in opened)

    def test_retry_starts_text_over(self, catalog, fake_chunks, monkeypatch):
        """Попытка оборвалась на середине — повтор не дописывается к обрывку."""
        monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)

        class Broken(FakeStreamedResponse):
            def iter_lines(self):
                lines = list(super().iter_lines())
                yield from lines[:6]
                raise ConnectionError("обрыв")

        agent = make_agent(catalog, QUESTION_SCRIPT, fake_chunks)
        streaming = agent.client.chat.completions.with_streaming_response
        original = streaming.create
        attempts = []

        def create(**kwargs):
            content = original(**kwargs)._content
            attempts.append(1)
            return (Broken if len(attempts) == 1 else FakeStreamedResponse)(content)

        monkeypatch.setattr(streaming, "create", create)
        events: list[dict] = []
        answer = agent.run("Сколько белка?", on_event=events.append)

        texts = [e["raw"] for e in events if e["type"] == "text"]
        restart = next(i for i in range(1, len(texts)) if len(texts[i]) < len(texts[i - 1]))
        assert ANSWER.startswith(texts[restart]), "новая попытка — с начала ответа"
        assert texts[-1] == ANSWER and answer.answer == ANSWER


@pytest.mark.parametrize("raw, shown", [
    ("Белка нужно больше [PM", "Белка нужно больше"),
    ("Белка нужно больше (PMID 1", "Белка нужно больше"),
    ("Белка нужно больше PMI", "Белка нужно больше"),
    ("Белка (около 1.6 г/кг) в день", "Белка (около 1.6 г/кг) в день"),
])
def test_partial_text_never_shows_pmid(raw, shown):
    assert for_reader_partial(raw) == shown


@pytest.fixture
def client(catalog, fake_chunks, tmp_path):
    def make(script=QUESTION_SCRIPT, **kwargs):
        api._state.update(
            agent=make_agent(catalog, script, fake_chunks, **kwargs), catalog=catalog,
            storage=MemoryStorage(), chat_log=ChatLog(tmp_path),
        )
        return consented(TestClient(app), improve_service=True)
    yield make
    api._state.clear()


def sse(response) -> list[dict]:
    return [json.loads(line[5:]) for line in response.text.splitlines()
            if line.startswith("data:")]


class TestStreamEndpoint:
    def test_done_matches_chat(self, client):
        """Итог потока — ровно ответ /chat: страница рисует его той же функцией."""
        plain = client().post("/chat", json={"message": "Сколько белка?"}).json()
        events = sse(client().post("/chat/stream", json={"message": "Сколько белка?"}))

        done = events[-1]
        assert done["type"] == "done"
        for key in ("answer", "sources", "checked_fragments", "profile"):
            assert done["response"][key] == plain[key], key

    def test_no_pmid_in_any_event(self, client):
        """Ссылка рвётся на куски по 7 символов — проверяем каждое событие."""
        events = sse(client().post("/chat/stream", json={"message": "Сколько белка?"}))
        texts = [e["text"] for e in events if e["type"] == "text"]
        assert texts, "текст должен приходить до итога"
        for text in texts:
            assert "PMID" not in text.upper() and "111" not in text and "222" not in text, text

    def test_stages_before_text(self, client):
        events = sse(client().post("/chat/stream", json={"message": "Сколько белка?"}))
        kinds = [e["type"] for e in events]
        assert kinds[0] == "stage" and kinds.index("text") > kinds.index("stage")
        assert {e["label"] for e in events if e["type"] == "stage"} >= {"ищу в статьях"}

    def test_session_continues_and_journal_written(self, client, tmp_path):
        http = client()
        http.post("/chat/stream", json={"message": "Привет"})
        http.post("/chat/stream", json={"message": "Ещё"})  # та же cookie

        records = [json.loads(line) for path in tmp_path.glob("chat-*.jsonl")
                   for line in path.read_text(encoding="utf-8").splitlines()]
        assert [r["turn"] for r in records] == [1, 2]
        assert "PMID" not in records[0]["shown"]

    def test_failure_is_an_event_and_logged(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)
        events = sse(client(fail_times=99).post("/chat/stream", json={"message": "Сколько белка?"}))

        assert events[-1]["type"] == "error"
        assert "недоступна" in events[-1]["detail"]
        (record,) = [json.loads(line) for path in tmp_path.glob("chat-*.jsonl")
                     for line in path.read_text(encoding="utf-8").splitlines()]
        assert record["status"] == "error"
