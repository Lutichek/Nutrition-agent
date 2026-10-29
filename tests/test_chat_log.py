"""
Тесты журнала чата.

Журнал нужен для разбора тихих багов, поэтому проверяется не «файл
появился», а то, ради чего он заведён: в записи есть то, что агент понял,
по записи план повторяется до блюда, и сам журнал не может уронить ответ.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from conftest import FakeLLMClient, FakeRetriever
from fastapi.testclient import TestClient

import agent as agent_module
import api
from agent import NutritionAgent
from api import app
from chat_log import ChatLog, dialogues, main, replay_plan
from storage import MemoryStorage

FULL_PROFILE = {
    "sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74,
    "activity": "sedentary", "goal": "lose",
}

QUESTION_SCRIPT = {
    "маршрутизатор": json.dumps({"intent": "question", "red_flag": ""}),
    "has_relevant": '{"has_relevant": true, "keep": [1, 2]}',
    "ассистент по питанию. Ответь": "Белка нужно больше [PMID 111].",
}


def make_client(catalog, chunks, chat_log, llm=None):
    agent = NutritionAgent(
        client=llm or FakeLLMClient(script=QUESTION_SCRIPT),
        model="fake",
        retriever=FakeRetriever(chunks),
        catalog=catalog,
    )
    api._state.update(
        agent=agent, catalog=catalog, storage=MemoryStorage(), chat_log=chat_log,
    )
    return TestClient(app, raise_server_exceptions=False)


def read_records(directory) -> list[dict]:
    lines = []
    for path in sorted(directory.glob("chat-*.jsonl")):
        lines += path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture(autouse=True)
def _clean_state():
    yield
    api._state.clear()


class TestRecordContent:
    def test_question_turn_records_what_agent_understood(self, catalog, fake_chunks, tmp_path):
        """По паре «вопрос → ответ» тихий баг не найти — нужна середина."""
        client = make_client(catalog, fake_chunks, ChatLog(tmp_path))
        client.post("/chat", json={"message": "Сколько белка?"})

        (record,) = read_records(tmp_path)
        assert record["status"] == "ok"
        assert record["message"] == "Сколько белка?"
        assert record["turn"] == 1

        trace = record["trace"]
        assert trace["intent"] == "question"
        assert len(trace["chunks"]) == 2
        nodes = [step["node"] for step in trace["nodes"]]
        assert nodes[:3] == ["classify", "rewrite", "retrieve"]
        # Сырой ответ модели виден в узле: именно он выдаёт тихие баги.
        classify = trace["nodes"][0]
        assert '"intent": "question"' in classify["llm"][0]["response"]

    def test_shown_text_kept_when_it_differs(self, catalog, fake_chunks, tmp_path):
        """Человек жалуется на то, что ВИДЕЛ: номера PMID там вырезаны."""
        client = make_client(catalog, fake_chunks, ChatLog(tmp_path))
        client.post("/chat", json={"message": "Сколько белка?"})

        (record,) = read_records(tmp_path)
        assert "PMID" in record["answer"]
        assert record["shown"] == "Белка нужно больше."
        assert record["sources"]

    def test_turns_are_numbered_within_session(self, catalog, fake_chunks, tmp_path):
        client = make_client(catalog, fake_chunks, ChatLog(tmp_path))
        client.post("/chat", json={"message": "Привет"})
        client.post("/chat", json={"message": "Ещё"})  # та же cookie

        records = read_records(tmp_path)
        assert [r["turn"] for r in records] == [1, 2]
        assert len({r["session_id"] for r in records}) == 1

    def test_no_secrets_in_record(self, catalog, fake_chunks, tmp_path, monkeypatch):
        """В запись идут только поля из списка — окружение туда не утекает."""
        monkeypatch.setenv("POLZA_AI_API_KEY", "sk-test-secret-value")
        client = make_client(catalog, fake_chunks, ChatLog(tmp_path))
        client.post("/chat", json={"message": "Сколько белка?"})

        text = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("*.jsonl"))
        assert "sk-test-secret-value" not in text

    def test_trace_not_in_saved_answers(self, catalog, fake_chunks):
        """Сохранённые ответы для метрик остаются в прежнем формате."""
        agent = NutritionAgent(
            client=FakeLLMClient(script=QUESTION_SCRIPT), model="fake",
            retriever=FakeRetriever(fake_chunks), catalog=catalog,
        )
        answer = agent.run("Сколько белка?")
        assert answer.trace is not None
        assert "trace" not in answer.model_dump()


class TestFailures:
    def test_provider_failure_is_logged_with_message(
        self, catalog, fake_chunks, tmp_path, monkeypatch
    ):
        """Трассировка в логе uvicorn не говорит, на какой фразе упало."""
        monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)
        client = make_client(
            catalog, fake_chunks, ChatLog(tmp_path), llm=FakeLLMClient(fail_times=99)
        )
        response = client.post("/chat", json={"message": "Сколько белка?"})
        assert response.status_code == 502

        (record,) = read_records(tmp_path)
        assert record["status"] == "error"
        assert record["message"] == "Сколько белка?"
        assert "RuntimeError" in record["error"]
        assert "Traceback" in record["traceback"]

    def test_unwritable_directory_does_not_break_chat(self, catalog, fake_chunks, tmp_path):
        """Повтор истории с Embedder и mkdir на read-only диске.

        Там сервис не поднимался вовсе. Здесь журнал обязан выключиться
        сам, а человек — получить ответ.
        """
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("файл на месте папки")
        chat_log = ChatLog(blocker / "logs")
        client = make_client(catalog, fake_chunks, chat_log)

        for _ in range(2):
            response = client.post("/chat", json={"message": "Сколько белка?"})
            assert response.status_code == 200
            assert response.json()["answer"]
        assert not chat_log.enabled

    def test_constructor_touches_nothing(self, tmp_path):
        """Создание журнала не трогает диск — только первая запись."""
        ChatLog(tmp_path / "logs")
        assert not (tmp_path / "logs").exists()


def utc_today():
    """Сутки журнала — по UTC, как в ChatLog.write.

    С местной датой тест падал в первые часы суток (UTC+5: с 00:00 до 05:00):
    «31 день назад» по местному календарю для журнала был ровно 30-м,
    и файл законно оставался.
    """
    return datetime.now(UTC).date()


class TestRetention:
    def test_old_files_removed_recent_kept(self, tmp_path):
        today = utc_today()
        old = tmp_path / f"chat-{(today - timedelta(days=31)).isoformat()}.jsonl"
        recent = tmp_path / f"chat-{(today - timedelta(days=5)).isoformat()}.jsonl"
        foreign = tmp_path / "notes.jsonl"
        for path in (old, recent, foreign):
            path.write_text("{}\n")

        ChatLog(tmp_path, retention_days=30).write({"message": "x"})

        assert not old.exists()
        assert recent.exists()
        assert foreign.exists(), "чужие файлы журнал не трогает"

    def test_old_reading_copy_removed_too(self, tmp_path):
        """Переписка .md — те же вес и диагнозы, что в журнале.

        Не попади она под очистку, срок хранения в 30 дней соблюдался бы
        только на бумаге: журнал удалён, а его читаемая копия лежит вечно.
        """
        today = utc_today()
        old = tmp_path / f"chat-{(today - timedelta(days=31)).isoformat()}.md"
        recent = tmp_path / f"chat-{(today - timedelta(days=5)).isoformat()}.md"
        for path in (old, recent):
            path.write_text("# переписка\n", encoding="utf-8")

        ChatLog(tmp_path, retention_days=30).write({"message": "x"})

        assert not old.exists()
        assert recent.exists()


def _record(session, turn, message, **extra):
    return {"ts": "2026-10-22T09:37:18+00:00", "session_id": session, "turn": turn,
            "status": "ok", "elapsed_ms": 1500, "message": message, **extra}


class TestDialogues:
    def test_turns_grouped_by_conversation(self):
        """В файле реплики разговоров перемешаны по времени — читать
        надо разговор целиком, иначе второй вопрос оторван от первого."""
        records = [
            _record("a", 1, "первый вопрос А", answer="ответ А1"),
            _record("b", 1, "вопрос Б", answer="ответ Б1"),
            _record("a", 2, "второй вопрос А", answer="ответ А2"),
        ]
        text = dialogues(records)

        assert "Разговоров: 2, реплик: 3" in text
        first_a, second_a, b = (text.index(s) for s in ("ответ А1", "ответ А2", "ответ Б1"))
        assert first_a < second_a < b

    def test_shows_what_the_person_saw(self):
        """Жалуются на показанный текст: в сыром ответе ещё стоят PMID."""
        record = _record("a", 1, "белок?", answer="Нужно больше [PMID 111].",
                         shown="Нужно больше.", sources=["111"])
        text = dialogues([record])

        assert "Нужно больше." in text
        assert "[PMID 111]" not in text
        assert "pubmed.ncbi.nlm.nih.gov/111/" in text

    def test_failed_turn_visible_in_place(self):
        """Реплика со сбоем ответа не имеет — она не должна пропасть молча."""
        record = _record("a", 1, "белок?", status="error", error="RuntimeError: provider down")
        text = dialogues([record])

        assert "белок?" in text
        assert "RuntimeError: provider down" in text

    def test_plan_without_cost_is_not_zero(self):
        """Стоимость дня бывает None — «не посчитана», а не 0 ₽."""
        record = _record("a", 1, "план", answer="план готов",
                         plan={"kcal": 1800, "target_kcal": 1810, "cost": None, "budget": 500})
        text = dialogues([record])

        assert "1800 из 1810 ккал" in text
        assert "≈" not in text, "сумма не названа, раз её нет"

    def test_writes_reading_copy_next_to_log(self, tmp_path, capsys):
        """Копия рядом с журналом: там её найдёт очистка по сроку."""
        log = tmp_path / "chat-2026-10-22.jsonl"
        log.write_text(
            json.dumps(_record("a", 1, "Сколько белка?", answer="1.6 г/кг"), ensure_ascii=False)
            + "\n{битая строка\n",
            encoding="utf-8",
        )
        main([str(log)])

        copy = tmp_path / "chat-2026-10-22.md"
        assert "Сколько белка?" in copy.read_text(encoding="utf-8")
        assert str(copy) in capsys.readouterr().out


class TestFromEnv:
    @pytest.mark.parametrize("value", ["", "off", "OFF", "0"])
    def test_can_be_disabled(self, monkeypatch, value):
        monkeypatch.setenv("CHAT_LOG_DIR", value)
        assert not ChatLog.from_env().enabled

    def test_custom_dir_and_retention(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHAT_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("CHAT_LOG_RETENTION_DAYS", "7")
        chat_log = ChatLog.from_env()
        assert chat_log.directory == tmp_path
        assert chat_log.retention_days == 7


class TestReplay:
    def test_plan_reproduced_from_record(self, catalog, tmp_path):
        """Ради этого в записи профиль, seed и days: баг с планом
        повторяется до блюда без обращения к модели."""
        llm = FakeLLMClient(script={
            "маршрутизатор": json.dumps({"intent": "plan", "red_flag": ""}),
            "Извлеки параметры": json.dumps({**FULL_PROFILE, "exclude": ["pork"]}),
            "Объясни человеку": "объяснение",
        })
        client = make_client(catalog, [], ChatLog(tmp_path), llm=llm)
        client.post("/chat", json={"message": "Мне 31, женщина, 168, 74, худею, без свинины"})

        (record,) = read_records(tmp_path)
        trace = record["trace"]
        assert trace["plan_days"]
        assert "pork" in trace["profile"]["exclude"]

        assert replay_plan(api._state["agent"], trace) == trace["plan_days"]
