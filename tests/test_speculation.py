"""
Вызовы модели наперёд: extract_profile и rewrite спрашивают модель, пока
classify решает маршрут.

Опережение обязано менять ТОЛЬКО время: тот же промпт — тот же ответ,
и узел получает его просто раньше. Отсюда проверки: ответы с опережением
и без совпадают, чужой промпт не подставляется, потраченный впустую
вызов не пропадает из журнала (его токены оплачены), вызовы действительно
идут одновременно — ради этого всё и делалось.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future

import pytest
from conftest import FakeLLMClient, FakeRetriever

from agent import NutritionAgent, _Speculation

PROFILE = {
    "sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74,
    "activity": "sedentary", "goal": "lose",
}

PLAN_SCRIPT = {
    "маршрутизатор": json.dumps({"intent": "plan", "red_flag": ""}),
    "Извлеки параметры": json.dumps({**PROFILE, "exclude": ["pork"]}),
    "Объясни человеку": "Норма посчитана.",
}
QUESTION_SCRIPT = {
    "маршрутизатор": json.dumps({"intent": "question", "red_flag": ""}),
    "поисковый запрос": "protein intake weight loss",
    "has_relevant": '{"has_relevant": true, "keep": [1]}',
    "ассистент по питанию. Ответь": "Белка нужно больше [PMID 111].",
}


def make_agent(catalog, script, chunks=None, speculate=True) -> NutritionAgent:
    return NutritionAgent(
        client=FakeLLMClient(script=dict(script)), model="fake",
        retriever=FakeRetriever(chunks or []), catalog=catalog, speculate=speculate,
    )


@pytest.mark.parametrize("script, message", [
    (PLAN_SCRIPT, "Мне 31, женщина, 168 см, 74 кг, свинину не ем"),
    (QUESTION_SCRIPT, "Сколько белка нужно при похудении?"),
])
def test_same_answer_with_and_without(catalog, fake_chunks, script, message):
    """Опережение меняет время, а не ответ, маршрут или профиль."""
    plain = make_agent(catalog, script, fake_chunks, speculate=False).run(message)
    ahead = make_agent(catalog, script, fake_chunks, speculate=True).run(message)

    assert ahead.answer == plain.answer
    for key in ("intent", "profile", "search_query", "plan_days"):
        assert ahead.trace[key] == plain.trace[key], key


def test_calls_run_while_classify_waits(catalog):
    """Классификатор не ответит, пока модель не спросили об извлечении
    параметров. Последовательный граф упёрся бы в таймаут."""
    asked_extract = threading.Event()
    agent = make_agent(catalog, PLAN_SCRIPT)
    create = agent.client.chat.completions.create
    seen: list[bool] = []

    def watched(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "Извлеки параметры" in prompt:
            asked_extract.set()
        if "маршрутизатор" in prompt:
            seen.append(asked_extract.wait(timeout=10))
        return create(**kwargs)

    agent.client.chat.completions.create = watched
    answer = agent.run("Мне 31, женщина, 168 см, 74 кг")

    assert seen == [True]
    assert answer.plan is not None
    extract = next(s for s in answer.trace["nodes"] if s["node"] == "extract_profile")
    assert extract["llm"][0]["speculative"] is True


def test_unused_call_stays_in_trace(catalog, fake_chunks):
    """Вопрос не нуждается в профиле, но вызов extract_profile оплачен.

    Без записи о нём журнал занижал бы расход ровно на опережение,
    и «стало быстрее» выглядело бы ещё и «дешевле».
    """
    answer = make_agent(catalog, QUESTION_SCRIPT, fake_chunks).run("Сколько белка?")

    unused = [s for s in answer.trace["nodes"] if s["node"] == "speculative_unused"]
    assert [s["for"] for s in unused] == ["extract_profile"]
    assert unused[0]["seconds"] == 0.0, "ответ человеку он не задерживал"
    assert "extract_profile" not in [s["node"] for s in answer.trace["nodes"]]


def _done(value) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


def test_other_prompt_is_not_substituted():
    """Состояние успело поменяться — ответ на другой промпт не годится."""
    speculation = _Speculation()
    speculation._calls["rewrite"] = ("промпт А", _done(("ответ на А", [], None)))

    assert speculation.take("rewrite", "промпт Б") is None
    assert speculation.take("rewrite", "промпт А") == ("ответ на А", [])


def test_failure_surfaces_in_the_node_that_needs_it():
    """Ошибка фонового вызова поднимается там, где ответ нужен, — как
    поднялась бы без опережения, — а не теряется в потоке пула."""
    speculation = _Speculation()
    error = RuntimeError("Модель не ответила после 3 попыток")
    speculation._calls["extract_profile"] = ("п", _done((None, [{"error": "x"}], error)))

    with pytest.raises(RuntimeError, match="не ответила"):
        speculation.take("extract_profile", "п")


def test_queued_call_is_cancelled_not_awaited():
    """Пул занят — вызов ещё в очереди. Ждать его дольше, чем спросить самому."""
    speculation = _Speculation()
    speculation._calls["rewrite"] = ("п", Future())  # не начат

    assert speculation.take("rewrite", "п") is None
    assert speculation.unused() == [], "отменённый вызов ничего не стоил"
