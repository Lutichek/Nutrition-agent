"""
Тесты HTTP-слоя.

Сервис поднимается без прогрева: тяжёлые ресурсы (каталог, индекс, LLM)
подменяются заглушками прямо в состоянии приложения. Поэтому тесты
не ходят в сеть и идут за секунды.

Приём с TestClient: lifespan запускается только при входе в контекстный
менеджер (``with TestClient(app)``). Мы его не используем — значит,
настоящий агент не собирается, а работают наши подмены.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLMClient, FakeRetriever
from fastapi.testclient import TestClient

import api
from agent import NutritionAgent
from api import SessionStore, app

PROFILE = {
    "sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74,
    "activity": "sedentary", "goal": "lose",
}


@pytest.fixture
def client(catalog, fake_chunks):
    """Клиент с подменёнными зависимостями приложения."""
    agent = NutritionAgent(
        client=FakeLLMClient(
            script={
                "маршрутизатор": json.dumps({"intent": "question", "red_flag": ""}),
                "has_relevant": '{"has_relevant": true, "keep": [1, 2]}',
                "ассистент по питанию. Ответь": "Белка нужно больше [PMID 111].",
            }
        ),
        model="fake",
        retriever=FakeRetriever(fake_chunks),
        catalog=catalog,
    )

    api._state["agent"] = agent
    api._state["catalog"] = catalog
    api._state["sessions"] = SessionStore()
    api._state["menus"] = api.MenuStore()

    yield TestClient(app)

    api._state.clear()


class TestHealth:
    def test_reports_loaded_data(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["foods_in_catalog"] > 4000

    def test_503_before_warmup(self, catalog):
        """Пока ресурсы не загружены, сервис обязан честно сказать 503."""
        api._state.clear()
        response = TestClient(app, raise_server_exceptions=False).get("/health")
        assert response.status_code == 503


class TestTargetsEndpoint:
    def test_returns_computed_norm(self, client):
        body = client.post("/targets", json=PROFILE).json()
        assert body["targets"]["kcal"] > 1000
        assert body["explanation"]

    def test_is_deterministic(self, client):
        """Ручка без LLM: два одинаковых запроса дают идентичный ответ."""
        first = client.post("/targets", json=PROFILE).json()
        second = client.post("/targets", json=PROFILE).json()
        assert first == second

    def test_safety_adjustments_are_exposed(self, client):
        """Сработавший предохранитель должен быть виден клиенту, а не съеден."""
        aggressive = {**PROFILE, "rate_kg_per_week": 2.0}
        body = client.post("/targets", json=aggressive).json()
        assert body["targets"]["adjustments"]

    @pytest.mark.parametrize(
        "broken",
        [{"age": 250}, {"height_cm": 10}, {"sex": "нечто"}, {"weight_kg": -5}],
        ids=["возраст", "рост", "пол", "вес"],
    )
    def test_invalid_profile_is_422(self, client, broken):
        assert client.post("/targets", json={**PROFILE, **broken}).status_code == 422


class TestPlanEndpoint:
    def test_returns_plan_that_passes_checks(self, client):
        body = client.post("/plan", json={"profile": PROFILE, "seed": 7}).json()
        assert body["checks_passed"] is True
        assert len(body["plan"]["items"]) == 7

    def test_same_seed_same_plan(self, client):
        first = client.post("/plan", json={"profile": PROFILE, "seed": 3}).json()
        second = client.post("/plan", json={"profile": PROFILE, "seed": 3}).json()
        assert first["plan"] == second["plan"]

    def test_respects_exclusions(self, client):
        payload = {"profile": {**PROFILE, "exclude": ["pork", "bacon"]}, "seed": 5}
        body = client.post("/plan", json=payload).json()
        names = " ".join(item["name"].lower() for item in body["plan"]["items"])
        assert "pork" not in names and "bacon" not in names

    def test_unbuildable_plan_is_422_not_500(self, client, catalog):
        """Невозможный расчёт — это не поломка сервиса.

        Ограничениями каталог больше не задушить: после перехода на поиск
        по границам слова даже «веган без глютена и орехов» оставляет
        достаточно блюд. Поэтому недостачу изображаем самим каталогом.
        """
        full = api._state["catalog"]
        api._state["catalog"] = catalog[catalog["role"] == "main"]
        try:
            response = client.post("/plan", json={"profile": PROFILE, "seed": 0})
        finally:
            api._state["catalog"] = full

        assert response.status_code == 422
        assert "исключений" in response.json()["detail"]


class TestFoodsEndpoint:
    def test_search_by_name(self, client):
        found = client.get("/foods", params={"q": "oatmeal"}).json()
        assert found and all("oatmeal" in item["name"].lower() for item in found)

    def test_filter_by_role(self, client):
        found = client.get("/foods", params={"q": "chicken", "role": "main"}).json()
        assert all(item["role"] == "main" for item in found)

    def test_limit_is_respected(self, client):
        assert len(client.get("/foods", params={"q": "ch", "limit": 3}).json()) == 3

    def test_too_short_query_rejected(self, client):
        assert client.get("/foods", params={"q": "a"}).status_code == 422


class TestChatEndpoint:
    def test_returns_answer_and_sources(self, client):
        body = client.post("/chat", json={"message": "Сколько белка?"}).json()
        assert body["answer"]
        assert body["session_id"]
        assert len(body["sources"]) == 2

    def test_sources_carry_pmid(self, client):
        body = client.post("/chat", json={"message": "Сколько белка?"}).json()
        assert all(source["pmid"] for source in body["sources"])

    def test_session_is_reused(self, client):
        first = client.post("/chat", json={"message": "Привет"}).json()
        second = client.post(
            "/chat", json={"message": "Ещё вопрос", "session_id": first["session_id"]}
        ).json()
        assert second["session_id"] == first["session_id"]

    def test_sessions_are_isolated(self, client):
        """Профиль одного пользователя не должен попасть другому."""
        first = client.post("/chat", json={"message": "Привет"}).json()
        second = client.post("/chat", json={"message": "Привет"}).json()
        assert first["session_id"] != second["session_id"]

    def test_unknown_session_does_not_crash(self, client):
        """После перезапуска сервиса id у клиентов протухают — не 500."""
        body = client.post(
            "/chat", json={"message": "Привет", "session_id": "давно-протухший"}
        ).json()
        assert body["session_id"] == "давно-протухший"

    def test_empty_message_rejected(self, client):
        assert client.post("/chat", json={"message": ""}).status_code == 422

    def test_provider_outage_is_502(self, catalog):
        """Внешний сбой — 502, а не 500: сервис-то исправен."""
        api._state["agent"] = NutritionAgent(
            client=FakeLLMClient(fail_times=99),
            model="fake",
            retriever=FakeRetriever(),
            catalog=catalog,
            max_retries=1,
        )
        api._state["catalog"] = catalog
        api._state["sessions"] = SessionStore()
        api._state["menus"] = api.MenuStore()

        response = TestClient(app).post("/chat", json={"message": "Привет"})
        assert response.status_code == 502
        api._state.clear()


class TestSessionStore:
    def test_delete_removes_session(self, client):
        session_id = client.post("/chat", json={"message": "Привет"}).json()["session_id"]
        assert client.delete(f"/sessions/{session_id}").status_code == 200
        assert client.delete(f"/sessions/{session_id}").status_code == 404

    def test_eviction_keeps_store_bounded(self):
        """Иначе словарь сессий растёт до бесконечности."""
        store = SessionStore(max_sessions=3)
        ids = [store.get_or_create(None)[0] for _ in range(5)]
        assert store.count() == 3
        assert store.drop(ids[0]) is False   # самый старый вытеснен
        assert store.drop(ids[-1]) is True   # самый свежий на месте
