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
from api import SESSION_COOKIE, app
from storage import MemoryStorage, SqliteStorage

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
    api._state["storage"] = MemoryStorage()

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
        assert "session_id" not in body, "пропуск — в cookie httpOnly, не в теле"
        assert len(body["sources"]) == 2

    def test_sources_carry_pmid(self, client):
        body = client.post("/chat", json={"message": "Сколько белка?"}).json()
        assert all(source["pmid"] for source in body["sources"])

    def test_answer_hides_pmid_but_keeps_sources(self, client):
        """Номер PMID человеку не говорит ничего: в тексте его быть не должно.

        Статьи при этом никуда не деваются — они приходят карточками
        в ``sources``, собранными из найденных фрагментов, а не из текста.
        """
        body = client.post("/chat", json={"message": "Сколько белка?"}).json()
        assert "PMID" not in body["answer"]
        assert body["answer"] == "Белка нужно больше."
        assert len(body["sources"]) == 2

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
        api._state["storage"] = MemoryStorage()

        response = TestClient(app).post("/chat", json={"message": "Привет"})
        assert response.status_code == 502
        api._state.clear()


class TestSessions:
    """Пропуск в разговор выдаёт сервер (ревью R2, R3).

    Раньше id присылал клиент, и неизвестный id становился id новой
    сессии — его можно было навязать (тест прямо закреплял это:
    «давно-протухший» принимался как есть). Разговоры жили в памяти
    и пропадали при рестарте.
    """

    def _token(self, http: TestClient) -> str:
        return http.cookies[SESSION_COOKIE]

    def test_cookie_is_httponly_and_lax(self, client):
        response = client.post("/chat", json={"message": "Привет"})
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=lax" in cookie

    def test_conversation_continues_by_cookie(self, client):
        client.post("/chat", json={"message": "Привет"})
        client.post("/chat", json={"message": "Ещё вопрос"})
        storage = api._state["storage"]
        assert storage.count_sessions() == 1
        assert len(storage.load_session(self._token(client))["state"]["history"]) == 4

    def test_sessions_are_isolated(self, client):
        """Профиль одного пользователя не должен попасть другому."""
        client.post("/chat", json={"message": "Привет"})
        TestClient(app).post("/chat", json={"message": "Привет"})
        assert api._state["storage"].count_sessions() == 2

    def test_imposed_token_is_not_adopted(self, client):
        """Неизвестный пропуск — новая сессия с НОВЫМ пропуском, не с этим."""
        client.cookies.set(SESSION_COOKIE, "attacker-chosen-id")
        response = client.post("/chat", json={"message": "Привет"})
        assert response.status_code == 200
        issued = response.cookies.get(SESSION_COOKIE)
        assert issued and issued != "attacker-chosen-id"
        assert api._state["storage"].load_session("attacker-chosen-id") is None

    def test_body_session_id_rejected_loudly(self, client):
        """Молча проигнорированное поле прежний клиент принял бы за рабочее."""
        response = client.post("/chat", json={"message": "Привет", "session_id": "x"})
        assert response.status_code == 400
        assert "cookie" in response.json()["detail"]

    def test_header_token_for_clients_without_cookies(self, client):
        client.post("/chat", json={"message": "Привет"})
        token = self._token(client)
        TestClient(app).post("/chat", json={"message": "Ещё"},
                             headers={api.SESSION_HEADER: token})
        assert api._state["storage"].count_sessions() == 1

    def _menu_for(self, http: TestClient, catalog) -> str:
        """Меню во владении сессии этого клиента."""
        from solver import build_menu
        from targets import Profile, compute_targets

        storage = api._state["storage"]
        owner = storage.load_session(self._token(http))["public_id"]
        menu = build_menu(catalog, compute_targets(Profile(**PROFILE)), days=1, seed=0)
        return storage.put_menu(owner, menu.model_dump_json(), PROFILE)

    def test_menu_only_for_its_owner(self, client, catalog):
        """В книге — профиль с весом. Чужому отвечаем как несуществующему."""
        client.post("/chat", json={"message": "Привет"})
        menu_id = self._menu_for(client, catalog)

        assert client.get(f"/menu/{menu_id}.xlsx").status_code == 200
        stranger = TestClient(app)
        stranger.post("/chat", json={"message": "Привет"})
        assert stranger.get(f"/menu/{menu_id}.xlsx").status_code == 404
        assert TestClient(app).get(f"/menu/{menu_id}.xlsx").status_code == 404

    def test_delete_my_data(self, client, catalog):
        client.post("/chat", json={"message": "Привет"})
        menu_id = self._menu_for(client, catalog)

        response = client.delete("/session")
        assert response.status_code == 200 and response.json()["menus_deleted"] == 1
        assert api._state["storage"].get_menu(menu_id) is None
        assert client.delete("/session").status_code == 404

    def test_no_delete_by_someone_elses_id(self, client):
        """Старая ручка /sessions/{id} удаляла чужой разговор по id."""
        assert client.delete("/sessions/любой-id").status_code in (404, 405)

    def test_conversation_survives_restart(self, client, tmp_path):
        path = tmp_path / "state.sqlite3"
        api._state["storage"] = SqliteStorage(path)
        client.post("/chat", json={"message": "Привет"})

        api._state["storage"] = SqliteStorage(path)  # «рестарт»: новый процесс, тот же том
        client.post("/chat", json={"message": "Ещё вопрос"})
        storage = api._state["storage"]
        assert storage.count_sessions() == 1
        assert len(storage.load_session(self._token(client))["state"]["history"]) == 4
