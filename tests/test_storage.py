"""
Хранилище разговоров, меню и суточного счётчика (storage.py).

Ревью: сессии и меню жили в словарях процесса — рестарт стирал все
разговоры, а при 1000 сессиях старые вытеснялись молча. Отсюда проверки:
SQLite переживает повторное открытие, токен не лежит на диске открытым
текстом, просроченное удаляется, а недоступный файл роняет запуск,
а не выключает хранение молча.
"""

from __future__ import annotations

import pytest

from storage import MemoryStorage, SqliteStorage, open_storage


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return MemoryStorage()
    return SqliteStorage(tmp_path / "state.sqlite3")


class TestSessions:
    def test_roundtrip(self, store):
        token = store.create_session({"profile": {"age": 31}})
        store.save_session(token, {"profile": {"age": 32}, "history": ["x"]})
        assert store.load_session(token) == {"profile": {"age": 32}, "history": ["x"]}

    def test_tokens_unguessable_and_distinct(self, store):
        tokens = {store.create_session({}) for _ in range(50)}
        assert len(tokens) == 50
        assert all(len(t) >= 40 for t in tokens), "256 бит в base64 — 43 символа"

    def test_unknown_token_is_none(self, store):
        assert store.load_session("чужой-или-выдуманный") is None

    def test_delete(self, store):
        token = store.create_session({"a": 1})
        assert store.delete_session(token) == {"a": 1}
        assert store.load_session(token) is None
        assert store.delete_session(token) is None

    def test_expired_session_is_gone(self, store):
        token = store.create_session({})
        store.session_ttl = -1
        assert store.load_session(token) is None

    def test_loaded_copy_is_not_live(self, store):
        """Правка загруженного без save не должна просачиваться в хранилище —
        в SQLite так по природе, и в памяти обязано быть так же, иначе тесты
        на памяти прятали бы забытый save."""
        token = store.create_session({"profile": {}})
        store.load_session(token)["profile"]["age"] = 99
        assert store.load_session(token) == {"profile": {}}


class TestMenus:
    def test_roundtrip_and_owner(self, store):
        menu_id = store.put_menu("владелец", '{"days": []}', {"age": 31})
        assert store.get_menu(menu_id) == ("владелец", '{"days": []}', {"age": 31})

    def test_delete_by_owner(self, store):
        mine = store.put_menu("я", "{}", {})
        other = store.put_menu("другой", "{}", {})
        assert store.delete_menus("я") == 1
        assert store.get_menu(mine) is None
        assert store.get_menu(other) is not None

    def test_expired_menu_is_gone(self, store):
        menu_id = store.put_menu("я", "{}", {})
        store.menu_ttl = -1
        assert store.get_menu(menu_id) is None


def test_usage_accumulates(store):
    assert store.add_usage("2026-10-29", 3) == 3
    assert store.add_usage("2026-10-29", 2) == 5
    assert store.get_usage("2026-10-30") == 0


class TestSqliteSurvivesRestart:
    def test_everything_survives_reopen(self, tmp_path):
        path = tmp_path / "state.sqlite3"
        first = SqliteStorage(path)
        token = first.create_session({"profile": {"weight_kg": 74}})
        menu_id = first.put_menu("я", "{}", {})
        first.add_usage("2026-10-29", 7)

        again = SqliteStorage(path)
        assert again.load_session(token) == {"profile": {"weight_kg": 74}}
        assert again.get_menu(menu_id) is not None
        assert again.get_usage("2026-10-29") == 7, "рестарт не обнуляет суточный потолок"

    def test_token_not_on_disk(self, tmp_path):
        path = tmp_path / "state.sqlite3"
        store = SqliteStorage(path)
        token = store.create_session({})
        store._db.execute("PRAGMA wal_checkpoint(FULL)")
        raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.is_file())
        assert token.encode() not in raw

    def test_purge_removes_expired_on_open(self, tmp_path):
        path = tmp_path / "state.sqlite3"
        SqliteStorage(path).create_session({})
        store = SqliteStorage(path, session_ttl_days=-1)
        assert store.count_sessions() == 0

    def test_unwritable_path_fails_loudly(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("не папка")
        with pytest.raises(RuntimeError, match="Хранилище недоступно"):
            SqliteStorage(blocker / "state.sqlite3")


def test_open_storage_choices(tmp_path):
    assert isinstance(open_storage("memory"), MemoryStorage)
    assert isinstance(open_storage(str(tmp_path / "s.sqlite3")), SqliteStorage)


def test_memory_store_stays_bounded():
    store = MemoryStorage(max_sessions=3)
    tokens = [store.create_session({}) for _ in range(5)]
    assert store.count_sessions() == 3
    assert store.load_session(tokens[0]) is None
    assert store.load_session(tokens[-1]) is not None
