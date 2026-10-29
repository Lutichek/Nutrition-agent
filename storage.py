"""
Хранилище сервиса: разговоры, собранные меню, суточный счётчик вызовов модели.

Зачем. Раньше всё жило в словарях процесса: при рестарте пропадали все
разговоры, при 1000 сессиях самые старые вытеснялись молча. Теперь —
SQLite в отдельном томе (контейнер работает с файловой системой только
на чтение, пишет он только в тома). Реализация в памяти осталась для
тестов: у обеих один набор методов, и ``api.py`` не знает, какая под ним.

Три правила, каждое — про тихий отказ:

* **Токен сессии на диске не лежит**, только его SHA-256. Токен — это
  пропуск в разговор с весом и диагнозами; утечка файла базы не должна
  давать войти в чужие разговоры.
* **Хранилище не знает, что в сессии.** Оно хранит JSON-словарь; что в нём
  профиль и история — знает ``api.py``. Иначе модуль хранения тянул бы
  за собой граф агента, а смена формата профиля — миграцию здесь.
* **Сломанное хранилище роняет запуск**, а не выключается молча (в отличие
  от журнала, ``chat_log.py``): без него сервис терял бы разговоры при
  каждом рестарте и выглядел бы при этом исправным.

Сроки жизни: сессия — ``SESSION_TTL_DAYS`` с последней реплики, меню —
``MENU_TTL_DAYS`` с момента сборки. Просроченное удаляется при запуске
и затем не чаще раза в час, по ходу записи.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

DAY = 24 * 3600
PURGE_EVERY = 3600


def new_token() -> str:
    """Пропуск в разговор: 256 случайных бит — не угадать и не навязать."""
    return secrets.token_urlsafe(32)


def _key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class MemoryStorage:
    """То же самое в памяти процесса — для тестов и запуска без диска.

    ``max_sessions`` — потолок, чтобы словарь не рос бесконечно: самая
    давно не использованная сессия вытесняется первой.
    """

    def __init__(self, session_ttl_days: float = 30, menu_ttl_days: float = 7,
                 max_sessions: int = 1000, max_menus: int = 200) -> None:
        self._sessions: dict[str, tuple[dict, float]] = {}
        self._menus: dict[str, tuple[str, str, dict, float]] = {}
        self._usage: dict[str, int] = {}
        self._lock = Lock()
        self.session_ttl = session_ttl_days * DAY
        self.menu_ttl = menu_ttl_days * DAY
        self._max_sessions = max_sessions
        self._max_menus = max_menus

    # ── разговоры ──
    def create_session(self, data: dict) -> str:
        token = new_token()
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                del self._sessions[next(iter(self._sessions))]
            self._sessions[_key(token)] = (data, time.time())
        return token

    def load_session(self, token: str) -> dict | None:
        with self._lock:
            found = self._sessions.get(_key(token))
            if found is None or time.time() - found[1] > self.session_ttl:
                return None
            return json.loads(json.dumps(found[0]))  # копия, как из базы

    def save_session(self, token: str, data: dict) -> None:
        with self._lock:
            key = _key(token)
            self._sessions.pop(key, None)  # в конец: свежая вытесняется последней
            self._sessions[key] = (data, time.time())

    def delete_session(self, token: str) -> dict | None:
        with self._lock:
            found = self._sessions.pop(_key(token), None)
            return found[0] if found else None

    def count_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ── меню ──
    def put_menu(self, owner: str, menu_json: str, profile: dict) -> str:
        menu_id = uuid.uuid4().hex
        with self._lock:
            if len(self._menus) >= self._max_menus:
                del self._menus[next(iter(self._menus))]
            self._menus[menu_id] = (owner, menu_json, profile, time.time())
        return menu_id

    def get_menu(self, menu_id: str) -> tuple[str, str, dict] | None:
        """(владелец, меню JSON, профиль) или None."""
        with self._lock:
            found = self._menus.get(menu_id)
            if found is None or time.time() - found[3] > self.menu_ttl:
                return None
            return found[0], found[1], found[2]

    def delete_menus(self, owner: str) -> int:
        with self._lock:
            mine = [k for k, v in self._menus.items() if v[0] == owner]
            for k in mine:
                del self._menus[k]
            return len(mine)

    # ── траты ──
    def add_usage(self, day: str, calls: int) -> int:
        with self._lock:
            self._usage[day] = self._usage.get(day, 0) + calls
            return self._usage[day]

    def get_usage(self, day: str) -> int:
        with self._lock:
            return self._usage.get(day, 0)


class SqliteStorage:
    """Постоянное хранилище в одном файле SQLite.

    Одно соединение на процесс под блокировкой: ручки FastAPI идут в пуле
    потоков, а записей — единицы в секунду. WAL — чтобы чтение не ждало записи.
    """

    def __init__(self, path: str | Path, session_ttl_days: float = 30,
                 menu_ttl_days: float = 7) -> None:
        self.path = Path(path)
        self.session_ttl = session_ttl_days * DAY
        self.menu_ttl = menu_ttl_days * DAY
        self._lock = Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    key TEXT PRIMARY KEY, data TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS menus (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, menu TEXT NOT NULL,
                    profile TEXT NOT NULL, created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS menus_owner ON menus(owner);
                CREATE TABLE IF NOT EXISTS usage (
                    day TEXT PRIMARY KEY, calls INTEGER NOT NULL);
            """)
        except (OSError, sqlite3.Error) as error:
            raise RuntimeError(
                f"Хранилище недоступно: {self.path} ({error}). В контейнере это "
                "том /app/state (docker-compose.yml); локально — переменная STATE_DB."
            ) from error
        self._purged_at = 0.0
        self.purge()

    def _write(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._db.execute(sql, args)
            self._db.commit()
        if time.time() - self._purged_at > PURGE_EVERY:
            self.purge()
        return cursor

    def _read(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def purge(self) -> None:
        """Удалить просроченные разговоры и меню."""
        now = time.time()
        with self._lock:
            self._db.execute("DELETE FROM sessions WHERE updated < ?", (now - self.session_ttl,))
            self._db.execute("DELETE FROM menus WHERE created < ?", (now - self.menu_ttl,))
            self._db.commit()
        self._purged_at = now

    # ── разговоры ──
    def create_session(self, data: dict) -> str:
        token = new_token()
        now = time.time()
        self._write("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                    (_key(token), json.dumps(data, ensure_ascii=False), now, now))
        return token

    def load_session(self, token: str) -> dict | None:
        rows = self._read("SELECT data, updated FROM sessions WHERE key = ?", (_key(token),))
        if not rows or time.time() - rows[0][1] > self.session_ttl:
            return None
        return json.loads(rows[0][0])

    def save_session(self, token: str, data: dict) -> None:
        now = time.time()
        self._write(
            "INSERT INTO sessions VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET data = excluded.data, updated = excluded.updated",
            (_key(token), json.dumps(data, ensure_ascii=False), now, now))

    def delete_session(self, token: str) -> dict | None:
        found = self.load_session(token)
        self._write("DELETE FROM sessions WHERE key = ?", (_key(token),))
        return found

    def count_sessions(self) -> int:
        return self._read("SELECT COUNT(*) FROM sessions")[0][0]

    # ── меню ──
    def put_menu(self, owner: str, menu_json: str, profile: dict) -> str:
        menu_id = uuid.uuid4().hex
        self._write("INSERT INTO menus VALUES (?, ?, ?, ?, ?)",
                    (menu_id, owner, menu_json, json.dumps(profile, ensure_ascii=False),
                     time.time()))
        return menu_id

    def get_menu(self, menu_id: str) -> tuple[str, str, dict] | None:
        rows = self._read("SELECT owner, menu, profile, created FROM menus WHERE id = ?",
                          (menu_id,))
        if not rows or time.time() - rows[0][3] > self.menu_ttl:
            return None
        owner, menu, profile, _ = rows[0]
        return owner, menu, json.loads(profile)

    def delete_menus(self, owner: str) -> int:
        return self._write("DELETE FROM menus WHERE owner = ?", (owner,)).rowcount

    # ── траты ──
    def add_usage(self, day: str, calls: int) -> int:
        self._write("INSERT INTO usage VALUES (?, ?) "
                    "ON CONFLICT(day) DO UPDATE SET calls = calls + excluded.calls",
                    (day, calls))
        return self.get_usage(day)

    def get_usage(self, day: str) -> int:
        rows = self._read("SELECT calls FROM usage WHERE day = ?", (day,))
        return rows[0][0] if rows else 0


def open_storage(path: str | None, **ttl: Any) -> MemoryStorage | SqliteStorage:
    """Хранилище по настройке STATE_DB: путь к файлу или «memory»."""
    if not path or path.strip().lower() == "memory":
        return MemoryStorage(**ttl)
    return SqliteStorage(path, **ttl)
