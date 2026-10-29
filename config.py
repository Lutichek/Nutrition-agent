"""
Настройки проекта: ключи API и параметры провайдеров.

Значения читаются из переменных окружения, а при их отсутствии — из файла
``.env`` рядом с этим модулем. Образец переменных — в ``.env.example``.

Оба ключа объявлены необязательными намеренно. Проекту достаточно одного
провайдера: если вы работаете только через polza.ai, требовать ключ GigaChat
незачем, и наоборот. Проверка происходит в момент, когда провайдер реально
понадобился (см. ``providers.get_clients``), а не при импорте модуля —
иначе половина проекта, которой LLM вообще не нужна (каталог блюд, расчёт
норм, солвер), не запускалась бы без ключей.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Конфигурация приложения."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── polza.ai (OpenAI-совместимый) ─────────────────────────
    # Используется для эмбеддингов всегда, для генерации — по умолчанию.
    polza_ai_api_key: str = Field(default="", validation_alias="POLZA_AI_API_KEY")

    # ── GigaChat (необязательная альтернатива для генерации) ──
    gigachat_api_key: str = Field(default="", validation_alias="GIGACHAT_API_KEY")
    gigachat_auth_url: str = "https://ngw.devices.sberbank.ru/api/v2/oauth"
    gigachat_ruid: str = Field(
        default_factory=lambda: str(uuid4()), validation_alias="GIGACHAT_RQUID"
    )
    gigachat_ssl_verify: bool = Field(default=False, validation_alias="GIGACHAT_SSL_VERIFY")

    # ── Хранилище разговоров и меню (storage.py) ──────────────
    # Путь к файлу SQLite или «memory». В контейнере — том /app/state:
    # файловая система там только на чтение.
    state_db: str = Field(default=str(PROJECT_ROOT / "state" / "agent.sqlite3"),
                          validation_alias="STATE_DB")
    # Сессия живёт столько дней с последней реплики — как журнал чата.
    session_ttl_days: float = Field(default=30, validation_alias="SESSION_TTL_DAYS")
    # Меню на скачивание — столько дней с момента сборки.
    menu_ttl_days: float = Field(default=7, validation_alias="MENU_TTL_DAYS")
    # Cookie сессии только по HTTPS. Включать за обратным прокси с TLS;
    # по голому http://localhost браузер такую cookie не вернёт.
    session_cookie_secure: bool = Field(default=False, validation_alias="SESSION_COOKIE_SECURE")

    # ── Ограничения (limits.py) ───────────────────────────────
    # Реплика в чате — 4-7 платных вызовов модели. По сессии — чтобы один
    # разговор не съел всё; по IP — потому что скрипт без cookie каждый раз
    # получает НОВУЮ сессию, и держит его только лимит по адресу.
    chat_per_minute_session: int = Field(default=6, validation_alias="CHAT_PER_MINUTE_SESSION")
    chat_per_hour_session: int = Field(default=60, validation_alias="CHAT_PER_HOUR_SESSION")
    chat_per_minute_ip: int = Field(default=20, validation_alias="CHAT_PER_MINUTE_IP")
    chat_per_hour_ip: int = Field(default=200, validation_alias="CHAT_PER_HOUR_IP")
    # Одновременных ответов по частям с одного адреса: каждый — поток процесса.
    streams_per_ip: int = Field(default=3, validation_alias="STREAMS_PER_IP")
    # Ручки без модели: /targets, /plan, /foods — дёшевы; /menu на месяц — 15 с
    # процессора, поэтому у неё свой, строже.
    calc_per_minute_ip: int = Field(default=60, validation_alias="CALC_PER_MINUTE_IP")
    menu_per_minute_ip: int = Field(default=6, validation_alias="MENU_PER_MINUTE_IP")
    # Потолок вызовов модели на ВЕСЬ сервис в сутки (UTC). ~5 вызовов на
    # реплику — около 600 реплик; у gpt-4o-mini это порядка долларов в день.
    llm_calls_per_day: int = Field(default=3000, validation_alias="LLM_CALLS_PER_DAY")


@lru_cache
def get_settings() -> Settings:
    """Настройки в единственном экземпляре."""
    return Settings()


settings = get_settings()
