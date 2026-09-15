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


@lru_cache
def get_settings() -> Settings:
    """Настройки в единственном экземпляре."""
    return Settings()


settings = get_settings()
