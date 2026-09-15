"""
Настройка провайдеров: откуда берём LLM и откуда — эмбеддинги.

В проекте это два независимых клиента, и это сделано намеренно:

* **эмбеддинги** нужны и при заливке корпуса, и на каждый поисковый запрос;
* **LLM** нужен для перефразирования, классификации, оценки найденного
  и генерации ответа.

Их можно брать у разных провайдеров — например, эмбеддинги у polza.ai,
а генерацию у GigaChat. Так проект продолжает работать, даже если у одного
из провайдеров закончился лимит.

Использование::

    from providers import get_clients

    llm_client, model, embed_client = get_clients("polza")
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from config import settings

POLZA_BASE_URL = "https://api.polza.ai/api/v1"
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"

# Модель эмбеддингов. Её размерность зашита в индекс LanceDB — менять
# без пересборки индекса нельзя.
EMBEDDING_MODEL = "text-embedding-3-small"

DEFAULT_POLZA_MODEL = "openai/gpt-4o-mini"
DEFAULT_GIGACHAT_MODEL = "GigaChat-2"


def get_polza_client() -> OpenAI:
    """Клиент polza.ai (OpenAI-совместимый)."""
    if not settings.polza_ai_api_key:
        raise RuntimeError(
            "Не задан POLZA_AI_API_KEY. Скопируйте .env.example в .env "
            "и укажите ключ."
        )
    return OpenAI(base_url=POLZA_BASE_URL, api_key=settings.polza_ai_api_key)


def get_gigachat_client() -> OpenAI:
    """Клиент GigaChat (через OpenAI-совместимый интерфейс)."""
    from gigachat_auth import GigaTokenManager

    token_manager = GigaTokenManager()
    return OpenAI(
        api_key=token_manager.get_access_token(),
        base_url=GIGACHAT_BASE_URL,
        http_client=token_manager.http_client,
    )


def get_clients(llm_provider: str = "polza") -> tuple[Any, str, Any]:
    """Вернуть (llm_client, llm_model, embed_client).

    Args:
        llm_provider: "polza" или "gigachat" — кто отвечает за генерацию.

    Эмбеддинги всегда берём у polza.ai: именно этой моделью построен индекс,
    и подменить её нельзя без пересборки базы.
    """
    embed_client = get_polza_client()

    if llm_provider == "gigachat":
        return get_gigachat_client(), DEFAULT_GIGACHAT_MODEL, embed_client

    if llm_provider == "polza":
        return get_polza_client(), DEFAULT_POLZA_MODEL, embed_client

    raise ValueError(f"Неизвестный провайдер: {llm_provider}. Доступны: polza, gigachat")


# ────────────────────────────────────────────────────────────
# Обёртка для строгого JSON
# ────────────────────────────────────────────────────────────


class _DefencedMessage:
    """Сообщение с очищенным от markdown-обёртки содержимым."""

    def __init__(self, content: str) -> None:
        self.content = content


class _DefencedChoice:
    def __init__(self, message: Any) -> None:
        self.message = message


class _DefencedResponse:
    def __init__(self, choices: list[Any]) -> None:
        self.choices = choices


class _CompletionsProxy:
    def __init__(self, real_completions: Any) -> None:
        self._real = real_completions

    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._real.create(*args, **kwargs)
        content = (response.choices[0].message.content or "").strip()

        # Снимаем markdown-обёртку ```json ... ```
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        return _DefencedResponse([_DefencedChoice(_DefencedMessage(content))])


class _ChatProxy:
    def __init__(self, real_chat: Any) -> None:
        self.completions = _CompletionsProxy(real_chat.completions)


class JsonSafeClient:
    """Обёртка над LLM-клиентом, снимающая markdown-обёртку с JSON-ответов.

    Зачем: GigaChat даже при ``response_format="json_object"`` возвращает JSON
    внутри блока ```json ... ```, из-за чего ``json.loads`` в вызывающем коде
    падает. Обёртка чинит это, не требуя правок самого кода оценки.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = _ChatProxy(client.chat)
        self.base_url = getattr(client, "base_url", None)
        self.api_key = getattr(client, "api_key", None)

    def __getattr__(self, name: str) -> Any:
        # всё остальное (например, embeddings) отдаём исходному клиенту
        return getattr(self._client, name)
