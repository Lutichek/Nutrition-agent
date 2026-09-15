"""
Авторизация в GigaChat: обмен ключа на временный access-токен.

GigaChat не принимает API-ключ напрямую, как OpenAI-совместимые сервисы.
Нужен отдельный шаг: ключ обменивается на токен со сроком жизни около
получаса, и обновлять его приходится самостоятельно.

Модуль нужен, только если генерация идёт через GigaChat. При работе
через polza.ai он не импортируется вовсе.
"""

from __future__ import annotations

import time

import httpx

from config import settings

# Обновляем токен заранее, а не в момент истечения: запрос может уйти
# за секунду до конца срока и получить отказ уже на стороне сервиса.
REFRESH_MARGIN_SECONDS = 120


class GigaTokenManager:
    """Хранит access-токен и обновляет его по истечении срока."""

    def __init__(
        self,
        auth_key: str | None = None,
        url: str | None = None,
        ruid: str | None = None,
    ) -> None:
        self.auth_key = auth_key or settings.gigachat_api_key
        self.url = url or settings.gigachat_auth_url
        self.ruid = ruid or settings.gigachat_ruid

        if not self.auth_key:
            raise RuntimeError(
                "Не задан GIGACHAT_API_KEY. Укажите ключ в .env или используйте "
                "провайдера polza (LLM_PROVIDER=polza)."
            )

        # verify берётся из настроек: у сервиса сертификат от НУЦ Минцифры,
        # которого нет в стандартном хранилище — без него запрос не пройдёт.
        self.http_client = httpx.Client(
            verify=settings.gigachat_ssl_verify,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        self._access_token: str | None = None
        self._expires_at_ms: int = 0

    def _needs_refresh(self) -> bool:
        if self._access_token is None:
            return True
        now_ms = int(time.time() * 1000)
        return (self._expires_at_ms - now_ms) <= REFRESH_MARGIN_SECONDS * 1000

    def get_access_token(self) -> str:
        """Действующий токен: из кэша либо свежий."""
        if not self._needs_refresh():
            return self._access_token  # type: ignore[return-value]

        response = self.http_client.post(
            self.url,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "RqUID": self.ruid,
                "Authorization": f"Basic {self.auth_key}",
            },
            data={"scope": "GIGACHAT_API_PERS"},
        )
        response.raise_for_status()

        payload = response.json()
        self._access_token = payload["access_token"]
        self._expires_at_ms = int(payload["expires_at"])
        return self._access_token
