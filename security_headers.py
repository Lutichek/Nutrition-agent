"""
Заголовки безопасности для всех ответов (OWASP ASVS 14.4, LLM05).

Зачем. Ответы модели и данные корпуса попадают на страницу через
innerHTML. Экранирование там есть, но это один слой: ошибка в нём (как
было с кавычками в атрибуте) сразу становилась XSS. CSP — второй слой:
даже если разметка просочится, браузер не выполнит ни чужой скрипт,
ни обработчик событий, и не отправит данные на чужой адрес.

Скрипт страницы встроенный, поэтому он разрешён по ХЕШУ, а не
'unsafe-inline': подменённый или дописанный скрипт хеша не совпадёт.
Две тихие ловушки:

* хеш — по БАЙТАМ файла, как его получит браузер. read_text превращает
  CRLF в LF, и на рабочей копии Windows хеш не совпал бы — страница
  молча перестала бы работать целиком;
* хеш пересчитывается, когда файл меняется: посчитанный один раз при
  запуске выключал бы интерфейс после правки index.html без перезапуска.

Стили встроенные и в атрибутах style= (полосы КБЖУ), для них
'unsafe-inline' оставлен: через стиль не выполнить код, а хеши атрибутов
требуют 'unsafe-hashes' и переписывания разметки.

/docs и /redoc (Swagger) грузят скрипты с CDN — CSP им не ставится,
остальные заголовки — ставятся.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

_INLINE_SCRIPT = re.compile(rb"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)
_NO_CSP_PATHS = ("/docs", "/redoc")
_STATIC_SUFFIXES = (".html", ".css", ".js", ".svg", ".ico", ".png")


class _ScriptHashes:
    """'sha256-…' встроенных скриптов страниц, с пересчётом по mtime."""

    def __init__(self, static_dir: Path) -> None:
        self.static_dir = static_dir
        self._key: tuple = ()
        self._value = ""

    def __call__(self) -> str:
        pages = sorted(self.static_dir.glob("*.html")) if self.static_dir.is_dir() else []
        key = tuple((p.name, p.stat().st_mtime_ns) for p in pages)
        if key != self._key:
            hashes = sorted({
                "'sha256-" + base64.b64encode(hashlib.sha256(body).digest()).decode() + "'"
                for page in pages for body in _INLINE_SCRIPT.findall(page.read_bytes())})
            self._key, self._value = key, " ".join(hashes)
        return self._value


def content_security_policy(script_hashes: str) -> str:
    return "; ".join([
        "default-src 'none'",
        f"script-src 'self' {script_hashes}".rstrip(),
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "font-src 'self'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "object-src 'none'",
    ])


class SecurityHeaders:
    """ASGI-обёртка: дописывает заголовки в начало каждого ответа.

    Чистый ASGI, а не BaseHTTPMiddleware: ответ по частям (/chat/stream)
    идёт насквозь, без буферизации.
    """

    def __init__(self, app: Any, static_dir: Path, https: bool = False) -> None:
        self.app = app
        self.hashes = _ScriptHashes(static_dir)
        self.https = https

    async def __call__(self, scope: dict, receive: Callable[[], Awaitable[dict]],
                       send: Callable[[dict], Awaitable[None]]) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")

        async def with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                for name, value in self._headers(path):
                    if name not in present:
                        headers.append((name, value))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, with_headers)

    def _headers(self, path: str) -> list[tuple[bytes, bytes]]:
        headers = {
            "x-content-type-options": "nosniff",
            "referrer-policy": "no-referrer",
            "x-frame-options": "DENY",
            "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=()",
            "cross-origin-opener-policy": "same-origin",
        }
        if not path.startswith(_NO_CSP_PATHS):
            headers["content-security-policy"] = content_security_policy(self.hashes())
        # Ответы API несут вес, возраст и диагнозы: ни браузер, ни прокси
        # не должны класть их в кэш. Статика страницы кэшируется как раньше.
        if not (path == "/" or path.endswith(_STATIC_SUFFIXES)):
            headers["cache-control"] = "no-store"
        if self.https:
            headers["strict-transport-security"] = "max-age=31536000; includeSubDomains"
        return [(k.encode(), v.encode()) for k, v in headers.items()]
