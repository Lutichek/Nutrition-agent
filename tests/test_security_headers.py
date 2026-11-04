"""
Заголовки безопасности (security_headers.py).

Наблюдалось при разборе безопасности: ни одного заголовка — ни CSP, ни
nosniff, ни запрета встраивания в чужой фрейм, а ответы API с весом и
диагнозами не запрещали кэш. Экранирование на странице было единственным
слоем, и его ошибка (кавычки в атрибуте) сразу становилась XSS.

Проверено вживую в Edge headless: с этим CSP скрипт страницы работает
(окно согласия открывается, нарушений нет), с заведомо неверным хешем —
браузер его блокирует. Тесты ниже стерегут то, что ломает это молча.
"""

from __future__ import annotations

import base64
import hashlib
import re

import pytest
from conftest import consented
from fastapi.testclient import TestClient

import api
from api import STATIC_DIR, app
from security_headers import _ScriptHashes
from storage import MemoryStorage


@pytest.fixture
def http(catalog):
    api._state["catalog"] = catalog
    api._state["storage"] = MemoryStorage()
    yield consented(TestClient(app))
    api._state.clear()


def _csp(response) -> str:
    return response.headers["content-security-policy"]


def test_page_script_is_allowed_by_its_hash(http):
    """Хеш — по байтам файла, как его получит браузер. Посчитанный по
    read_text (CRLF → LF), на рабочей копии Windows он бы не совпал, и
    интерфейс молча перестал бы работать целиком."""
    raw = (STATIC_DIR / "index.html").read_bytes()
    body = re.search(rb"<script>(.*?)</script>", raw, re.S).group(1)
    expected = base64.b64encode(hashlib.sha256(body).digest()).decode()
    assert f"'sha256-{expected}'" in _csp(http.get("/"))


def test_no_unsafe_inline_scripts(http):
    script_src = next(d for d in _csp(http.get("/")).split("; ") if d.startswith("script-src"))
    assert "'unsafe-inline'" not in script_src and "'unsafe-eval'" not in script_src
    assert "frame-ancestors 'none'" in _csp(http.get("/"))


def test_hash_follows_the_edited_page(tmp_path):
    """Хеш, посчитанный один раз при запуске, выключал бы интерфейс после
    правки страницы без перезапуска."""
    page = tmp_path / "index.html"
    page.write_bytes(b"<script>one()</script>")
    hashes = _ScriptHashes(tmp_path)
    before = hashes()
    page.write_bytes(b"<script>two()</script>")
    import os
    os.utime(page, ns=(1, 10**18))
    assert hashes() != before


def test_api_responses_are_not_cached(http):
    """В ответах API вес, возраст и диагнозы — ни браузеру, ни прокси."""
    assert http.get("/session").headers["cache-control"] == "no-store"
    assert http.get("/").headers.get("cache-control") != "no-store", "страница кэшируется"


def test_basic_headers_everywhere(http):
    for path in ("/", "/health", "/privacy.html"):
        headers = http.get(path).headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        assert headers["referrer-policy"] == "no-referrer"


def test_swagger_keeps_working(http):
    """Swagger грузит скрипты с CDN: строгий CSP его сломал бы."""
    response = http.get("/docs")
    assert "content-security-policy" not in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"
