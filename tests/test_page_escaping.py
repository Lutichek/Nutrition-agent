"""
Экранирование на странице (OWASP LLM05: небезопасная обработка вывода).

Наблюдалось при разборе безопасности: escape в static/index.html не трогал
кавычки, а стоит и внутри атрибута — title="${escape(s.title)}". Название
статьи с кавычкой закрывало атрибут, дальше шёл чужой обработчик событий.
Номер PMID шёл в ссылку и в текст вовсе без экранирования.

Тест выполняет НАСТОЯЩИЕ функции страницы в Node.js, а не ищет строку
в файле: правка, которая сохранит «&quot;» в тексте и сломает результат,
тоже упадёт.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "static" / "index.html"
NODE = shutil.which("node")

# Источник, каким его мог бы подсунуть отравленный корпус.
EVIL = {
    "title": 'Protein" onmouseover="alert(1)',
    "journal": "<img src=x onerror=alert(2)>",
    "year": 2019,
    "pmid": '1"><script>alert(3)</script>',
    "quote": "<b onclick=alert(4)>цитата</b>",
}

pytestmark = pytest.mark.skipif(NODE is None, reason="нужен Node.js")


def _function(source: str, name: str) -> str:
    """Исходник функции страницы: от объявления до скобки в начале строки."""
    match = re.search(rf"^function {name}\(.*?^}}$", source, re.S | re.M)
    assert match, f"на странице нет функции {name}"
    return match.group(0)


def _render(call: str) -> str:
    page = PAGE.read_text(encoding="utf-8")
    escape = re.search(r"^const escape = .*?;$", page, re.S | re.M).group(0)
    code = "\n".join([escape, _function(page, "renderSources"),
                      _function(page, "renderPlanFoot"),
                      f"process.stdout.write({call});"])
    done = subprocess.run([NODE, "-e", code], capture_output=True, text=True,
                          encoding="utf-8", timeout=30, check=True)
    return done.stdout


def _assert_inert(html: str) -> None:
    """Данные не стали ни тегом, ни атрибутом."""
    assert not re.search(r"<(script|img|b\s)", html, re.I), html
    # Внутри каждого тега убираем значения атрибутов в кавычках: обработчик,
    # оставшийся после этого, — настоящий атрибут, открытый кавычкой из
    # данных. Текст между тегами не смотрим: экранированный, он безвреден.
    for tag in re.findall(r"<[^>]+>", html):
        outside = re.sub(r'"[^"]*"', '""', tag)
        assert not re.search(r"\son\w+\s*=", outside), tag


def test_sources_card_is_inert():
    _assert_inert(_render(f"renderSources([{json.dumps(EVIL)}])"))


def test_plan_foot_title_attribute_is_not_broken():
    _assert_inert(_render(f"renderPlanFoot([{json.dumps(EVIL)}])"))
