"""
Общее для агента и узлов графа: журнал ответов модели в текущем узле
и разбор JSON из ответа модели.

Вынесено из agent.py (ревью R6), чтобы модули узлов не импортировали
agent.py и не замыкали круг. agent.py реэкспортирует имена.
"""

from __future__ import annotations

import json
from contextvars import ContextVar

# Ответы модели внутри текущего узла графа — для журнала чата.
#
# Пишется в _complete, собирается обёрткой узла (_traced). Через контекст,
# а не через self: агент один на весь сервис, и реплики разных людей идут
# в соседних потоках. Вне графа (скрипты замеров) переменная пуста,
# и _complete ничего не копит.
_LLM_CALLS: ContextVar[list[dict] | None] = ContextVar("_LLM_CALLS", default=None)


def parse_json_response(raw: str) -> dict:
    """Разобрать JSON из ответа модели.

    Некоторые провайдеры (например, GigaChat) оборачивают JSON в markdown-блок
    даже при явной просьбе вернуть чистый JSON, поэтому обёртку снимаем сами.
    """
    text = (raw or "").strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Последняя попытка: выкусить объект между первой { и последней }.
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}
