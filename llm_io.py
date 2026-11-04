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


# Потолок длины ответа модели — на каждый шаг свой (OWASP LLM10).
#
# Раньше потолка не было вовсе: реплика «повтори слово 5000 раз» тратила
# деньги и держала соединение, а суточный потолок (limits.py) считает
# вызовы, не токены. Числа — замеренная длина ответа с запасом ~2× и
# больше: обрезать нормальный ответ хуже, чем недоэкономить. Замер по
# журналу чата и 210 ответам прогона метрик (токены gpt-4o-mini):
#
#     шаг              наблюдалось макс   потолок
#     generate          ~820 (p99 ~740)     1500
#     explain_plan       248                1500
#     extract_profile    105                 800   (разбор упоминаний растёт с репликой)
#     eaten              115                 800
#     classify            13                 200   (red_flag — короткая фраза)
#     grade               22                 300
#     rewrite/refine       6                 150
#
# Ответ, упёршийся в потолок, помечается в trace (truncated) — по журналу
# видно, если потолок начнёт резать живые ответы.
MAX_TOKENS: dict[str, int] = {
    "classify": 200,
    "extract_profile": 800,
    "rewrite": 150,
    "grade": 300,
    "refine": 150,
    "generate": 1500,
    "eaten": 800,
    "eaten_pick": 200,
    "ask_profile": 500,
    "explain_plan": 1500,
}
# Шаг, забывший свой потолок, всё равно ограничен.
DEFAULT_MAX_TOKENS = 1500

# Приписка к тексту, упёршемуся в потолок (у JSON-шагов её нет).
TRUNCATED_NOTE = ("…\n\n(Ответ получился слишком длинным и сокращён. "
                  "Спросите о конкретной части — отвечу подробнее.)")


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
