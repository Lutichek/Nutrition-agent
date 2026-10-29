"""
Типы агента: состояние графа, память разговора, ответ.

Вынесены из agent.py (ревью R6), чтобы модули узлов графа могли опираться
на них без циклического импорта. agent.py реэкспортирует имена.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Один найденный фрагмент из векторного хранилища."""

    text: str
    chunk_id: int
    year: int  # здесь лежит doc_id — идентификатор статьи-источника
    distance: float


class RAGAgentAnswer(BaseModel):
    """Контракт ответа агента для системы оценки."""

    dataset_row_id: str | None = None
    answer: str
    retrieved_chunks: list[Chunk] | None = None

    # Собранное меню, если агент его строил. В переписку оно не печатается
    # (на месяц это несколько тысяч строк) — API отдаёт его файлом.
    menu: Any = None
    menu_days: int = 0

    # Первый день плана и нормы — СТРУКТУРОЙ, а не текстом.
    #
    # Интерфейсу нужны сами числа: нарисовать полосы «факт к норме» из абзаца
    # невозможно, а именно они и есть суть продукта. Текстовое объяснение
    # рядом остаётся — его пишет модель, и оно отвечает на «почему столько».
    plan: Any = None
    targets: Any = None
    plan_targets: Any = None
    cycled: Any = None

    # Что агент понял по дороге к ответу — для журнала чата (chat_log.py).
    # Из model_dump исключено: сохранённые ответы для метрик остаются
    # в прежнем формате и не разрастаются служебным.
    trace: dict | None = Field(default=None, exclude=True)


class ConversationState(BaseModel):
    """Память одного разговора: накопленный профиль и история реплик.

    Вынесена из агента наружу намеренно. Сам агент держит дорогие ресурсы —
    каталог блюд и индекс LanceDB — и в сервисе он один
    на всех. Если хранить профиль внутри него, параметры одного пользователя
    утекут в расчёт другому.

    Поэтому агент без состояния, а состояние живёт снаружи: в ``demo.py`` —
    в самом объекте для удобства, в ``api.py`` — в хранилище сессий.
    """

    profile: dict = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)

    # На сколько дней просили меню. Запоминается наравне с профилем:
    # человек говорит «составь на месяц» в первой реплике, а данные
    # дописывает во второй — и период не должен теряться по дороге.
    days: int = 1

    # Последний показанный план и что в нём уже заменяли: на «замени
    # творог» меняется ОДНО блюдо, а не собирается новый день (ревью R8).
    last_plan: dict | None = None
    swap_rejected: list[int] = Field(default_factory=list)
    swap_index: int | None = None  # какой слот меняли последним — для «ещё раз»

    # Съеденное не по плану за день (этап 2): копится между репликами дня
    # и сбрасывается новым планом или новыми сутками.
    eaten: list[dict] = Field(default_factory=list)
    eaten_date: str | None = None
    # Съеденное, которое ждёт ответа «на какой приём пищи?»: {items, date}.
    # Живёт одну реплику — следующая либо отвечает, либо о другом.
    eaten_pending: dict | None = None

    # Для адаптивной нормы (adaptive.py, ревью R9): взвешивания и съеденное
    # по дням (дата ISO → кг / ккал) и последняя оценка расхода {tdee, at}.
    weighins: dict[str, float] = Field(default_factory=dict)
    intake: dict[str, float] = Field(default_factory=dict)
    adaptive: dict | None = None


Intent = Literal["plan", "question", "off_topic"]


# Что показать человеку, пока идёт шаг (событие «stage» в run(on_event=...)).
# Раньше страница угадывала этап по таймеру: «ищу в статьях» на 4-й секунде
# показывалось и тогда, когда агент собирал рацион и в статьи не ходил.
STAGE_LABELS = {
    "classify": "разбираю вопрос",
    "extract_profile": "уточняю параметры",
    "ask_profile": "формулирую вопрос",
    "make_plan": "подбираю блюда",
    "explain_plan": "объясняю расчёт",
    "rewrite": "формулирую запрос",
    "retrieve": "ищу в статьях",
    "grade": "отбираю фрагменты",
    "refine": "ищу другими словами",
    "generate": "пишу ответ",
    "match_eaten": "ищу съеденное в справочнике",
}


class AgentState(TypedDict, total=False):
    """Данные, которые узлы графа передают друг другу."""

    question: str          # реплика пользователя
    history: list[dict]    # предыдущие реплики: [{"role": ..., "content": ...}]

    intent: Intent
    red_flag: str          # непустая строка — нужен врач, рацион не выдаём

    profile: dict          # накопленный профиль (может быть неполным)
    missing: list[str]     # каких полей не хватает для расчёта

    targets: Any           # Targets — обычная норма
    plan_targets: Any      # Targets — норма показанного дня (он может быть тренировочным)
    cycled: Any            # CycledTargets — нормы тренировочного дня и отдыха
    plan: Any              # DayPlan — первый день, его показываем в ответе
    menu: Any              # Menu — всё меню на запрошенный период
    menu_rest: Any         # Future[list[DayPlan]] — дни со второго, пока собираются
    speculative: Any       # _Speculation — вызовы модели, начатые в classify наперёд
    on_event: Any          # Callable[[dict], None] — события для ответа по частям
    last_plan: dict        # последний показанный план — для «замени творог»
    swap_rejected: list[int]  # что уже заменяли в этом плане: не предлагать снова
    swap: dict             # {index, old} — эта реплика заменила одно блюдо
    swap_index: int        # слот последней замены — «замени ещё раз» меняет его же
    eaten_new: list[dict]  # съеденное из этой реплики: {fdc_id, grams, meal}
    eaten_before: list[dict]  # съеденное сегодня в ПРОШЛЫХ репликах (вход)
    eaten_all: list[dict]  # всё съеденное за сегодня — только если был пересчёт (выход)
    eaten_shown: list      # MealItem съеденного из этой реплики — для ответа
    eaten_pending: dict    # съеденное, ждущее ответа «какой приём пищи» (вход)
    eaten_pending_new: dict  # то же, если эта реплика задала уточнение (выход)
    weighins: dict         # дата → вес, кг (вход)
    intake: dict           # дата → съедено, ккал (вход)
    adaptive: dict         # {tdee, at} — оценка расхода по динамике веса
    days: int              # на сколько дней просили меню
    checks: dict[str, bool]

    search_query: str
    chunks: list[dict]
    has_relevant: bool
    refine_count: int

    answer: str

    # Пройденные узлы: время и ответы модели в каждом. Копится сложением
    # (operator.add), а не перезаписью — каждый узел дописывает свою строку.
    trace: Annotated[list[dict], operator.add]
