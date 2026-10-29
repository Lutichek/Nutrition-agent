"""
Агент-нутрициолог на LangGraph: разговор, расчёт рациона и ответы по литературе.

Главная идея проекта — разделение труда между кодом и моделью:

    считает      — код   (targets.py, solver.py, foods.py)
    объясняет    — LLM   (по найденным в корпусе абстрактам)

Модель не складывает калории и не придумывает нормы. Она понимает, что человек
хочет, вытаскивает из разговора профиль, а потом объясняет уже посчитанный план
и отвечает на вопросы, опираясь на найденные статьи. Всё, что можно проверить
арифметикой, проверяется арифметикой — см. ``solver.check_plan``.

Граф:

    classify ─┬─→ off_topic ──────────────────────────────────────→ END
              ├─→ red_flag ───────────────────────────────────────→ END
              ├─→ extract_profile ─┬─→ ask_profile ───────────────→ END
              │                    └─→ make_plan → explain_plan ──→ END
              └─→ rewrite → retrieve → grade ─┬─→ generate ───────→ END
                                              ├─→ refine → retrieve
                                              └─→ cannot_answer ──→ END

Что делает каждый узел:
  • classify        — чего хочет пользователь: план, вопрос или не по теме;
                      заодно ловит состояния, с которыми нужен врач
  • extract_profile — собирает пол, возраст, рост, вес, активность, цель
                      из всего разговора, а не только из последней реплики
  • ask_profile     — не хватает данных для расчёта: спрашивает недостающее
  • make_plan       — детерминированный расчёт: нормы + сборка дня + проверки
  • explain_plan    — показывает план и объясняет цифры
  • rewrite         — переводит вопрос на английский и превращает в запрос
  • retrieve        — ищет абстракты по векторам (плотный поиск, top-5)
  • grade           — проверяет, отвечает ли найденное на вопрос
  • refine          — переформулирует и отправляет на повторный поиск
  • generate        — пишет ответ строго по найденному, со ссылками на PMID
  • cannot_answer   — честно сообщает, что в корпусе такого нет
  • off_topic       — вежливо отказывается от нерелевантных тем
  • red_flag        — отправляет к врачу вместо выдачи рациона

⚠️ Про перевод запроса. Корпус англоязычный, пользователь пишет по-русски.
Узел ``rewrite`` переводит вопрос на английский ДО поиска: на русском запросе
векторный поиск работает вполсилы. Это самый дорогой по отдаче шаг всего
поиска — precision@5 0.421 → 0.573, ndcg 0.465 → 0.596 (замер
``experiments/retrieval.py``, конфигурации «0» и «1»).

Контракт ответа (``RAGAgentAnswer``) — единый формат для модуля оценки
``evaluation.py``: по нему считаются и метрики поиска, и метрики ответа.

⚠️ Расчёты и ответы носят справочный характер и не заменяют консультацию
врача или диетолога.
"""

from __future__ import annotations

import json
import logging
import operator
import re
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

import food_search
from adaptive import adapted_tdee
from data_preporation import load_corpus_index
from foods import load_catalog
from portions import eaten_grams
from restrictions import expand, is_restriction_phrase, matches_terms, matches_words, to_pattern
from shopping import build_shopping_list
from solver import (
    DAY_TEMPLATE,
    DayPlan,
    MealItem,
    Menu,
    PlanNotFeasible,
    _filter_catalog,
    check_plan,
    day_word,
    eaten_item,
    mean_targets,
    menu_days,
    menu_targets,
    plan_cautions,
    preferred_ids,
    rebuild_rest,
    swap_item,
)
from targets import (
    ACTIVITY_FACTORS,
    ACTIVITY_LABELS,
    GOAL_LABELS,
    Profile,
    Targets,
    compute_targets,
    cycle_targets,
    explain,
    total_expenditure,
)

logger = logging.getLogger(__name__)

# Ответы модели внутри текущего узла графа — для журнала чата.
#
# Пишется в _complete, собирается обёрткой узла (_traced). Через контекст,
# а не через self: агент один на весь сервис, и реплики разных людей идут
# в соседних потоках. Вне графа (скрипты замеров) переменная пуста,
# и _complete ничего не копит.
_LLM_CALLS: ContextVar[list[dict] | None] = ContextVar("_LLM_CALLS", default=None)

# Дни меню со второго собираются здесь, пока модель объясняет первый
# (_node_make_plan). Солвер считает на чистом Python, а поток запроса в это
# время ждёт сеть — GIL свободен, и сборка идёт почти без помех. Пул общий
# на сервис: по потоку на одновременный расчёт, лишние ждут в очереди,
# а не плодят потоки.
_MENU_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="menu")

# Вызовы модели наперёд (NutritionAgent.speculate): пока classify решает,
# куда идти, extract_profile и rewrite уже спрашивают модель. Потоки
# ждут сеть, а не считают, — восьми хватает на несколько разговоров сразу.
_LLM_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm")


class _Speculation:
    """Вызовы модели, запущенные до того, как стало ясно, понадобятся ли они.

    Ответ отдаётся узлу, ТОЛЬКО если его промпт совпал символ в символ:
    тогда это тот же вызов, просто начатый раньше. Не совпал (состояние
    успело поменяться) или вызов ещё стоит в очереди — узел спрашивает
    модель сам, и опережение ничего не меняет, кроме потраченного вызова.
    """

    def __init__(self) -> None:
        self._calls: dict[str, tuple[str, Future]] = {}
        self._taken: set[str] = set()

    def start(self, name: str, prompt: str, call) -> None:
        self._calls[name] = (prompt, _LLM_POOL.submit(call))

    def take(self, name: str, prompt: str) -> tuple[str, list[dict]] | None:
        """(ответ, вызовы для trace) или None — тогда узел вызывает модель сам."""
        entry = self._calls.get(name)
        if entry is None or entry[0] != prompt:
            return None
        future = entry[1]
        if future.cancel():  # ещё не начался: ждать очередь дольше, чем спросить
            return None
        self._taken.add(name)
        raw, calls, error = future.result()
        if error is not None:
            raise error
        return raw, calls

    def unused(self) -> list[dict]:
        """Невостребованные вызовы — для trace: их токены тоже потрачены.

        Незавершённые не ждутся: ответ человеку из-за учёта не задерживается,
        и такой вызов помечается pending, а не выдаётся за бесплатный.
        """
        steps = []
        for name, (_prompt, future) in self._calls.items():
            if name in self._taken or future.cancelled():
                continue
            # Своё имя шага: под именем настоящего узла запись смешалась бы
            # с ним в разборе журнала. seconds = 0 — ответ он не задерживал.
            step = {"node": "speculative_unused", "for": name, "seconds": 0.0}
            if future.done():
                step["llm"] = future.result()[1]
            else:
                step["pending"] = True
            steps.append(step)
        return steps

# Сколько символов ответа модели хранить в журнале. Нужное для разбора —
# JSON маршрутизатора и профиля — укладывается в сотни символов; длинные
# ответы генерации и так лежат в записи целиком полем answer.
TRACE_RESPONSE_CHARS = 2000


# Признаки того, что деньги кончились НАСОВСЕМ и повторять бессмысленно.
#
# Главная тонкость — отличить это от временной нехватки. polza.ai резервирует
# сумму под каждый запрос в полёте, поэтому при работе в несколько потоков
# приходит 402 вида:
#
#   Требуется: 1.00 ₽, доступно: 0.87 ₽ (баланс: 3.87 ₽, зарезервировано: 3.00 ₽)
#
# Денег тут хватает (3.87 ₽), просто три рубля заняты соседними запросами.
# Через секунду они освободятся. Наблюдалось вживую: прогон в 4 потока упал,
# а тот же вызов по одному прошёл сразу же.
#
# Поэтому 402 сам по себе НЕ повод сдаваться — смотрим на баланс из тела
# ответа. Если и он меньше требуемого, деньги действительно кончились.
#
# Второй случай — квота OpenAI: там резервирования нет, и `insufficient_quota`
# означает именно исчерпание. Код при этом 429, тот же, что у обычного
# троттлинга, который повторять как раз нужно; различает их только тело.
_OUT_OF_FUNDS_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
)

_BALANCE_RE = re.compile(r"требуется:\s*([\d.,]+).*?баланс:\s*([\d.,]+)", re.IGNORECASE | re.DOTALL)


def _is_out_of_funds(error: Exception) -> bool:
    """Деньги кончились насовсем — в отличие от «заняты соседними запросами»."""
    text = str(error).lower()

    if any(marker in text for marker in _OUT_OF_FUNDS_MARKERS):
        return True

    match = _BALANCE_RE.search(text)
    if match:
        required, balance = (float(g.replace(",", ".")) for g in match.groups())
        # Баланса не хватает даже без учёта резерва — ждать нечего.
        return balance < required

    # 402 без разбора баланса: считаем временным и повторяем. Ошибиться
    # в эту сторону дешевле — лишние три попытки против оборванного
    # получасового прогона.
    return False


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


# ────────────────────────────────────────────────────────────
# Контракт ответа (совместим с evaluation.py)
# ────────────────────────────────────────────────────────────


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

    # Для адаптивной нормы (adaptive.py, ревью R9): взвешивания и съеденное
    # по дням (дата ISO → кг / ккал) и последняя оценка расхода {tdee, at}.
    weighins: dict[str, float] = Field(default_factory=dict)
    intake: dict[str, float] = Field(default_factory=dict)
    adaptive: dict | None = None


# ────────────────────────────────────────────────────────────
# Состояние графа
# ────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────
# Промпты
# ────────────────────────────────────────────────────────────

# Правило про ОБЩИЙ вопрос — из журнала чата. После показанного плана
# «Сколько белка нужно спортсменам на массе?» уходило в plan (2 прогона
# из 2), и объяснение плана отвечало личной нормой человека как общей
# рекомендацией. Замер с историей плана: ошибок на 8 общих вопросах
# 6/16 → 3/16, просьбы изменить план по-прежнему 6/6 в plan; на 210
# вопросах оценки без истории маршрут сменился у одного (off_topic →
# question, вопрос про йогу при травме — по теме).
CLASSIFY_PROMPT = """Ты — маршрутизатор запросов ассистента по питанию.

Реплика пользователя: {question}

Контекст разговора:
{history}

Верни ТОЛЬКО JSON:
{{"intent": "plan"|"question"|"off_topic", "red_flag": ""}}

intent = "plan"      — человек просит составить рацион, меню, план питания,
                       посчитать калории под свою цель, или отвечает на вопрос
                       о своих параметрах (рост, вес, возраст, активность).

                       ⚠️ Если в реплике названы ЛИЧНЫЕ ПАРАМЕТРЫ — возраст,
                       рост, вес, пол, уровень активности — это ВСЕГДА "plan",
                       независимо от того, о чём шёл разговор раньше.
                       «Мне 31, женщина, 168 см, 74 кг, хочу похудеть» — это
                       plan, даже если перед этим спрашивали про белок.
                       Человек называет свои данные только для расчёта.
intent = "question"  — вопрос о питании, нутриентах, диетах, похудении,
                       продуктах — но без просьбы составить план.
                       ⚠️ ОБЩИЙ вопрос — не о самом человеке, а о людях
                       вообще, о группе или о продукте («сколько белка
                       нужно спортсменам?», «полезна ли гречка?», «вредно
                       ли есть на ночь?») — это ВСЕГДА "question", даже если
                       до этого обсуждали его собственный план.
intent = "off_topic" — не о питании вообще (погода, код, политика).

red_flag — непустая строка ТОЛЬКО если выполнены ОБА условия:
  (а) человек просит рацион ДЛЯ СЕБЯ, и
  (б) он сообщает о СВОЁМ состоянии, при котором питание должен подбирать врач:
      беременность, кормление грудью, расстройство пищевого поведения,
      сахарный диабет, болезни почек или печени, онкология, возраст младше 18,
      либо просит калорийность заведомо опасного уровня (менее 1000 ккал).

ВАЖНО: вопрос о том, что известно науке о питании при каком-либо заболевании, —
это НЕ red_flag. «Что говорят исследования о питании при диабете?» — обычный
вопрос (intent = "question", red_flag = ""). Флаг нужен там, где человек
просит персональное предписание при своём диагнозе, а не там, где упомянута
болезнь.

В red_flag положи КОРОТКОЕ описание причины на русском. Иначе — пустая строка."""

EXTRACT_PROFILE_PROMPT = """Извлеки параметры человека из разговора о питании.

Уже известно: {known}

Новая реплика: {question}

Контекст разговора:
{history}

Верни ТОЛЬКО JSON с теми полями, которые ЯВНО названы (остальные не включай):
{{"sex": "female"|"male",
  "age": число полных лет,
  "height_cm": число,
  "weight_kg": число,
  "activity": "sedentary"|"light"|"moderate"|"high"|"athlete",
  "goal": "lose"|"maintain"|"gain",
  "rate_kg_per_week": число,
  "days": на сколько дней просят меню (1, 7, 30 и т.п.),
  "exclude": ["английские названия продуктов, которые человек НЕ ест"],
  "allergies": ["те из exclude, на которые у человека АЛЛЕРГИЯ"],
  "include": ["английские названия продуктов, которые человек ХОЧЕТ видеть"],
  "include_ru": {{"слово из include": "оно же по-русски"}}}}

Правила:
- НЕ придумывай значения. Не названо — не включай поле.
- activity по числу тренировок в неделю:
    нет тренировок, сидячая работа  → "sedentary"
    1 раз в неделю                  → "light"
    2-3 раза в неделю               → "moderate"
    4 раза в неделю и больше        → "high"
    две тренировки в день, спортсмен → "athlete"
  «Хожу в зал» без числа — НЕ повод угадывать, поле не включай.
- goal: похудеть = lose, удержать вес = maintain, набрать = gain.
  «Рекомпозиция», «recomp», «сжечь жир и набрать мышцы», «подсушиться
  без потери мышц» = recomp: калорийность поддерживающая, а белка больше.
  «Привести себя в форму» без уточнений = maintain.
- exclude — ТОЛЬКО английские слова, они ищутся в англоязычных названиях блюд.
  Для целой группы продуктов ставь ОДНО слово-категорию из списка ниже,
  НЕ перечисляй её содержимое сам — категорию раскроет программа:
    сладости, десерты, сладкое     → "sweets"
    мучное, выпечка, хлебобулочное → "flour"
    молочное, лактоза, аллергия
      на молоко                    → "lactose"
    глютен, целиакия               → "gluten"
    вегетарианец                   → "vegetarian"
    веган                          → "vegan"
    жареное                        → "fried"
    орехи                          → "nuts"
    рыба и морепродукты            → "seafood"
    яйца                           → "eggs"
    сахар, сладкие напитки         → "sugar"
  «Без сладкого», «не ем сладкое» — это "sweets" (десерты, выпечка,
  конфеты). "sugar" — только сахар как продукт и сладкие напитки.
    алкоголь                       → "alcohol"
    халяль, только халяльное       → "halal"
  Диета или диагноз, названные утвердительно («только халяль», «я веган»,
  «у меня целиакия»), — это exclude, а не include.
- allergies — только то, что человек назвал АЛЛЕРГИЕЙ («аллергия на арахис»
  → exclude ["peanut"], allergies ["peanut"]). «Не ем», «не люблю»,
  «непереносимость лактозы» — это НЕ аллергия: только exclude.
  Отдельный продукт называй как есть: «не ем свинину» → ["pork"],
  «без грибов» → ["mushroom"].
  «Не ем свинину, сладости, мучное и аллергия на лактозу» →
  ["pork","sweets","flour","lactose"].
- include — то же самое, но с ОБРАТНЫМ знаком: продукты, которые человек
  хочет видеть в рационе. Те же английские слова и те же категории.
  «Люблю гречку и творог» → ["buckwheat","cottage cheese"].
  «Хочу побольше овощей» → ["vegetables"].
  include_ru — словарь: ключ — слово из include, значение — тот же продукт
  по-русски в именительном падеже: «хочу гречку и свёклу» →
  include ["buckwheat","beet"], include_ru {{"buckwheat":"гречка","beet":"свёкла"}}.
  Не путай знак: «без грибов» это exclude, «хочу грибы» это include.
  Оборот «ем всё, кроме рыбы» — это exclude ["fish"], а НЕ include.
  Просьба «добавь больше белка» — это НЕ include: речь о нутриенте,
  а не о продукте, норму белка считает программа. Поле не включай.
- days: «на неделю» = 7, «на месяц» = 30, «на две недели» = 14. Не сказано —
  не включай поле.
- Уже известные значения переписывай только если человек их явно поправил."""

# ── Извлечение профиля по схеме (Schema-Guided Reasoning) ──────────────
#
# Модель сначала разбирает КАЖДОЕ упоминание в реплике со знаком (mentions:
# не ест / хочет / нутриент / диета), и только потом заполняет поля. Строгая
# схема держит типы и перечисления.
#
# Замер на живой модели (experiments/extract_profile.py, 38 фраз × 3 прогона,
# итог ПОСЛЕ кода агента, промпт уже с правилом про диеты):
#
#     набор                    свободный JSON   схема + разбор
#     простые (18)                 54/54            54/54
#     трудные (11)                 33/33            32/33   (не нашёл пожелание)
#     смешанный знак (9)           22/27            25/27
#     утечек ограничений             5                2
#     после уже названного (6)     13/18            15/18   (после правок ниже;
#                                                             до них 12/18 и 12/18)
#     время шага                 1.5-1.7 с        2.1-2.8 с
#
# Выигрыш — на смешанном знаке: «…а свинину и колбасу не ем» теряло колбасу
# 2/3, с разбором 0/3. Пилот до правила про диеты: та же картина плюс
# «Молочку не ем, мясо ем, но не свинину» (свинина 3/3 → 0/3). Строгая
# схема БЕЗ разбора не дала ничего и сама потеряла мёд у «я веган»: дело
# в разборе, а не в схеме. Время шага на ответ не легло: шаг идёт наперёд,
# параллельно classify, и первый текст плана — 4.2-4.6 с в обоих режимах
# (latency.py --free-json-extract, попеременно).
_NULLABLE = {"integer": {"type": ["integer", "null"]}, "number": {"type": ["number", "null"]}}
_EXTRACT_FIELDS: dict[str, Any] = {
    "mentions": {
        "type": "array",
        "description": "Сначала разбери КАЖДОЕ упоминание еды, нутриента, диеты "
                       "или состояния в новой реплике",
        "items": {"type": "object", "additionalProperties": False, "properties": {
            "phrase_ru": {"type": "string"},
            "kind": {"type": "string", "enum": [
                "food_excluded", "allergy", "food_wanted", "nutrient",
                "diet_or_condition", "other"]},
            "english": {"type": "string",
                        "description": "английское слово или слово-категория из правил"},
        }, "required": ["phrase_ru", "kind", "english"]},
    },
    "sex": {"type": ["string", "null"], "enum": ["female", "male", None]},
    "age": _NULLABLE["integer"],
    "height_cm": _NULLABLE["number"],
    "weight_kg": _NULLABLE["number"],
    "activity": {"type": ["string", "null"],
                 "enum": ["sedentary", "light", "moderate", "high", "athlete", None]},
    "goal": {"type": ["string", "null"], "enum": ["lose", "maintain", "gain", "recomp", None]},
    "rate_kg_per_week": _NULLABLE["number"],
    "days": _NULLABLE["integer"],
    "exclude": {"type": "array", "items": {"type": "string"}},
    "allergies": {"type": "array", "items": {"type": "string"}},
    "include": {"type": "array", "items": {"type": "string"}},
    # Список пар, а не словарь: строгая схема не допускает произвольных ключей.
    "include_ru": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"en": {"type": "string"}, "ru": {"type": "string"}},
        "required": ["en", "ru"]}},
}
EXTRACT_PROFILE_SCHEMA = {
    "name": "profile",
    "strict": True,
    "schema": {"type": "object", "additionalProperties": False,
               "properties": _EXTRACT_FIELDS, "required": list(_EXTRACT_FIELDS)},
}
_FREE_JSON_LINE = "Верни ТОЛЬКО JSON с теми полями, которые ЯВНО названы (остальные не включай):"
assert _FREE_JSON_LINE in EXTRACT_PROFILE_PROMPT
# Тот же промпт: меняется одна строка о формате, правила те же.
EXTRACT_PROFILE_PROMPT_SGR = EXTRACT_PROFILE_PROMPT.replace(
    _FREE_JSON_LINE,
    "Заполни поля. Чего ЯВНО нет в разговоре — null или пустой список. Смысл полей:",
)


# ── Аллергия: не то же, что «не ем» (ревью R5) ─────────────────────────
#
# Фильтр одинаков для обоих: аллерген отсекает exclude по названию,
# разделу и полному составу. Но для аллергии этого мало, и человек должен
# это услышать: справочник не знает о следах и перекрёстном загрязнении,
# а состав в магазине бывает другим. Текст — шаблон кода, не модели.
_ALLERGY_RE = re.compile(r"аллерг|анафилак", re.IGNORECASE)
UNSPECIFIED_ALLERGY = "unspecified"
_ALLERGEN_RU = {
    "peanut": "арахис", "peanuts": "арахис", "nuts": "орехи", "nut": "орехи",
    "tree nuts": "орехи", "eggs": "яйца", "egg": "яйца", "lactose": "молочное",
    "milk": "молоко", "dairy": "молочное", "seafood": "рыбу и морепродукты",
    "fish": "рыбу", "shellfish": "моллюсков и ракообразных", "shrimp": "креветки",
    "gluten": "глютен", "wheat": "пшеницу", "soy": "сою", "sesame": "кунжут",
    "honey": "мёд", "chocolate": "шоколад", "strawberry": "клубнику",
    "citrus": "цитрусовые",
}
ALLERGY_NOTE = (
    "\n\n⚠️ Про аллергию{names}: блюда подобраны так, чтобы аллергена не было "
    "ни в названии, ни в составе по справочнику USDA. Это не гарантия: справочник "
    "не знает о следовых количествах и перекрёстном загрязнении на производстве, "
    "а состав продукта в магазине может отличаться. Сверяйтесь с этикеткой, "
    "а при тяжёлой аллергии согласуйте рацион с аллергологом.{ask}"
)


# ── Замена одного блюда (ревью R8) ─────────────────────────────────────
_SWAP_RE = re.compile(
    r"замен|помен|вместо|другое блюд|другим блюд|не хочу|не нрав|убер", re.IGNORECASE)
# «Замени ещё раз», «другое» — без названия: тот же слот, что в прошлый раз.
_AGAIN_RE = re.compile(r"ещё раз|еще раз|друго[ейм]|не то|опять", re.IGNORECASE)
_SLOT_WORDS = {"завтрак": "Завтрак", "обед": "Обед — основное", "ужин": "Ужин — основное",
               "гарнир": "Обед — гарнир", "перекус": "Перекус"}


def _swap_index(question: str, plan: DayPlan) -> int | None:
    """Какое блюдо плана человек просит заменить — или None.

    Сначала по названию (основа первых слов русского названия блюда:
    «творог» → «Творог, нежирный»), потом по приёму пищи («замени ужин»).
    Название важнее: в «замени творог на завтрак» речь о твороге.
    """
    text = question.lower()
    for index, item in enumerate(plan.items):
        if item.slot.endswith("(съедено)"):
            continue  # съеденное уже не заменить
        words = re.findall(r"[а-яё]{4,}", item.title.lower())[:2]
        if any(word[:5] in text for word in words):
            return index
    for word, slot in _SLOT_WORDS.items():
        if word in text:
            for index, item in enumerate(plan.items):
                if item.slot == slot:
                    return index
    return None


def swap_note(old: MealItem, plan: DayPlan, before: DayPlan) -> str:
    """Ответ на замену — шаблон кода: числа плана модель не пересказывает."""
    index = next(i for i, it in enumerate(before.items) if it.fdc_id == old.fdc_id)
    new = plan.items[index]
    text = (f"Заменил «{old.title}» на «{new.title}» — {new.grams:.0f} г, "
            f"{new.kcal:.0f} ккал.")
    tuned = [f"{after.title} — {after.grams:.0f} г" for j, (was, after)
             in enumerate(zip(before.items, plan.items, strict=True))
             if j != index and was.servings != after.servings]
    if tuned:
        text += " Чтобы день остался в норме, подправил порции: " + "; ".join(tuned) + "."
    text += (f"\n\nИтого за день {plan.totals['kcal']:.0f} ккал из "
             f"{plan.targets['kcal']:.0f} — все проверки нормы пройдены. "
             "Если и это не подходит — скажите «замени ещё раз».")
    return text


# ── «Съел не по плану — пересчитай день» (этап 2) ──────────────────────
#
# Модель здесь только СОПОСТАВЛЯЕТ: выписывает, что съедено, и выбирает
# блюдо из найденных кодом кандидатов. Граммы не оценивает: не названы —
# справочная порция × названное количество. Числа дня — солвер.
_EATEN_RE = re.compile(
    r"\bсъел|\bсъела|\bпоел|\bпоела|перекусил|\bвыпил|\bвыпила|не по плану"
    r"|на (?:завтрак|обед|ужин) (?:был|была|были)", re.IGNORECASE)

EATEN_PROMPT = """Человек рассказывает, что съел. Выпиши каждую съеденную позицию.

Реплика: {question}

Верни ТОЛЬКО JSON:
{{"items": [{{"phrase_ru": "как сказал человек",
  "query_en": "1-3 английских слова для поиска в справочнике USDA (pizza, cheese pancake)",
  "grams": число граммов — ТОЛЬКО если человек его назвал, иначе null,
  "count": сколько штук или порций — ТОЛЬКО если назвал, иначе null,
  "unit": "piece"|"slice"|"bar"|"cup"|"small"|"medium"|"large"|null — какую
    единицу НАЗВАЛ человек («кусок» → slice, «батончик» → bar, «стакан» → cup),
  "meal": "breakfast"|"lunch"|"dinner"|"snack"|"unknown"}}]}}

Правила:
- НЕ оценивай граммы и калории сам: не названо — null.
- Составное разбивай на продукты: «блины со сметаной» → блины и сметана,
  «гречка с курицей» → гречка и курица. Готовое блюдо (пицца, борщ,
  сэндвич, котлета) — одна позиция.
- «Два куска», «три блина», «батончик» — это count (2, 3, 1), не grams.
- meal — только если назван приём пищи (завтрак, обед, ужин, перекус);
  иначе "unknown"."""

EATEN_PICK_PROMPT = """Сопоставь съеденное с блюдами справочника. Для каждой позиции
выбери номер самого похожего блюда из ЕЁ списка или 0, если подходящего нет.

Выбирай то, что И ЕСТЬ названный продукт или блюдо. Если в варианте он лишь
ингредиент («сметана» → «картофель со сметаной») или это его часть
(«курица» → «куриная кожа»), это не подходит — ставь 0.

{blocks}

Верни ТОЛЬКО JSON: {{"choices": [номер для позиции 1, номер для позиции 2, ...]}}"""

# Приём пищи → какие слоты DAY_TEMPLATE он закрывает. Обед и ужин — оба
# слота (основное и гарнир): «на обед съел пиццу» — это весь обед.
_MEAL_SLOTS = {"breakfast": [0], "lunch": [1, 2], "dinner": [3, 4], "snack": [5, 6]}

CHECK_RU = {
    "kcal_in_range": "калорийность дня вне нормы",
    "protein_g_in_range": "белок вне нормы",
    "fat_g_in_range": "жиры вне нормы",
    "carb_g_in_range": "углеводы вне нормы",
    "fiber_sufficient": "клетчатки мало",
    "sodium_within_limit": "соли больше суточного предела",
    "no_duplicates": "блюда повторяются",
    "portions_sane": "порции вне разумных границ",
}


def eaten_note(new: list[MealItem], plan: DayPlan, checks: dict[str, bool]) -> str:
    """Ответ на «съел не по плану» — шаблон кода, числа только из солвера."""
    eaten = "; ".join(f"{it.title} — {it.grams:.0f} г, {it.kcal:.0f} ккал" for it in new)
    # Сопоставление со справочником — работа модели, и она ошибается
    # («сметана» → «картофель со сметаной» на живом прогоне). Поэтому
    # человек видит, ЧТО учтено и сколько граммов, и может поправить.
    text = (f"Учёл съеденное: {eaten}. Если я понял не так — напишите, что именно "
            f"было и сколько, и пересчитаю.\n\nПересобрал остаток дня под оставшуюся "
            f"норму: итого за день {plan.totals['kcal']:.0f} ккал из "
            f"{plan.targets['kcal']:.0f}.")
    failed = [CHECK_RU.get(name, name) for name, ok in checks.items() if not ok]
    if failed:
        text += (" Честно: из-за съеденного день целиком в норму не уложился — "
                 + ", ".join(failed) + ". Остальные приёмы пищи я подобрал так, "
                 "чтобы отклонение было как можно меньше.")
    return text


def _today_utc() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


# Число, похожее на вес в реплике: «вешу 79,4», «79.4 кг».
_WEIGHT_RE = re.compile(r"(\d{2,3}(?:[.,]\d{1,2})?)\s*(?:кг|kg)|вешу\s*(?:около\s*)?(\d{2,3}(?:[.,]\d)?)",
                        re.IGNORECASE)


def _merge_allergies(question: str, known: list, named: Any, exclude: list,
                     said_before: bool = False) -> tuple[list[str], list[str]]:
    """Аллергии профиля после реплики и exclude, в котором есть каждая из них.

    Три источника, потому что ни одному нельзя верить в одиночку:
    модель (поле allergies), формулировка в exclude («peanut allergy» —
    это уже аллергия) и сам текст («аллергия», а модель не отметила
    ничего — тогда UNSPECIFIED, предупреждение без названия аллергена).

    Аллергии копятся, а не заменяются: молча потерять аллергию из-за того,
    что модель в следующей реплике её не повторила, нельзя. И каждая
    обязана стоять в exclude — по той же причине.

    Аллергию, отмеченную моделью, верим, только если слово «аллергия»
    звучало — в этой реплике или раньше (``said_before``). Наблюдалось
    (experiments/extract_profile.py): «Хочу творог, но у меня непереносимость
    лактозы» модель 3/3 отмечала аллергией. «И на кунжут тоже» после
    «аллергия на арахис» при этом проходит — слово было раньше.
    """
    allergies = [str(a).strip().lower() for a in known if str(a).strip()]
    if not (said_before or _ALLERGY_RE.search(question)):
        named = []
    new = [str(a).strip().lower() for a in named or [] if isinstance(a, str) and a.strip()]
    new += [t.strip().lower() for t in exclude
            if isinstance(t, str) and "allergy" in t.lower()]
    for item in new:
        if item not in allergies:
            allergies.append(item)
    if not allergies and _ALLERGY_RE.search(question):
        allergies = [UNSPECIFIED_ALLERGY]
    if len(allergies) > 1 and UNSPECIFIED_ALLERGY in allergies:
        allergies.remove(UNSPECIFIED_ALLERGY)

    exclude = list(exclude)
    lowered = {t.strip().lower() for t in exclude if isinstance(t, str)}
    for item in allergies:
        if item != UNSPECIFIED_ALLERGY and item not in lowered:
            exclude.append(item)
    return allergies, exclude


def allergy_note(allergies: list[str]) -> str:
    """Предупреждение для ответа с планом — или пусто, если аллергии нет."""
    if not allergies:
        return ""
    names = []
    for item in allergies:
        if item == UNSPECIFIED_ALLERGY:
            continue
        name = _ALLERGEN_RU.get(item.removesuffix(" allergy"), item)
        if name not in names:
            names.append(name)
    return ALLERGY_NOTE.format(
        names=f" ({', '.join(names)})" if names else "",
        ask="" if names else " Напишите, на что именно аллергия, — исключу эти продукты.",
    )


# «Без сладкого» — это sweets (десерты, выпечка, конфеты), а не sugar
# (сахар, сироп, сладкие напитки). Наблюдалось на живой модели 3/3 в обоих
# режимах: «Ещё без сладкого» → sugar, и выпечка оставалась в меню. Знак
# (нельзя / можно) модель определяет верно — путает словарь, поэтому
# категорию поправляет код, и только у той фразы-упоминания, где сказано
# «сладк…»: вся реплика «Сахар не кладу, а сладкое ем» исключила бы
# сладкое, которое человек ест.
_SWEET_PHRASE = re.compile(r"сладк|сладост|десерт", re.IGNORECASE)
_SUGAR_PHRASE = re.compile(r"сахар|напит", re.IGNORECASE)
FIX_SWEETS = True


def _category_fix(word: str, phrase_ru: str) -> str:
    """Слово упоминания с поправленной категорией (sugar → sweets)."""
    if (FIX_SWEETS and word.lower() == "sugar" and _SWEET_PHRASE.search(phrase_ru)
            and not _SUGAR_PHRASE.search(phrase_ru)):
        return "sweets"
    return word


# Какие разобранные упоминания (mentions) — в какие поля профиля.
_MENTION_FIELDS = {"food_excluded": ("exclude",), "allergy": ("exclude", "allergies"),
                   "food_wanted": ("include",)}


def _from_structured(parsed: dict) -> dict:
    """Ответ по схеме → прежний вид профиля; свободный JSON проходит как есть.

    Исключения и пожелания ДОПОЛНЯЮТСЯ из разобранных упоминаний.
    Наблюдалось на живой модели, 3 прогона из 3: на «Ещё не ем грибы»
    после «Не ем свинину» разбор верный — mushroom, food_excluded, — а в
    поле exclude модель переписала только известное ["pork"], и грибы
    молча оставались в каталоге. Разбор фактов верен, ошибается сборка
    итога — значит, собирает код. Свободный JSON так не ошибался.

    Пары include_ru снова становятся словарём, а null и пустое — «не
    названо», как раньше отсутствие поля: иначе пустой exclude затёр бы
    известные ограничения.
    """
    parsed = dict(parsed)
    for mention in parsed.pop("mentions", None) or []:
        if not isinstance(mention, dict):
            continue
        word = _category_fix(str(mention.get("english") or "").strip(),
                             str(mention.get("phrase_ru") or ""))
        for field in _MENTION_FIELDS.get(mention.get("kind"), ()) if word else ():
            current = parsed.get(field)
            current = [current] if isinstance(current, str) else list(current or [])
            if word.lower() not in {str(t).lower() for t in current}:
                current.append(word)
            parsed[field] = current
    pairs = parsed.get("include_ru")
    if isinstance(pairs, list):
        parsed["include_ru"] = {
            pair["en"]: pair["ru"] for pair in pairs
            if isinstance(pair, dict) and pair.get("en") and pair.get("ru")
        }
    return {key: value for key, value in parsed.items() if value not in (None, [], {}, "")}


REWRITE_PROMPT = """Преврати вопрос пользователя в короткий поисковый запрос
НА АНГЛИЙСКОМ языке для базы научных абстрактов по нутрициологии.

Правила:
- переведи на английский, используй принятую научную терминологию
- убери разговорные обороты
- не придумывай того, чего нет в вопросе
- верни ТОЛЬКО текст запроса, без пояснений и кавычек

Вопрос: {question}"""

GRADE_PROMPT = """Оцени, помогают ли найденные фрагменты ответить на вопрос.

Вопрос: {question}

Фрагменты:
{context}

Верни ТОЛЬКО JSON: {{"has_relevant": true/false, "keep": [номера полезных фрагментов]}}

has_relevant = true, если хотя бы один фрагмент содержит информацию по существу
вопроса. Нумерация фрагментов начинается с 1."""

# Строгий вариант того же узла. Мягкий признавал релевантным всё, что просто
# «про питание»: на вопрос «какое влияние оказывают блюда с историей» агент
# находил фрагмент про средиземноморскую диету и уверенно отвечал. Здесь
# критерий сформулирован от противного — перечислено, что полезным НЕ считается.
GRADE_PROMPT_STRICT = """Оцени, можно ли ОТВЕТИТЬ на вопрос по этим фрагментам.

Вопрос: {question}

Фрагменты:
{context}

Верни ТОЛЬКО JSON: {{"has_relevant": true/false, "keep": [номера полезных фрагментов]}}

Фрагмент полезен, ТОЛЬКО если содержит конкретные сведения, отвечающие
именно на заданный вопрос.

Фрагмент НЕ полезен, если он:
- просто о той же общей теме (питание, здоровье, диета), но не о том, что спрашивают;
- отвечает на соседний вопрос, а не на заданный;
- позволяет лишь строить догадки и достраивать выводы.

Отдельно: если вопрос опирается на предположение, которого в научной литературе
нет (эзотерика, несуществующие эффекты, надуманные связи), верни
has_relevant = false, даже если фрагменты тоже про питание.

Лучше честно вернуть false, чем притянуть фрагмент за уши.
Нумерация фрагментов начинается с 1."""

REFINE_PROMPT = """Поиск по этому запросу не дал полезного результата.

Исходный вопрос: {question}
Запрос, который не сработал: {search_query}

Составь ДРУГОЙ поисковый запрос на английском: смени формулировку,
используй синонимы и более общие научные термины.
Верни ТОЛЬКО текст запроса."""

GENERATE_PROMPT = """Ты — ассистент по питанию. Ответь на вопрос пользователя,
опираясь ТОЛЬКО на приведённые фрагменты научных статей.

Вопрос: {question}

Фрагменты (у каждого указан источник):
{context}

Правила:
- отвечай по-русски, простым языком, без научного жаргона
- опирайся только на фрагменты; если чего-то в них нет — так и скажи
- НЕ ВЫДУМЫВАЙ числа: приводи только те, что есть во фрагментах
- ссылайся ТОЛЬКО номером PMID из строки «Источник», например [PMID 12345678];
  НИКОГДА не ссылайся на номер фрагмента ([1], [2]) — за пределами этого
  сообщения такие номера ничего не значат
- в конце ответа перечисли использованные PMID
- 3-6 предложений
- если вопрос касается здоровья, напомни, что это не замена консультации врача"""

# Вариант с требованием полноты. Отличается от базового ровно двумя строками —
# намеренно: попытка переписать промпт генерации целиком ухудшала все метрики
# сразу, а сработала одна добавленная строка.
#
# Что чиним: замер показал, что у 17 вопросов из 80 нужный фрагмент был найден,
# а ответ всё равно вышел неполным — судья писал «упоминает X, но не отражает Y».
# Подозрение на потолок «3-6 предложений»: он заставляет выбирать между фактами.
GENERATE_PROMPT_FULL = """Ты — ассистент по питанию. Ответь на вопрос пользователя,
опираясь ТОЛЬКО на приведённые фрагменты научных статей.

Вопрос: {question}

Фрагменты (у каждого указан источник):
{context}

Правила:
- отвечай по-русски, простым языком, без научного жаргона
- опирайся только на фрагменты; если чего-то в них нет — так и скажи
- НЕ ВЫДУМЫВАЙ числа: приводи только те, что есть во фрагментах
- ПЕРЕЧИСЛИ ВСЕ относящиеся к вопросу сведения из фрагментов: величины,
  сравнения групп, оговорки и условия. Неполный ответ — такая же ошибка,
  как неверный: человек примет решение, не зная половины картины
- ПЕРЕСЧИТЫВАЙ единицы под вопрос. Если спрашивают «сколько грамм на кг веса»,
  а во фрагменте даны проценты от калорийности — переведи их в г/кг и покажи
  расчёт коротко (1 г белка и углеводов = 4 ккал, 1 г жира = 9 ккал). Отдавать
  проценты в ответ на вопрос о граммах — значит не ответить
- ссылайся ТОЛЬКО номером PMID из строки «Источник», например [PMID 12345678];
  НИКОГДА не ссылайся на номер фрагмента ([1], [2]) — за пределами этого
  сообщения такие номера ничего не значат
- в конце ответа перечисли использованные PMID
- столько предложений, сколько нужно, чтобы не потерять факты
- если вопрос касается здоровья, напомни, что это не замена консультации врача"""

# Кандидат, проверяемый экспериментом (experiments/answer_prompt.py).
#
# Откуда взялся. Разбор 210 вопросов: 45 ответов получили 0.5, и это 76%
# всех потерь correctness. У 89% из них поиск сработал (первый фрагмент
# релевантен) — то есть материал был, а в ответ не попал. Обоснования судьи
# однотипны: «не указывает на мета-анализ с 15 исследованиями», «не упоминает
# исследование PREDIMED», «не включает формулы Cunningham и Ten-Haaf»,
# «не отражает, что участников было 281».
#
# Действующий промпт уже требует перечислить «величины, сравнения групп,
# оговорки и условия» — но модель читает это как размеры эффектов, а не как
# состав доказательной базы. Поэтому категория названа отдельно.
#
# Отличие от GENERATE_PROMPT_FULL — ровно один пункт, и это намеренно:
# на шаге 22 попытка переписать промпт целиком ухудшала все метрики разом.
# Текст собирается вставкой, чтобы разница физически не могла оказаться
# больше заявленной.
_EVIDENCE_RULE = """- СОХРАНЯЙ СОСТАВ ДОКАЗАТЕЛЬСТВ, если он есть во фрагментах: сколько было
  исследований и какого они типа (мета-анализ, когортное, рандомизированное),
  сколько участников, как назывались испытания, когорты и формулы. «В одном
  из исследований» вместо «в мета-анализе 15 исследований на 281 участнике» —
  потеря: по такому ответу нельзя понять, насколько выводу можно верить.
  Чего во фрагментах нет — не придумывай
"""

# Расчёт нормы человека — вставляется в промпт генерации перед фрагментами,
# только если профиль собран (NutritionAgent._personal_norms). Без профиля
# промпт не меняется вовсе: на нём сняты все метрики оценки.
FRAGMENTS_HEADER = "Фрагменты (у каждого указан источник):"

PERSONAL_NORMS_BLOCK = """О человеке известно: {profile}

Его норма (посчитана формулами, НЕ меняй эти числа):
{targets_explained}

Как этим пользоваться:
- вопрос о его СОБСТВЕННОЙ норме или рационе («почему у меня…», «сколько
  мне…», «почему так мало…») — начни с ответа по этому расчёту, фрагменты —
  обоснование;
- вопрос ОБЩИЙ (о людях вообще, группе, продукте) — отвечай только по
  фрагментам и его норму НЕ упоминай вовсе;
- НИКОГДА не пересчитывай рекомендации из фрагментов под его вес или рост
  сам: числа о нём есть только в расчёте выше.

"""

# Формулировка «как этим пользоваться» — из сравнения трёх вариантов
# на одних и тех же фрагментах. Мягкая («если вопрос общий — как обычно»)
# втягивала личное в общее: на «сколько клетчатки нужно взрослому» —
# «ваша норма 17 г», на «белок спортсменам» модель сама умножала 1.3 г/кг
# на вес человека. Без блока личные вопросы получали ерунду («углеводов
# в веганской диете меньше…»). Строгая — личное по расчёту, общее без него.

GENERATE_PROMPT_EVIDENCE = GENERATE_PROMPT_FULL.replace(
    "- ПЕРЕСЧИТЫВАЙ единицы под вопрос.",
    _EVIDENCE_RULE + "- ПЕРЕСЧИТЫВАЙ единицы под вопрос.",
)

# Варианты промпта генерации. Короткий («3-6 предложений») оставлен как
# отклонённая на шаге 22 базовая линия: без него нечем показать, чего стоило
# требование полноты.
ANSWER_PROMPTS = {
    "short": GENERATE_PROMPT,
    "full": GENERATE_PROMPT_FULL,
    "evidence": GENERATE_PROMPT_EVIDENCE,
}


# План модель НЕ перепечатывает (пункт 2). Раньше просили «приведи план как
# есть»: копия занимала 44-72% ответа, хотя карточка в интерфейсе показывает
# тот же план из данных и подробнее. Выходной текст — самая медленная часть
# ответа, и это единственное место, где модель держала в руках числа плана.
# Замер — experiments/latency.py.
#
# Примера фразы о плане в пункте 2 нет намеренно: был «в плане упор на белок
# за завтраком», и модель вставляла его дословно — про план, где это не так.
# Пример в промпте модель читает как готовый текст.
EXPLAIN_PLAN_PROMPT = """Ты — ассистент по питанию. Объясни человеку его план на день.

Реплика человека: {question}

Профиль: {profile}

Расчёт норм (выполнен формулами, НЕ меняй эти числа):
{targets_explained}

План на день (собран программой, НЕ меняй состав и числа):
{plan}

Результат автоматических проверок: {checks}

Фрагменты научных статей, если нужно обосновать нормы:
{context}

Напиши ответ по-русски:
0. Если в реплике есть ВОПРОС, а не только данные о себе или просьба
   составить план, — начни с прямого ответа на него одним-двумя
   предложениями. Числа бери ТОЛЬКО из расчёта норм и плана выше:
   «сколько белка» — это норма белка из расчёта. Если ответа в расчёте
   нет, скажи, что это не считал, и не прикидывай сам.
   Расчёт — про ЭТОГО человека. Если вопрос о других людях («сколько
   белка нужно спортсменам?»), НЕ выдавай его норму за общую
   рекомендацию: скажи прямо, что посчитал это для него.
1. Коротко — какая норма получилась и почему именно такая.
2. План человек видит карточкой рядом с ответом: блюда, граммы, калории.
   НЕ перечисляй блюда и граммовки и не пересказывай план по приёмам
   пищи. Ничего не утверждай о составе плана, чего нельзя прямо
   прочитать в нём выше.
3. Если в проверках что-то не прошло — честно скажи, что именно.
4. Если во фрагментах есть обоснование норм — перескажи его своими словами:
   что показали исследования. Номера PMID и номера фрагментов НЕ пиши:
   статьи человек видит карточками под ответом, а номер ему ничего не говорит.
5. Напомни, что блюда взяты из справочника USDA и это ориентир по составу,
   а не предписание; при заболеваниях нужен врач.

Не выдумывай блюда и числа, которых нет выше."""

ASK_PROFILE_PROMPT = """Ты — ассистент по питанию. Чтобы посчитать рацион,
не хватает данных.

Уже известно: {known}
Не хватает: {missing}

Напиши короткое дружелюбное сообщение по-русски: перечисли, что нужно уточнить.
Не задавай лишних вопросов, спроси только недостающее. 1-3 предложения.

Если среди недостающего есть activity, спроси про тренировки и ОБЯЗАТЕЛЬНО
приведи варианты списком — человек не обязан знать, что такое «moderate»:
{activity_options}"""

OFF_TOPIC_ANSWER = (
    "Я помогаю только с питанием: рассчитать норму калорий, собрать рацион "
    "или разобраться, что об этом говорят исследования. С этим вопросом "
    "я, к сожалению, не помогу."
)

CANNOT_ANSWER_TEMPLATE = (
    "В моей базе научных статей нет информации, чтобы ответить на этот вопрос. "
    "Могу рассчитать норму калорий и собрать рацион под вашу цель — или "
    "попробуйте переформулировать вопрос."
)

RED_FLAG_TEMPLATE = (
    "Здесь я не могу составить рацион: {reason}. В такой ситуации питание "
    "должен подбирать врач или дипломированный диетолог — цена ошибки слишком "
    "высока. Могу вместо этого рассказать, что об этом пишут исследования."
)

# Номера PMID в тексте ответа — для модели, а не для человека.
#
# Промпт генерации требует ссылаться номером PMID: без этого модель пишет
# «[1][2]» — номера фрагментов, которые вне промпта ничего не значат.
# Человеку же номер не говорит ничего: статьи он видит карточками под
# ответом (название, журнал, цитата и ссылка на PubMed), а карточки
# собираются из найденных фрагментов, а не из текста. Поэтому номера
# вырезаются ТОЛЬКО при показе (``for_reader``), а ответ агента остаётся
# как есть — по нему считаются метрики, и сохранённые прогоны сравнимы.
#
# Формы взяты из 2400 сохранённых ответов: «[PMID N]», перечни
# «[PMID N], [PMID N]» и «[PMID N][PMID N]», хвост «Использованные PMID: …»,
# изредка голое «статьи с PMID N».
#
# Перечень через «и» в сохранённых ответах не встречался: они все из ветки
# вопросов. Встретился в объяснении плана — «например [PMID 1] и [PMID 2],
# которые…» — и оставлял «например и, которые…». С тех пор объяснение
# плана номеров не пишет вовсе (EXPLAIN_PLAN_PROMPT, п. 4), а здесь
# остаётся страховка.
_CITATION_GROUP = re.compile(
    # \b перед «и» обязателен: без него «сытыми [PMID 1]» теряло окончание
    # и становилось «сытым» — 321 искажённое слово на сохранённых ответах.
    r"(?:\s*(?:[,;]|\bи(?=\s*[\[(]\s*PMID))?\s*[\[(]\s*PMID\b[^\])\n]*[\])])+"
)
_BARE_PMID = re.compile(
    r"(?:\s+(?:с|в|из))?\s*\bPMID\b:?\s*\d+(?:\s*(?:[,;]|и)\s*\d+)*", re.IGNORECASE
)
_SOURCES_LINE = re.compile(
    r"^[\s*_]*(?:использован\w*|источник\w*)?(?:\s*PMID)?[\s*_:.,;—-]*$", re.IGNORECASE
)


# Предлог, державшийся только за ссылку: «например, в [PMID 1] и [PMID 2].»
# давало «например, в.». Снимается вместе с группой; если от оборота
# осталось голое «например», уходит и оно.
_PREPOSITION_BEFORE_CITATION = re.compile(
    r"\s+\b(?:в|во|из|на|по|с|со)(?=\s*[\[(]\s*PMID)", re.IGNORECASE
)
_DANGLING_EXAMPLE = re.compile(r",\s*например,?\s*(?=[.;!?])", re.IGNORECASE)


def for_reader(text: str) -> str:
    """Ответ в том виде, в каком его читает человек: без номеров PMID."""
    lines = []
    for line in text.split("\n"):
        had_pmid = "PMID" in line.upper()
        line = _PREPOSITION_BEFORE_CITATION.sub("", line)
        line = _CITATION_GROUP.sub("", line)
        line = _BARE_PMID.sub("", line)
        if had_pmid and _SOURCES_LINE.match(line):
            # От «Использованные PMID: 123, 456» осталась одна подпись.
            continue
        line = re.sub(r"\s+([.,;:!?)])", r"\1", line)
        if had_pmid:
            line = _DANGLING_EXAMPLE.sub("", line)
            line = re.sub(r",([.;!?])", r"\1", line)
        lines.append(re.sub(r"(?<=\S) {2,}", " ", line).rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def for_reader_partial(raw: str) -> str:
    """``for_reader`` для ответа, который модель ещё пишет.

    Хвост придерживается до конца ссылки: она приходит кусками, и
    «[PMID 12» или «(PMID 1» мелькнули бы на экране раньше, чем их
    вырежут. Держится всё от последней незакрытой скобки — обычная
    скобка в тексте лишь чуть позже появится — и недописанное «PMID»
    без скобки: _BARE_PMID снимает его только вместе с цифрами.
    """
    for opening, closing in (("[", "]"), ("(", ")")):
        opened = raw.rfind(opening)
        if opened != -1 and closing not in raw[opened:]:
            raw = raw[:opened]
    raw = _PMID_TAIL.sub("", raw)
    return for_reader(raw)


# «P», «PM», «PMI», «PMID», «PMID: 12» в самом конце недописанного текста.
_PMID_TAIL = re.compile(r"\bP(?:M(?:I(?:D)?)?)?\b:?[\s\d,;]*$")

# Предложение назвать любимые продукты. Показывается ОДИН раз и только
# после того, как человек увидел собранный рацион.
#
# Текст фиксирован намеренно: по нему же проверяется, не предлагали ли
# уже в этом разговоре (см. _preferences_offer). Перепишете формулировку —
# предложение начнёт повторяться на каждый пересчёт плана, и молча:
# ошибка выглядит как назойливость, а не как поломка.
PREFERENCES_OFFER = (
    "Если есть продукты, которые вы хотели бы видеть в рационе, — скажите, "
    "и я пересоберу меню с ними. Например: «люблю гречку и творог»."
)

# Поля профиля, без которых расчёт невозможен.
#
# activity здесь не случайно. У Profile есть значение по умолчанию
# («sedentary»), и раньше агент молча его подставлял: человек писал
# «хочу рекомпозицию», получал норму как для сидячего образа жизни
# и не понимал, почему калорий так мало. Между «сидячим» и «высоким»
# у мужчины 65 кг лежит 830 ккал в день — это не мелочь, о которой
# можно догадаться за пользователя.
REQUIRED_PROFILE_FIELDS = ["sex", "age", "height_cm", "weight_kg", "activity"]

# Дольше месяца меню не собираем: смысла мало, а времени уходит много —
# подбор линеен по числу дней.
MAX_MENU_DAYS = 31


def _goal_from_text(text: str) -> str | None:
    """Цель по формулировке — там, где модель ошибается устойчиво.

    Проверено на семи формулировках: «рекомпозицию тела» и «рекомпозицию»
    модель понимает, а «мягкая рекомпозиция» отдаёт как gain, «сжечь жир
    и набрать мышцы» — тоже gain, «подсушиться, но не потерять мышцы» —
    как lose. Две последние стоят в промпте прямым примером и всё равно
    не срабатывают.

    Человек при этом получает не то, что просил: белок 1.4 г/кг вместо 2.0,
    то есть обычное поддержание вместо смены состава тела.

    Здесь разбирается только рекомпозиция — единственная цель, которую
    модель путает. Остальные три она различает уверенно, и отбирать
    их у неё незачем.
    """
    lowered = text.lower()

    # Слово названо прямо — вопросов нет.
    if re.search(r"рекомпозиц|recomp", lowered):
        return "recomp"

    # Названа суть: одновременно убрать жир и нарастить мышцы. Порознь это
    # разные цели, вместе — рекомпозиция.
    losing = re.search(r"сжечь жир|убрать жир|снизить жир|подсушит|сушк|"
                       r"уменьшить жир|похудеть", lowered)
    keeping = re.search(r"набрать мышц|нарастить мышц|не потерять мышц|"
                        r"сохранить мышц|без потери мышц|набрать масс", lowered)
    if losing and keeping:
        return "recomp"

    return None


def _activity_from_text(text: str) -> str | None:
    """Уровень активности по формулировке, которую предложил сам агент.

    Наблюдавшийся случай, и он особенно обиден. Узел ``ask_profile``
    показывает человеку пять вариантов словами: «сидячий образ жизни»,
    «малая активность», «средняя активность», «высокая активность»,
    «спортсмен». Человек отвечает подписью варианта — это единственное
    разумное поведение, раз ему дали список.

    А промпт извлечения сопоставляет только ЧИСЛО тренировок. Замер
    на сырых ответах модели, по три прогона:

        «тренируюсь 2-3 раза в неделю»  → moderate   ✓
        «высокая активность»            → high       ✓
        «средняя активность»            → sedentary  ✗ 3 из 3
        «средняя»                       → sedentary  ✗ 3 из 3

    «Высокую» модель понимает, «среднюю» устойчиво роняет в «сидячий» —
    то есть в противоположный край шкалы.

    Цена ошибки: коэффициент 1.2 вместо 1.55, это около 500 ккал в день.
    И отказ тихий — поле не пропадает, а приходит НЕВЕРНЫМ, поэтому
    проверка обязательных полей его не ловит и переспросить агент
    не догадывается. Заодно пропадает циклирование: при sedentary
    тренировочных дней ноль, и разводить нечего.

    Здесь разбираются ровно те подписи, которые агент показывает сам.
    Свободные формулировки («хожу в зал трижды») остаются за моделью —
    их она различает уверенно.
    """
    lowered = text.lower()

    # Порядок важен: «высокая» проверяется раньше «малой», иначе
    # «не малая, а высокая» ушло бы не туда.
    patterns = [
        ("athlete", r"спортсмен|две тренировки в день|тяжёлый физический труд"),
        ("high", r"высок\w*\s+активн|4 раза|четыре раза|больше четырёх"),
        ("moderate", r"средн\w*\s+активн|^\s*средн\w*\s*$|2-3 раза|2–3 раза|"
                     r"два-три раза|три раза в недел"),
        ("light", r"мал\w*\s+активн|^\s*мал\w*\s*$|1 раз в недел|один раз в недел"),
        ("sedentary", r"сидяч\w*\s+(образ|работ)|без тренировок|не трениру|"
                      r"тренировок нет"),
    ]
    for level, pattern in patterns:
        if re.search(pattern, lowered):
            return level
    return None


# Нутриенты, которые модель принимает за продукты.
#
# Наблюдавшийся случай: «добавь больше белка» → include=["protein"],
# 2 прогона из 3, при том что промпт это запрещает прямым текстом
# («речь о нутриенте, а не о продукте, поле не включай»).
#
# Отказ был бы тихим и вредным: "protein" находит в каталоге протеиновые
# батончики и порошки, и человек, попросивший больше белка, получил бы
# в меню спортпит. Норму белка считает targets.py, и просьба «побольше
# белка» — это разговор о цели, а не о продукте.
#
# Тот же приём, что у _goal_from_text и _activity_from_text: там, где
# модель устойчиво ошибается, решает код.
_NUTRIENT_WORDS = {
    "protein", "proteins", "fat", "fats", "carb", "carbs", "carbohydrate",
    "carbohydrates", "calorie", "calories", "kcal", "energy",
    "fiber", "fibre", "vitamin", "vitamins", "mineral", "minerals",
    "macros", "micronutrients", "nutrients",
}


def _resolve_preferences(profile: Profile) -> list[str]:
    """Убрать из пожеланий то, что противоречит ограничениям.

    Человек способен попросить и то и другое: «хочу молочку» вместе
    с «аллергия на лактозу» — это не выдумка, а обычный разговор,
    где пожелание названо раньше ограничения или в другой реплике.

    Побеждает ОГРАНИЧЕНИЕ, и всегда. Нарушенное исключение — аллерген
    в тарелке, несработавшее пожелание — только огорчение. Цена ошибки
    несопоставима, поэтому выбор не обсуждается.

    Сравниваем ПОСЛЕ раскрытия категорий, иначе конфликт не виден:
    «lactose» и «milk» — разные слова, но первое раскрывается во второе.
    Проверка идёт ТЕМ ЖЕ сопоставлением, что и фильтр каталога
    (restrictions.matches_terms): с границами слова — чтобы «хочу ham»
    не выживало при исключении «graham», — и с исключениями. Раньше
    здесь стоял голый re.search без исключений, и при запрете лактозы
    снимались «almond milk» и «coconut milk», а при глютене — «rice
    noodles»; человеку при этом говорилось «снято вашим ограничением»,
    хотя фильтр каталога эти блюда пропускал.

    Сам солвер этот конфликт тоже переживёт (предпочтения он ищет
    в УЖЕ отфильтрованном каталоге), но тогда пожелание просто молча
    не сработает. Здесь оно снимается явно — и это видно в профиле.
    """
    # Нутриенты отсеиваем ДО всего остального: они не продукты, и вопрос
    # конфликта с ограничениями к ним не относится.
    wanted = [t for t in profile.include if t.strip().lower() not in _NUTRIENT_WORDS]

    if not wanted or not profile.exclude:
        return wanted

    kept: list[str] = []
    for term in wanted:
        # Раскрываем пожелание и смотрим, не попало ли хоть одно слово
        # под запрет. Достаточно одного: «молочка» без молока — уже
        # не то, о чём просили.
        words = pd.Series(expand([term]) or [term], dtype=object)
        if matches_terms(words, profile.exclude).any():
            continue
        kept.append(term)

    return kept


def _with_russian_fallback(terms: list[str], profile: Profile, catalog: Any) -> list[str]:
    """Заменить английское пожелание русским, если английское не нашло ничего.

    Модель переводит пожелание словом, которого в USDA нет: «semolina»
    вместо «farina», «millet porridge» вместо «millet». Блюдо в каталоге
    есть, а человек слышит «не нашлось». Русские названия есть у всех
    блюд, и на наборе из 48 пожеланий запасной путь закрыл все такие
    промахи (132/132 против 124/132 без него), не дав ни одного ложного
    совпадения там, где продукта в каталоге нет.

    Только ЗАПАСНОЙ путь: поиск по основе русского слова шире английского,
    и там, где английское слово сработало, ему и верим.

    Работает на уже разрешённых пожеланиях: снятое ограничением слово сюда
    не доходит, и его русская пара не воскрешает его.
    """
    result: list[str] = []
    for term in terms:
        russian = profile.include_ru.get(term)
        if russian and not preferred_ids(catalog, [term]):
            result.append(russian)
        else:
            result.append(term)
    return result


# Во сколько дней пересчитывать сумму, названную за период.
#
# Месяц здесь 30 дней, а не 30.44: человек, говорящий «20 тысяч в месяц»,
# имеет в виду круглое число, и точность календаря тут ложная.
_BUDGET_PERIODS: list[tuple[str, float]] = [
    (r"в\s*день|в\s*сутки|ежедневн|/\s*день|в\s*дн", 1.0),
    (r"в\s*недел|за\s*недел|/\s*недел|еженедельн", 7.0),
    (r"в\s*месяц|за\s*месяц|/\s*месяц|ежемесячн", 30.0),
]

# Число с пробелами-разделителями («15 000») и необязательным «тысяч».
_BUDGET_AMOUNT = re.compile(
    r"(\d[\d\s  ]*(?:[.,]\d+)?)\s*(тыс\.?|тысяч[аи]?)?\s*"
    r"(?:руб\w*|₽|р\.)?",
    re.IGNORECASE,
)

# Признаки того, что речь вообще о деньгах. Без них «500 в день» можно
# принять за калории или граммы.
_BUDGET_TRIGGER = re.compile(
    r"руб|₽|деньг|бюджет|потрат|тратить|уложит|стоим|цен[аеу]|дорог|дешёв|дешев",
    re.IGNORECASE,
)


def _budget_from_text(text: str) -> float | None:
    """Дневной бюджет из реплики: «20 тысяч в месяц» → 666.7 ₽ в день.

    Разбор детерминированный, а не моделью, и причина та же, что
    у ``_activity_from_text``: здесь нужен ПЕРЕСЧЁТ, а на арифметике
    модель врёт молча. «15 000 в месяц» она возвращала то как 15000,
    то как 500 — и оба раза уверенно.

    Требуется признак денег («рублей», «бюджет», «потратить»): голое
    «500 в день» — это с тем же успехом калории.

    Возвращает сумму НА ДЕНЬ. Пересчёт делается ровно здесь и один раз,
    дальше по коду ходит только дневная величина.
    """
    if not _BUDGET_TRIGGER.search(text):
        return None

    lowered = text.lower()

    # Ищем период: он задаёт делитель. Без периода считаем, что назван
    # день — «готов тратить 600 рублей» почти всегда про день.
    per_days = 1.0
    period_at = len(lowered)
    for pattern, days in _BUDGET_PERIODS:
        found = re.search(pattern, lowered)
        if found and found.start() < period_at:
            per_days, period_at = days, found.start()

    # Из всех чисел берём то, что стоит ближе всего ПЕРЕД периодом:
    # в «рацион на 7 дней, бюджет 5000 в неделю» чисел два.
    best: float | None = None
    for match in _BUDGET_AMOUNT.finditer(lowered):
        raw = match.group(1).strip()
        if not raw or not raw[0].isdigit():
            continue
        digits = re.sub(r"[\s  ]", "", raw).replace(",", ".")
        try:
            amount = float(digits)
        except ValueError:
            continue
        if match.group(2):  # «тысяч»
            amount *= 1000
        if match.start() <= period_at:
            best = amount

    if best is None or best <= 0:
        return None

    per_day = best / per_days

    # Отсев бессмыслицы: меньше полусотни рублей в день не бывает даже
    # на хлебе, больше двадцати тысяч — это уже не про продукты.
    # Такие числа обычно означают, что разбор зацепил чужую цифру.
    if not 50 <= per_day <= 20_000:
        return None

    return round(per_day, 2)


def _days_from_text(text: str) -> int | None:
    """На сколько дней просят меню — разбором фразы, а не моделью.

    Модель этот параметр теряла: инструкция «не названо — не включай поле»
    перевешивала, и «на месяц» в JSON не попадало ни разу из трёх.
    Здесь терять нечего — либо шаблон совпал, либо нет.
    """
    lowered = text.lower()

    # Явное число: «на 10 дней», «на 3 недели».
    match = re.search(r"на\s+(\d{1,3})\s*(дн|день|дня|дней|недел|месяц)", lowered)
    if match:
        count = int(match.group(1))
        unit = match.group(2)
        if unit.startswith("недел"):
            count *= 7
        elif unit.startswith("месяц"):
            count *= 30
        return count if 1 <= count <= MAX_MENU_DAYS else None

    # Словами. Порядок важен: «на две недели» должно сработать раньше,
    # чем «недел», иначе получится 7 вместо 14.
    for pattern, count in (
        (r"на\s+пол(торы|тора)\s+недел", 10),
        (r"на\s+(две|пару)\s+недел", 14),
        (r"на\s+недел", 7),
        (r"на\s+месяц", 30),
        (r"на\s+день|на\s+один\s+день|на\s+сутки", 1),
    ):
        if re.search(pattern, lowered):
            return count

    return None


def _day_word(count: int) -> str:
    """«1 день», «7 дней», «22 дня»."""
    if 11 <= count % 100 <= 14:
        return "дней"
    return {1: "день", 2: "дня", 3: "дня", 4: "дня"}.get(count % 10, "дней")

FIELD_NAMES_RU = {
    "sex": "пол",
    "age": "возраст",
    "height_cm": "рост в сантиметрах",
    "weight_kg": "вес в килограммах",
    "activity": "как часто тренируетесь",
    "goal": "цель (похудеть / удержать вес / набрать / рекомпозиция)",
}


# ────────────────────────────────────────────────────────────
# Агент
# ────────────────────────────────────────────────────────────


class NutritionAgent:
    """Разговорный агент-нутрициолог.

    Держит профиль пользователя между репликами: человек редко называет
    все параметры сразу, и переспрашивать по кругу то, что уже сказано, —
    худшее, что может делать такой ассистент.

    Args:
        client: LLM-клиент (OpenAI-совместимый).
        model: имя модели для генерации.
        retriever: объект Retriever для поиска по корпусу.
        catalog: каталог блюд; если None — загрузится сам.
        top_k: сколько фрагментов держать в контексте.
        max_refines: сколько раз переформулировать неудачный запрос.
        seed: фиксирует случайность солвера.
        max_retries: сколько раз повторять вызов модели при сбое.
        strict_grade: строгий критерий релевантности в узле ``grade``.
            Размен между полнотой и честностью отказов померен
            в ``experiments/grade.py``, цифры — в EVOLUTION.md.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        retriever: Any,
        catalog: Any = None,
        top_k: int = 5,
        max_refines: int = 1,
        seed: int = 0,
        max_retries: int = 3,
        strict_grade: bool = True,
        answer_style: str = "full",
        speculate: bool = False,
        structured_extract: bool = True,
    ) -> None:
        self.client = client
        # Опережение (см. _Speculation) включает сервис: там ждёт человек.
        # В прогонах метрик и тестах оно выключено — время там никого
        # не держит, а невостребованный вызов на каждом вопросе стоил бы
        # денег без пользы.
        self.speculate = speculate
        # Извлечение профиля по схеме с разбором упоминаний (SGR) —
        # см. EXTRACT_PROFILE_SCHEMA. False — прежний свободный JSON: для
        # провайдера без строгих схем и для замера «до/после».
        self.structured_extract = structured_extract
        # Суточный потолок вызовов модели (limits.LLMBudget) — ставит сервис.
        # В скриптах и прогонах метрик его нет: там тратит владелец сам.
        self.budget: Any = None
        self.model = model
        self.retriever = retriever
        self.catalog = catalog if catalog is not None else load_catalog()
        self.top_k = top_k
        self.max_refines = max_refines
        self.seed = seed
        self.max_retries = max_retries
        self.strict_grade = strict_grade
        self.answer_style = answer_style

        self.corpus_index = load_corpus_index()

        # Память «по умолчанию» — для скриптов и ноутбуков, где разговор один.
        # В сервисе состояние передают в run() явно, см. ConversationState.
        self._own_state = ConversationState()

        self.graph = self._build_graph()

    @property
    def profile(self) -> dict:
        """Профиль разговора по умолчанию (удобный доступ для скриптов)."""
        return self._own_state.profile

    @property
    def history(self) -> list[dict]:
        """История разговора по умолчанию."""
        return self._own_state.history

    @property
    def days(self) -> int:
        """Запрошенный период меню в разговоре по умолчанию."""
        return self._own_state.days

    # ------------------------------------------------------------------
    # Вспомогательное
    # ------------------------------------------------------------------
    def _complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        json_mode: bool = False,
        on_text: Callable[[str], None] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Один вызов модели с повторами при сетевых сбоях.

        Ретраи здесь обязательны: у графа до семи обращений к модели на реплику,
        и без повторов один случайный таймаут провайдера ронял бы весь ответ.

        ``on_text`` — ответ по частям: получает весь накопленный текст после
        каждого куска. При повторе попытки текст начинается заново, и первый
        же кусок новой попытки заменяет показанное — склейки обрывка
        с новым ответом не бывает.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        calls = _LLM_CALLS.get()
        started = time.perf_counter()

        for attempt in range(self.max_retries):
            # Суточный потолок (limits.LLMBudget) — на КАЖДУЮ попытку, вне try:
            # исчерпан — BudgetExhausted уходит наружу без повторов, как
            # «кончились деньги». Повтор после сбоя тоже платный.
            if self.budget is not None:
                self.budget.charge()
            try:
                if on_text is not None:
                    content, usage = self._stream(kwargs, on_text)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content or ""
                    usage = getattr(response, "usage", None)
                if calls is not None:
                    call = {
                        "response": content[:TRACE_RESPONSE_CHARS],
                        "attempts": attempt + 1,
                        "seconds": round(time.perf_counter() - started, 2),
                    }
                    if on_text is not None:
                        call["streamed"] = True
                    # Токены — только если провайдер их вернул. Нет счётчика —
                    # нет поля, а не ноль: «0 токенов» читалось бы как бесплатный вызов.
                    for field in ("prompt_tokens", "completion_tokens"):
                        value = (usage.get(field) if isinstance(usage, dict)
                                 else getattr(usage, field, None))
                        if isinstance(value, int):
                            call[field] = value
                    calls.append(call)
                return content
            except Exception as error:
                last_error = error
                if calls is not None:
                    calls.append({"error": f"{type(error).__name__}: {error}"[:500],
                                  "attempt": attempt + 1})

                # Кончились деньги или квота — повторять бессмысленно.
                # Баланс не пополнится за четыре секунды паузы, а каждая
                # попытка тратит время и засоряет лог трассировкой, из-за
                # которой настоящая причина теряется. Выходим сразу
                # и говорим, что делать.
                if _is_out_of_funds(error):
                    raise RuntimeError(
                        "У провайдера кончились средства или квота — прогон "
                        "остановлен.\nПополните баланс и запустите заново. "
                        "Повторять запрос бесполезно.\n"
                        f"Ответ провайдера: {error}"
                    ) from error

                logger.warning(
                    "Вызов модели не удался (попытка %d из %d): %s",
                    attempt + 1,
                    self.max_retries,
                    error,
                )
                # Экспоненциальная пауза: 1с, 2с, 4с, ...
                time.sleep(2**attempt)

        raise RuntimeError(
            f"Модель не ответила после {self.max_retries} попыток: {last_error}"
        ) from last_error

    @staticmethod
    def _text_sink(state: AgentState) -> Callable[[str], None] | None:
        """Куда отдавать финальный текст по частям — или None, если некуда.

        Только для узлов, чей ответ модели и есть ответ человеку. Сырой
        текст, с номерами PMID: вырезает их тот, кто показывает (api.py),
        — тем же for_reader, что и готовый ответ.
        """
        emit = state.get("on_event")
        if emit is None:
            return None
        return lambda text: emit({"type": "text", "raw": text})

    def _stream(self, kwargs: dict[str, Any], on_text: Callable[[str], None]):
        """Ответ модели по частям. Возвращает (текст, usage или None).

        Поток читается ВРУЧНУЮ и до конца, а не итератором SDK. Замер:
        после итератора SDK в наборе соединений оставалось 0 — он бросает
        чтение на «[DONE]» и закрывает недочитанное, и соединение
        выбрасывается. С зависающим рукопожатием (providers.py) это лишние
        секунды на КАЖДОМ потоковом ответе и на следующем вызове за ним.
        """
        text = ""
        usage = None
        with self.client.chat.completions.with_streaming_response.create(
            **kwargs, stream=True, stream_options={"include_usage": True},
        ) as response:
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        text += piece
                        on_text(text)
        return text, usage

    def _call_options(self, name: str) -> dict[str, Any]:
        """Параметры вызова модели для шага — одни на сам шаг и на вызов наперёд.

        Одно место, а не два: разойдись они — ответ наперёд был бы получен
        в другом режиме и всё равно подставился бы (сверяется только промпт).
        """
        if name == "extract_profile":
            return ({"schema": EXTRACT_PROFILE_SCHEMA} if self.structured_extract
                    else {"json_mode": True})
        return {}

    def _call_in_background(self, prompt: str, options: dict[str, Any]):
        """Вызов модели в потоке пула. Возвращает (ответ, вызовы, ошибка).

        Вызовы для trace копятся в СВОЁМ списке: ContextVar в поток пула
        не переходит, и без этого вызов пропал бы из журнала вместе
        с токенами. Ошибка возвращается, а не бросается, — чтобы узел,
        которому вызов нужен, поднял её у себя, с теми же записями о попытках.
        """
        token = _LLM_CALLS.set([])
        try:
            try:
                return self._complete(prompt, **options), _LLM_CALLS.get(), None
            except Exception as error:
                return None, _LLM_CALLS.get(), error
        finally:
            _LLM_CALLS.reset(token)

    def _ask(self, state: AgentState, name: str, prompt: str) -> str:
        """Ответ модели для узла: готовый из опережения или новый вызов."""
        speculation: _Speculation | None = state.get("speculative")
        ready = speculation.take(name, prompt) if speculation is not None else None
        if ready is None:
            return self._complete(prompt, **self._call_options(name))
        raw, calls = ready
        current = _LLM_CALLS.get()
        if current is not None:
            current.extend({**call, "speculative": True} for call in calls)
        return raw

    def _extract_prompt(self, state: AgentState) -> str:
        known = dict(state.get("profile") or {})
        template = (EXTRACT_PROFILE_PROMPT_SGR if self.structured_extract
                    else EXTRACT_PROFILE_PROMPT)
        return template.format(
            known=json.dumps(known, ensure_ascii=False) if known else "(ничего)",
            question=state["question"],
            history=self._format_history(state.get("history")),
        )

    @staticmethod
    def _rewrite_prompt(state: AgentState) -> str:
        return REWRITE_PROMPT.format(question=state["question"])

    # Сколько символов ответа агента оставлять в истории.
    #
    # Историю получают два узла — classify и extract_profile, — и оба задают
    # по ней вопросы, для которых длинный ответ не нужен: чего хочет человек
    # и какие параметры он назвал. Параметры называет пользователь, а не агент.
    #
    # Замер на реальных ответах: шесть сообщений — это 3280 токенов, и они
    # уходят дважды за реплику. При обрезке до 200 символов остаётся 573,
    # то есть 5414 сэкономленных токенов на каждую реплику разговора.
    #
    # Реплики ПОЛЬЗОВАТЕЛЯ не трогаются: именно в них лежит профиль, и потерять
    # «рост 168» из-за обрезки нельзя.
    HISTORY_ANSWER_CHARS = 200

    def _format_history(self, history: list[dict] | None, limit: int = 6) -> str:
        if not history:
            return "(разговор только начался)"

        lines = []
        for item in history[-limit:]:
            content = item["content"]
            if item["role"] == "assistant" and len(content) > self.HISTORY_ANSWER_CHARS:
                content = content[: self.HISTORY_ANSWER_CHARS] + "…"
            lines.append(f"{item['role']}: {content}")
        return "\n".join(lines)

    def _citation(self, doc_id: int) -> str:
        """Ссылка на статью по её doc_id."""
        meta = self.corpus_index.get(str(doc_id))
        if not meta:
            return "[источник не найден]"
        return f"[PMID {meta['pmid']}] {meta['title']} ({meta['journal']}, {meta['year']})"

    def _format_context(self, chunks: list[dict]) -> str:
        """Собрать фрагменты в текст для промпта — с источником у каждого.

        Номер фрагмента и ссылка на источник намеренно разнесены по разным
        строкам и по-разному подписаны. Когда они стояли рядом в виде
        «[1] [PMID 12345678] Название», модель в ответе ссылалась на номер
        фрагмента — писала «[1][2][3]» вместо PMID, и ссылки становились
        бесполезны: за пределами промпта эти номера ничего не значат.
        """
        blocks = []
        for number, chunk in enumerate(chunks, start=1):
            source = self._citation(int(chunk["doc_id"]))
            blocks.append(f"Фрагмент {number}\nИсточник: {source}\nТекст: {chunk['text']}")
        return "\n\n".join(blocks) if blocks else "(ничего не найдено)"

    # ------------------------------------------------------------------
    # Узлы: маршрутизация
    # ------------------------------------------------------------------
    def _node_classify(self, state: AgentState) -> AgentState:
        """Понять, чего хочет пользователь, и проверить на противопоказания.

        С ``speculate`` следующие шаги обеих веток спрашивают модель, пока
        идёт этот вызов: какой понадобится, станет ясно только из его ответа,
        а ждать его — 1.4-2.3 с на каждой реплике (experiments/latency.py).
        """
        speculation = None
        if self.speculate:
            speculation = _Speculation()
            for name, prompt in (
                ("extract_profile", self._extract_prompt(state)),
                ("rewrite", self._rewrite_prompt(state)),
            ):
                speculation.start(name, prompt, lambda p=prompt, o=self._call_options(name):
                                  self._call_in_background(p, o))

        raw = self._complete(
            CLASSIFY_PROMPT.format(
                question=state["question"],
                history=self._format_history(state.get("history")),
            ),
            json_mode=True,
        )
        parsed = parse_json_response(raw)

        intent = parsed.get("intent")
        if intent not in ("plan", "question", "off_topic"):
            # Модель ответила чем-то неожиданным — считаем это вопросом:
            # худший исход здесь всего лишь лишний поиск по корпусу.
            intent = "question"

        return {
            "intent": intent,
            "red_flag": str(parsed.get("red_flag") or "").strip(),
            "speculative": speculation,
        }

    def _route_after_classify(self, state: AgentState) -> str:
        """Куда идти после классификации.

        ⚠️ Красный флаг блокирует ТОЛЬКО ветку расчёта рациона, но не вопросы
        по литературе. Разница принципиальная:

            «Составь мне рацион, у меня гестационный диабет» — отказ,
            персональное предписание при таком диагнозе даёт врач.

            «Что исследования говорят о питании при гестационном диабете?» —
            обычный вопрос, на него отвечают со ссылками на статьи.

        Пока флаг перехватывал всё подряд, агент отказывался отвечать на 11
        вопросов из 80 в тестовом прогоне — просто потому, что в вопросе
        упоминалась болезнь. Это не осторожность, а бесполезность.
        """
        if state.get("intent") == "off_topic":
            return "off_topic"

        if state.get("intent") == "plan":
            return "red_flag" if state.get("red_flag") else "extract_profile"

        return "rewrite"

    def _node_off_topic(self, state: AgentState) -> AgentState:
        return {"answer": OFF_TOPIC_ANSWER}

    def _node_red_flag(self, state: AgentState) -> AgentState:
        return {"answer": RED_FLAG_TEMPLATE.format(reason=state.get("red_flag", "нужен врач"))}

    # ------------------------------------------------------------------
    # Узлы: ветка расчёта рациона
    # ------------------------------------------------------------------
    def _node_extract_profile(self, state: AgentState) -> AgentState:
        """Достать из разговора параметры человека и дополнить ими профиль."""
        known = dict(state.get("profile") or {})

        raw = self._ask(state, "extract_profile", self._extract_prompt(state))
        extracted = _from_structured(parse_json_response(raw))

        # days — это параметр запроса, а не свойство человека, поэтому
        # в профиль он не кладётся: иначе Profile не прошёл бы валидацию.
        #
        # Разбор фразы идёт первым, ответ модели — только запасной вариант.
        # Модель этот параметр устойчиво теряла: на «составь план питания
        # на месяц» она три раза из трёх не возвращала days вовсе, потому
        # что правило «не названо — не включай» перевешивало. «На месяц» —
        # это не суждение, а сопоставление с шаблоном.
        days = _days_from_text(state["question"]) or extracted.pop("days", None)
        extracted.pop("days", None)

        # Цель рекомпозиции разбираем сами: модель путает её с набором массы
        # и с похудением даже на примерах из собственного промпта.
        # Разбор фразы главнее — он не зависит от настроения модели.
        goal = _goal_from_text(state["question"])
        if goal:
            extracted["goal"] = goal

        # Уровень активности — там же и по той же причине. Отличие в том,
        # что здесь модель не теряет поле, а возвращает неверное значение,
        # и потому ошибка не ловится проверкой обязательных полей.
        activity = _activity_from_text(state["question"])
        if activity:
            extracted["activity"] = activity

        # Бюджет модели не доверяем вовсе: тут нужен ПЕРЕСЧЁТ периода,
        # а на арифметике она врёт молча — «15 000 в месяц» возвращались
        # то как 15000, то как 500, и оба раза уверенно.
        budget = _budget_from_text(state["question"])
        if budget:
            extracted["budget_rub_per_day"] = budget

        # Диета, попавшая в пожелания, — это ограничение (is_restriction_phrase).
        # Промпт об этом просит, но «Только халяль» модель 9 раз из 9 клала
        # в include, и свинина с алкоголем оставались в каталоге. Перенос
        # кодом — по той же причине, что разбор периода и бюджета: правило,
        # которое модель нарушает, держится не промптом.
        wanted = extracted.get("include")
        diets = [t for t in wanted or [] if isinstance(t, str) and is_restriction_phrase(t)]
        if diets:
            extracted["include"] = [t for t in wanted if t not in diets]
            excluded = extracted.get("exclude") or []
            # Строку, а не список, модель тоже присылает — не разбить по буквам.
            excluded = [excluded] if isinstance(excluded, str) else list(excluded)
            extracted["exclude"] = [*excluded, *diets]

        # Русские пары пожеланий — вспомогательное поле, и ронять из-за него
        # весь расчёт нельзя: не словарь строк — просто отбрасываем.
        pairs = extracted.pop("include_ru", None)
        if isinstance(pairs, dict):
            pairs = {str(k): v for k, v in pairs.items()
                     if isinstance(v, str) and v.strip() and k not in diets}
            if pairs:
                extracted["include_ru"] = {**known.get("include_ru", {}), **pairs}

        named_allergies = extracted.pop("allergies", None)

        # Модель возвращает только явно названное, поэтому просто накладываем
        # новое поверх старого. Кроме exclude: его — копить.
        #
        # Наблюдалось (experiments/extract_profile.py, набор multi): на
        # «Люблю курицу, но не ем грибы» после «Молочку не ем» свободный JSON
        # 3/3 возвращал exclude ["mushroom"] — и замена списка молча снимала
        # молочное. Снять исключение репликой («теперь ем свинину») поэтому
        # нельзя — только «удалить мои данные»; аллерген в тарелке дороже.
        for field, value in extracted.items():
            if value in (None, "", []):
                continue
            if field == "exclude":
                value = [value] if isinstance(value, str) else list(value)
                before = list(known.get("exclude") or [])
                seen = {str(t).lower() for t in before}
                value = before + [t for t in value if str(t).lower() not in seen]
            known[field] = value

        said_before = any(
            item.get("role") == "user" and _ALLERGY_RE.search(str(item.get("content", "")))
            for item in state.get("history") or [])
        allergies, exclude = _merge_allergies(
            state["question"], known.get("allergies") or [], named_allergies,
            known.get("exclude") or [], said_before=said_before)
        if allergies:
            known["allergies"], known["exclude"] = allergies, exclude

        missing = [field for field in REQUIRED_PROFILE_FIELDS if field not in known]

        update: AgentState = {"profile": known, "missing": missing}
        if isinstance(days, (int, float)) and 1 <= int(days) <= MAX_MENU_DAYS:
            update["days"] = int(days)

        return update

    def _route_after_profile(self, state: AgentState) -> str:
        if state.get("missing"):
            return "ask_profile"
        # «Съел не по плану» имеет смысл, только когда план уже был.
        if state.get("last_plan") and _EATEN_RE.search(state["question"]):
            return "match_eaten"
        return "make_plan"

    def _route_after_eaten(self, state: AgentState) -> str:
        # Уточнение («на какой приём пищи?») уже написано — дальше не идём.
        return END if state.get("answer") else "make_plan"

    def _eaten_candidates(self, item: dict) -> list[dict]:
        """До двенадцати блюд каталога для одной съеденной позиции — ищет КОД.

        По смыслу (food_search), если есть векторы названий и эмбеддер:
        на живых фразах это 10/16 верных против 7/16 у поиска по словам
        и одно неверное блюдо против четырёх. Иначе — по словам.
        """
        catalog = self.catalog
        embedder = getattr(self.retriever, "embedder", None)
        if embedder is not None and food_search.available():
            text = f"{item.get('phrase_ru', '')} | {item.get('query_en', '')}"
            ids = food_search.nearest(text, embedder)
            rows = catalog.set_index("fdc_id", drop=False)
            return rows.loc[[i for i in ids if i in rows.index],
                            ["fdc_id", "name", "name_ru", "portion_g"]].to_dict("records")
        hits = pd.Series(False, index=catalog.index)
        query = str(item.get("query_en") or "").strip()
        if query and (pattern := to_pattern([query])):
            hits |= matches_words(catalog["name"], pattern)
        # Основа — 4 буквы: «куриц» из «курица» не находил «Куриная грудка».
        stems = [w[:4] for w in re.findall(r"[а-яё]{4,}", str(item.get("phrase_ru", "")).lower())]
        if stems and "name_ru" in catalog.columns:
            names = catalog["name_ru"].fillna("").astype(str).str.lower()
            hits |= names.map(lambda name: any(stem in name for stem in stems)).astype(bool)
        found = catalog[hits]
        # Выше — названия, которые НАЧИНАЮТСЯ со слова человека (по-русски,
        # потом по-английски), затем короткие. Живой прогон: при сортировке
        # только по длине на «курицу» восемь мест заняли кожа, лапки и хвост,
        # а на «гречку с курицей» первым шёл «Soup, chicken».
        first = query.split()[0].lower() if query else "\0"
        ru_first = stems[0] if stems else "\0"
        ru_names = (found["name_ru"].fillna("").astype(str).str.lower()
                    if "name_ru" in found.columns else pd.Series("", index=found.index))
        found = found.assign(
            _ru=~ru_names.str.startswith(ru_first),
            _en=~found["name"].str.lower().str.startswith(first),
            _len=found["name"].str.len(),
        ).sort_values(["_ru", "_en", "_len", "fdc_id"])
        return found.head(12)[["fdc_id", "name", "name_ru", "portion_g"]].to_dict("records")

    def _node_match_eaten(self, state: AgentState) -> AgentState:
        """Сопоставить съеденное с каталогом. Модель выписывает и выбирает;
        ищет кандидатов и считает граммы код."""
        raw = self._complete(EATEN_PROMPT.format(question=state["question"]), json_mode=True)
        items = [i for i in parse_json_response(raw).get("items") or [] if isinstance(i, dict)]
        if not items:
            return {"answer": "Не понял, что именно съедено. Напишите, например: "
                              "«на обед съел пиццу, 300 г»."}
        unknown = [str(i.get("phrase_ru") or "?") for i in items
                   if i.get("meal") not in _MEAL_SLOTS]
        if unknown:
            return {"answer": (f"Уточните, на какой приём пищи было: {', '.join(unknown)} — "
                               "завтрак, обед, ужин или перекус? Тогда пересчитаю остаток дня.")}

        candidates = [self._eaten_candidates(item) for item in items]
        blocks = "\n\n".join(
            f"Позиция {n}: «{item.get('phrase_ru')}»\n" + "\n".join(
                f"  {k}. {c['name']} ({c.get('name_ru') or ''})" for k, c in enumerate(found, 1))
            for n, (item, found) in enumerate(zip(items, candidates, strict=True), 1))
        picked = parse_json_response(self._complete(
            EATEN_PICK_PROMPT.format(blocks=blocks), json_mode=True)).get("choices") or []

        eaten, missing = [], []
        for n, (item, found) in enumerate(zip(items, candidates, strict=True)):
            choice = picked[n] if n < len(picked) else 0
            if not isinstance(choice, int) or not 1 <= choice <= len(found):
                missing.append(str(item.get("phrase_ru") or "?"))
                continue
            record = found[choice - 1]
            # Граммы — по словам человека и таблице порций USDA (portions.py),
            # не справочная порция каталога: та — «1 cup» или целая пицца.
            grams = eaten_grams(int(record["fdc_id"]), item.get("grams"), item.get("count"),
                                float(record["portion_g"]), unit=item.get("unit"))
            eaten.append({"fdc_id": int(record["fdc_id"]), "grams": float(grams),
                          "meal": item["meal"]})
        if missing:
            return {"answer": (f"Не нашёл в справочнике: {', '.join(missing)}. Назовите проще "
                               "(например, «пицца с сыром») — и пересчитаю остаток дня.")}
        return {"eaten_new": eaten}

    def _node_ask_profile(self, state: AgentState) -> AgentState:
        """Спросить недостающие параметры."""
        missing_ru = [FIELD_NAMES_RU.get(field, field) for field in state.get("missing", [])]
        known = state.get("profile") or {}

        answer = self._complete(
            ASK_PROFILE_PROMPT.format(
                known=json.dumps(known, ensure_ascii=False) if known else "(ничего)",
                missing=", ".join(missing_ru),
                activity_options="\n".join(
                    f"  • {label}" for label in ACTIVITY_LABELS.values()
                ),
            ),
            temperature=0.3,
            on_text=self._text_sink(state),
        )
        return {"answer": answer.strip()}

    def _node_make_plan(self, state: AgentState, defer_rest: bool = False) -> AgentState:
        """Детерминированная часть: нормы, сборка дня и проверки.

        Здесь нет ни одного обращения к модели — и это принципиально.
        """
        raw_profile = dict(state.get("profile") or {})

        try:
            profile = Profile(**raw_profile)
        except ValidationError as error:
            # Модель могла вытащить чушь («возраст 250»). Не считаем наугад,
            # а честно возвращаемся к уточнению.
            #
            # У проверок уровня модели (model_validator) поле loc пустое —
            # тогда перечислять «какие поля неверны» нечего, и нужно показать
            # саму причину. Без этой ветки пользователь получал обрубок:
            # «Не получилось разобрать параметры: . Уточните, пожалуйста.»
            bad_fields = sorted({str(item["loc"][0]) for item in error.errors() if item["loc"]})

            if bad_fields:
                # Кривые поля сбрасываем: иначе агент будет спотыкаться о них
                # на каждой следующей реплике.
                for field in bad_fields:
                    raw_profile.pop(field, None)
                reason = ", ".join(FIELD_NAMES_RU.get(field, field) for field in bad_fields)
                message = f"Не получилось разобрать параметры: {reason}. Уточните, пожалуйста."
            else:
                reasons = "; ".join(item.get("msg", "") for item in error.errors())
                message = f"Параметры противоречат друг другу: {reasons}. Уточните, пожалуйста."

            return {"profile": raw_profile, "missing": bad_fields or ["profile"], "answer": message}

        # Расход — по динамике веса, когда данных хватает (adaptive.py),
        # иначе формула. Предохранители нормы действуют в обоих случаях.
        adaptive = adapted_tdee(
            state.get("weighins") or {}, state.get("intake") or {},
            formula_tdee=total_expenditure(profile),
            fallback_kcal=compute_targets(profile).kcal,
            previous=state.get("adaptive"), today=datetime.now(UTC).date())
        targets = compute_targets(profile, tdee_override=adaptive["tdee"] if adaptive else None)

        # Период запрашивают в реплике («на месяц»); по умолчанию — один день.
        days = int(state.get("days") or 1)

        # Если человек тренируется, дни разводятся: в тренировочный больше
        # углеводов и калорий, в выходной меньше, недельное среднее — норма.
        # Для сидячего образа жизни разводить нечего, и cycle_targets
        # возвращает None.
        cycled = cycle_targets(profile, targets)
        per_day = ([cycled.for_day(i) for i in range(days)] if cycled else targets)

        # Норма ИМЕННО ТОГО дня, который показываем. Без неё карточка плана
        # сравнивала тренировочный день с базовой нормой и писала «отклонение
        # 10%» там, где день сошёлся со своей целью с точностью 0.1%.
        shown_targets = per_day[0] if cycled else targets

        if state.get("eaten_new"):
            return {"profile": profile.model_dump(), "targets": targets,
                    "plan_targets": shown_targets, "cycled": cycled, "days": days,
                    "adaptive": adaptive,
                    **self._rebuild_after_eaten(state, profile, shown_targets)}

        swapped = self._try_swap(state, profile)
        if swapped is not None:
            return {"profile": profile.model_dump(), "targets": targets,
                    "plan_targets": shown_targets, "cycled": cycled, "days": days,
                    "adaptive": adaptive, **swapped}

        all_targets = menu_targets(per_day, days)
        day_plans = menu_days(
            self.catalog,
            all_targets,
            exclude=profile.exclude,
            include=_with_russian_fallback(
                _resolve_preferences(profile), profile, self.catalog),
            budget=profile.budget_rub_per_day,
            seed=self.seed,
        )
        try:
            first = next(day_plans)
            # Остальные дни — в фоне, пока модель объясняет первый: объяснение
            # ничего, кроме первого дня, не читает. Ждёт их explain_plan уже
            # ПОСЛЕ вызова модели (_finish_menu). Вне графа (replay_plan,
            # скрипты) — сразу, чтобы узел сам по себе отдавал меню целиком.
            if defer_rest and days > 1:
                rest: Future | list[DayPlan] = _MENU_POOL.submit(list, day_plans)
            else:
                rest = list(day_plans)
        except PlanNotFeasible as error:
            return self._infeasible(profile, targets, error)

        result = {
            "profile": profile.model_dump(),
            "targets": targets,
            "plan_targets": shown_targets,
            "cycled": cycled,
            "plan": first,
            "days": days,
            "checks": check_plan(first),
            "adaptive": adaptive,
        }
        if isinstance(rest, Future):
            result["menu_rest"] = (rest, all_targets)
        else:
            result["menu"] = Menu(days=[first, *rest], targets=mean_targets(all_targets))
        return result

    def _rebuild_after_eaten(self, state: AgentState, profile: Profile,
                             targets: Targets) -> AgentState:
        """Съеденное за день зафиксировать, остаток дня собрать под остаток нормы."""
        everything = [*(state.get("eaten_before") or []), *state["eaten_new"]]
        covered: set[int] = set()
        slots: list[int] = []
        for entry in everything:
            options = _MEAL_SLOTS[entry["meal"]]
            if entry["meal"] == "snack":
                # Перекусов два слота: второй перекус — во второй.
                options = [next((i for i in options if i not in covered), options[-1])]
            covered.update(options)
            slots.append(options[0])
        items = [eaten_item(self.catalog, e["fdc_id"], e["grams"], DAY_TEMPLATE[slot][0])
                 for e, slot in zip(everything, slots, strict=True)]
        shown = items[len(everything) - len(state["eaten_new"]):]
        try:
            plan = rebuild_rest(
                self.catalog, targets, items, covered, exclude=profile.exclude,
                include=_with_russian_fallback(
                    _resolve_preferences(profile), profile, self.catalog),
                budget=profile.budget_rub_per_day, seed=self.seed)
        except PlanNotFeasible as error:
            # Съеденное всё равно запоминается: оно было, и для адаптивной
            # нормы важно, сколько человек на самом деле съел.
            return {"eaten_all": everything, "plan": None, "answer": (
                f"Учёл съеденное. {error} На сегодня норма набрана — новых приёмов "
                "пищи я не добавляю.")}
        return {"plan": plan, "checks": check_plan(plan), "eaten_all": everything,
                "eaten_shown": shown, "menu": Menu(days=[plan], targets=dict(plan.targets))}

    def _try_swap(self, state: AgentState, profile: Profile) -> AgentState | None:
        """«Замени творог на что-нибудь» — заменить ОДНО блюдо последнего плана.

        None — это не просьба о замене (или заменять не во что), и день
        собирается заново, как раньше. Блюдо ищет код — по русскому названию
        в плане или по приёму пищи; промпт маршрутизатора не меняется.

        Если в той же реплике появилось ограничение, под которое не подходят
        ДРУГИЕ блюда плана («замени творог, и я не ем молочное»), одной
        заменой не обойтись — день собирается заново.
        """
        last = state.get("last_plan")
        if not last or not _SWAP_RE.search(state["question"]):
            return None
        plan = DayPlan(**last)
        index = _swap_index(state["question"], plan)
        if index is None and _AGAIN_RE.search(state["question"]):
            index = state.get("swap_index")
        if index is None:
            return None
        allowed = set(_filter_catalog(self.catalog, profile.exclude)["fdc_id"].astype(int))
        if any(item.fdc_id not in allowed for j, item in enumerate(plan.items) if j != index):
            return None

        rejected = [int(i) for i in state.get("swap_rejected") or []]
        old = plan.items[index]
        try:
            new = swap_item(self.catalog, plan, index, exclude=profile.exclude,
                            seed=self.seed + len(rejected), avoid_ids=set(rejected))
        except PlanNotFeasible as error:
            return {"plan": plan, "checks": check_plan(plan), "answer": (
                f"Не получилось заменить «{old.title}»: {error} План оставил прежним — "
                "можно попросить заменить другое блюдо или собрать день заново.")}
        return {"plan": new, "checks": check_plan(new),
                "menu": Menu(days=[new], targets=dict(new.targets)),
                "swap": {"index": index, "old": old.fdc_id}}

    def _node_make_plan_in_graph(self, state: AgentState) -> AgentState:
        return self._node_make_plan(state, defer_rest=True)

    @staticmethod
    def _finish_menu(state: AgentState) -> Menu | None:
        """Меню целиком: дождаться дней, собранных в фоне за время вызова модели."""
        if state.get("menu") is not None:
            return state["menu"]
        pending = state.get("menu_rest")
        if pending is None:
            return None
        future, all_targets = pending
        return Menu(days=[state["plan"], *future.result()], targets=mean_targets(all_targets))

    @staticmethod
    def _infeasible(profile: Profile, targets: Targets, error: PlanNotFeasible) -> AgentState:
        # Каталог после исключений слишком узкий. Это не поломка, а нормальный
        # исход: человек мог исключить половину еды. Отвечаем по-человечески.
        excluded = ", ".join(profile.exclude) if profile.exclude else "—"
        return {
            "profile": profile.model_dump(),
            "targets": targets,
            "answer": (
                "Не получилось собрать рацион: после ваших ограничений "
                f"({excluded}) в справочнике почти не осталось подходящих блюд. "
                f"{error} Попробуйте убрать часть ограничений.\n\n"
                f"Норму при этом я посчитал:\n{explain(profile, targets)}"
            ),
        }

    def _route_after_plan(self, state: AgentState) -> str:
        # Если make_plan упёрся в кривой профиль — он уже написал ответ.
        return END if state.get("answer") else "explain_plan"

    def _explain_with_model(self, state: AgentState, plan: DayPlan, targets: Targets,
                            profile: Profile, days: int) -> tuple[str, list, str]:
        """Объяснение плана моделью: (текст, фрагменты статей, поисковый запрос)."""

        # Ищем обоснование норм — но провал поиска не должен ломать выдачу
        # плана: план посчитан детерминированно и верен сам по себе.
        query = self._norms_query(profile)
        try:
            chunks = self.retriever.retrieve(query, k=self.top_k)
        except Exception:
            chunks = []

        failed = [name for name, passed in state.get("checks", {}).items() if not passed]
        checks_text = "все проверки пройдены" if not failed else f"не прошли: {', '.join(failed)}"

        # На длинный период в чат идёт ОДИН день как образец. Печатать
        # тридцать дней в переписку бессмысленно: это несколько тысяч строк,
        # которые невозможно прочитать и некуда сохранить. Полное меню
        # отдаётся файлом — см. Menu.render() и кнопку скачивания.
        period_note = ""
        if days > 1:
            period_note = (
                f"\n\nЭто меню на {days} {day_word(days)}. В карточке показан "
                f"первый день как образец — скажи об этом и добавь, что полное "
                f"меню можно скачать кнопкой рядом с ответом. Не перечисляй "
                f"остальные дни в тексте."
            )

        # Реплика идёт в промпт, иначе вопрос, пришедший в ветку плана,
        # остаётся без ответа. Наблюдалось в журнале чата: после плана
        # «Сколько белка нужно при похудении?» маршрутизатор стабильно
        # считает просьбой о расчёте (цель уже известна), и человек получал
        # тот же план повторно — ни слова о белке.
        answer = self._complete(
            EXPLAIN_PLAN_PROMPT.format(
                question=state.get("question", ""),
                profile=self._describe_profile(profile),
                targets_explained=explain(profile, targets),
                plan=plan.render() + period_note,
                checks=checks_text,
                context=self._format_context(chunks),
            ),
            temperature=0.3,
            on_text=self._text_sink(state),
        )

        return answer.strip(), chunks, query

    def _node_explain_plan(self, state: AgentState) -> AgentState:
        """Показать план и объяснить цифры, подкрепив их литературой."""
        targets: Targets = state["targets"]
        plan: DayPlan = state["plan"]
        days = int(state.get("days") or 1)
        profile = Profile(**state["profile"])

        # Замена одного блюда — ответ шаблоном кода, без модели: план тот же,
        # кроме одного блюда, и пересказывать его незачем (ревью R8).
        swap = state.get("swap")
        if swap:
            before = DayPlan(**state["last_plan"])
            text = swap_note(before.items[swap["index"]], plan, before)
            chunks, query = [], None
        elif state.get("eaten_shown"):
            # Съеденное — тоже шаблоном: числа дня только из солвера.
            text = eaten_note(state["eaten_shown"], plan, state.get("checks") or {})
            chunks, query = [], None
        else:
            text, chunks, query = self._explain_with_model(state, plan, targets, profile, days)

        # Дни со второго собирались, пока модель писала объяснение.
        # Не собрались — ответ тот же, что раньше давал make_plan: объяснение
        # первого дня без остального меню выдало бы неделю, которой нет.
        try:
            menu = self._finish_menu(state)
        except PlanNotFeasible as error:
            # План первого дня снимается: иначе карточка показала бы его
            # рядом с текстом «не получилось собрать рацион».
            return {**self._infeasible(profile, targets, error),
                    "plan": None, "menu": None, "menu_rest": None}

        # Бюджет — раньше общей стоимости: человек спрашивал «уложился ли»,
        # а не «сколько вышло». Сумма идёт следом как обоснование.
        budget = self._budget_note(plan, days)
        if budget:
            text += "\n\n" + budget

        cost = self._cost_note(menu or plan)
        if cost:
            text += cost

        caution = self._caution_note(profile, menu, plan)
        if caution:
            text += caution

        # В КАЖДОМ ответе с планом, пока в профиле есть аллергия: новый
        # план — новые блюда, и про этикетку надо сказать для них тоже.
        text += allergy_note(profile.allergies)

        # Сначала — о непонятых пожеланиях, потом предложение назвать новые.
        # Обратный порядок выглядел бы издевательством: «назовите любимые
        # продукты» сразу после «ваш любимый продукт я не нашёл».
        note = self._unmatched_note(profile)
        if note:
            text += note

        offer = self._preferences_offer(profile, state.get("history") or [])
        if offer:
            text += offer

        return {"answer": text, "chunks": chunks, "search_query": query,
                "menu": menu, "menu_rest": None}

    @staticmethod
    def _budget_note(plan: DayPlan, days: int) -> str:
        """Уложились ли в названный бюджет — прямым текстом.

        Молчать нельзя. Человек назвал сумму, и если рацион в неё не влез,
        он обязан узнать это от агента, а не сложив цены сам. Это ровно
        тот класс тихих отказов, который в проекте запрещён.

        Перерасход называется в рублях и в процентах: «на 40 ₽» и «на 16%»
        отвечают на разные вопросы, а вместе показывают, стоит ли вообще
        что-то менять.
        """
        budget = plan.budget
        if not budget:
            return ""

        # Стоимость не посчитана — так и говорим, а не «уложились»
        # или «не уложились»: ни то ни другое не известно.
        if plan.cost is None:
            return ("⚠️ Проверить бюджет не получилось: для части блюд нет цены, "
                    "и сумму дня я называть не стану.")

        if plan.within_budget:
            left = budget - plan.cost
            text = f"В ваш бюджет уложился: {plan.cost:.0f} ₽ из {budget:.0f} ₽ в день"
            if left >= budget * 0.1:
                text += f" — остаётся ещё около {left:.0f} ₽"
            return text + "."

        over = plan.cost - budget
        text = (f"⚠️ В бюджет уложиться не удалось: {plan.cost:.0f} ₽ в день "
                f"против ваших {budget:.0f} ₽ — перерасход {over:.0f} ₽ "
                f"({over / budget:.0%}).")

        # Называем причину, а не только факт: при такой норме дешевле
        # уже не собрать из того, что есть в справочнике.
        text += (" Это минимум, который получился при вашей норме калорий "
                 "и ограничениях. Снизить можно, если убрать ограничения "
                 "или пересмотреть цель по весу.")

        if days > 1:
            text += f" За {days} {day_word(days)} выйдет около {plan.cost * days:.0f} ₽."

        return text

    def _cost_note(self, menu_or_plan: Any) -> str:
        """Во что примерно обойдётся рацион.

        Текст собирает `ShoppingList.cost_estimate` — код, а не модель.
        Просить модель назвать сумму нельзя по главному правилу проекта:
        ни одно число в ответе не приходит от LLM. К тому же она охотно
        округлит «около 4 300» до «4 297 ₽» и тем соврёт точностью.

        Провал оценки не должен ронять выдачу плана: план посчитан
        детерминированно и верен сам по себе, а цена — довесок.
        """
        try:
            shopping = build_shopping_list(menu_or_plan, self.catalog)
            estimate = shopping.cost_estimate()
        except Exception:
            logger.warning("Не удалось оценить стоимость рациона", exc_info=True)
            return ""

        return "\n\n" + estimate if estimate else ""

    @staticmethod
    def _caution_note(profile: Profile, menu: Any, plan: DayPlan) -> str:
        """Назвать спорные продукты плана: овёс при глютене, желатин при свинине.

        Текстом из кода, а не просьбой к модели: предупреждение об аллергене
        не должно зависеть от того, вспомнит ли о нём генерация.
        """
        days = menu.days if menu is not None else [plan]
        items = [item for day in days for item in day.items]
        lines = []
        for label, text, names in plan_cautions(items, profile.exclude):
            shown = ", ".join(f"«{name}»" for name in names[:4])
            more = f" и ещё {len(names) - 4}" if len(names) > 4 else ""
            lines.append(f"⚠️ {label.capitalize()} в плане: {shown}{more}. {text}.")
        return "\n\n" + "\n".join(lines) if lines else ""

    def _unmatched_note(self, profile: Profile) -> str:
        """Сказать вслух, какие пожелания не удалось выполнить.

        Иначе отказ получается молчаливым, а он здесь вероятен: каталог
        построен на USDA и русские привычные продукты покрывает неровно.
        Гречка в нём — ОДНО блюдо, творог — четыре, тогда как курица —
        четыреста. Человек, попросивший гречку, план без неё прочтёт
        как «меня проигнорировали», и будет прав.

        Две разные причины молчания и оба случая названы своими словами:

        * пожелание снято ограничением — человек сам просил и то и другое;
        * в справочнике такого блюда нет — вина каталога, не человека.

        Про «нашлось, но не попало в день» не пишем: пожелание мягкое
        и в конкретный день попасть не обязано, а извинение за каждый
        такой случай превратит ответ в отчёт солвера.
        """
        if not profile.include:
            return ""

        allowed = _resolve_preferences(profile)

        # Нутриенты («побольше белка») из отчёта исключаем молча: норму
        # по ним считает targets.py, и фраза «не добавил в рацион protein»
        # человека только запутает — он просил не продукт.
        blocked = [
            term for term in profile.include
            if term not in allowed and term.strip().lower() not in _NUTRIENT_WORDS
        ]

        # Не нашлось ни по-английски, ни запасным путём по-русски. Слово
        # после запасного пути уже русское, если пара была, — так человеку
        # и показываем: он просил «ряженку», а не «ryazhenka».
        missing = [
            term for term in _with_russian_fallback(allowed, profile, self.catalog)
            if not preferred_ids(self.catalog, [term])
        ]

        parts: list[str] = []
        if blocked:
            parts.append(
                f"Не добавил в рацион {', '.join(blocked)} — это противоречит "
                f"вашим ограничениям ({', '.join(profile.exclude)})."
            )
        if missing:
            parts.append(
                f"В справочнике не нашлось подходящих блюд: {', '.join(missing)}. "
                f"Каталог собран на базе USDA, и часть привычных продуктов "
                f"в нём представлена скудно."
            )

        return "\n\n" + " ".join(parts) if parts else ""

    @staticmethod
    def _preferences_offer(profile: Profile, history: list[dict]) -> str:
        """Предложить назвать любимые продукты — один раз и ПОСЛЕ плана.

        Почему после, а не в опроснике. Пока человек не увидел рацион,
        вопрос «какие продукты хотите видеть» отвечать не на что: набор
        блюд неизвестен, и в ответ приходит либо молчание, либо «ну,
        обычные». Увидев конкретный список, человек сразу знает, чего
        в нём не хватает.

        Почему текст фиксированный, а не от модели. Предложение должно
        быть дословно одним и тем же, иначе проверка «не предлагали ли
        уже» по истории перестанет срабатывать. Это ровно тот случай,
        когда формулировка — часть контракта, а не украшение.
        """
        if profile.include:
            return ""

        # Предлагали в этом разговоре — не повторяем. Иначе на каждый
        # пересчёт плана человек получал бы одно и то же предложение.
        if any(PREFERENCES_OFFER in (turn.get("content") or "")
               for turn in history if turn.get("role") == "assistant"):
            return ""

        return "\n\n" + PREFERENCES_OFFER

    def _describe_profile(self, profile: Profile) -> str:
        text = (
            f"{'женщина' if profile.sex == 'female' else 'мужчина'}, {profile.age} лет, "
            f"{profile.height_cm:.0f} см, {profile.weight_kg:.0f} кг, "
            f"активность {profile.activity} (×{ACTIVITY_FACTORS[profile.activity]}), "
            f"цель — {GOAL_LABELS[profile.goal]}"
        )
        if profile.exclude:
            text += f", не ест: {', '.join(profile.exclude)}"

        # Показываем пожелания ПОСЛЕ снятия конфликтов, а не как пришли
        # от модели: иначе в карточке окажется «хочет молоко, не ест
        # лактозу», и человек будет искать молоко в плане, где его нет.
        wanted = _resolve_preferences(profile)
        if wanted:
            text += f", хочет видеть: {', '.join(wanted)}"
        return text

    def _norms_query(self, profile: Profile) -> str:
        """Поисковый запрос под обоснование норм — сразу на английском."""
        if profile.goal == "lose":
            return "protein requirement and safe rate of weight loss during energy restriction"
        if profile.goal == "gain":
            return "protein and energy requirements for lean mass gain"
        return "energy and protein requirements for weight maintenance in adults"

    # ------------------------------------------------------------------
    # Узлы: ветка вопроса по литературе
    # ------------------------------------------------------------------
    def _node_rewrite(self, state: AgentState) -> AgentState:
        """Перевести вопрос на английский и превратить в поисковый запрос."""
        query = self._ask(state, "rewrite", self._rewrite_prompt(state))
        query = query.strip().strip('"')
        return {"search_query": query or state["question"], "refine_count": 0}

    def _node_retrieve(self, state: AgentState) -> AgentState:
        chunks = self.retriever.retrieve(state["search_query"], k=self.top_k)
        return {"chunks": chunks}

    def _node_grade(self, state: AgentState) -> AgentState:
        """Проверить, отвечает ли найденное на вопрос."""
        chunks = state.get("chunks") or []
        if not chunks:
            return {"has_relevant": False}

        prompt = GRADE_PROMPT_STRICT if self.strict_grade else GRADE_PROMPT
        raw = self._complete(
            prompt.format(
                question=state["question"],
                context=self._format_context(chunks),
            ),
            json_mode=True,
        )
        parsed = parse_json_response(raw)

        has_relevant = bool(parsed.get("has_relevant"))

        # Оставляем только те фрагменты, которые модель назвала полезными:
        # мусор в контексте портит и ответ, и faithfulness.
        keep = parsed.get("keep") or []
        if has_relevant and keep:
            indices = [int(number) - 1 for number in keep if isinstance(number, (int, float))]
            filtered = [chunks[i] for i in indices if 0 <= i < len(chunks)]
            if filtered:
                return {"has_relevant": True, "chunks": filtered}

        return {"has_relevant": has_relevant}

    def _route_after_grade(self, state: AgentState) -> str:
        if state.get("has_relevant"):
            return "generate"
        if state.get("refine_count", 0) < self.max_refines:
            return "refine"
        return "cannot_answer"

    def _node_refine(self, state: AgentState) -> AgentState:
        query = self._complete(
            REFINE_PROMPT.format(
                question=state["question"],
                search_query=state.get("search_query", ""),
            )
        )
        return {
            "search_query": query.strip().strip('"'),
            "refine_count": state.get("refine_count", 0) + 1,
        }

    def _node_generate(self, state: AgentState) -> AgentState:
        prompt = ANSWER_PROMPTS[self.answer_style].format(
            question=state["question"],
            context=self._format_context(state.get("chunks") or []),
        )
        personal = self._personal_norms(state.get("profile"),
                                        (state.get("adaptive") or {}).get("tdee"))
        if personal:
            prompt = prompt.replace(FRAGMENTS_HEADER, personal + FRAGMENTS_HEADER, 1)
        answer = self._complete(prompt, temperature=0.2, on_text=self._text_sink(state))
        return {"answer": answer.strip()}

    def _personal_norms(self, raw_profile: dict | None, tdee: float | None = None) -> str:
        """Расчёт нормы человека для ветки вопросов — или пусто.

        Наблюдалось: после плана «А почему так мало углеводов?» уходило
        в вопросы, и человек получал рассказ о низкоуглеводных диетах
        вообще, а не о своей норме — хотя она посчитана и объяснима.
        Маршрутизатор тут не трогается: на вопросах оценки профиля нет,
        блок не вставляется, и промпт совпадает с прежним до байта.
        """
        # Неполный профиль — не считаем: у Profile есть умолчания (активность),
        # и норма по ним была бы домыслом, а не расчётом.
        if not raw_profile or any(f not in raw_profile for f in REQUIRED_PROFILE_FIELDS):
            return ""
        try:
            profile = Profile(**raw_profile)
        except ValidationError:
            return ""
        return PERSONAL_NORMS_BLOCK.format(
            profile=self._describe_profile(profile),
            # Та же норма, что в плане: с фактическим расходом, если он есть.
            targets_explained=explain(profile, compute_targets(profile, tdee_override=tdee)),
        )

    def _node_cannot_answer(self, state: AgentState) -> AgentState:
        return {"answer": CANNOT_ANSWER_TEMPLATE}

    # ------------------------------------------------------------------
    # Сборка графа
    # ------------------------------------------------------------------
    @staticmethod
    def _traced(name: str, node):
        """Обернуть узел: записать в trace его время и ответы модели в нём.

        Ответ модели — главное для разбора тихих багов: «include=["protein"]»
        отсеивается дальше по коду, и по итоговому профилю его уже не видно.
        """
        def wrapper(state: AgentState) -> AgentState:
            emit = state.get("on_event")
            if emit is not None and name in STAGE_LABELS:
                emit({"type": "stage", "node": name, "label": STAGE_LABELS[name]})
            token = _LLM_CALLS.set([])
            started = time.perf_counter()
            try:
                update = node(state) or {}
            finally:
                calls = _LLM_CALLS.get() or []
                _LLM_CALLS.reset(token)
            step = {"node": name, "seconds": round(time.perf_counter() - started, 2)}
            if calls:
                step["llm"] = calls
            return {**update, "trace": [step]}

        return wrapper

    def _build_graph(self):
        builder = StateGraph(AgentState)
        nodes = {
            "classify": self._node_classify,
            "off_topic": self._node_off_topic,
            "red_flag": self._node_red_flag,

            "extract_profile": self._node_extract_profile,
            "match_eaten": self._node_match_eaten,
            "ask_profile": self._node_ask_profile,
            "make_plan": self._node_make_plan_in_graph,
            "explain_plan": self._node_explain_plan,

            "rewrite": self._node_rewrite,
            "retrieve": self._node_retrieve,
            "grade": self._node_grade,
            "refine": self._node_refine,
            "generate": self._node_generate,
            "cannot_answer": self._node_cannot_answer,
        }
        for name, node in nodes.items():
            builder.add_node(name, self._traced(name, node))

        builder.set_entry_point("classify")

        builder.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "off_topic": "off_topic",
                "red_flag": "red_flag",
                "extract_profile": "extract_profile",
                "rewrite": "rewrite",
            },
        )

        builder.add_conditional_edges(
            "extract_profile",
            self._route_after_profile,
            {"ask_profile": "ask_profile", "make_plan": "make_plan",
             "match_eaten": "match_eaten"},
        )
        builder.add_conditional_edges(
            "match_eaten",
            self._route_after_eaten,
            {"make_plan": "make_plan", END: END},
        )
        builder.add_conditional_edges(
            "make_plan",
            self._route_after_plan,
            {"explain_plan": "explain_plan", END: END},
        )

        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"generate": "generate", "refine": "refine", "cannot_answer": "cannot_answer"},
        )
        builder.add_edge("refine", "retrieve")

        for node in ("off_topic", "red_flag", "ask_profile", "explain_plan",
                     "generate", "cannot_answer"):
            builder.add_edge(node, END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------
    def run(
        self,
        question: str,
        verbose: bool = False,
        dataset_row_id: str | None = None,
        remember: bool = True,
        state: ConversationState | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> RAGAgentAnswer:
        """Ответить на реплику пользователя.

        Args:
            on_event: события для ответа по частям — {"type": "stage", ...}
                в начале каждого шага и {"type": "text", "raw": ...} по мере
                того, как модель пишет финальный ответ. Вызывается из потока,
                в котором идёт ответ. Итоговый ответ от этого не меняется.
            remember: сохранять ли профиль и историю для следующих реплик.
                На прогоне метрик это выключают: вопросы там независимы,
                и профиль от одного не должен протекать в другой.
            state: память конкретного разговора. Если не передана, агент
                использует собственную — так удобнее в скриптах и ноутбуках,
                но в сервисе на несколько пользователей всегда передавайте
                состояние явно, иначе профили перемешаются.

        Returns:
            RAGAgentAnswer — контракт, который принимает модуль evaluation.py.
        """
        memory = state if state is not None else self._own_state

        final_state = self.graph.invoke(
            {
                "question": question,
                "history": list(memory.history) if remember else [],
                "profile": dict(memory.profile) if remember else {},
                "days": memory.days if remember else 1,
                "on_event": on_event,
                "last_plan": memory.last_plan if remember else None,
                "swap_rejected": list(memory.swap_rejected) if remember else [],
                "swap_index": memory.swap_index if remember else None,
                # Вчерашнее съеденное к сегодняшнему дню отношения не имеет.
                "eaten_before": (list(memory.eaten)
                                 if remember and memory.eaten_date == _today_utc() else []),
                "weighins": dict(memory.weighins) if remember else {},
                "intake": dict(memory.intake) if remember else {},
                "adaptive": memory.adaptive if remember else None,
            }
        )

        answer = final_state.get("answer", "")

        if remember:
            memory.profile = final_state.get("profile") or memory.profile
            memory.days = int(final_state.get("days") or memory.days)
            shown = final_state.get("plan")
            if shown is not None:
                swap = final_state.get("swap")
                # Новый день — заменять в нём ещё нечего; замена — копим
                # отвергнутые блюда, чтобы «замени ещё раз» их не вернуло.
                memory.swap_rejected = (
                    [*memory.swap_rejected, swap["old"]] if swap else [])
                memory.swap_index = swap["index"] if swap else None
                memory.last_plan = shown.model_dump(mode="json")
            # Съеденное: копится в пределах суток; новый день целиком
            # (не замена и не пересчёт после съеденного) его сбрасывает.
            if final_state.get("eaten_all") is not None:
                memory.eaten, memory.eaten_date = final_state["eaten_all"], _today_utc()
            elif shown is not None and not final_state.get("swap"):
                memory.eaten, memory.eaten_date = [], None
            self._remember_progress(memory, question, final_state)
            memory.history.append({"role": "user", "content": question})
            memory.history.append({"role": "assistant", "content": answer})

        if verbose:
            print(f"Реплика: {question}")
            print(f"  намерение: {final_state.get('intent')}")
            if final_state.get("red_flag"):
                print(f"  красный флаг: {final_state['red_flag']}")
            if final_state.get("missing"):
                print(f"  не хватает: {final_state['missing']}")
            if final_state.get("plan") is not None:
                print(f"  проверки плана: {final_state.get('checks')}")
            print(f"  поисковый запрос: {final_state.get('search_query')}")
            print(f"  чанков в контексте: {len(final_state.get('chunks') or [])}")

        chunks = final_state.get("chunks") or []
        retrieved = [
            Chunk(
                text=chunk["text"],
                chunk_id=int(chunk["chunk_id"]),
                year=int(chunk["doc_id"]),
                distance=float(chunk.get("distance", 0.0)),
            )
            for chunk in chunks
        ]

        menu = final_state.get("menu")

        # Достаточно, чтобы повторить план без модели: числа считает
        # детерминированный код, и профиль + seed + days дают то же меню
        # до блюда (chat_log.replay_plan). Через /plan и /menu — НЕ то же:
        # они не учитывают пожелания, бюджет и тренировочные дни.
        trace = {
            "intent": final_state.get("intent"),
            "red_flag": final_state.get("red_flag") or None,
            "missing": final_state.get("missing") or None,
            "profile": final_state.get("profile"),
            "days": final_state.get("days"),
            "seed": self.seed,
            "model": self.model,
            "search_query": final_state.get("search_query"),
            "refine_count": final_state.get("refine_count"),
            "chunks": [
                {"doc_id": int(c["doc_id"]), "chunk_id": int(c["chunk_id"]),
                 "distance": round(float(c.get("distance", 0.0)), 4)}
                for c in chunks
            ] or None,
            "checks": final_state.get("checks"),
            "plan_days": [[item.fdc_id for item in day.items] for day in menu.days]
            if menu is not None else None,
            "nodes": (final_state.get("trace") or []) + (
                final_state["speculative"].unused()
                if final_state.get("speculative") is not None else []),
        }

        return RAGAgentAnswer(
            dataset_row_id=dataset_row_id,
            answer=answer,
            retrieved_chunks=retrieved or None,
            plan=final_state.get("plan"),
            targets=final_state.get("targets"),
            plan_targets=final_state.get("plan_targets"),
            cycled=final_state.get("cycled"),
            menu=menu,
            menu_days=len(menu.days) if menu is not None else 0,
            trace=trace,
        )

    @staticmethod
    def _remember_progress(memory: ConversationState, question: str, final_state: dict) -> None:
        """Взвешивание и съеденное за сегодня — для адаптивной нормы (adaptive.py).

        Вес записывается, только если число из ЭТОЙ реплики совпало с весом
        профиля: вопрос «сколько белка при весе 80 кг?» про другого человека
        в журнал взвешиваний не попадёт. Съеденное — калорийность показанного
        дня (после «съел не по плану» — с поправкой). Хранится 90 дней.
        """
        today = _today_utc()
        weight = (final_state.get("profile") or {}).get("weight_kg")
        said = [float((m.group(1) or m.group(2)).replace(",", "."))
                for m in _WEIGHT_RE.finditer(question)]
        if weight and any(abs(n - float(weight)) < 0.05 for n in said):
            memory.weighins[today] = float(weight)
        plan = final_state.get("plan")
        if plan is not None:
            memory.intake[today] = float(plan.totals["kcal"])
        if final_state.get("adaptive"):
            memory.adaptive = final_state["adaptive"]
        border = (datetime.now(UTC).date() - timedelta(days=90)).isoformat()
        memory.weighins = {d: w for d, w in memory.weighins.items() if d >= border}
        memory.intake = {d: k for d, k in memory.intake.items() if d >= border}

    def reset(self) -> None:
        """Забыть разговор и профиль по умолчанию — начать с чистого листа."""
        self._own_state = ConversationState()


# ────────────────────────────────────────────────────────────
# Удобная сборка «под ключ»
# ────────────────────────────────────────────────────────────


def build_agent(
    client: Any,
    model: str = "openai/gpt-4o-mini",
    db_path: str = "lance_db/vectorstore",
    table_name: str = "chunks",
    # BM25 ВЫКЛЮЧЕН. Раньше он держался не как вывод из замера, а как ставка
    # на расширение корпуса: «на запросе бесплатен, вреда не доказано».
    # Ставка не подтвердилась — вред доказан, причём тремя независимыми
    # замерами, каждый из которых закрывает лазейку предыдущего.
    #
    # 1. Токенизатор (experiments/tokenizer.py). Проверяли гипотезу, что
    #    виновата обрезка слова до 6 символов и BM25 просто плохо настроен.
    #    Гипотеза опровергнута: без обрезки становится хуже, а не лучше.
    #    Заодно выяснилось главное — BM25 при равном голосе тянет выдачу
    #    ВНИЗ под обеими разметками, а не просто ничего не даёт.
    # 2. Разбор по вопросам. Выигрывает на 64 вопросах (+0.133 NDCG),
    #    проигрывает на 88 (−0.198). Выигрыши настоящие и приходятся
    #    на точные термины, но признака, чтобы отличить их заранее, нет:
    #    максимальный IDF слова запроса 4.62 против 4.31, а наличие числа
    #    работает в обратную сторону (3% против 8%).
    # 3. Сетка весов (experiments/bm25_weight.py). Последняя лазейка: может,
    #    с малым весом BM25 поднимает своё, но не перебивает чужое. Точки
    #    нет. НИ ОДИН вес не лучше нуля ни под одной разметкой, а от 0.2
    #    заметно хуже. Ноль — единственный, который нигде не проигрывает.
    #
    # РЕШАЮЩИЕ ЦИФРЫ — отсюда, из пункта 3, и только отсюда. Этот замер
    # детерминирован: обе конфигурации получают ОДИН И ТОТ ЖЕ английский
    # запрос из кэша, модель не вызывается вовсе. Вес 1.0 против нуля:
    #
    #     ndcg −0.042 [−0.066, −0.020]  родственная разметка
    #     ndcg −0.023 [−0.047, −0.000]  независимая (впритык, но значимо)
    #
    # Сквозной прогон (run_eval.py --hybrid) дал precision +0.031
    # [+0.001, +0.063] в пользу выключения, но в обоснование НЕ засчитан:
    # путь до поиска идёт через `rewrite`, а это вызов LLM. Разные прогоны
    # дают разные английские запросы, то есть разную выдачу, и отделить
    # эффект BM25 от шума прогона там нечем. Это подтверждение направления,
    # а не доказательство.
    #
    # На correctness выключение дало +0.031 — тоже не засчитано, по той же
    # причине. Контрольная группа: вопросы, где обе конфигурации отдали
    # в генерацию одинаковый контекст (59 из 210 — сравнивать надо состав
    # И ПОРЯДОК чанков, по одному составу в контроль просачиваются
    # переставленные, а перестановка это уже эффект BM25). Контроль дал
    # +0.008, разность между группами +0.031 [−0.006, +0.070] — значимости
    # не достигает, хотя и впритык.
    use_hybrid: bool = False,
    # Реранкинг ВЫКЛЮЧЕН — но не по той причине, по которой был выключен
    # раньше. Прежнее обоснование («метрики поиска не растут») перемерено
    # и оказалось неверным: оно опиралось на разметку с одним эталонным
    # чанком на вопрос, под которой польза реранкера невидима в принципе.
    #
    # Под пулинговой разметкой он значимо улучшает поиск, и это
    # подтверждено независимым судьёй (шаг 24 в EVOLUTION.md):
    #
    #   precision@5   0.839 → 0.881   [+0.010, +0.074]
    #   recall@5      0.436 → 0.526   [+0.065, +0.115]
    #   NDCG@5        0.854 → 0.898   [+0.015, +0.074]
    #
    # А выключен он потому, что до ОТВЕТА этот прирост не доходит:
    #
    #   correctness   0.860 → 0.867   [-0.019, +0.033]
    #   faithfulness  0.919 → 0.954   [-0.005, +0.030]
    #
    # Сдвиг correctness +0.007 лежит ниже измеренного порога различимости
    # судьи (0.010, см. experiments_judge_noise.py). Между поиском и ответом
    # стоит grade, и он отбрасывает нерелевантное сам — реранкер делает ту же
    # работу раньше, но к генерации разница уже съедена.
    #
    # Цена при этом настоящая: +2.1 с к ответу (медиана 8.9 → 10.9 с, +23%)
    # и лишний вызов LLM на каждый запрос.
    #
    # Включать имеет смысл, если важна чистота панели источников: она
    # показывает найденные фрагменты, и там прирост precision виден глазами.
    use_rerank: bool = False,
    top_k: int = 5,
    max_refines: int = 1,
    embed_client: Any = None,
    seed: int = 0,
    # Строгий критерий релевантности включён по результатам замеров: честность
    # отказов на бессмысленных вопросах выросла с 86% до 93% (на трудных —
    # с 76% до 84%), precision — на 0.062, обе разницы значимы. Полнота при этом
    # просела на 0.013 recall, что укладывается в шум. См. experiments/grade.py.
    strict_grade: bool = True,
    # Какой промпт генерации использовать: "short" / "full" / "evidence".
    #
    # "full" включён по замерам на 210 вопросах: correctness +0.069
    # [+0.038, +0.100] — значимо, faithfulness при этом не просела (+0.014).
    # Цена — ответы длиннее на 85%. См. experiments/answer_prompt.py.
    #
    # "short" — отклонённая базовая линия, оставлена, чтобы было с чем
    # сравнивать. "evidence" — кандидат, см. GENERATE_PROMPT_EVIDENCE.
    answer_style: str = "full",
    # Вызовы модели наперёд — для сервиса (см. NutritionAgent.speculate).
    speculate: bool = False,
    # Извлечение профиля по схеме (EXTRACT_PROFILE_SCHEMA). None — по модели:
    # строгие схемы проверены у polza (gpt-4o-mini), у GigaChat — нет,
    # и там остаётся прежний свободный JSON.
    structured_extract: bool | None = None,
) -> NutritionAgent:
    """Собрать готового агента: БД → эмбеддер → ретривер → каталог → граф.

    Args:
        client: клиент LLM (генерация, классификация, реранкинг).
        embed_client: клиент для эмбеддингов. Если None — используется client.
            Разделение полезно, когда эмбеддинги и генерация живут
            у разных провайдеров.
    """
    from embedder import Embedder
    from lance_db import open_table
    from retriever import Retriever

    table = open_table(db_path, table_name)
    embedder = Embedder(embed_client or client)
    retriever = Retriever(
        table,
        embedder,
        client=client,
        model=model,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
    )

    return NutritionAgent(
        client=client,
        model=model,
        retriever=retriever,
        top_k=top_k,
        max_refines=max_refines,
        seed=seed,
        strict_grade=strict_grade,
        answer_style=answer_style,
        speculate=speculate,
        structured_extract=(structured_extract if structured_extract is not None
                            else not model.lower().startswith("gigachat")),
    )
