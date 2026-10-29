"""
HTTP-сервис поверх агента.

Запуск::

    uvicorn api:app --reload
    # документация: http://127.0.0.1:8000/docs

Ручки разделены по стоимости и предсказуемости — это то же разделение,
что и внутри проекта:

* ``/targets`` и ``/plan`` — **детерминированные**. Ни одного обращения
  к LLM, ответ за миллисекунды, бесплатно, при том же seed всегда одинаков.
  Именно их удобно дёргать из интеграционных тестов и из чужого кода.
* ``/chat`` — **разговорная**. Здесь работает граф агента, а значит есть
  и задержка, и стоимость токенов.

Состояние разговора живёт в хранилище сессий, а не в агенте: агент один
на весь процесс (он держит каталог блюд и индекс), и профиль
одного пользователя не должен попадать в расчёт другому.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path as FilePath
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import (
    CANNOT_ANSWER_TEMPLATE,
    OFF_TOPIC_ANSWER,
    RED_FLAG_TEMPLATE,
    ConversationState,
    NutritionAgent,
    build_agent,
    for_reader,
    for_reader_partial,
)
from chat_log import ChatLog, turn_record
from config import settings
from export import menu_to_xlsx
from foods import load_catalog
from limits import BudgetExhausted, Limits, LLMBudget, RateLimiter, seconds_to_utc_midnight
from providers import get_clients
from shopping import ShoppingList, build_shopping_list
from solver import (
    DayPlan,
    Menu,
    PlanNotFeasible,
    build_day,
    build_menu,
    check_plan,
    swap_item,
)
from storage import open_storage
from targets import CycledTargets, Profile, Targets, compute_targets, explain

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("nutrition_agent.api")

# ────────────────────────────────────────────────────────────
# Сессии
# ────────────────────────────────────────────────────────────
#
# Пропуск в разговор выдаёт ТОЛЬКО сервер: 256 случайных бит в cookie
# httpOnly (JavaScript страницы его не видит — кража через XSS исключена),
# SameSite=Lax. Ревью (R2): раньше id присылал клиент, и неизвестный id
# становился id новой сессии — его можно было навязать. Теперь неизвестный
# или протухший пропуск — это новая сессия с НОВЫМ пропуском.
#
# Клиентам без браузера — тот же пропуск в заголовке X-Session-Token.
# В журнал и к меню пишется не пропуск, а public_id: журнал не должен
# хранить ключи от чужих разговоров.
SESSION_COOKIE = "nutrition_session"
SESSION_HEADER = "X-Session-Token"


@dataclass
class Session:
    token: str
    public_id: str
    state: ConversationState
    is_new: bool = False
    consent: dict[str, Any] = field(default_factory=dict)

    def dump(self) -> dict[str, Any]:
        return {"public_id": self.public_id, "state": self.state.model_dump(),
                "consent": self.consent}


def _find_session(request: Request, storage: Any) -> Session | None:
    """Сессия по пропуску из cookie или заголовка — или None."""
    token = request.cookies.get(SESSION_COOKIE) or request.headers.get(SESSION_HEADER)
    data = storage.load_session(token) if token else None
    if data is None:
        return None
    return Session(token=token, public_id=data["public_id"],
                   state=ConversationState(**data["state"]),
                   consent=data.get("consent") or {})


def _open_session(request: Request, storage: Any) -> Session:
    """Сессия по пропуску; нет такой — новая, с НОВЫМ пропуском (не с присланным)."""
    session = _find_session(request, storage)
    if session is not None:
        return session
    session = Session(token="", public_id=uuid.uuid4().hex, state=ConversationState(),
                      is_new=True)
    session.token = storage.create_session(session.dump())
    return session


# Ревью R4: в разговоре — вес, возраст, диагнозы, то есть данные о здоровье
# (специальная категория по 152-ФЗ). Без согласия разговор не начинается:
# проверяет СЕРВЕР, а не только окно на странице — иначе любой клиент API
# его обходит. Версия текста политики — рядом с согласием: поменяется
# текст — видно, на какой из них человек соглашался.
POLICY_VERSION = "2026-10-29-черновик"
CONSENT_REQUIRED = (
    "Сначала нужно согласие на обработку данных о здоровье — "
    "см. /privacy.html и POST /consent."
)


def _consented_session(request: Request, storage: Any) -> Session:
    """Сессия с согласием — иначе 403, и новая сессия НЕ заводится."""
    session = _find_session(request, storage)
    if session is None or not session.consent.get("health_data_at"):
        raise HTTPException(status_code=403, detail=CONSENT_REQUIRED)
    return session


def _set_session_cookie(response: Response, session: Session) -> None:
    if session.is_new:
        response.set_cookie(
            SESSION_COOKIE, session.token,
            max_age=int(settings.session_ttl_days * 24 * 3600),
            httponly=True, samesite="lax", secure=settings.session_cookie_secure, path="/",
        )


# ────────────────────────────────────────────────────────────
# Схемы запросов и ответов
# ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    # Только чтобы отклонить его ГРОМКО (400): молча игнорируемое поле
    # прежние клиенты API приняли бы за работающее и теряли бы профиль
    # на каждой реплике, не понимая почему.
    session_id: str | None = Field(
        default=None, description="Больше не принимается: разговор держит cookie сервера.",
    )


class SourceOut(BaseModel):
    """Источник, на который опирался ответ.

    Вместе с выходными данными отдаётся сам найденный фрагмент. Это
    единственное, что нельзя подделать генерацией: либо предложение есть
    в статье, либо его нет. Без цитаты ответ агента ничем не отличается
    от ответа обычной модели — человеку остаётся верить на слово.
    """

    pmid: str
    title: str
    journal: str
    year: str
    quote: str = ""


class ChatResponse(BaseModel):
    # Пропуска в ответе нет: он в cookie httpOnly, и тело ответа, которое
    # читает JavaScript страницы, не должно его раскрывать.
    answer: str
    sources: list[SourceOut] = Field(default_factory=list)
    profile: dict = Field(default_factory=dict)
    elapsed_ms: int

    # Если агент собрал меню, здесь ссылка на его выгрузку в Excel:
    # в ответе показан один день как образец, а скачать можно всё.
    menu_url: str | None = None
    menu_days: int = 0

    # План и нормы структурой — интерфейс рисует по ним карточки приёмов пищи
    # и полосы «факт к норме». Из текста ответа эти числа не достать.
    plan: DayPlan | None = None
    targets: Targets | None = None

    # Вердикт check_plan по показанному дню. Карточка судит по нему, а не
    # по своему порогу: раньше она красила всё, что дальше ±10%, а код
    # допускает жиры и углеводы ±35% — после замены завтрака текст говорил
    # «все проверки пройдены», а карточка показывала три отклонения.
    checks: dict[str, bool] | None = None

    # Норма показанного дня. При циклировании первый день недели —
    # тренировочный, и сравнивать его с базовой нормой неверно.
    plan_targets: Targets | None = None

    # Нормы тренировочного дня и дня отдыха, если человек тренируется.
    cycled: CycledTargets | None = None

    # Сколько фрагментов посмотрел агент. Нужно для отказов: «в базе нет
    # информации» выглядит как провал, хотя это работающая проверка.
    # Число превращает пустоту в видимую работу.
    checked_fragments: int = 0


class TargetsResponse(BaseModel):
    targets: Targets
    explanation: str


class PlanRequest(BaseModel):
    profile: Profile
    seed: int = Field(default=0, description="Один seed — один и тот же план.")


class PlanResponse(BaseModel):
    targets: Targets
    plan: DayPlan
    checks: dict[str, bool]
    checks_passed: bool


class MenuRequest(BaseModel):
    profile: Profile
    days: int = Field(default=1, ge=1, le=31, description="На сколько дней собрать меню")
    seed: int = Field(default=0, description="Один seed — одно и то же меню.")


class MenuResponse(BaseModel):
    targets: Targets
    menu: Menu
    checks: list[dict[str, bool]]
    checks_passed: bool
    unique_dishes: int
    text: str = Field(description="Меню обычным текстом — для скачивания")

    # Список покупок: то же меню, свёрнутое по блюдам и разложенное
    # по отделам магазина. В магазине нужен не «что я ем в среду»,
    # а «сколько всего курицы взять».
    shopping: ShoppingList


class FoodOut(BaseModel):
    fdc_id: int
    name: str
    role: str
    portion: str
    portion_g: float
    portion_kcal: float
    portion_protein_g: float


class HealthResponse(BaseModel):
    status: str
    foods_in_catalog: int
    chunks_in_index: int
    articles_in_corpus: int
    active_sessions: int


# ────────────────────────────────────────────────────────────
# Жизненный цикл приложения
# ────────────────────────────────────────────────────────────

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Поднять тяжёлые ресурсы один раз на старте, а не на каждый запрос.

    Каталог блюд и индекс LanceDB собираются секунды —
    делать это в обработчике запроса нельзя.
    """
    logger.info("Загружаю каталог блюд и индекс...")
    started = time.perf_counter()

    llm_client, model, embed_client = get_clients(os.getenv("LLM_PROVIDER", "polza"))

    _state["catalog"] = load_catalog()
    # speculate — здесь ждёт человек: следующие шаги спрашивают модель,
    # пока classify решает маршрут (experiments/latency.py).
    _state["agent"] = build_agent(
        llm_client, model=model, embed_client=embed_client, speculate=True)
    # Сессии, меню и суточный счётчик — в постоянном хранилище: рестарт
    # больше не стирает разговоры (storage.py). Недоступно — запуск падает.
    _state["storage"] = open_storage(
        settings.state_db, session_ttl_days=settings.session_ttl_days,
        menu_ttl_days=settings.menu_ttl_days,
    )
    _state["chat_log"] = ChatLog.from_env()
    _state["limits"] = Limits.from_settings(settings)
    _state["limiter"] = RateLimiter()
    # Потолок — у агента: списывается на каждый вызов модели (limits.py).
    _state["agent"].budget = LLMBudget(_state["storage"], _state["limits"].llm_calls_per_day)

    chat_log = _state["chat_log"]
    logger.info(
        "Журнал чата: %s",
        f"{chat_log.directory}, хранится {chat_log.retention_days} дн."
        if chat_log.enabled else "выключен",
    )
    logger.info(
        "Готово за %.1f с: %d блюд, %d чанков",
        time.perf_counter() - started,
        len(_state["catalog"]),
        _state["agent"].retriever.table.count_rows(),
    )
    yield
    _state.clear()


app = FastAPI(
    title="Агент-нутрициолог",
    description=(
        "Расчёт нормы КБЖУ, сборка рациона из справочника USDA "
        "и ответы по научной литературе. Не заменяет консультацию врача."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def get_agent() -> NutritionAgent:
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Сервис ещё не прогрелся")
    return agent


def get_storage() -> Any:
    return _state["storage"]


# ── Ограничения (ревью R1, limits.py) ──
def _limits() -> Limits:
    return _state.get("limits") or Limits.from_settings(settings)


def _limiter() -> RateLimiter:
    return _state.setdefault("limiter", RateLimiter())


def _client_ip(http: Request) -> str:
    # За обратным прокси адрес клиента даёт uvicorn --proxy-headers;
    # без него здесь окажется адрес прокси, и лимит станет общим на всех.
    return http.client.host if http.client else "unknown"


def _too_many(seconds: float, text: str) -> HTTPException:
    seconds = int(seconds + 0.999)
    return HTTPException(status_code=429, detail=text.format(seconds=seconds),
                         headers={"Retry-After": str(seconds)})


def _hit(key: str, limit: int, window: float, text: str) -> None:
    wait = _limiter().hit(key, limit, window)
    if wait is not None:
        raise _too_many(wait, text)


def limit_chat(http: Request) -> None:
    """Частота реплик: по адресу и по сессии.

    По адресу — обязательно: лимит по сессии держит один разговор,
    а скрипт может завести их сколько угодно через /consent.
    """
    limits, ip = _limits(), _client_ip(http)
    text = "Слишком много сообщений подряд — подождите {seconds} с и попробуйте снова."
    _hit(f"chat-ip-m:{ip}", limits.chat_per_minute_ip, 60, text)
    _hit(f"chat-ip-h:{ip}", limits.chat_per_hour_ip, 3600, text)
    token = http.cookies.get(SESSION_COOKIE) or http.headers.get(SESSION_HEADER)
    if token:
        _hit_session(token)


def _hit_session(token: str) -> None:
    """Засчитать реплику сессии."""
    limits = _limits()
    text = "Слишком много сообщений подряд — подождите {seconds} с и попробуйте снова."
    _hit(f"chat-s-m:{token}", limits.chat_per_minute_session, 60, text)
    _hit(f"chat-s-h:{token}", limits.chat_per_hour_session, 3600, text)


def limit_calc(http: Request) -> None:
    """Ручки без модели — мягко: они дешёвые, но не бесплатные."""
    _hit(f"calc:{_client_ip(http)}", _limits().calc_per_minute_ip, 60,
         "Слишком много запросов — подождите {seconds} с.")


def limit_menu(http: Request) -> None:
    """Меню на месяц — 15 с процессора: десяток параллельных кладут сервис."""
    _hit(f"menu:{_client_ip(http)}", _limits().menu_per_minute_ip, 60,
         "Меню собирается долго, поэтому их не больше нескольких в минуту — "
         "подождите {seconds} с.")


BUDGET_TEXT = (
    "Сегодня сервис исчерпал дневной лимит ответов. Попробуйте завтра — "
    "норму КБЖУ и рацион без разговора можно посчитать и сейчас."
)


def _budget_exhausted() -> HTTPException:
    return HTTPException(status_code=429, detail=BUDGET_TEXT,
                         headers={"Retry-After": str(seconds_to_utc_midnight())})


def get_catalog():
    return _state["catalog"]


def get_chat_log() -> ChatLog:
    # Без прогрева (тесты) журнал выключен: на диск никто не пишет.
    return _state.get("chat_log") or ChatLog(None)


# ────────────────────────────────────────────────────────────
# Служебное
# ────────────────────────────────────────────────────────────


def _index_size(agent: NutritionAgent) -> int:
    """Сколько чанков в индексе.

    Через getattr, а не напрямую: health-ручка не должна отдавать 500 из-за
    того, что у ретривера поменялось внутреннее устройство. Она отвечает
    на вопрос «сервис жив?», и этот ответ важнее одной цифры в теле.
    """
    table = getattr(agent.retriever, "table", None)
    if table is None:
        return 0
    try:
        return int(table.count_rows())
    except Exception:  # pragma: no cover — зависит от версии LanceDB
        logger.warning("Не удалось прочитать размер индекса", exc_info=True)
        return 0


@app.get("/health", response_model=HealthResponse, tags=["служебное"])
def health(agent: NutritionAgent = Depends(get_agent)) -> HealthResponse:
    """Готовность сервиса и размеры загруженных данных."""
    return HealthResponse(
        status="ok",
        foods_in_catalog=len(agent.catalog),
        chunks_in_index=_index_size(agent),
        articles_in_corpus=len(agent.corpus_index),
        active_sessions=get_storage().count_sessions(),
    )


# ────────────────────────────────────────────────────────────
# Детерминированные ручки (без LLM)
# ────────────────────────────────────────────────────────────


@app.post("/targets", response_model=TargetsResponse, tags=["расчёт без LLM"],
          dependencies=[Depends(limit_calc)])
def post_targets(profile: Profile) -> TargetsResponse:
    """Посчитать норму КБЖУ по профилю.

    Чистая формула: ни одного обращения к модели, ответ за миллисекунды.
    Сработавшие предохранители возвращаются в ``targets.adjustments`` —
    урезанную цифру сервис молча не отдаёт.
    """
    targets = compute_targets(profile)
    return TargetsResponse(targets=targets, explanation=explain(profile, targets))


@app.post("/plan", response_model=PlanResponse, tags=["расчёт без LLM"],
          dependencies=[Depends(limit_calc)])
def post_plan(request: PlanRequest, catalog=Depends(get_catalog)) -> PlanResponse:
    """Собрать рацион на день под норму профиля.

    Тоже без LLM. При одном и том же ``seed`` ответ побайтово одинаков.
    """
    targets = compute_targets(request.profile)

    try:
        plan = build_day(
            catalog, targets, exclude=request.profile.exclude, seed=request.seed
        )
    except PlanNotFeasible as error:
        # Это не сбой сервиса, а неудачные входные данные: человек исключил
        # слишком много. 422, а не 500.
        raise HTTPException(status_code=422, detail=str(error)) from error

    checks = check_plan(plan)
    return PlanResponse(
        targets=targets, plan=plan, checks=checks, checks_passed=all(checks.values())
    )


class SwapRequest(BaseModel):
    profile: Profile
    plan: DayPlan = Field(description="План из ответа /plan (или прошлого /plan/swap)")
    index: int = Field(ge=0, description="Номер позиции в plan.items, с нуля")
    seed: int = Field(default=0, description="Один seed — одна и та же замена.")
    avoid_ids: list[int] = Field(default_factory=list, description=(
        "fdc_id блюд, которые в этот слот уже предлагались и не подошли"))


class SwapResponse(PlanResponse):
    replaced: dict[str, Any]


@app.post("/plan/swap", response_model=SwapResponse, tags=["расчёт без LLM"],
          dependencies=[Depends(limit_calc)])
def post_plan_swap(request: SwapRequest, catalog=Depends(get_catalog)) -> SwapResponse:
    """Заменить одно блюдо плана другим той же роли (ревью R8).

    Без LLM и детерминированно по ``seed``. Новый план проходит все
    проверки ``check_plan`` — иначе 422. Числа блюд и нормы сервер
    пересчитывает сам (каталог и профиль), а не берёт из присланного плана.
    """
    targets = compute_targets(request.profile)
    plan = request.plan.model_copy(update={"targets": targets.as_dict()})
    try:
        swapped = swap_item(catalog, plan, request.index, exclude=request.profile.exclude,
                            seed=request.seed, avoid_ids=set(request.avoid_ids))
    except PlanNotFeasible as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    old, new = plan.items[request.index], swapped.items[request.index]
    checks = check_plan(swapped)
    return SwapResponse(
        targets=targets, plan=swapped, checks=checks, checks_passed=all(checks.values()),
        replaced={"index": request.index,
                  "from": {"fdc_id": old.fdc_id, "name": old.title},
                  "to": {"fdc_id": new.fdc_id, "name": new.title}},
    )


@app.post("/menu", response_model=MenuResponse, tags=["расчёт без LLM"],
          dependencies=[Depends(limit_menu)])
def post_menu(request: MenuRequest, catalog=Depends(get_catalog)) -> MenuResponse:
    """Собрать меню на несколько дней.

    Тоже без единого обращения к модели. Дни различаются между собой, а блюда
    последних двух дней временно исключаются из подбора, чтобы одно и то же
    не повторялось подряд.

    Время растёт линейно: день — около полусекунды, месяц — порядка 15 секунд.
    """
    targets = compute_targets(request.profile)

    try:
        menu = build_menu(
            catalog,
            targets,
            days=request.days,
            exclude=request.profile.exclude,
            seed=request.seed,
        )
    except PlanNotFeasible as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    per_day = [check_plan(day) for day in menu.days]

    return MenuResponse(
        targets=targets,
        menu=menu,
        checks=per_day,
        checks_passed=all(all(day.values()) for day in per_day),
        unique_dishes=menu.unique_dishes,
        text=menu.render(),
        shopping=build_shopping_list(menu, catalog),
    )


@app.get("/foods", response_model=list[FoodOut], tags=["расчёт без LLM"],
         dependencies=[Depends(limit_calc)])
def get_foods(
    q: str = Query(min_length=2, description="Часть названия блюда"),
    role: str | None = Query(default=None, description="breakfast / main / side / snack / drink"),
    limit: int = Query(default=20, ge=1, le=100),
    catalog=Depends(get_catalog),
) -> list[FoodOut]:
    """Поиск по каталогу блюд — посмотреть, из чего вообще собирается рацион."""
    found = catalog[catalog["name"].str.contains(q, case=False, regex=False, na=False)]
    if role:
        found = found[found["role"] == role]

    columns = ["fdc_id", "name", "role", "portion", "portion_g", "portion_kcal", "portion_protein_g"]
    return [FoodOut(**row) for row in found.head(limit)[columns].to_dict("records")]


# ────────────────────────────────────────────────────────────
# Разговорная ручка (с LLM)
# ────────────────────────────────────────────────────────────


def _reject_client_session_id(request: ChatRequest) -> None:
    if request.session_id is not None:
        raise HTTPException(status_code=400, detail=(
            "session_id больше не принимается: разговор держит cookie, которую "
            f"ставит сервер (для клиентов без cookie — заголовок {SESSION_HEADER})."))


@app.post("/chat", response_model=ChatResponse, tags=["разговор"],
          dependencies=[Depends(limit_chat)])
def post_chat(
    http: Request,
    request: ChatRequest = Body(...),
    agent: NutritionAgent = Depends(get_agent),
    storage: Any = Depends(get_storage),
    chat_log: ChatLog = Depends(get_chat_log),
) -> ChatResponse:
    """Реплика в разговоре с агентом.

    Профиль накапливается между репликами: человек редко называет все
    параметры сразу. Разговор держит cookie, которую ставит POST /consent;
    клиенту без cookie — тот же пропуск в заголовке X-Session-Token.
    Без согласия — 403, и разговор не начинается.
    """
    _reject_client_session_id(request)
    session = _consented_session(http, storage)
    return _run_turn(request, session, agent, storage, chat_log)


@app.post("/chat/stream", tags=["разговор"], dependencies=[Depends(limit_chat)])
def post_chat_stream(
    http: Request,
    request: ChatRequest = Body(...),
    agent: NutritionAgent = Depends(get_agent),
    storage: Any = Depends(get_storage),
    chat_log: ChatLog = Depends(get_chat_log),
) -> StreamingResponse:
    """То же, что ``/chat``, но ответ приходит по частям (Server-Sent Events).

    События, по одному JSON в строке ``data:``:

    * ``{"type": "stage", "label": "ищу в статьях"}`` — начался шаг;
    * ``{"type": "text", "text": "..."}`` — ВЕСЬ показанный текст на сейчас,
      уже без номеров PMID (не приращение: вырезание ссылки может убрать
      и уже показанный предлог перед ней);
    * ``{"type": "done", "response": {...}}`` — итог, ровно как ответ ``/chat``;
      его текст заменяет показанный: в конец ответа с планом код дописывает
      стоимость и примечания;
    * ``{"type": "error", "detail": "..."}``.

    Зачем. Ответ по статьям пишется моделью 4-6 с, объяснение плана 3-5 с,
    и всё это время человек смотрел на «думаю…». Первый текст теперь
    появляется, как только модель начала писать.
    """
    _reject_client_session_id(request)
    session = _consented_session(http, storage)
    # До потока, а не внутри: внутри отказ ушёл бы событием при статусе 200.
    if agent.budget is not None and agent.budget.exhausted():
        raise _budget_exhausted()
    # Каждый ответ по частям — свой поток процесса. Без потолка на адрес
    # десяток параллельных запросов держал бы десяток потоков с моделью.
    stream_key = f"stream:{_client_ip(http)}"
    if not _limiter().acquire(stream_key, _limits().streams_per_ip):
        raise HTTPException(status_code=429, detail=(
            "С вашего адреса уже идёт несколько ответов — дождитесь их."))
    events: queue.Queue[dict | None] = queue.Queue()

    def work() -> None:
        try:
            response = _run_turn(request, session, agent, storage, chat_log,
                                 on_event=events.put)
            events.put({"type": "done", "response": response.model_dump(mode="json")})
        except HTTPException as error:
            events.put({"type": "error", "detail": str(error.detail)})
        except Exception:
            logger.exception("Сбой ответа по частям")
            events.put({"type": "error", "detail": "Внутренняя ошибка, попробуйте ещё раз"})
        finally:
            _limiter().release(stream_key)
            events.put(None)

    # Ответ — в своём потоке, а не в генераторе: человек может закрыть
    # вкладку, генератор остановится, а реплика всё равно должна дойти
    # до конца и попасть в журнал и историю разговора.
    threading.Thread(target=work, daemon=True, name="chat-stream").start()
    streaming = StreamingResponse(
        _sse(events), media_type="text/event-stream",
        # no-transform/X-Accel-Buffering: прокси (nginx) иначе копит поток
        # и отдаёт одним куском — стриминг молча перестаёт быть стримингом.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
    return streaming


# Не чаще раза в 80 мс: модель шлёт куски по 1-3 токена, и перерисовка
# на каждый — сотни событий в секунду без видимой разницы для глаза.
TEXT_EVENT_INTERVAL = 0.08


def _sse(events: queue.Queue) -> Iterator[str]:
    """События агента → строки SSE. Текст — без PMID и не чаще интервала."""
    def line(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    pending: str | None = None
    shown = ""
    sent_at = 0.0
    while True:
        try:
            event = events.get(timeout=TEXT_EVENT_INTERVAL)
        except queue.Empty:
            event = {"type": "flush"}
        if event is None:
            return
        if event["type"] == "text":
            pending = for_reader_partial(event["raw"])
        elif event["type"] != "flush":
            pending = None  # итог или новый шаг: недосланный текст устарел
        due = time.perf_counter() - sent_at >= TEXT_EVENT_INTERVAL
        if pending is not None and pending != shown and (due or event["type"] == "flush"):
            shown = pending
            sent_at = time.perf_counter()
            yield line({"type": "text", "text": shown})
        if event["type"] not in ("text", "flush"):
            yield line({k: v for k, v in event.items() if k != "raw"})


def _run_turn(
    request: ChatRequest,
    session: Session,
    agent: NutritionAgent,
    storage: Any,
    chat_log: ChatLog,
    on_event: Callable[[dict], None] | None = None,
) -> ChatResponse:
    """Реплика целиком: ответ агента, журнал, меню на скачивание."""
    state = session.state
    session_id = session.public_id  # в журнал — не пропуск
    # Журнал — только с ОТДЕЛЬНЫМ согласием «помогать улучшать сервис»:
    # в нём текст реплик, то есть диагнозы и вес словами человека.
    if not session.consent.get("improve_service"):
        chat_log = ChatLog(None)
    started = time.perf_counter()
    # Номер реплики в разговоре — до ответа: при сбое история не растёт.
    turn = len(state.history) // 2 + 1

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    # Потолок исчерпан — отвечаем сразу, не начиная: иначе каждая реплика
    # ещё успевала бы до отказа потратить пару вызовов наперёд.
    if agent.budget is not None and agent.budget.exhausted():
        raise _budget_exhausted()

    try:
        answer = agent.run(request.message, state=state, on_event=on_event)
    except Exception as error:
        # Сбой пишется в журнал ВМЕСТЕ с репликой: по одной трассировке
        # в логе uvicorn не понять, на какой фразе агент упал.
        chat_log.write(turn_record(
            session_id=session_id, turn=turn, message=request.message,
            elapsed_ms=elapsed_ms(), error=error,
        ))
        if isinstance(error, BudgetExhausted):
            # Потолок кончился посреди реплики — тот же отказ, что и до неё.
            logger.warning("Суточный потолок вызовов модели исчерпан")
            raise _budget_exhausted() from error
        if isinstance(error, RuntimeError):
            # Провайдер не ответил даже после ретраев — это внешний сбой, 502.
            logger.error("Провайдер недоступен: %s", error)
            raise HTTPException(
                status_code=502, detail="Модель недоступна, попробуйте позже"
            ) from error
        raise

    # Сохранить сразу после ответа: агент дописал профиль и историю
    # в state, но в постоянное хранилище сам ничего не пишет.
    storage.save_session(session.token, session.dump())

    sources = _collect_sources(agent, answer)
    shown = for_reader(answer.answer)
    chat_log.write(turn_record(
        session_id=session_id, turn=turn, message=request.message,
        answer=answer, shown=shown, sources=[source.pmid for source in sources],
        elapsed_ms=elapsed_ms(),
    ))

    menu_url = None
    if answer.menu is not None:
        menu_id = storage.put_menu(session.public_id, answer.menu.model_dump_json(),
                                   state.profile)
        menu_url = f"/menu/{menu_id}.xlsx"

    return ChatResponse(
        # Номера PMID вырезаются из текста: статьи человек видит в sources.
        answer=shown,
        sources=sources,
        profile=state.profile,
        menu_url=menu_url,
        menu_days=answer.menu_days,
        plan=answer.plan,
        checks=check_plan(answer.plan) if answer.plan is not None else None,
        targets=answer.targets,
        plan_targets=answer.plan_targets or answer.targets,
        cycled=answer.cycled,
        checked_fragments=len(answer.retrieved_chunks or []),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def _is_canned_refusal(text: str) -> bool:
    """Готовый отказ агента, а не ответ по найденному.

    Сверяемся с началом шаблонов из agent.py: у RED_FLAG_TEMPLATE внутри
    подставляется причина, поэтому сравнивать целиком нельзя.
    """
    prefixes = (
        CANNOT_ANSWER_TEMPLATE[:40],
        OFF_TOPIC_ANSWER[:40],
        RED_FLAG_TEMPLATE[:30],
    )
    stripped = (text or "").strip()
    return any(stripped.startswith(prefix) for prefix in prefixes)


def _collect_sources(agent: NutritionAgent, answer: Any) -> list[SourceOut]:
    """Собрать выходные данные статей, на которые опирался ответ.

    В контракте ответа лежит только doc_id (поле ``year`` — так требует
    evaluation.py), поэтому выходные данные достаём из справочника корпуса.

    ⚠️ У готовых отказов источники не показываются, хотя фрагменты в ответе
    есть. Так получается по устройству графа: ``retrieve`` отработал, а
    ``grade`` признал найденное негодным и отправил в ``cannot_answer``.
    Фрагменты остаются в состоянии — и это правильно, по ним считаются
    метрики поиска. Но пользователю показывать их нельзя: под фразой
    «в моей базе нет информации» две ссылки на PubMed выглядят как
    противоречие самому себе. Вместо ссылок интерфейс показывает,
    сколько фрагментов было проверено, — см. ``checked_fragments``.
    """
    if _is_canned_refusal(answer.answer):
        return []

    sources: list[SourceOut] = []
    seen: set[str] = set()

    for chunk in answer.retrieved_chunks or []:
        meta = agent.corpus_index.get(str(chunk.year))
        if not meta or meta["pmid"] in seen:
            continue
        seen.add(meta["pmid"])
        sources.append(SourceOut(**meta, quote=chunk.text.strip()))

    return sources


def _xlsx_response(menu: Menu, profile: dict[str, Any]) -> Response:
    """Книга Excel с правильными заголовками для скачивания."""
    name = f"меню-{len(menu.days)}-дн.xlsx"
    # Имя файла кириллицей: в заголовке HTTP допустим только latin-1,
    # поэтому по RFC 6266 отдаём и ASCII-запасной вариант, и filename*
    # в процентной кодировке. Без второго браузер сохранит «Ð¼ÐµÐ½Ñ».
    quoted = urllib.parse.quote(name)
    return Response(
        content=menu_to_xlsx(menu, profile),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition":
                f"attachment; filename=menu-{len(menu.days)}d.xlsx; "
                f"filename*=UTF-8''{quoted}"
        },
    )


@app.get("/menu/{menu_id}.xlsx", tags=["разговор"], dependencies=[Depends(limit_calc)])
def get_menu_xlsx(
    http: Request,
    menu_id: str = Path(description="Идентификатор меню из ответа /chat"),
    storage: Any = Depends(get_storage),
) -> Response:
    """Скачать меню, собранное в разговоре, книгой Excel — только своё.

    Ссылка без пропуска своей сессии не работает: в книге — профиль
    с весом и возрастом. Чужое меню отвечает тем же 404, что и
    несуществующее, — не подсказывая, что такое меню есть.
    """
    token = http.cookies.get(SESSION_COOKIE) or http.headers.get(SESSION_HEADER)
    owner = (storage.load_session(token) or {}).get("public_id") if token else None
    found = storage.get_menu(menu_id)
    if found is None or owner is None or found[0] != owner:
        # Меню живут ограниченный срок (MENU_TTL_DAYS) — истёкшая ссылка
        # это нормальный, а не аварийный случай.
        raise HTTPException(
            status_code=404,
            detail="Меню больше недоступно — попросите агента собрать его заново",
        )
    _, menu_json, profile = found
    return _xlsx_response(Menu.model_validate_json(menu_json), profile)


@app.post("/menu.xlsx", tags=["расчёт без LLM"], dependencies=[Depends(limit_menu)])
def post_menu_xlsx(request: MenuRequest, catalog=Depends(get_catalog)) -> Response:
    """Собрать меню и сразу отдать книгой Excel, без обращения к модели."""
    try:
        menu = build_menu(
            catalog,
            compute_targets(request.profile),
            days=request.days,
            exclude=request.profile.exclude,
            seed=request.seed,
        )
    except PlanNotFeasible as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return _xlsx_response(menu, request.profile.model_dump())


class ConsentRequest(BaseModel):
    health_data: bool = Field(description=(
        "Согласие на обработку данных о здоровье (вес, рост, возраст, заболевания) "
        "для расчёта нормы и рациона. Без него разговор не начинается."))
    improve_service: bool = Field(default=False, description=(
        "Отдельно и необязательно: хранить переписку до 30 дней, чтобы находить ошибки."))


@app.post("/consent", tags=["разговор"], dependencies=[Depends(limit_calc)])
def post_consent(
    http: Request,
    response: Response,
    request: ConsentRequest = Body(...),
    storage: Any = Depends(get_storage),
) -> dict[str, Any]:
    """Дать согласие и начать разговор: сервер выдаёт пропуск в cookie."""
    if not request.health_data:
        raise HTTPException(status_code=400, detail=(
            "Без согласия на обработку данных о здоровье посчитать норму "
            "и рацион нельзя: для них нужны вес, рост и возраст."))
    session = _open_session(http, storage)
    session.consent = {
        "health_data_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "improve_service": request.improve_service,
        "policy": POLICY_VERSION,
    }
    storage.save_session(session.token, session.dump())
    _set_session_cookie(response, session)
    return {"consent": session.consent}


class WeightRequest(BaseModel):
    weight_kg: float = Field(ge=35, le=250)
    day: str | None = Field(default=None, description="ГГГГ-ММ-ДД; не задана — сегодня (UTC)")


@app.post("/weight", tags=["разговор"], dependencies=[Depends(limit_calc)])
def post_weight(
    http: Request,
    request: WeightRequest = Body(...),
    storage: Any = Depends(get_storage),
) -> dict[str, Any]:
    """Записать взвешивание (ревью R9) — для адаптивной нормы.

    Только в свою сессию с согласием: вес — данные о здоровье. Взвешивание
    за сегодня обновляет и вес в профиле: от него считается норма.
    """
    from adaptive import MIN_SPAN_DAYS, MIN_WEIGHINS

    session = _consented_session(http, storage)
    today = datetime.now(UTC).date().isoformat()
    day = request.day or today
    try:
        if datetime.fromisoformat(day).date().isoformat() > today:
            raise ValueError
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Дата взвешивания — не позже сегодня") from error
    session.state.weighins[day] = request.weight_kg
    if day == today:
        session.state.profile["weight_kg"] = request.weight_kg
    storage.save_session(session.token, session.dump())
    days = sorted(session.state.weighins)
    span = (datetime.fromisoformat(days[-1]) - datetime.fromisoformat(days[0])).days
    return {"weighins": len(days),
            "adaptive_ready": len(days) >= MIN_WEIGHINS and span >= MIN_SPAN_DAYS}


@app.get("/session", tags=["разговор"], dependencies=[Depends(limit_calc)])
def get_session(http: Request, storage: Any = Depends(get_storage)) -> dict[str, Any]:
    """Есть ли согласие у текущего разговора — без самих данных."""
    session = _find_session(http, storage)
    consent = session.consent if session else {}
    return {"consented": bool(consent.get("health_data_at")),
            "improve_service": bool(consent.get("improve_service"))}


@app.delete("/session", tags=["разговор"])
def delete_session(
    http: Request,
    response: Response,
    storage: Any = Depends(get_storage),
    chat_log: ChatLog = Depends(get_chat_log),
) -> dict[str, Any]:
    """Удалить мои данные: разговор, профиль, меню, записи журнала, согласие.

    Только СВОЮ сессию — по пропуску из cookie. Раньше была ручка
    /sessions/{id}: чужой разговор мог удалить любой, кто знает id,
    а журнал и меню при этом не трогались вовсе.
    """
    token = http.cookies.get(SESSION_COOKIE) or http.headers.get(SESSION_HEADER)
    data = storage.delete_session(token) if token else None
    response.delete_cookie(SESSION_COOKIE, path="/")
    if data is None:
        raise HTTPException(status_code=404, detail="Сессии нет — удалять нечего")
    menus = storage.delete_menus(data["public_id"])
    journal = chat_log.forget(data["public_id"])
    return {"status": "deleted", "menus_deleted": menus, "journal_deleted": journal}


# ────────────────────────────────────────────────────────────
# Веб-интерфейс
# ────────────────────────────────────────────────────────────
# Статика отдаётся тем же приложением: ни второго контейнера, ни сборки
# фронтенда, ни CORS.
#
# ⚠️ Монтируется ПОСЛЕДНИМ и это принципиально: Starlette сопоставляет
# маршруты в порядке регистрации, а mount на "/" совпадает с чем угодно.
# Объяви его выше — и все ручки API, описанные ниже, перестали бы работать.
STATIC_DIR = FilePath(__file__).parent / "static"

if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
else:  # pragma: no cover — сервис остаётся рабочим и без интерфейса
    logger.warning("Папка %s не найдена — интерфейс недоступен, API работает", STATIC_DIR)
