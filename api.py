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
на весь процесс (он держит каталог, индекс и BM25 по корпусу), и профиль
одного пользователя не должен попадать в расчёт другому.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path as FilePath
from threading import Lock
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Path, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import (
    CANNOT_ANSWER_TEMPLATE,
    OFF_TOPIC_ANSWER,
    RED_FLAG_TEMPLATE,
    ConversationState,
    NutritionAgent,
    build_agent,
)
from export import menu_to_xlsx
from foods import load_catalog
from providers import get_clients
from solver import DayPlan, Menu, PlanNotFeasible, build_day, build_menu, check_plan
from targets import CycledTargets, Profile, Targets, compute_targets, explain

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("nutrition_agent.api")

# Сколько разговоров держим в памяти. Хранилище намеренно простое: для
# продакшена сюда встанет Redis с TTL, но для демонстрации и одного процесса
# словаря достаточно, а лишняя зависимость только мешала бы запустить проект.
MAX_SESSIONS = 1000

# Меню тяжелее сессии, поэтому их держим меньше.
MAX_MENUS = 200


# ────────────────────────────────────────────────────────────
# Хранилище сессий
# ────────────────────────────────────────────────────────────


class SessionStore:
    """Разговоры по идентификатору сессии.

    Под блокировкой, потому что FastAPI выполняет синхронные ручки
    в пуле потоков — обращения к словарю идут из разных потоков.
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: dict[str, ConversationState] = {}
        self._lock = Lock()
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str | None) -> tuple[str, ConversationState]:
        with self._lock:
            if session_id and session_id in self._sessions:
                return session_id, self._sessions[session_id]

            # Клиент прислал неизвестный id — не падаем, а заводим новую сессию:
            # после перезапуска сервиса у всех клиентов id «протухают».
            new_id = session_id or str(uuid.uuid4())

            if len(self._sessions) >= self._max_sessions:
                # Простейшее вытеснение: словари в Python сохраняют порядок
                # вставки, поэтому первый ключ — самый старый.
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]
                logger.info("Вытеснена старая сессия %s", oldest)

            state = ConversationState()
            self._sessions[new_id] = state
            return new_id, state

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


class MenuStore:
    """Меню, собранные в разговорах, — чтобы отдать их файлом.

    Хранится объект ``Menu``, а не готовый файл: собирать книгу Excel
    для меню, которое никто не скачает, незачем. А пересобирать меню
    заново на скачивании нельзя — это ещё 15 секунд подбора, и человек
    получил бы не то меню, которое ему показали.
    """

    def __init__(self, max_menus: int = MAX_MENUS) -> None:
        self._menus: dict[str, tuple[Menu, dict[str, Any]]] = {}
        self._lock = Lock()
        self._max_menus = max_menus

    def put(self, menu: Menu, profile: dict[str, Any]) -> str:
        menu_id = str(uuid.uuid4())
        with self._lock:
            if len(self._menus) >= self._max_menus:
                del self._menus[next(iter(self._menus))]
            self._menus[menu_id] = (menu, profile)
        return menu_id

    def get(self, menu_id: str) -> tuple[Menu, dict[str, Any]] | None:
        with self._lock:
            return self._menus.get(menu_id)


# ────────────────────────────────────────────────────────────
# Схемы запросов и ответов
# ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None, description="Идентификатор разговора. Не передан — начнётся новый."
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
    session_id: str
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

    Каталог, индекс LanceDB и BM25 по всему корпусу собираются секунды —
    делать это в обработчике запроса нельзя.
    """
    logger.info("Загружаю каталог блюд и индекс...")
    started = time.perf_counter()

    llm_client, model, embed_client = get_clients(os.getenv("LLM_PROVIDER", "polza"))

    _state["catalog"] = load_catalog()
    _state["agent"] = build_agent(llm_client, model=model, embed_client=embed_client)
    _state["sessions"] = SessionStore()
    _state["menus"] = MenuStore()

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


def get_sessions() -> SessionStore:
    return _state["sessions"]


def get_menus() -> MenuStore:
    return _state["menus"]


def get_catalog():
    return _state["catalog"]


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
        active_sessions=get_sessions().count(),
    )


# ────────────────────────────────────────────────────────────
# Детерминированные ручки (без LLM)
# ────────────────────────────────────────────────────────────


@app.post("/targets", response_model=TargetsResponse, tags=["расчёт без LLM"])
def post_targets(profile: Profile) -> TargetsResponse:
    """Посчитать норму КБЖУ по профилю.

    Чистая формула: ни одного обращения к модели, ответ за миллисекунды.
    Сработавшие предохранители возвращаются в ``targets.adjustments`` —
    урезанную цифру сервис молча не отдаёт.
    """
    targets = compute_targets(profile)
    return TargetsResponse(targets=targets, explanation=explain(profile, targets))


@app.post("/plan", response_model=PlanResponse, tags=["расчёт без LLM"])
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


@app.post("/menu", response_model=MenuResponse, tags=["расчёт без LLM"])
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
    )


@app.get("/foods", response_model=list[FoodOut], tags=["расчёт без LLM"])
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


@app.post("/chat", response_model=ChatResponse, tags=["разговор"])
def post_chat(
    request: ChatRequest = Body(...),
    agent: NutritionAgent = Depends(get_agent),
    sessions: SessionStore = Depends(get_sessions),
    menus: MenuStore = Depends(get_menus),
) -> ChatResponse:
    """Реплика в разговоре с агентом.

    Профиль накапливается между репликами: человек редко называет все
    параметры сразу. Чтобы продолжить разговор, передавайте ``session_id``
    из предыдущего ответа.
    """
    session_id, state = sessions.get_or_create(request.session_id)
    started = time.perf_counter()

    try:
        answer = agent.run(request.message, state=state)
    except RuntimeError as error:
        # Провайдер не ответил даже после ретраев — это внешний сбой, 502.
        logger.error("Провайдер недоступен: %s", error)
        raise HTTPException(status_code=502, detail="Модель недоступна, попробуйте позже") from error

    sources = _collect_sources(agent, answer)

    menu_url = None
    if answer.menu is not None:
        menu_id = menus.put(answer.menu, state.profile)
        menu_url = f"/menu/{menu_id}.xlsx"

    return ChatResponse(
        session_id=session_id,
        answer=answer.answer,
        sources=sources,
        profile=state.profile,
        menu_url=menu_url,
        menu_days=answer.menu_days,
        plan=answer.plan,
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


@app.get("/menu/{menu_id}.xlsx", tags=["разговор"])
def get_menu_xlsx(
    menu_id: str = Path(description="Идентификатор меню из ответа /chat"),
    menus: MenuStore = Depends(get_menus),
) -> Response:
    """Скачать меню, собранное в разговоре, книгой Excel."""
    found = menus.get(menu_id)
    if found is None:
        # Меню живут в памяти и вытесняются — после перезапуска сервиса
        # ссылка перестанет работать, и это нормальный, а не аварийный случай.
        raise HTTPException(
            status_code=404,
            detail="Меню больше недоступно — попросите агента собрать его заново",
        )
    return _xlsx_response(*found)


@app.post("/menu.xlsx", tags=["расчёт без LLM"])
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


@app.delete("/sessions/{session_id}", tags=["разговор"])
def delete_session(
    session_id: str = Path(description="Идентификатор разговора"),
    sessions: SessionStore = Depends(get_sessions),
) -> dict[str, str]:
    """Забыть разговор: профиль и историю."""
    if not sessions.drop(session_id):
        raise HTTPException(status_code=404, detail="Такой сессии нет")
    return {"status": "deleted", "session_id": session_id}


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
