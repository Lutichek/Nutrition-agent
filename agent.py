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

# agent.py разбит на модули (ревью R6): промпты — prompts.py, разбор текста —
# text_parsing.py, типы — agent_state.py, правила профиля — profile_rules.py,
# текст для читателя — reader.py, узлы — nodes_plan.py и nodes_question.py,
# граф — graph.py. Импорты вида «X as X» ниже — реэкспорт: api.py, тесты
# и замеры по-прежнему берут эти имена из agent.
import json
import logging
import re
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

from agent_state import AgentState, Chunk, ConversationState, RAGAgentAnswer
from data_preparation import load_corpus_index
from foods import load_catalog
from graph import AgentGraph
from llm_io import _LLM_CALLS, parse_json_response
from nodes_plan import PlanNodes
from nodes_question import QuestionNodes
from profile_rules import _NUTRIENT_WORDS as _NUTRIENT_WORDS
from profile_rules import _WEIGHT_RE, _today_utc, meal_from_answer
from profile_rules import REQUIRED_PROFILE_FIELDS as REQUIRED_PROFILE_FIELDS
from profile_rules import _category_fix as _category_fix
from profile_rules import _from_structured as _from_structured
from profile_rules import _merge_allergies as _merge_allergies
from profile_rules import _resolve_preferences as _resolve_preferences
from profile_rules import _swap_index as _swap_index
from profile_rules import _with_russian_fallback as _with_russian_fallback
from profile_rules import allergy_note as allergy_note
from prompts import ALLERGY_NOTE as ALLERGY_NOTE
from prompts import ANSWER_PROMPTS as ANSWER_PROMPTS
from prompts import ASK_PROFILE_PROMPT as ASK_PROFILE_PROMPT
from prompts import CANNOT_ANSWER_TEMPLATE as CANNOT_ANSWER_TEMPLATE
from prompts import (
    CLASSIFY_PROMPT,
    EXTRACT_PROFILE_PROMPT,
    EXTRACT_PROFILE_PROMPT_SGR,
    EXTRACT_PROFILE_SCHEMA,
    OFF_TOPIC_ANSWER,
    RED_FLAG_TEMPLATE,
    REWRITE_PROMPT,
)
from prompts import FRAGMENTS_HEADER as FRAGMENTS_HEADER
from prompts import GENERATE_PROMPT as GENERATE_PROMPT
from prompts import GENERATE_PROMPT_EVIDENCE as GENERATE_PROMPT_EVIDENCE
from prompts import GENERATE_PROMPT_FULL as GENERATE_PROMPT_FULL
from prompts import PREFERENCES_OFFER as PREFERENCES_OFFER
from reader import for_reader as for_reader
from reader import for_reader_partial as for_reader_partial
from text_parsing import _activity_from_text as _activity_from_text
from text_parsing import _budget_from_text as _budget_from_text
from text_parsing import _days_from_text as _days_from_text
from text_parsing import _goal_from_text as _goal_from_text

logger = logging.getLogger(__name__)

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



# ────────────────────────────────────────────────────────────
# Контракт ответа (совместим с evaluation.py)
# ────────────────────────────────────────────────────────────



# ────────────────────────────────────────────────────────────
# Состояние графа
# ────────────────────────────────────────────────────────────


# Промпты и готовые тексты ответов — в prompts.py.



# ────────────────────────────────────────────────────────────
# Агент
# ────────────────────────────────────────────────────────────


class NutritionAgent(PlanNodes, QuestionNodes, AgentGraph):
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
        # «На обед» в ответ на «на какой приём пищи было?» — по одной
        # реплике модель не поймёт, что это про рацион. Решает код.
        resumes_eaten = meal_from_answer(state["question"], state.get("eaten_pending"))
        if resumes_eaten and not state.get("red_flag"):
            return "extract_profile"

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
                "eaten_pending": (memory.eaten_pending if remember and memory.eaten_pending
                                  and memory.eaten_pending.get("date") == _today_utc() else None),
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
            # Ждёт ответа «на какой приём пищи» ровно одну реплику.
            pending = final_state.get("eaten_pending_new")
            memory.eaten_pending = {**pending, "date": _today_utc()} if pending else None
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
