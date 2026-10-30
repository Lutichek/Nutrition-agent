"""
Узлы ветки рациона: извлечь профиль, сопоставить съеденное, переспросить,
собрать план и объяснить его.

Вынесено из agent.py (ревью R6). Примесь к NutritionAgent: модель,
каталог и вспомогательные методы (_complete, _ask и др.) остаются
в агенте и вызываются через self. Числа здесь считает код — solver.py,
targets.py, shopping.py; модель только понимает и объясняет.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from langgraph.graph import END
from pydantic import ValidationError

import food_search
from adaptive import adapted_tdee
from agent_state import AgentState
from llm_io import parse_json_response
from portions import eaten_grams, grams_per_ml, is_drink, unspecified_grams
from profile_rules import (
    _AGAIN_RE,
    _ALLERGY_RE,
    _EATEN_RE,
    _MEAL_ORDER,
    _MEAL_SLOTS,
    _NUTRIENT_WORDS,
    _SWAP_RE,
    FIELD_NAMES_RU,
    REQUIRED_PROFILE_FIELDS,
    _eaten_turn_filter,
    _from_structured,
    _merge_allergies,
    _resolve_preferences,
    _swap_index,
    _with_russian_fallback,
    allergy_note,
    drink_amount,
    eaten_note,
    pending_answer,
    swap_note,
)
from prompts import (
    ASK_PROFILE_PROMPT,
    EATEN_PICK_PROMPT,
    EATEN_PROMPT,
    EXPLAIN_PLAN_PROMPT,
    PREFERENCES_OFFER,
)
from restrictions import is_restriction_phrase, matches_words, to_pattern
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
from text_parsing import (
    MAX_MENU_DAYS,
    _activity_from_text,
    _budget_from_text,
    _days_from_text,
    _goal_from_text,
)

# Имя журнала прежнее: узлы жили в agent.py, и настройки логирования
# (и привычный поиск по журналу) ищут их записи под «agent».
logger = logging.getLogger("agent")

# Дни меню со второго собираются здесь, пока модель объясняет первый
# (_node_make_plan). Солвер считает на чистом Python, а поток запроса в это
# время ждёт сеть — GIL свободен, и сборка идёт почти без помех. Пул общий
# на сервис: по потоку на одновременный расчёт, лишние ждут в очереди,
# а не плодят потоки.
_MENU_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="menu")

# Обобщённая запись («Soup, NFS») первой в кандидатах на съеденное —
# см. _generic_first; флаг для замера experiments/eaten_match.py.
#
# Живой прогон: на «две куски пиццы» поиск дал 12 видов пиццы, модель
# ответила «ничего не подходит» — вид не назван. Замер (2 прогона):
#
#     подсказка выбора                 вид назван   вид не назван
#     прежняя                            12/16          4/16
#     прежняя + строка «выбери обычный»  12/16          4/16
#     прежняя + «X, NFS» первым          12/16          8/16
#     переписанная + «X, NFS» первым     12/16         14/16
#
# Одной строки в подсказке мало: запрет «ставь 0» перевешивал. Помогло
# поставить «вид не назван — любой обычный» ПЕРВЫМ правилом, с примерами.
GENERIC_FIRST = True


class PlanNodes:
    """Узлы ветки «рацион»."""

    # ------------------------------------------------------------------
    # Узлы: ветка расчёта рациона
    # ------------------------------------------------------------------
    def _node_extract_profile(self, state: AgentState) -> AgentState:
        """Достать из разговора параметры человека и дополнить ими профиль."""
        known = dict(state.get("profile") or {})

        raw = self._ask(state, "extract_profile", self._extract_prompt(state))
        extracted = _eaten_turn_filter(
            _from_structured(parse_json_response(raw)), state["question"],
            about_eaten=pending_answer(state["question"], state.get("eaten_pending")) is not None)

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
        # «На обед», «стакан» — ответ на уточнение о съеденном, а не просьба о плане.
        question = state["question"]
        if state.get("last_plan") and (
                _EATEN_RE.search(question)
                or pending_answer(question, state.get("eaten_pending"))):
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
            found = rows.loc[[i for i in ids if i in rows.index],
                             ["fdc_id", "name", "name_ru", "portion_g"]].to_dict("records")
            return self._generic_first(found, item) if GENERIC_FIRST else found
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

    def _generic_first(self, found: list[dict], item: dict) -> list[dict]:
        """Обобщённая запись каталога («Soup, NFS») — первой, если она есть.

        «На обед был суп»: смысловой поиск дал двенадцать конкретных супов,
        а модель, не зная какой, отвечала «ничего не подходит».
        """
        query = str(item.get("query_en") or "").strip()
        if not query:
            return found
        generic = self.catalog[self.catalog["name"].str.lower() == f"{query.lower()}, nfs"]
        if generic.empty:
            return found
        record = generic.iloc[0][["fdc_id", "name", "name_ru", "portion_g"]].to_dict()
        rest = [r for r in found if int(r["fdc_id"]) != int(record["fdc_id"])]
        return [record, *rest][:12]

    def _node_match_eaten(self, state: AgentState) -> AgentState:
        """Сопоставить съеденное с каталогом. Модель выписывает и выбирает;
        ищет кандидатов и считает граммы код."""
        pending = state.get("eaten_pending")
        reply = pending_answer(state["question"], pending)
        if reply is not None:
            # Ответ на уточнение: позиции выписаны и сопоставлены в прошлой
            # реплике, модель второй раз не спрашиваем — дописываем ответ.
            items = [dict(i) for i in pending["items"]]
            picked = pending.get("picked")
            need = list(pending.get("need_amount") or [])
            if reply.get("meal"):
                for item in items:
                    if item.get("meal") not in _MEAL_SLOTS:
                        item["meal"] = reply["meal"]
            need = self._apply_amounts(items, need, reply)
        else:
            raw = self._complete(EATEN_PROMPT.format(question=state["question"]), json_mode=True)
            items = [i for i in parse_json_response(raw).get("items") or [] if isinstance(i, dict)]
            picked, need = None, []
        if not items:
            return {"answer": "Не понял, что именно съедено. Напишите, например: "
                              "«на обед съел пиццу, 300 г»."}

        if picked is None:
            # Сопоставляем ДО уточнений: напиток ли это, знает только каталог,
            # а на ответ «стакан» модель второй раз уже не зовётся.
            picked, missing = self._pick_eaten(items)
            if missing:
                return {"answer": (f"Не нашёл в справочнике: {', '.join(missing)}. Назовите "
                                   "проще (например, «пицца с сыром») — и пересчитаю остаток дня.")}
            need = [n for n, (item, record) in enumerate(zip(items, picked, strict=True))
                    if self._needs_amount(item, record)]

        question = self._eaten_question(items, need)
        if question:
            return {"answer": question,
                    "eaten_pending_new": {"items": items, "picked": picked, "need_amount": need}}
        return {"eaten_new": [
            {"fdc_id": record["fdc_id"], "grams": self._eaten_grams(item, record),
             "meal": item["meal"]}
            for item, record in zip(items, picked, strict=True)]}

    def _pick_eaten(self, items: list[dict]) -> tuple[list[dict], list[str]]:
        """Блюдо каталога для каждой позиции: кандидатов ищет код, выбирает модель."""
        candidates = [self._eaten_candidates(item) for item in items]
        blocks = "\n\n".join(
            f"Позиция {n}: «{item.get('phrase_ru')}»\n" + "\n".join(
                f"  {k}. {c['name']} ({c.get('name_ru') or ''})" for k, c in enumerate(found, 1))
            for n, (item, found) in enumerate(zip(items, candidates, strict=True), 1))
        choices = parse_json_response(self._complete(
            EATEN_PICK_PROMPT.format(blocks=blocks), json_mode=True)).get("choices") or []
        picked, missing = [], []
        for n, (item, found) in enumerate(zip(items, candidates, strict=True)):
            choice = choices[n] if n < len(choices) else 0
            if not isinstance(choice, int) or not 1 <= choice <= len(found):
                missing.append(str(item.get("phrase_ru") or "?"))
                continue
            record = found[choice - 1]
            # Только числа JSON: выбранное ждёт ответа в сохранённом состоянии.
            picked.append({"fdc_id": int(record["fdc_id"]),
                           "portion_g": float(record["portion_g"])})
        return picked, missing

    def _needs_amount(self, item: dict, record: dict) -> bool:
        """Калорийный напиток, о количестве которого человек не сказал ни слова."""
        return (self._is_drink(record) and drink_amount(item) is None
                and not item.get("ml") and not item.get("unspecified"))

    def _is_drink(self, record: dict) -> bool:
        row = self.catalog.loc[self.catalog["fdc_id"] == record["fdc_id"]]
        return not row.empty and is_drink(row.iloc[0].get("category"), row.iloc[0].get("kcal"))

    @staticmethod
    def _apply_amounts(items: list[dict], need: list[int], reply: dict) -> list[int]:
        """Дописать названные количества; вернуть, у каких напитков их всё ещё нет.

        Количеств столько же, сколько напитков, — по порядку; одно на
        несколько — каждому («по стакану»); иначе не угадываем и спросим снова.
        """
        if not need:
            return need
        if reply.get("unspecified"):
            for n in need:
                items[n]["unspecified"] = True
            return []
        amounts = reply.get("amounts") or []
        if len(amounts) == 1:
            amounts = amounts * len(need)
        if len(amounts) != len(need):
            return need
        for n, amount in zip(need, amounts, strict=True):
            items[n].update(amount)
        return []

    @staticmethod
    def _eaten_question(items: list[dict], need: list[int]) -> str | None:
        """Одно уточнение на всё, чего не хватает: приём пищи и количество напитков."""
        unknown = [str(i.get("phrase_ru") or "?") for i in items
                   if i.get("meal") not in _MEAL_SLOTS]
        drinks = [str(items[n].get("phrase_ru") or "?") for n in need]
        parts = []
        if unknown:
            parts.append(f"на какой приём пищи было: {', '.join(unknown)} — "
                         "завтрак, обед, ужин или перекус?")
        if drinks:
            parts.append(f"сколько выпили: {', '.join(drinks)}? Подойдут миллилитры, "
                         "стаканы или граммы — например, «330 мл» или «стакан».")
        if not parts:
            return None
        return "Уточните, " + " И ".join(parts) + " Тогда пересчитаю остаток дня."

    def _eaten_grams(self, item: dict, record: dict) -> float:
        """Граммы съеденного — по словам человека и таблице порций USDA.

        Не справочная порция каталога: та — «1 cup» или целая пицца.
        Объём напитка — через его плотность из той же таблицы.
        """
        fdc_id, portion_g = int(record["fdc_id"]), float(record["portion_g"])
        if item.get("unspecified"):
            return unspecified_grams(fdc_id, portion_g)
        if not item.get("ml") and not item.get("grams") and self._is_drink(record):
            # Количество напитка — из слов человека (drink_amount): «стакан
            # колы» без этого становился «средним напитком» — 512 г.
            item = {**item, **(drink_amount(item) or {})}
        if item.get("ml"):
            return float(item["ml"]) * grams_per_ml(fdc_id)
        return float(eaten_grams(fdc_id, item.get("grams"), item.get("count"), portion_g,
                                 unit=item.get("unit")))

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
        kept = self._kept_before(state, profile, everything, covered)
        try:
            plan = rebuild_rest(
                self.catalog, targets, items, covered, exclude=profile.exclude,
                include=_with_russian_fallback(
                    _resolve_preferences(profile), profile, self.catalog),
                budget=profile.budget_rub_per_day, seed=self.seed, kept=kept)
        except PlanNotFeasible as error:
            # Съеденное всё равно запоминается: оно было, и для адаптивной
            # нормы важно, сколько человек на самом деле съел.
            return {"eaten_all": everything, "plan": None, "answer": (
                f"Учёл съеденное. {error} На сегодня норма набрана — новых приёмов "
                "пищи я не добавляю.")}
        return {"plan": plan, "checks": check_plan(plan), "eaten_all": everything,
                "eaten_shown": shown, "eaten_kept": list(kept.values()),
                "menu": Menu(days=[plan], targets=dict(plan.targets))}

    def _kept_before(self, state: AgentState, profile: Profile, everything: list[dict],
                     covered: set[int]) -> dict[int, MealItem]:
        """Блюда прежнего плана на приёмах пищи ДО съеденного — {слот: блюдо}.

        Живой прогон: «на обед съел пиццу» пересобрал и завтрак — йогурт
        сменился вафлей, хотя к обеду завтрак уже съеден. «До» — по
        основным приёмам (завтрак → обед → ужин). Когда был перекус, не
        знает никто, поэтому перекусы не оставляются и сами ничего не
        оставляют. Блюдо, которое не проходит ограничений этой реплики,
        тоже не оставляется: его соберут заново.
        """
        last = state.get("last_plan")
        eaten_main = [_MEAL_ORDER.index(e["meal"]) for e in everything if e["meal"] in _MEAL_ORDER]
        if not last or not eaten_main:
            return {}
        earlier = {i for meal in _MEAL_ORDER[:max(eaten_main)] for i in _MEAL_SLOTS[meal]}
        earlier -= covered
        if not earlier:
            return {}
        allowed = set(_filter_catalog(self.catalog, profile.exclude)["fdc_id"].astype(int))
        kept: dict[int, MealItem] = {}
        for item in DayPlan(**last).items:
            if item.slot.endswith("(съедено)") or item.fdc_id not in allowed:
                continue
            index = next((i for i in sorted(earlier)
                          if DAY_TEMPLATE[i][0] == item.slot and i not in kept), None)
            if index is not None:
                kept[index] = item
        return kept

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
            text = eaten_note(state["eaten_shown"], plan, state.get("checks") or {},
                              state.get("eaten_kept") or [])
            chunks, query = [], None
        else:
            # Норма ПОКАЗАННОГО дня, а не средняя за неделю: живой прогон —
            # текст называл 2720 ккал, карточка рядом 2992 (тренировочный
            # день). Замер experiments/explain_norm.py: со средней текст спорил
            # с карточкой в 9 ответах из 24, с нормой дня — ни в одном.
            shown = state.get("plan_targets") or targets
            text, chunks, query = self._explain_with_model(state, plan, shown, profile, days)

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
        # После «съел» в карточке ОДИН сегодняшний день: живой прогон писал
        # «за 7 дней выйдет около 2910 ₽», умножая на неделю пересобранный день.
        after_eaten = bool(state.get("eaten_shown"))
        budget = self._budget_note(plan, 1 if after_eaten else days, after_eaten=after_eaten)
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
    def _budget_note(plan: DayPlan, days: int, after_eaten: bool = False) -> str:
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
            text = f"Уложился в ваш бюджет: {plan.cost:.0f} ₽ из {budget:.0f} ₽ в день"
            if left >= budget * 0.1:
                text += f" — остаётся ещё около {left:.0f} ₽"
            return text + "."

        over = plan.cost - budget
        text = (f"⚠️ В бюджет уложиться не удалось: {plan.cost:.0f} ₽ в день "
                f"против ваших {budget:.0f} ₽ — перерасход {over:.0f} ₽ "
                f"({over / budget:.0%}).")

        # Называем причину, а не только факт: при такой норме дешевле
        # уже не собрать из того, что есть в справочнике. После «съел» причина
        # другая: живой прогон советовал «убрать ограничения» человеку,
        # у которого их не было, — сумму подняло съеденное.
        if after_eaten:
            text += (" В сумме уже съеденное и приёмы пищи до него; остаток дня "
                     "я собрал так, чтобы день уложился в норму, — она важнее бюджета.")
        else:
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
