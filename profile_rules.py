"""
Правила профиля и плана, которые держит код, а не модель.

Аллергии, замена блюда, «съел не по плану», разбор ответа по схеме,
пожелания и их конфликты с ограничениями. У каждого правила рядом —
история тихой ошибки, из-за которой оно появилось.

Вынесено из agent.py (ревью R6), чтобы модули узлов графа могли опираться
на эти правила без циклического импорта. agent.py реэкспортирует имена.
"""

from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd

from prompts import ALLERGY_NOTE
from restrictions import expand, matches_terms
from solver import DayPlan, MealItem, preferred_ids
from targets import Profile

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


# Поля профиля, без которых расчёт невозможен.
#
# activity здесь не случайно. У Profile есть значение по умолчанию
# («sedentary»), и раньше агент молча его подставлял: человек писал
# «хочу рекомпозицию», получал норму как для сидячего образа жизни
# и не понимал, почему калорий так мало. Между «сидячим» и «высоким»
# у мужчины 65 кг лежит 830 ккал в день — это не мелочь, о которой
# можно догадаться за пользователя.
REQUIRED_PROFILE_FIELDS = ["sex", "age", "height_cm", "weight_kg", "activity"]


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
