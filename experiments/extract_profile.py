"""
Извлечение профиля: схема с разбором упоминаний (SGR) против свободного JSON.

Что меряется — ИТОГ после кода агента, а не сырой ответ модели: фраза
идёт в настоящий ``_node_extract_profile`` (там разбор периода, цели,
активности, бюджета, перенос диет из пожеланий). Исключения сравниваются
после раскрытия категорий (``restrictions.expand``) и с учётом числа слова,
как их сравнивает сам фильтр; пожелания — после отсева нутриентов.

Промах — это почти всегда утечка ограничения: продукт, от которого человек
отказался, остаётся в каталоге. Поэтому решающее число — «верно», а время
и токены — цена.

Наборы:

* ``easy`` — случаи из CLAUDE.md, на которых модель уже ошибалась
  (категории, «кроме», целиакия, «больше белка», манка и пшено);
* ``hard`` — разговорные формы, халяль, арахис, нутриенты в паре;
* ``mixed`` — смешанный знак в одной фразе: «ем / не ем / люблю»;
* ``multi`` — реплика после уже названного: старое обязано остаться.

Режимы чередуются внутри прогона — дрейф провайдера ложится поровну.

Запуск::

    python -m experiments.extract_profile               # все наборы, 3 прогона
    python -m experiments.extract_profile --sets mixed --repeats 5

Стоит денег: 38 фраз × 2 режима × прогоны (gpt-4o-mini — копейки).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent as agent_module  # noqa: E402
from agent import _NUTRIENT_WORDS, NutritionAgent  # noqa: E402
from providers import get_clients  # noqa: E402
from restrictions import expand  # noqa: E402

# (фраза, ожидаемые поля, exclude, include: [(англ. варианты, рус. основа)],
#  include обязан быть пуст)
EASY = [
    ("Мне 31, женщина, 168 см, 74 кг, сидячая работа, хочу похудеть",
     {"sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74}, [], [], True),
    ("Мужчина 45 лет, рост 182, вес 96",
     {"sex": "male", "age": 45, "height_cm": 182, "weight_kg": 96}, [], [], True),
    ("Не ем свинину, сладости, мучное и аллергия на лактозу", {},
     ["pork", "sweets", "flour", "lactose"], [], True),
    ("Ем всё, кроме рыбы", {}, ["fish"], [], True),
    ("У меня целиакия", {}, ["gluten"], [], True),
    ("Я веган", {}, ["vegan"], [], True),
    ("Аллергия на орехи и яйца", {}, ["nuts", "eggs"], [], True),
    ("Без грибов, пожалуйста", {}, ["mushroom"], [], True),
    ("Люблю гречку и творог", {}, [],
     [({"buckwheat"}, "греч"), ({"cottage cheese"}, "творог")], False),
    ("Хочу видеть в меню манку и пшено", {}, [],
     [({"farina", "semolina", "cream of wheat"}, "манк"), ({"millet"}, "пшен")], False),
    ("Добавь больше белка", {}, [], [], True),
    ("Хочу больше овощей и рыбы", {}, [],
     [({"vegetables", "vegetable"}, "овощ"), ({"fish"}, "рыб")], False),
    ("Не пью алкоголь и не ем жареное", {}, ["alcohol", "fried"], [], True),
    ("Хочу творог, но у меня непереносимость лактозы", {}, ["lactose"],
     [({"cottage cheese"}, "творог")], False),
    ("Не ем говядину и баранину, хочу курицу", {}, ["beef", "lamb"],
     [({"chicken"}, "куриц")], False),
    ("Не ем мясо", {}, ["meat"], [], True),
    ("Мне 28, женщина, 160 см, 55 кг, вегетарианка, люблю сыр",
     {"sex": "female", "age": 28, "height_cm": 160, "weight_kg": 55},
     ["vegetarian"], [({"cheese"}, "сыр")], False),
    ("Без глютена и без молочки", {}, ["gluten", "lactose"], [], True),
]
HARD = [
    ("Не переношу глютен, но хочу гречку", {}, ["gluten"], [({"buckwheat"}, "греч")], False),
    ("Хочу похудеть к лету: 34 года, 172 см, 81 кг, мужик",
     {"sex": "male", "age": 34, "height_cm": 172, "weight_kg": 81}, [], [], True),
    ("Вес 70, рост 175, 40 лет, женщина. Не ем морепродукты и свинину.",
     {"sex": "female", "age": 40, "height_cm": 175, "weight_kg": 70},
     ["seafood", "pork"], [], True),
    ("Люблю овсянку на завтрак, а вот рыбу терпеть не могу", {}, ["fish"],
     [({"oatmeal", "oats", "oat"}, "овся")], False),
    ("Только халяль", {}, ["halal"], [], True),
    ("Можно без яиц? И побольше творога", {}, ["eggs"], [({"cottage cheese"}, "творог")], False),
    ("Аллергия на арахис", {}, ["peanut"], [], True),
    ("Без сахара, пожалуйста", {}, ["sugar"], [], True),
    ("Хочу больше белка и клетчатки", {}, [], [], True),
    ("Молочку не ем, мясо ем, но не свинину, а рыбу люблю", {}, ["lactose", "pork"],
     [({"fish"}, "рыб")], False),
    ("Добавь в рацион больше орехов, но без арахиса", {}, ["peanut"],
     [({"nuts", "nut"}, "орех")], False),
]
MIXED = [
    ("Курицу люблю, свинину не ем, рыбу ем редко, но можно", {}, ["pork"],
     [({"chicken"}, "куриц")], False),
    ("Ем всё кроме грибов и баклажанов, особенно люблю гречку", {},
     ["mushroom", "eggplant"], [({"buckwheat"}, "греч")], False),
    ("Мясо да, свинина нет, сладкое нет, хлеб можно", {}, ["pork", "sweets"], [], False),
    ("Говядину не ем по религиозным причинам, свинину тоже, баранину ем", {},
     ["beef", "pork"], [], False),
    ("Люблю творог и яйца, не ем орехи", {}, ["nuts"],
     [({"cottage cheese"}, "творог"), ({"egg", "eggs"}, "яйц")], False),
    ("Овощи — да, мучное — нет, жареное тоже нет", {}, ["flour", "fried"],
     [({"vegetables", "vegetable"}, "овощ")], False),
    ("Пью кофе без молока, молочное вообще не ем, мясо ем любое кроме свинины", {},
     ["lactose", "pork"], [], False),
    ("Рыбу обожаю, курицу тоже, а свинину и колбасу не ем", {}, ["pork", "sausage"],
     [({"fish"}, "рыб"), ({"chicken"}, "куриц")], False),
    ("Сладкое можно, но без шоколада; молочку ем; глютен нельзя", {},
     ["chocolate", "gluten"], [], False),
]
# Реплика ПОСЛЕ уже названного: (что сказано раньше, профиль после него).
# Ожидаемое — старое вместе с новым. Одиночные фразы этого не ловят:
# схема с разбором на «Ещё не ем грибы» после «Не ем свинину» 3/3 теряла
# грибы — разбор был верный, а в exclude модель переписывала только известное.
_PORK = ("Не ем свинину", {"exclude": ["pork"]})
MULTI = [
    ("Ещё не ем грибы", {}, ["pork", "mushroom"], [], True, _PORK),
    ("И аллергия на арахис", {}, ["pork", "peanut"], [], True, _PORK),
    ("Хочу гречку", {}, ["pork"], [({"buckwheat"}, "греч")], False, _PORK),
    ("Мне 30, мужчина, 180 см, 80 кг",
     {"sex": "male", "age": 30, "height_cm": 180, "weight_kg": 80}, ["pork"], [], True, _PORK),
    ("Ещё без сладкого", {}, ["pork", "sweets"], [], True, _PORK),
    ("Люблю курицу, но не ем грибы", {}, ["lactose", "mushroom"],
     [({"chicken"}, "куриц")], False, ("Молочку не ем", {"exclude": ["lactose"]})),
]
SETS = {"easy": EASY, "hard": HARD, "mixed": MIXED, "multi": MULTI}
MODES = {"схема + разбор": True, "свободный JSON": False}


def _norm(word: str) -> str:
    # Число слова фильтр ловит в обе стороны (peanut/peanuts) — сравниваем так же.
    return word.lower().removesuffix("s")


def problems(profile: dict, nums: dict, exclude: list, include: list, empty: bool) -> list[str]:
    found = []
    for field, value in nums.items():
        if profile.get(field) != value:
            found.append(f"{field}={profile.get(field)}")
    got = {_norm(w) for w in expand(profile.get("exclude") or [])}
    leak = sorted(w for w in expand(exclude) if _norm(w) not in got)
    if leak:
        found.append(f"утечка {leak[:3]} (exclude={profile.get('exclude')})")
    wanted = [t for t in profile.get("include") or [] if t.strip().lower() not in _NUTRIENT_WORDS]
    russian = " ".join((profile.get("include_ru") or {}).values()).lower()
    for variants, stem in include:
        if not any(t.lower() in variants for t in wanted) and stem not in russian:
            found.append(f"нет пожелания {sorted(variants)[0]}")
    if empty and wanted:
        found.append(f"лишнее include={wanted}")
    return found


def run_once(agent: NutritionAgent, phrase: str,
             prior: tuple[str, dict] | None = None) -> tuple[dict, float, int | None, int | None]:
    history, known = [], {}
    if prior:
        history = [{"role": "user", "content": prior[0]},
                   {"role": "assistant", "content": "Учту."}]
        known = dict(prior[1])
    calls_token = agent_module._LLM_CALLS.set([])
    started = time.perf_counter()
    try:
        profile = agent._node_extract_profile(
            {"question": phrase, "history": history, "profile": known})["profile"]
        calls = agent_module._LLM_CALLS.get() or []
    finally:
        agent_module._LLM_CALLS.reset(calls_token)
    seconds = time.perf_counter() - started
    prompt = sum(c.get("prompt_tokens", 0) for c in calls) or None
    completion = sum(c.get("completion_tokens", 0) for c in calls) or None
    return profile, seconds, prompt, completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sets", nargs="*", default=list(SETS), choices=list(SETS))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    client, model, _ = get_clients("polza")
    agent = NutritionAgent(client=client, model=model, retriever=None)

    for set_name in args.sets:
        cases = SETS[set_name]
        stats = {mode: {"ok": 0, "n": 0, "sec": [], "in": [], "out": []} for mode in MODES}
        for repeat in range(args.repeats):
            for phrase, nums, exclude, include, empty, *prior in cases:
                for mode, structured in MODES.items():
                    agent.structured_extract = structured
                    profile, seconds, tokens_in, tokens_out = run_once(
                        agent, phrase, prior[0] if prior else None)
                    found = problems(profile, nums, exclude, include, empty)
                    s = stats[mode]
                    s["n"] += 1
                    s["ok"] += not found
                    s["sec"].append(seconds)
                    if tokens_in:
                        s["in"].append(tokens_in)
                    if tokens_out:
                        s["out"].append(tokens_out)
                    if found:
                        print(f"  [{set_name} {repeat + 1}] {mode:<15} ✗ «{phrase[:42]}»: "
                              f"{'; '.join(found)}", flush=True)

        print(f"\n{set_name} ({len(cases)} фраз × {args.repeats}):")
        for mode, s in stats.items():
            med = lambda xs: f"{statistics.median(xs):.0f}" if xs else "—"  # noqa: E731
            print(f"  {mode:<15} верно {s['ok']}/{s['n']}   время {statistics.median(s['sec']):.2f} с"
                  f"   токены {med(s['in'])} / {med(s['out'])}")
        print()


if __name__ == "__main__":
    main()
