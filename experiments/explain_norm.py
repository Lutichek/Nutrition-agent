"""
Норма в тексте объяснения плана: какую давать модели и нужна ли она вообще.

Живой прогон: текст называл 2720 ккал — среднюю за неделю норму, а
карточка рядом показывала 2992 — норму тренировочного дня, который в ней
и собран. Модели в промпт шёл расчёт по СРЕДНЕЙ норме (state["targets"]).
Числа нормы карточка показывает и так, поэтому вопрос двойной: какую
норму давать модели и повторять ли числа в тексте.

Варианты:

* «средняя (было)» — расчёт по средней норме, промпт прежний;
* «норма дня» — расчёт по норме показанного дня, промпт прежний;
* «дня + не повторяй» — то же, а пункт 1 промпта просит объяснить,
  ПОЧЕМУ норма такая, не повторяя чисел карточки (число — только если
  о нём спросили).

Метрики на профилях с тренировками и без, по двум репликам — просьба
о плане и вопрос «сколько белка мне нужно»:

* «спорит с карточкой» — в тексте калорийность, которой нет в карточке
  (средняя вместо дневной);
* «повторяет карточку» — числа нормы (ккал, Б, Ж, У) из карточки в тексте
  ответа на просьбу о плане, где о них не спрашивали;
* «ответ на вопрос» — на «сколько белка» названа норма белка из карточки;
* длина ответа, символов.

    python -m experiments.explain_norm

Результат (4 профиля × 2 реплики × 3 прогона = 24 ответа на вариант):

    вариант                         спорит  повторяет  белок   символов
    средняя (было)                     9      12/12    12/12     842
    норма дня                          0      12/12    12/12     859
    дня + «не повторяй» (принято)      0       9/12    12/12     714
    дня, строка нормы кодом — только
      по вопросу                       1      11/12    12/12     923
    то же + «не повторяй»              0       8/12    12/12     703

Спор с карточкой снимает норма показанного дня. Убирать строку «Норма
на день» кодом бесполезно: числа модель видит и в самом плане, выигрыш
в пределах шума — вариант отклонён и из кода убран.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nodes_plan  # noqa: E402
from agent import build_agent  # noqa: E402
from prompts import EXPLAIN_PLAN_PROMPT  # noqa: E402
from providers import get_clients  # noqa: E402
from targets import Profile  # noqa: E402

RUNS = 3
PROFILES = {
    "мужчина, 3 тренировки": Profile(sex="male", age=35, height_cm=180, weight_kg=80,
                                     activity="moderate", goal="maintain"),
    "женщина, похудение, 1 тренировка": Profile(sex="female", age=31, height_cm=168,
                                                weight_kg=74, activity="light", goal="lose"),
    "спортсмен, 6 тренировок": Profile(sex="male", age=25, height_cm=178, weight_kg=75,
                                       activity="athlete", goal="gain"),
    "без тренировок": Profile(sex="female", age=55, height_cm=160, weight_kg=58,
                              activity="sedentary", goal="maintain"),
}
QUESTIONS = {"план": "Составь план питания на день",
             "белок": "Сколько белка мне нужно в день?"}

OLD_POINT = "1. Коротко — какая норма получилась и почему именно такая."
# Дословно как в prompts.py: замер переключает пункт заменой строки.
NEW_POINT = (
    "1. Коротко — почему норма именно такая: что учтено (обмен, активность, цель,\n"
    "   тренировочный это день или день отдыха). Числа нормы человек видит\n"
    "   в карточке рядом — не повторяй их; число называй, только если о нём\n"
    "   спросили (п. 0).")
_OLD = EXPLAIN_PLAN_PROMPT.replace(NEW_POINT, OLD_POINT)
_NEW = EXPLAIN_PLAN_PROMPT.replace(OLD_POINT, NEW_POINT)
# (какая норма, промпт)
VARIANTS = {
    "средняя (было)": ("base", _OLD),
    "норма дня": ("day", _OLD),
    "дня + «не повторяй»": ("day", _NEW),
}


def numbers(text: str) -> set[int]:
    return {int(n.replace(" ", "").replace(" ", ""))
            for n in re.findall(r"\d[\d  ]*\d|\d", text)}


def main() -> None:
    client, model, embed_client = get_clients("polza")
    agent = build_agent(client, model=model, embed_client=embed_client)
    states = {}
    for name, profile in PROFILES.items():
        made = agent._node_make_plan({"profile": profile.model_dump(), "question": ""})
        states[name] = made

    print(f"{'вариант':30} {'спорит':>7} {'повтор.':>8} {'белок':>6} {'символов':>9}")
    for variant, (which, prompt) in VARIANTS.items():
        nodes_plan.EXPLAIN_PLAN_PROMPT = prompt
        conflict = repeat = protein_ok = asked = plans = 0
        length = []
        for made in states.values():
            card = made["plan_targets"]
            targets = made["targets"] if which == "base" else card
            profile = Profile(**made["profile"])
            card_kcal = round(card.kcal)
            card_numbers = {card_kcal, round(card.protein_g), round(card.fat_g),
                            round(card.carb_g)}
            for kind, question in QUESTIONS.items():
                for _ in range(RUNS):
                    state = {**made, "question": question}
                    text, _, _ = agent._explain_with_model(state, made["plan"], targets,
                                                           profile, 1)
                    found = numbers(text)
                    length.append(len(text))
                    kcal_like = {n for n in found if 1000 <= n <= 6000}
                    # Калорийность нормы, которой нет в карточке: средняя за
                    # неделю (обмен и расход по формуле — не норма, их не считаем).
                    conflict += round(made["targets"].kcal) in kcal_like and \
                        round(made["targets"].kcal) != card_kcal
                    if kind == "план":
                        plans += 1
                        repeat += bool(found & card_numbers)
                    else:
                        asked += 1
                        protein_ok += round(card.protein_g) in found
        print(f"{variant:30} {conflict:>7} {repeat:>5}/{plans:<2} {protein_ok:>3}/{asked:<2} "
              f"{sum(length) / len(length):>9.0f}")


if __name__ == "__main__":
    main()
