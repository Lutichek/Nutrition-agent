"""
Демонстрация агента: один разговор, проходящий через все ветки графа.

Запуск::

    python demo.py                 # весь сценарий
    python demo.py --provider gigachat

Перед первым запуском нужно собрать данные и индекс::

    python data_sources.py
    python foods.py --rebuild
    python build_index.py
"""

from __future__ import annotations

import sys

from agent import build_agent
from providers import get_clients

# Сценарий подобран так, чтобы задеть каждую ветку графа: уточнение профиля,
# детерминированный расчёт, правку профиля по ходу разговора, вопрос
# по литературе, отказ по медицинским показаниям и отказ не по теме.
SCRIPT: list[tuple[str, str]] = [
    ("уточнение профиля", "Привет! Хочу похудеть, составь мне рацион на день"),
    ("расчёт плана", "Мне 31, я женщина, рост 168, вес 74, работа сидячая, свинину не ем"),
    ("правка профиля", "Забыла сказать — три раза в неделю хожу в зал"),
    ("вопрос по литературе", "А сколько белка нужно есть при похудении и почему?"),
    ("вопрос по литературе", "Правда, что есть после шести вредно?"),
    ("не по теме", "Какая завтра погода в Москве?"),
]


def main() -> None:
    provider = "polza"
    if "--provider" in sys.argv:
        provider = sys.argv[sys.argv.index("--provider") + 1]

    llm_client, model, embed_client = get_clients(provider)
    agent = build_agent(llm_client, model=model, embed_client=embed_client)

    for label, message in SCRIPT:
        print("=" * 78)
        print(f"[{label}]")
        print(f"ПОЛЬЗОВАТЕЛЬ: {message}")
        print("-" * 78)

        answer = agent.run(message, verbose=True)

        print()
        print(f"АГЕНТ: {answer.answer}")
        print()

    print("=" * 78)
    print("Профиль, накопленный за разговор:")
    print(agent.profile)


if __name__ == "__main__":
    main()
