"""
Проверка работоспособности: всё ли готово к запуску агента.

Проходит по слоям снизу вверх — от настроек до полного ответа агента —
и на каждом шаге говорит, что делать, если что-то не готово.

Слои проверяются именно в этом порядке не случайно: каждый следующий
зависит от предыдущего. Если не собран каталог, бессмысленно выяснять,
почему не отвечает агент.

Запуск::

    python check_setup.py           # полная проверка
    python check_setup.py --offline # без обращений к API (бесплатно)

Код возврата 0 — всё в порядке, 1 — что-то не готово. Удобно для CI.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

OK = "  [ OK ]"
FAIL = "  [ НЕТ ]"
SKIP = "  [ ... ]"

failures: list[str] = []


def report(passed: bool, title: str, detail: str = "", fix: str = "") -> bool:
    """Напечатать результат шага и запомнить провал."""
    print(f"{OK if passed else FAIL} {title}")
    if detail:
        print(f"         {detail}")
    if not passed:
        failures.append(title)
        if fix:
            print(f"         → {fix}")
    return passed


# ────────────────────────────────────────────────────────────
# 1. Настройки
# ────────────────────────────────────────────────────────────


def check_config(offline: bool) -> bool:
    print("\n1. Настройки и ключи")
    try:
        from config import settings
    except Exception as error:
        return report(False, "Модуль config", str(error)[:70],
                      "проверьте, что зависимости установлены: uv sync")

    env_exists = (HERE / ".env").exists()
    report(env_exists, "Файл .env",
           fix="скопируйте .env.example в .env и впишите ключ")

    has_key = bool(settings.polza_ai_api_key)
    if offline:
        print(f"{SKIP} Ключ POLZA_AI_API_KEY — пропущено (--offline)")
        return env_exists

    return report(has_key, "Ключ POLZA_AI_API_KEY",
                  fix="получите ключ на polza.ai и впишите в .env") and env_exists


# ────────────────────────────────────────────────────────────
# 2. Данные и каталог
# ────────────────────────────────────────────────────────────


def check_catalog() -> bool:
    print("\n2. Каталог блюд")
    from foods import CATALOG_PATH

    if not CATALOG_PATH.exists():
        return report(False, "Каталог собран", "",
                      "python data_sources.py && python foods.py --rebuild")

    from foods import check_catalog as verify
    from foods import load_catalog

    catalog = load_catalog()
    ok = report(len(catalog) > 1000, f"Каталог собран: {len(catalog)} блюд")

    # Сверка калорийности с составом: ловит порчу данных, которую
    # глазами не заметить.
    try:
        stats = verify(catalog)
        ok &= report(
            stats["atwater_within_25pct"] > 0.9,
            "Данные согласованы",
            f"калорийность сходится с составом у {stats['atwater_within_25pct']:.1%} блюд",
        )
    except AssertionError as error:
        ok &= report(False, "Данные согласованы", str(error)[:70],
                     "пересоберите каталог: python foods.py --rebuild")
    return ok


# ────────────────────────────────────────────────────────────
# 3. Поисковый индекс
# ────────────────────────────────────────────────────────────


def check_index() -> bool:
    print("\n3. Поисковый индекс")
    from lance_db import DEFAULT_DB_PATH, DEFAULT_TABLE_NAME, open_table, table_exists

    if not table_exists(DEFAULT_DB_PATH, DEFAULT_TABLE_NAME):
        return report(False, "Индекс собран", "", "python build_index.py")

    table = open_table(DEFAULT_DB_PATH, DEFAULT_TABLE_NAME)
    rows = table.count_rows()
    ok = report(rows > 100, f"Индекс собран: {rows} фрагментов")

    from data_preporation import CORPUS_INDEX_PATH
    ok &= report(CORPUS_INDEX_PATH.exists(), "Справочник источников на месте",
                 fix="python build_index.py")
    return ok


# ────────────────────────────────────────────────────────────
# 4. Расчёт без модели
# ────────────────────────────────────────────────────────────


def check_deterministic() -> bool:
    """Ядро проекта: нормы и подбор рациона. Модель здесь не участвует."""
    print("\n4. Расчёт нормы и рациона (без обращений к модели)")

    from foods import load_catalog
    from solver import build_day, check_plan
    from targets import Profile, compute_targets

    profile = Profile(sex="female", age=31, height_cm=168, weight_kg=74,
                      activity="sedentary", goal="lose")
    targets = compute_targets(profile)
    ok = report(targets.kcal > 1000, f"Норма посчитана: {targets.kcal:.0f} ккал")

    started = time.perf_counter()
    plan = build_day(load_catalog(), targets, seed=0)
    elapsed = time.perf_counter() - started

    ok &= report(len(plan.items) == 7,
                 f"Рацион собран: {len(plan.items)} блюд, {plan.totals['kcal']:.0f} ккал",
                 f"за {elapsed:.1f} с")

    checks = check_plan(plan)
    failed = [name for name, passed in checks.items() if not passed]
    ok &= report(not failed, f"Проверки рациона: {len(checks) - len(failed)}/{len(checks)}",
                 "не прошли: " + ", ".join(failed) if failed else "")

    # Один seed обязан давать один и тот же план — иначе воспроизводимости нет.
    again = build_day(load_catalog(), targets, seed=0)
    ok &= report([i.fdc_id for i in plan.items] == [i.fdc_id for i in again.items],
                 "Результат воспроизводим")
    return ok


# ────────────────────────────────────────────────────────────
# 5. Связь с провайдером
# ────────────────────────────────────────────────────────────


def check_providers() -> bool:
    print("\n5. Связь с провайдером моделей")
    from providers import get_clients

    try:
        client, model, embed_client = get_clients()
    except RuntimeError as error:
        return report(False, "Клиенты созданы", str(error)[:70])

    # Эмбеддинги: самый дешёвый запрос, которым можно проверить доступ.
    try:
        from embedder import Embedder
        vector = Embedder(embed_client).embed_query("проверка связи")
        ok = report(len(vector) == 1536, f"Эмбеддинги отвечают: вектор из {len(vector)} чисел")
    except Exception as error:
        return report(False, "Эмбеддинги отвечают", _short(error),
                      _hint(error))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Ответь одним словом: работает?"}],
            temperature=0,
        )
        ok &= report(bool(response.choices[0].message.content),
                     f"Модель {model} отвечает")
    except Exception as error:
        ok &= report(False, f"Модель {model} отвечает", _short(error), _hint(error))
    return ok


def _short(error: Exception) -> str:
    return str(error).split("\n")[0][:90]


def _hint(error: Exception) -> str:
    text = str(error)
    if "402" in text or "INSUFFICIENT" in text.upper():
        return "на счёте провайдера закончились средства — пополните баланс"
    if "401" in text or "403" in text:
        return "ключ неверный или отозван — проверьте POLZA_AI_API_KEY в .env"
    if "503" in text or "SERVICE_UNAVAILABLE" in text.upper():
        return "сервис провайдера временно недоступен — повторите позже"
    return "проверьте интернет и значение POLZA_AI_API_KEY в .env"


# ────────────────────────────────────────────────────────────
# 6. Агент целиком
# ────────────────────────────────────────────────────────────


def check_agent() -> bool:
    print("\n6. Агент целиком")
    from agent import build_agent
    from providers import get_clients

    client, model, embed_client = get_clients()
    agent = build_agent(client, model=model, embed_client=embed_client)

    try:
        started = time.perf_counter()
        answer = agent.run("Сколько белка нужно есть при похудении?", remember=False)
        elapsed = time.perf_counter() - started
    except Exception as error:
        return report(False, "Ответ на вопрос по литературе", _short(error), _hint(error))

    ok = report(len(answer.answer) > 50,
                "Ответ на вопрос по литературе",
                f"{len(answer.answer)} символов за {elapsed:.0f} с")
    ok &= report(bool(answer.retrieved_chunks),
                 f"Источники найдены: {len(answer.retrieved_chunks or [])} фрагментов")
    return ok


# ────────────────────────────────────────────────────────────


def main() -> None:
    offline = "--offline" in sys.argv

    print("=" * 62)
    print("ПРОВЕРКА ГОТОВНОСТИ АГЕНТА")
    if offline:
        print("режим --offline: обращений к API не будет")
    print("=" * 62)

    steps_ok = check_config(offline)
    steps_ok &= check_catalog()
    steps_ok &= check_index()
    steps_ok &= check_deterministic()

    if offline:
        print("\n5-6. Связь с провайдером и агент — пропущено (--offline)")
    else:
        if check_providers():
            check_agent()
        else:
            print("\n6. Агент целиком")
            print(f"{SKIP} Пропущено: сначала нужна связь с провайдером")

    print()
    print("=" * 62)
    if failures:
        print(f"НЕ ГОТОВО: не прошло проверок — {len(failures)}")
        for name in failures:
            print(f"  · {name}")
        print("\nПодсказки, что делать, выведены выше рядом с каждым пунктом.")
        sys.exit(1)

    print("ВСЁ ГОТОВО. Запускайте: uvicorn api:app")
    print("Интерфейс будет на http://127.0.0.1:8000")
    sys.exit(0)


if __name__ == "__main__":
    main()
