"""
Ревизия каталога: качество перевода и доступность блюд в России и СНГ.

Справочник USDA описывает то, что едят в США. Часть блюд у нас не купить
и не приготовить: гритс, окра, коллард, батат конкретных сортов, блюда
из сетей быстрого питания, пуэрто-риканская и соул-фуд кухня. Предлагать
их в рационе бессмысленно — человек просто не найдёт продукты.

Вторая задача того же прохода — проверить переводы. Их делала модель,
и ошибки вроде «Rice cake → Рисовый пирог» (на самом деле хлебцы)
поодиночке не выловить.

Обе проверки объединены в один запрос: модель и так читает название,
незачем платить дважды.

Скрипт добавляет в каталог колонку ``available_ru`` и правит ``name_ru``
там, где перевод неверен. Ничего не удаляется: недоступные блюда остаются
в справочнике (на вопрос о них можно ответить), но в подбор рациона
не идут — см. ``solver._filter_catalog``.

Запуск::

    python review_catalog.py            # проверить непроверенное
    python review_catalog.py --force    # проверить заново
    python review_catalog.py --report   # только показать итоги, без запросов
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from agent import parse_json_response
from foods import CATALOG_PATH, load_catalog
from providers import get_clients
from translation_fixes import apply as apply_translation_fixes

BATCH_SIZE = 40
MAX_WORKERS = 6

PROMPT = """Ты — редактор справочника питания для русскоязычных пользователей.

Проверь список блюд: у каждого оригинальное название на английском и перевод.

ЗАДАЧА 1. Найти переводы, искажающие смысл блюда.
Примеры ошибок: «Rice cake» → «Рисовый пирог» (это хлебцы);
«Abalone» → «Устрицы абалон» (это морское ушко, не устрица);
«Grits» → «Крупа» (это кукурузная каша конкретного вида).
Стилистику и длинноты не трогай — только смысловые ошибки.

ЗАДАЧА 2. Отметить блюда, которых НЕТ в обычном российском супермаркете
или которые невозможно приготовить из доступных у нас продуктов.

Отмечай недоступным:
- продукты, не продающиеся в России и СНГ: абалон, окра, коллард, гритс,
  батат особых сортов, плантан, джикама, черноглазый горох, кукурузная мука
  для корнбреда;
- позиции сетей быстрого питания и американских ресторанов;
- национальные блюда, продукты для которых у нас не найти: пуэрто-риканские,
  соул-фуд, гавайские, филиппинские (адобо), карибские;
- американские десерты и блюда, которых у нас не делают: амброзия, гритс,
  бисквиты южного типа, джамбалайя.

НЕ отмечай: курицу, рыбу, крупы, овощи, молочное, макароны, орехи, фрукты,
привычную выпечку, сэндвичи из обычных продуктов, блюда европейской
и азиатской кухни, продукты для которых есть в супермаркете.

Верни ТОЛЬКО JSON. Не используй двойные кавычки внутри строк:

{{"fix": [{{"n": номер, "name": "правильный перевод"}}],
  "unavailable": [номера недоступных]}}

Пустые списки, если ничего не нашлось.

Блюда ({count} шт.):
{items}"""


def review_batch(client, model: str, rows: list[dict], max_retries: int = 3) -> tuple[dict, set]:
    """Проверить пачку. Возвращает (исправления по номеру, множество недоступных)."""
    listing = "\n".join(
        f"{i}. {row['name']} | {row['name_ru']}" for i, row in enumerate(rows, start=1)
    )

    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": PROMPT.format(count=len(rows), items=listing)}],
                temperature=0.0,
                timeout=90.0,
            )
            parsed = parse_json_response(response.choices[0].message.content or "")
            if not parsed:
                continue

            fixes = {}
            for item in parsed.get("fix", []):
                number = item.get("n")
                name = str(item.get("name", "")).strip()
                # Номера вне диапазона игнорируем: модель иногда фантазирует
                # индексы, и без проверки правка уехала бы на чужое блюдо.
                if isinstance(number, (int, float)) and 1 <= int(number) <= len(rows) and name:
                    fixes[int(number) - 1] = name

            unavailable = {
                int(n) - 1
                for n in parsed.get("unavailable", [])
                if isinstance(n, (int, float)) and 1 <= int(n) <= len(rows)
            }
            return fixes, unavailable
        except Exception:
            continue

    return {}, set()


def report(catalog: pd.DataFrame) -> None:
    """Показать итоги ревизии."""
    if "available_ru" not in catalog.columns:
        print("Ревизия ещё не проводилась.")
        return

    unavailable = ~catalog["available_ru"].fillna(True).astype(bool)
    print(f"  всего блюд:        {len(catalog)}")
    print(f"  доступно в СНГ:    {(~unavailable).sum()} ({(~unavailable).mean():.1%})")
    print(f"  исключено:         {unavailable.sum()}")

    if unavailable.any():
        print("\n  примеры исключённых:")
        for name in catalog.loc[unavailable, "name_ru"].head(12):
            print(f"    {name}")


def main() -> None:
    catalog = load_catalog()

    if "--report" in sys.argv:
        report(catalog)
        return

    force = "--force" in sys.argv
    if "available_ru" not in catalog.columns:
        catalog["available_ru"] = pd.NA

    todo = catalog["available_ru"].isna()
    if force:
        todo = pd.Series(True, index=catalog.index)

    positions = list(catalog.index[todo])
    if not positions:
        print("Все блюда уже проверены. Для повтора: --force")
        report(catalog)
        return

    print(f"Блюд к проверке: {len(positions)}")
    client, model, _ = get_clients("polza")

    batches = [positions[i:i + BATCH_SIZE] for i in range(0, len(positions), BATCH_SIZE)]

    def work(batch: list) -> tuple[list, dict, set]:
        rows = catalog.loc[batch, ["name", "name_ru"]].to_dict("records")
        fixes, unavailable = review_batch(client, model, rows)
        return batch, fixes, unavailable

    fixed_count = 0
    unavailable_count = 0
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for batch, fixes, unavailable in pool.map(work, batches):
            for offset, name in fixes.items():
                catalog.loc[batch[offset], "name_ru"] = name
                fixed_count += 1
            for offset in range(len(batch)):
                catalog.loc[batch[offset], "available_ru"] = offset not in unavailable
            unavailable_count += len(unavailable)

            done += 1
            if done % 20 == 0:
                print(f"  проверено пачек {done}/{len(batches)}")

    catalog["available_ru"] = catalog["available_ru"].fillna(True).astype(bool)

    # Ревизия правит name_ru моделью, а значит, может занести те же
    # устойчивые ошибки заново. Исправления применяются последними.
    apply_translation_fixes(catalog)

    catalog.to_parquet(CATALOG_PATH, index=False)

    print(f"\nИсправлено переводов: {fixed_count}")
    print(f"Отмечено недоступными: {unavailable_count}")
    print(f"Сохранено → {CATALOG_PATH}\n")
    report(catalog)


if __name__ == "__main__":
    main()
