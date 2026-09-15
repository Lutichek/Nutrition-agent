"""
Второй проход ревизии каталога: едят ли это блюдо у нас.

Первый проход (``review_catalog.py``) отвечал на вопрос «можно ли достать
продукты». Он снял гритс, окру и позиции американских сетей, но пропустил
целый класс блюд, которые достать можно, а есть их никто не станет:
горькую дыню, розовую фасоль, семислойный салат, соус с колбасой,
рисово-зефирное печенье. Достать всё это в Москве при желании реально —
но человеку, попросившему меню, такое предлагать бессмысленно.

Здесь критерий другой: **окажется ли это блюдо на столе в России или СНГ**.

Речь не только о русской кухне. Итальянская, французская, грузинская,
японская, среднеазиатская давно свои: паста, пицца, ризотто, хачапури,
суши, плов, шаурма, хумус — всё это едят каждый день. Отсекается не
«иностранное», а незнакомое.

Заодно отсеиваются позиции, которые вообще не блюда: справочник USDA
содержит записи вроде «Огурец для сэндвича» и «Обёртка для сэндвича» —
они нужны для учёта съеденного, а не для составления меню.

Добавляет колонку ``familiar_ru``. Ничего не удаляет: непривычные блюда
остаются в справочнике и на вопрос о них можно ответить, но в подбор
рациона не идут — см. ``solver._filter_catalog``.

Запуск::

    python review_familiar.py            # проверить непроверенное
    python review_familiar.py --force    # проверить заново
    python review_familiar.py --report   # только итоги, без запросов

Около 3700 блюд пачками по 40 — примерно 95 коротких запросов.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from agent import parse_json_response
from foods import CATALOG_PATH, load_catalog
from providers import get_clients

MAX_WORKERS = 6

# Разделы справочника, которые модели не показывают вовсе: базовые продукты
# привычны по определению.
#
# Без этого списка первый полный прогон отсёк «Банан, сырой», «Творог,
# низкожирный», «Артишок» и 42 сыра из 57. Модель, которой показали список
# непривычного, начинает считать непривычным любое уточнение: «сухой»,
# «низкожирный», «безглютеновая». Спорить с ней промптом бесполезно —
# базовые продукты просто не должны зависеть от её суждения.
#
# Судить остаётся то, ради чего проверка и затевалась: составные блюда,
# выпечка, десерты, национальные кухни. Заодно вдвое меньше запросов.
ALWAYS_FAMILIAR = [
    # Молочное и яйца
    "cheese", "milk", "yogurt", "eggs and omelets",
    # Мясо, птица, рыба
    "beef", "pork", "chicken, whole pieces", "turkey, duck", "lamb, goat",
    "sausages", "bacon", "cold cuts", "fish", "shellfish",
    # Овощи
    "vegetables", "vegetable dishes", "potatoes", "carrots", "broccoli",
    "cabbage", "spinach", "onions", "tomatoes", "corn", "string beans",
    "lettuce", "coleslaw",
    # Фрукты и ягоды
    "apples", "bananas", "grapes", "pears", "peaches", "melons",
    "strawberries", "berries", "citrus fruits", "pineapple", "dried fruits",
    "fruits and fruit salads",
    # Крупы, бобовые, орехи
    "rice", "oatmeal", "pasta, noodles", "beans, peas, legumes",
    "nuts and seeds", "cooked cereals", "ready-to-eat cereal",
    # Напитки
    "coffee", "tea", "water", "juice",
]

# Пачка меньше, чем в первом проходе. На сорока блюдах модель перестаёт
# сверяться со списком и пропускает даже явные примеры из промпта.
BATCH_SIZE = 25

PROMPT = """Ты — шеф-повар, много лет составляющий меню для российских
заведений и семей.

Перед тобой блюда из американского справочника питания. По КАЖДОМУ реши:
окажется ли такое на столе в России, Беларуси, Казахстане или на Украине?

Речь НЕ только о русской кухне. Итальянская, французская, грузинская,
японская, среднеазиатская, ближневосточная давно свои: паста, пицца,
ризотто, лазанья, омлет, киш, хачапури, шашлык, плов, манты, суши,
шаурма, хумус, кебаб — всё это едят постоянно. Отсекается не иностранное,
а незнакомое.

Пройди по каждому блюду и проверь ДВА условия. Если сработало хотя бы
одно — блюдо в ответ, с кодом причины.

  «нет-продукта» — продукт, который у нас не едят.
    Примеры: горькая дыня, окра, коллард, плантан, джикама, маниока,
    черноглазый горох, розовая фасоль, васаби-горох, каштаны на гарнир.

  «не-готовят» — блюдо американской или иной кухни, которого у нас
    не подают и человек его не узнает.
    Примеры: семислойный салат, соус с колбасой, корнбред, гритс,
    амброзия, джамбалайя, гамбо, рисово-зефирное печенье, тыквенный
    пирог, банановый пудинг, мясной рулет американского типа,
    бисквиты южного типа, чизкейк-брауни, крабовый пирог.

Если ни одно условие не сработало — блюдо привычное, в ответ НЕ включай.
Обычные продукты, простая готовка, привычные десерты и снеки остаются.

Проверь КАЖДЫЙ номер по порядку и не пропускай: список примеров выше —
это ровно те случаи, которые чаще всего пропускают.

Верни ТОЛЬКО JSON, без пояснений:
{{"unfamiliar": [{{"n": номер, "why": "код"}}]}}

Пустой список, если все блюда привычные.

Блюда ({count} шт.):
{items}"""


def review_batch(client, model: str, rows: list[dict], max_retries: int = 3) -> dict[int, str]:
    """Проверить пачку. Возвращает {позиция: код причины} для непривычных блюд."""
    listing = "\n".join(
        f"{i}. {row['name_ru']} ({row['name']})"
        for i, row in enumerate(rows, start=1)
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
            if "unfamiliar" not in parsed:
                continue

            # Номера вне диапазона игнорируем: без этой проверки выдуманный
            # моделью индекс пометил бы чужое блюдо.
            found: dict[int, str] = {}
            for item in parsed["unfamiliar"]:
                number = item.get("n") if isinstance(item, dict) else item
                why = item.get("why", "?") if isinstance(item, dict) else "?"
                if isinstance(number, (int, float)) and 1 <= int(number) <= len(rows):
                    found[int(number) - 1] = str(why)
            return found
        except Exception:
            continue

    # Пачка не поддалась — считаем все блюда привычными. Ошибаться лучше
    # в сторону «оставить»: пустой каталог хуже, чем каталог с лишним.
    return {}


def _is_base_food(catalog: pd.DataFrame) -> pd.Series:
    """Базовые продукты: раздел справочника — обычная еда, а не блюдо.

    Составные блюда сюда не попадают: раздел «Meat mixed dishes» содержит
    «mixed dishes» и ни одному шаблону из ALWAYS_FAMILIAR не соответствует,
    а вот «Beef, excludes ground» соответствует «beef».
    """
    if "category" not in catalog.columns:
        return pd.Series(False, index=catalog.index)

    lowered = catalog["category"].str.lower().fillna("")

    # «Mixed dishes» — готовые составные блюда, их судит модель.
    #
    # Слово «combinations» в исключение НЕ входит: раздел овощей называется
    # «Other vegetables and combinations», и по нему из подбора вылетал
    # обычный артишок.
    composite = lowered.str.contains("mixed dish", regex=True, na=False)

    pattern = "|".join(ALWAYS_FAMILIAR)
    return lowered.str.contains(pattern, regex=True, na=False) & ~composite


def report(catalog: pd.DataFrame) -> None:
    if "familiar_ru" not in catalog.columns:
        print("Проверка ещё не проводилась.")
        return

    familiar = catalog["familiar_ru"].fillna(True).astype(bool)
    available = catalog.get("available_ru", pd.Series(True, index=catalog.index))
    available = available.fillna(True).astype(bool)
    mainstream = catalog.get("mainstream", pd.Series(True, index=catalog.index))

    both = familiar & available & mainstream.astype(bool)

    print(f"  всего блюд:          {len(catalog)}")
    print(f"  привычных:           {familiar.sum()} ({familiar.mean():.1%})")
    print(f"  идёт в подбор:       {both.sum()}")

    if (~familiar).any():
        print("\n  примеры отсеянных:")
        for name in catalog.loc[~familiar, "name_ru"].head(15):
            print(f"    {name}")

    if both.any():
        print("\n  осталось по ролям:")
        for role, count in catalog.loc[both, "role"].value_counts().items():
            print(f"    {role:<11} {count}")


def main() -> None:
    catalog = load_catalog()

    if "--report" in sys.argv:
        report(catalog)
        return

    if "familiar_ru" not in catalog.columns:
        catalog["familiar_ru"] = pd.NA

    todo = catalog["familiar_ru"].isna()
    if "--force" in sys.argv:
        todo = pd.Series(True, index=catalog.index)

    # Блюда, уже отсеянные первым проходом, проверять незачем: они и так
    # не попадут в подбор, а каждая пачка стоит запроса.
    if "available_ru" in catalog.columns:
        todo &= catalog["available_ru"].fillna(True).astype(bool)
    if "mainstream" in catalog.columns:
        todo &= catalog["mainstream"].astype(bool)

    # Базовые продукты помечаем привычными сразу и модели не показываем.
    base = _is_base_food(catalog)
    catalog.loc[base, "familiar_ru"] = True
    todo &= ~base

    positions = list(catalog.index[todo])
    if not positions:
        print("Все блюда уже проверены. Для повтора: --force")
        report(catalog)
        return

    print(f"Блюд к проверке: {len(positions)}")
    client, model, _ = get_clients("polza")

    batches = [positions[i:i + BATCH_SIZE] for i in range(0, len(positions), BATCH_SIZE)]

    def work(batch: list) -> tuple[list, set[int]]:
        rows = catalog.loc[batch, ["name", "name_ru"]].to_dict("records")
        return batch, review_batch(client, model, rows)

    unfamiliar_count = 0
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for batch, unfamiliar in pool.map(work, batches):
            for offset, index in enumerate(batch):
                catalog.loc[index, "familiar_ru"] = offset not in unfamiliar
            unfamiliar_count += len(unfamiliar)

            done += 1
            if done % 10 == 0:
                print(f"  проверено пачек {done}/{len(batches)}")

    catalog["familiar_ru"] = catalog["familiar_ru"].fillna(True).astype(bool)
    catalog.to_parquet(CATALOG_PATH, index=False)

    print(f"\nОтмечено непривычными: {unfamiliar_count}")
    print(f"Сохранено → {CATALOG_PATH}\n")
    report(catalog)


if __name__ == "__main__":
    main()
