"""
Перевод названий блюд на русский: разовая операция над каталогом.

Справочник USDA англоязычный, а разговор идёт по-русски — показывать человеку
«Yogurt, Greek, NS as to type of milk, plain» неправильно. Скрипт добавляет
в каталог колонку ``name_ru``.

Почему перевод делается заранее, а не на лету при выдаче плана: ручка ``/plan``
не должна обращаться к языковой модели. Это её главное свойство — она
детерминированная, бесплатная и при одном seed даёт один и тот же ответ.
Перевод в момент выдачи это свойство сломал бы.

Английское название при этом остаётся в колонке ``name`` и продолжает
использоваться для отбора: исключения вроде «pork» ищутся именно в нём.

Запуск::

    python translate_catalog.py          # перевести всё, чего ещё нет
    python translate_catalog.py --force  # перевести заново

Стоимость: около 4800 названий пачками по 60 — примерно 80 коротких запросов.
Повторный запуск бесплатен, уже переведённое не трогается.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from agent import parse_json_response
from foods import CATALOG_PATH, load_catalog
from providers import get_clients
from translation_fixes import apply as apply_translation_fixes

BATCH_SIZE = 60
MAX_WORKERS = 6

# Описания порций короткие и однотипные, но модель на больших пачках
# сбивается со счёта чаще, чем на названиях. Пачка поменьше надёжнее.
PORTION_BATCH_SIZE = 25

# Сколько раз пачку можно делить пополам при неудаче. Три уровня превращают
# 60 строк в группы по 7-8 — этого хватает, чтобы локализовать проблемную
# строку, и не хватает, чтобы устроить лавину запросов.
MAX_SPLIT_DEPTH = 3

# Слово, внутри которого смешаны кириллица и латиница, — признак порчи
# перевода: «Пound cake», «РыбныйWrap-сэндвич». Такие строки переводим заново.
#
# Границы слова (\b) намеренно нет: скобка с брендом («Батончик (Nature Valley)»)
# — это нормально, а вот кириллица и латиница ВПРИТЫК внутри слова — нет.
MANGLED_RE = r"[а-яА-ЯёЁ][A-Za-z]|[A-Za-z][а-яА-ЯёЁ]"

PORTION_PROMPT = """Переведи бытовые описания порций из справочника питания USDA
на русский язык.

Требования:
- коротко, как в кулинарной книге: «1 cup» → «1 стакан», «1 slice» → «1 ломтик»
- «fl oz» — жидкая унция, «oz» — унция; переводи как «унция», размер не меняй
- служебные пометки NFS и NS выбрасывай
- дюймы пиши СЛОВОМ: «11-12" diameter» → «диаметр 11-12 дюймов»

⚠️ НИКОГДА не используй символ двойной кавычки внутри перевода — он ломает
формат ответа. Если в исходной строке есть кавычка (обозначение дюймов),
замени её словом «дюйм».

Верни ТОЛЬКО JSON вида {{"translations": ["перевод 1", ...]}}
в том же порядке и того же размера, что список ниже.

Описания ({count} шт.):
{names}"""

PROMPT = """Переведи названия блюд из справочника питания USDA на русский язык.

Требования:
- коротко и по-бытовому, как назвал бы блюдо обычный человек
- сохраняй способ приготовления и важные уточнения (жареный, консервированный,
  без добавления жира), но выбрасывай служебные пометки вроде NS, NFS,
  «as to type of milk»
- не переводи в диетологический канцелярит, пиши просто
- если название и так понятно (например, «Pizza»), дай привычный русский вариант

Верни ТОЛЬКО JSON вида {{"translations": ["перевод 1", "перевод 2", ...]}}
в том же порядке и того же размера, что список ниже.

Названия ({count} шт.):
{names}"""


def translate_batch(
    client,
    model: str,
    names: list[str],
    max_retries: int = 3,
    prompt: str = PROMPT,
    depth: int = 0,
) -> list[str]:
    """Перевести одну пачку строк. При неудаче вернуть исходные.

    Args:
        prompt: шаблон запроса — PROMPT для названий блюд,
            PORTION_PROMPT для описаний порций.
        depth: текущая глубина дробления, см. MAX_SPLIT_DEPTH.
    """
    numbered = "\n".join(f"{i}. {name}" for i, name in enumerate(names, start=1))

    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": prompt.format(count=len(names), names=numbered)}],
                temperature=0.0,
                # response_format="json_object" здесь НЕ используется намеренно:
                # на этом провайдере такие запросы стабильно подвисали до
                # таймаута, тогда как без него ответ приходит за 4 секунды.
                # Формат вытягивается разбором, он и так устойчив к обёрткам.
                #
                # Таймаут обязателен: без него один подвисший запрос
                # останавливает весь скрипт, и ретраям нечего перехватывать.
                timeout=60.0,
            )
            raw = (response.choices[0].message.content or "").strip()
            translations = parse_json_response(raw).get("translations", [])
            # Размер обязан совпасть: иначе переводы съедут относительно блюд,
            # и в плане окажется курица под названием «яблоко».
            if len(translations) == len(names):
                return [str(t).strip() for t in translations]
        except Exception:
            continue

    # Пачка не поддалась: модель устойчиво возвращает не тот размер. Обычно
    # виновата одна-две строки, поэтому делим пополам и пробуем снова —
    # так проблема локализуется, а остальные строки всё-таки переводятся.
    #
    # ⚠️ Глубина ограничена намеренно. Без предела дробление вырождается
    # в лавину запросов: пачка на 60 делится до одиночных строк, и на каждом
    # уровне тратится ещё max_retries попыток — это сотни вызовов на одну
    # неудачную пачку. Первый прогон из-за этого завис на 40 минут.
    if len(names) > 1 and depth < MAX_SPLIT_DEPTH:
        middle = len(names) // 2
        return (
            translate_batch(client, model, names[:middle], max_retries, prompt, depth + 1)
            + translate_batch(client, model, names[middle:], max_retries, prompt, depth + 1)
        )

    return names


def translate_portions(catalog: pd.DataFrame, client, model: str, force: bool = False) -> None:
    """Перевести описания порций («1 cup» → «1 стакан»).

    Уникальных описаний всего пара сотен на пять тысяч блюд, поэтому
    переводим множество значений, а не каждую строку каталога.
    """
    if "portion_ru" not in catalog.columns:
        catalog["portion_ru"] = pd.NA

    known = {}
    if not force:
        # Переведённым считается только то, где есть кириллица. Английское
        # значение в portion_ru — это след запасного пути, а не перевод;
        # по «непустоте» такие строки не отличить, и они навсегда остались бы
        # английскими. Та же ловушка, что и с названиями блюд.
        filled = catalog[
            catalog["portion_ru"].astype(str).str.contains("[а-яА-ЯёЁ]", regex=True, na=False)
        ]
        known = dict(zip(filled["portion"], filled["portion_ru"], strict=True))

    unique = [p for p in catalog["portion"].dropna().unique() if p not in known]
    if not unique:
        return

    print(f"\nОписаний порции к переводу: {len(unique)}")
    batches = [unique[i:i + PORTION_BATCH_SIZE]
               for i in range(0, len(unique), PORTION_BATCH_SIZE)]

    translated: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for chunk in pool.map(
            lambda b: translate_batch(client, model, b, prompt=PORTION_PROMPT), batches
        ):
            translated.extend(chunk)

    known.update(dict(zip(unique, translated, strict=True)))
    catalog["portion_ru"] = catalog["portion"].map(known).fillna(catalog["portion"])
    print(f"Порции переведены: {len(unique)}")


def main() -> None:
    force = "--force" in sys.argv

    catalog = load_catalog()
    if "name_ru" not in catalog.columns:
        catalog["name_ru"] = pd.NA

    # Непереведённым считается не только пустое поле. Когда модель возвращает
    # пачку не того размера, срабатывает запасной путь и в name_ru попадает
    # исходное английское название — по пустоте такую строку не отличить.
    # Надёжный признак перевода — наличие кириллицы.
    has_cyrillic = catalog["name_ru"].astype(str).str.contains("[а-яА-ЯёЁ]", regex=True, na=False)
    mangled = catalog["name_ru"].astype(str).str.contains(MANGLED_RE, regex=True, na=False)
    todo = catalog["name_ru"].isna() | (catalog["name_ru"] == "") | ~has_cyrillic | mangled
    if force:
        todo = pd.Series(True, index=catalog.index)

    names = catalog.loc[todo, "name"].tolist()
    if not names:
        print("Все названия уже переведены. Для повтора: --force")
        return

    batches = [names[i:i + BATCH_SIZE] for i in range(0, len(names), BATCH_SIZE)]
    print(f"Названий к переводу: {len(names)}, пачек: {len(batches)}")

    client, model, _ = get_clients("polza")

    done = 0
    results: list[str] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for translated in pool.map(lambda b: translate_batch(client, model, b), batches):
            results.extend(translated)
            done += 1
            if done % 10 == 0:
                print(f"  переведено пачек {done}/{len(batches)}")

    catalog.loc[todo, "name_ru"] = results

    translate_portions(catalog, client, model, force=force)

    # Починка устойчивых ошибок перевода — последним шагом, по всему каталогу,
    # а не только по свежепереведённому: старые записи чинятся тоже.
    apply_translation_fixes(catalog)

    catalog.to_parquet(CATALOG_PATH, index=False)

    untouched = sum(1 for original, translated in zip(names, results, strict=True) if original == translated)
    print(f"\nГотово. Переведено: {len(results) - untouched}, осталось на английском: {untouched}")
    print(f"Сохранено → {CATALOG_PATH}\n")

    sample = catalog.loc[todo, ["name", "name_ru"]].head(8)
    for _, row in sample.iterrows():
        print(f"  {row['name'][:46]:<48} → {row['name_ru']}")


if __name__ == "__main__":
    main()
