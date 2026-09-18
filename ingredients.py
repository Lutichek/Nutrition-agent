"""
Состав блюд: из чего сделано каждое блюдо справочника и в какой пропорции.

Источник — ``input_food.csv`` из выгрузки FNDDS. Для каждого блюда там
лежит рецептура: список ингредиентов с граммовкой.

## Зачем

Две вещи сразу, и обе важные.

**Человеку — понятно, что он ест.** «Куриное бедро с рисом» ничего не говорит
о том, сколько там курицы; состав говорит.

**Цене — точность.** Пока цена бралась по ОДНОМУ главному ингредиенту,
она ошибалась в обе стороны, и сильно::

    Креветки с овощами, 200 г     199 ₽ → 94 ₽   креветок там 69 г, не 200
    Говядина тушёная, 200 г       219 ₽ → 92 ₽   мяса 69 г, воды 72 г
    Салат с макаронами и сыром    10 ₽  → 41 ₽   занижено: 26 г чеддера

Отдельно чинится вода: в рецептурах она стоит своей строкой и составляет
6% общей массы. По старому способу за неё платили как за основной продукт.

## Граммовки рецептурные, а не на 100 г

Ловушка, на которой легко ошибиться. ``gram_weight`` — это количество
в РЕЦЕПТЕ, и сумма по блюду совпадает с сотней лишь у половины записей::

    Milk shake, home recipe    488 г молока + 296 г мороженого = 784 г
    Bread, Puerto Rican style  4540 г муки + 2724 г воды + ... = 7729 г

Второе — производственная закладка на целую партию. Поэтому граммовки
приводятся к ДОЛЯМ, и доля умножается на реальный вес порции.

Чего это не учитывает: потерю воды при готовке. Если из 7729 г теста
выходит 6000 г хлеба, то на 100 г хлеба нужно 129 г продуктов, а мы
скажем 100. Занижение порядка 15-20% у выпечки, у остального меньше.

## Перевод названий

Ингредиентов 2077 штук, и человеку они показываются по-русски. Перевод —
разовая операция при сборке артефакта, ровно по тем же причинам, что
и у каталога: ручка ``/plan`` к языковой модели не обращается.

Запуск::

    python ingredients.py --rebuild   # собрать и перевести (нужен ключ)
    python ingredients.py             # показать, что собрано
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
COMPOSITION_PATH = PROCESSED / "ingredients.parquet"

RAW_INPUT_FOOD = HERE / "data" / "raw" / "usda" / "survey" / "input_food.csv"

# Доля ниже этой в состав не идёт. Соль и специи в рецептуре весят десятые
# доли грамма: в цене они ничего не меняют, а список превращают в простыню.
#
# Порог именно по доле, а не по граммам: у порции в 40 г и в 400 г
# «незначительное» — это разные граммовки.
MIN_SHARE = 0.01

# Сколько строк состава показывать человеку. Остальное сворачивается
# в «и ещё N ингредиентов»: рецептуры доходят до двадцати позиций,
# а читают из них первые три.
DISPLAY_LIMIT = 6


class Ingredient(BaseModel):
    """Одна строка состава блюда."""

    name: str           # английское название из FNDDS — по нему ищется цена
    name_ru: str        # русское — его видит человек
    share: float        # доля массы блюда, 0..1

    def grams_in(self, portion_grams: float) -> float:
        return self.share * portion_grams


@lru_cache(maxsize=4)
def load_composition(path: Path | None = None) -> dict[int, list[Ingredient]]:
    """Состав всех блюд: fdc_id → список ингредиентов.

    Пустой словарь, если артефакт не собран. Это не ошибка: без состава
    цена считается по старому способу, а состав просто не показывается.

    Кэшируется: файл неизменяемый, а зовут отсюда на каждый список покупок.
    Без кэша четырнадцать тысяч строк перечитывались бы с диска при каждом
    пересчёте плана.
    """
    path = path or COMPOSITION_PATH
    if not path.exists():
        return {}

    frame = pd.read_parquet(path)
    result: dict[int, list[Ingredient]] = {}
    for fdc_id, group in frame.groupby("fdc_id"):
        result[int(fdc_id)] = [
            Ingredient(name=row.name_en, name_ru=row.name_ru, share=float(row.share))
            for row in group.itertuples()
        ]
    return result


def build_shares(raw_path: Path | None = None,
                 keep_ids: set[int] | None = None) -> pd.DataFrame:
    """Привести рецептурные граммовки к долям массы блюда.

    Отдельной функцией, потому что это единственный содержательный расчёт
    модуля и его нужно проверять тестом без обращения к сети.
    """
    frame = pd.read_csv(raw_path or RAW_INPUT_FOOD)
    frame = frame[["fdc_id", "sr_description", "gram_weight"]].dropna()

    if keep_ids is not None:
        frame = frame[frame["fdc_id"].isin(keep_ids)]

    # Один ингредиент может встретиться в рецепте дважды (разные позиции
    # одного продукта) — складываем, иначе в составе будут дубли строк.
    frame = (frame.groupby(["fdc_id", "sr_description"], as_index=False)["gram_weight"]
                  .sum())

    totals = frame.groupby("fdc_id")["gram_weight"].transform("sum")
    # Блюдо с нулевой суммой — испорченная строка выгрузки. Делить на ноль
    # нельзя, а молча получить inf — тем более.
    frame = frame[totals > 0]
    totals = totals[totals > 0]
    frame = frame.assign(share=frame["gram_weight"] / totals)

    frame = frame[frame["share"] >= MIN_SHARE]

    # После отсева мелочи доли уже не дают единицу — нормируем повторно,
    # иначе стоимость блюда систематически занижалась бы на сумму отсева.
    totals = frame.groupby("fdc_id")["share"].transform("sum")
    frame = frame.assign(share=frame["share"] / totals)

    return (frame.rename(columns={"sr_description": "name_en"})
                 .sort_values(["fdc_id", "share"], ascending=[True, False])
                 [["fdc_id", "name_en", "share"]]
                 .reset_index(drop=True))


def describe(items: list[Ingredient], portion_grams: float,
             limit: int = DISPLAY_LIMIT) -> str:
    """Состав словами: «курица 69 г, рис 46 г, масло 12 г»."""
    if not items:
        return ""

    shown = items[:limit]
    parts = [f"{item.name_ru.lower()} {item.grams_in(portion_grams):.0f} г"
             for item in shown]
    text = ", ".join(parts)

    hidden = len(items) - len(shown)
    if hidden > 0:
        text += f" и ещё {hidden}"
    return text


# ────────────────────────────────────────────────────────────
# Сборка артефакта (разовая, требует ключа)
# ────────────────────────────────────────────────────────────


# Промпт свой, а не общий с каталогом, и это не дублирование.
#
# У блюда название — витрина, его читают целиком. У ингредиента название
# читают в строчке состава, где их шесть штук подряд, и «Молоко
# с пониженным содержанием жира, 2% жира, с добавлением витаминов A и D»
# превращает состав в нечитаемую простыню. Нужна короткая форма из магазина.
INGREDIENT_PROMPT = """Переведи названия продуктов из справочника USDA
на русский язык — КОРОТКО, как пишут в списке покупок.

Требования:
- два-три слова, не больше: «Milk, lowfat, fluid, 1% milkfat, with added
  vitamin A and vitamin D» → «молоко 1%»
- сохраняй только то, что отличает продукт от соседнего по полке:
  жирность, вид («гречка», «рис бурый»), способ («варёный», «копчёный»)
- выбрасывай: добавленные витамины, «fluid», «regular», «enriched»,
  «commercially prepared», NFS, NS, коды и пометки в скобках
- «Beverages, water, tap, drinking» → «вода из-под крана»
- «Chicken as ingredient in recipes» → «курица»
- с маленькой буквы, без точки в конце

Верни ТОЛЬКО JSON вида {{"translations": ["перевод 1", "перевод 2", ...]}}
в том же порядке и того же размера, что список ниже.

Названия ({count} шт.):
{names}"""


def rebuild(verbose: bool = True) -> pd.DataFrame:
    """Собрать состав и перевести названия ингредиентов."""
    from foods import load_catalog
    from providers import get_clients
    from translate_catalog import BATCH_SIZE, translate_batch

    if not RAW_INPUT_FOOD.exists():
        raise FileNotFoundError(
            f"Нет {RAW_INPUT_FOOD.name}. Выполните: python data_sources.py --usda"
        )

    catalog = load_catalog()
    keep = set(catalog["fdc_id"].astype(int))
    shares = build_shares(keep_ids=keep)

    names = sorted(shares["name_en"].unique())
    if verbose:
        print(f"Блюд с составом: {shares['fdc_id'].nunique():,}")
        print(f"Уникальных ингредиентов: {len(names):,}")
        print(f"Перевожу пачками по {BATCH_SIZE}...")

    client, model, _ = get_clients("polza")
    translated: dict[str, str] = {}
    for start in range(0, len(names), BATCH_SIZE):
        batch = names[start:start + BATCH_SIZE]
        translated.update(zip(
            batch,
            translate_batch(client, model, batch, prompt=INGREDIENT_PROMPT),
            strict=True,
        ))
        if verbose:
            print(f"  {min(start + BATCH_SIZE, len(names)):>5}/{len(names)}")

    shares = shares.assign(name_ru=shares["name_en"].map(translated))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    shares.to_parquet(COMPOSITION_PATH, index=False)
    if verbose:
        print(f"Сохранено → {COMPOSITION_PATH}")
    return shares


def main() -> None:
    if "--rebuild" in sys.argv[1:]:
        rebuild()
        return

    composition = load_composition()
    if not composition:
        print("Состав не собран. Выполните: python ingredients.py --rebuild")
        return

    print(f"Блюд с составом: {len(composition):,}")
    sizes = [len(items) for items in composition.values()]
    print(f"Ингредиентов на блюдо: в среднем {sum(sizes) / len(sizes):.1f}, "
          f"максимум {max(sizes)}")


if __name__ == "__main__":
    main()
