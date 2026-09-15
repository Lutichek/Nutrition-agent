"""
Каталог блюд: выгрузка FNDDS → одна опрятная таблица с КБЖУ и порциями.

Это детерминированный слой проекта. Никакой LLM здесь нет и быть не должно:
состав блюда — это факт из справочника, а не то, что модель припомнила.
Агент потом только выбирает строки из этой таблицы, а числа берёт как есть.

Что делает модуль:

* склеивает четыре CSV выгрузки в один каталог;
* переводит нутриенты из «длинного» формата в колонки (ккал, Б, Ж, У, ...);
* выбирает для каждого блюда бытовую порцию («1 cup» = 240 г);
* размечает блюда по роли в приёме пищи (завтрак / основное / гарнир / ...);
* отбрасывает то, что не еда для взрослого: детское питание, смеси, алкоголь.

⚠️ Ловушка исходных данных: в выгрузке FNDDS колонка ``food_nutrient.nutrient_id``
содержит НЕ идентификатор FDC (1008 = энергия), а старый номер нутриента
``nutrient_nbr`` (208 = энергия). При наивной склейке по ``nutrient.id``
таблица получается пустой — молча, без единой ошибки. Поэтому join идёт
по ``nutrient_nbr``, а тест ``check_catalog`` следит, чтобы покрытие
не обнулилось после обновления выгрузки.

Использование::

    from foods import load_catalog

    catalog = load_catalog()                     # DataFrame со всеми блюдами
    catalog[catalog.role == "breakfast"].head()

Запуск::

    python foods.py          # собрать каталог и показать сводку
    python foods.py --rebuild
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
CATALOG_PATH = PROCESSED / "foods.parquet"

# Номера нутриентов (nutrient_nbr), а не FDC id — см. предупреждение в docstring.
NUTRIENTS: dict[int, str] = {
    208: "kcal",
    203: "protein_g",
    204: "fat_g",
    205: "carb_g",
    291: "fiber_g",
    269: "sugar_g",
    307: "sodium_mg",
    606: "sat_fat_g",
}

# Нутриенты, без которых блюдо бесполезно для расчёта рациона.
REQUIRED_NUTRIENTS = ["kcal", "protein_g", "fat_g", "carb_g"]

# ────────────────────────────────────────────────────────────
# Роли блюд
# ────────────────────────────────────────────────────────────
# Категории WWEIA пронумерованы осмысленно: 1xxx — молочное, 2xxx — белковые
# продукты, 3xxx — составные блюда и сэндвичи, 4xxx — крупы и хлеб,
# 5xxx — снеки и сладкое, 6xxx — фрукты и овощи, 7xxx — напитки,
# 8xxx — жиры, соусы, сахар, 9xxx — детское питание и прочее.
# Поэтому роль выводится из диапазона номера, а не из 172 названий вручную.

# Точечные исключения, которые не ложатся в диапазон.
ROLE_OVERRIDES: dict[int, str] = {
    2502: "breakfast",  # Eggs and omelets
    3706: "breakfast",  # Egg/breakfast sandwiches
    1820: "breakfast",  # Yogurt, regular
    1822: "breakfast",  # Yogurt, Greek
    1904: "breakfast",  # Plant-based yogurt
    4404: "breakfast",  # Pancakes, waffles, French toast
    4402: "breakfast",  # Biscuits, muffins, quick breads
    2804: "snack",      # Nuts and seeds
    2802: "side",       # Beans, peas, legumes
}

# ────────────────────────────────────────────────────────────
# Что не предлагать без отдельной просьбы
# ────────────────────────────────────────────────────────────
# FNDDS — это перепись того, что американцы реально едят, поэтому в ней есть
# опоссум, бобр и белка. Формально это отличный источник белка, и солвер
# радостно ставил «Opossum» в обед: по калориям и белку попадание идеальное.
# Пользователю такой план показывать нельзя, поэтому дичь и субпродукты
# помечаются как немейнстрим и в автоматический подбор не идут.
#
# Шаблоны привязаны к началу названия: в FNDDS название начинается с самого
# продукта («Liver, beef»), а без якоря «kidney» цепляет «Kidney beans»,
# которых в каталоге два десятка.
EXOTIC_NAME_PATTERN = (
    r"^(?:opossum|beaver|raccoon|squirrel|muskrat|moose|caribou|elk|antelope|bear|"
    r"alligator|turtle|frog legs|snail|ostrich|emu|pheasant|quail|dove|squab|"
    r"rabbit|venison|goat\b(?! cheese)|"
    r"brains|gizzard|tripe|tongue|chitterlings|sweetbread|liverwurst|blood sausage|"
    r"liver\b|livers\b|heart\b|kidney\b(?! bean))"
)

# Записи, которые вообще не блюда, и позиции общепита.
#
# FNDDS — инструмент учёта съеденного, а не сборник рецептов, поэтому в нём
# есть строки вроде «Avocado, for use on a sandwich»: это не блюдо, а часть
# чужого блюда. Есть и пометки о месте: «Carrots, cooked, from restaurant» —
# та же морковь, что и обычная, только съеденная в кафе.
#
# Отсев здесь, а не в проверке моделью: это чистая работа для регулярного
# выражения. Модель на такие строки смотрит и всё равно пропускает их —
# проверено на выборке, где «Обёртка для сэндвича» и «Огурец для сэндвича»
# уцелели, хотя стояли в промпте прямым примером.
SERVICE_RECORD_PATTERN = (
    r"for use on a sandwich|, on a sandwich|"
    r"from fast food|from restaurant|fast food / restaurant|"
    r"school lunch|baby food|infant formula|"
    r"as ingredient in|filling only|"
    # Приправы и кислые цитрусы: формально это еда с честным КБЖУ, но порцией
    # их не едят. Солвер этого не знает и добирал калории 400 граммами лайма —
    # 120 ккал, в норму попадает, есть невозможно.
    r"^lime, raw|^lemon, raw|^vinegar|^horseradish|^mustard, prepared"
)

# Сырое мясо, рыба, моллюски и яйца.
#
# В FNDDS они есть законно: человек мог съесть тартар или сырое яйцо,
# и учесть это надо. Но ПРЕДЛАГАТЬ такое в плане питания нельзя — это уже
# не вопрос вкуса, а вопрос безопасности.
#
# Наблюдалось: «Клэмы, сырые» в качестве основного блюда на ужин.
# Рядом в каталоге лежали «Говядина, фарш, сырая» и «Яйцо, целое, сырое».
#
# Якорь на конец строки обязателен: у половины куриных позиций «raw» стоит
# в середине названия («prepared skinless, from raw») и означает ровно
# обратное — что блюдо приготовлено из сырого продукта.
RAW_ANIMAL_PATTERN = (
    r"^(?:beef|pork|chicken|turkey|lamb|veal|duck|goose|"
    r"clams?|oysters?|mussels?|scallops?|shrimp|crab|lobster|squid|"
    r"fish|tuna|salmon|cod|herring|egg)\b.*,\s*raw$"
)

# Минимальная доля калорий из белка для роли «основное блюдо».
#
# Значение подобрано замером (python -m experiments.measure_solver) — см. комментарий
# в месте применения. Ниже порога блюдо переводится в гарниры.
MAIN_MIN_PROTEIN_SHARE = 0.15


# Категории, которые в рацион взрослого не идут вовсе.
EXCLUDED_CATEGORIES: set[int] = {
    7502, 7504, 7506,  # пиво, вино, крепкий алкоголь
    8804,              # сахарозаменители
    9602,              # грудное молоко
    9999,              # «не отнесено к категории»
}


def _role_for_category(category_id: int) -> str | None:
    """Определить роль блюда по номеру категории WWEIA.

    Возвращает ``None`` для того, что не должно попадать в рацион
    как самостоятельная позиция (детское питание, алкоголь, приправы).
    """
    if category_id in EXCLUDED_CATEGORIES:
        return None
    if category_id in ROLE_OVERRIDES:
        return ROLE_OVERRIDES[category_id]

    thousand = category_id // 1000

    if thousand == 9:
        return None  # детское питание, смеси, порошки
    if thousand == 8:
        return None  # масла, соусы, сахар — добавка, а не блюдо
    if thousand == 7:
        return "drink"
    if thousand == 6:
        # 6002-6024 — фрукты, 64xx-68xx — овощи и картофель
        return "snack" if category_id < 6400 else "side"
    if thousand == 5:
        return "snack"
    if thousand == 4:
        # 46xx-48xx — сухие завтраки, овсянка, каши
        return "breakfast" if category_id >= 4600 else "side"
    if thousand == 3:
        return "main"  # составные блюда, сэндвичи, пицца, супы
    if thousand == 2:
        return "main"  # мясо, птица, рыба, яйца
    if thousand == 1:
        return "drink" if category_id < 1600 else "breakfast"  # молоко / сыр, творог

    return None


# ────────────────────────────────────────────────────────────
# Выбор бытовой порции
# ────────────────────────────────────────────────────────────
# У блюда бывает до двадцати вариантов порции — от «1 Nabisco Chips Ahoy!»
# до «1 cup». Нужна одна, самая понятная человеку. Список задаёт приоритет:
# чем раньше в списке, тем охотнее берём.
PORTION_PREFERENCE = [
    "1 cup",
    "1 medium",
    "1 serving",
    "1 piece",
    "1 slice",
    "1 sandwich",
    "1 bowl",
    "1 plate",
    "1 each",
    "1 egg",
    "1 fillet",
    "1 breast",
    "1 patty",
    "1 taco",
    "1 burrito",
    "1 bar",
    "1 muffin",
    "1 large",
    "1 small",
]

# Порция без веса или с бессмысленным описанием.
PORTION_JUNK = {"", "quantity not specified"}

# Описания, которые формально начинаются с удобного префикса, но порцией
# не являются: «1 cup, crumbs» — это стакан хлебных крошек, а не ломоть хлеба.
# Без этого списка префикс «1 cup» выигрывал у нормального «1 medium slice».
PORTION_DEMOTE = ("crumb", "cubic inch", "surface inch", "snack-size", "crust not eaten")


def _portion_rank(description: str) -> int:
    """Чем меньше число, тем удобнее порция для человека."""
    lowered = description.lower()

    if any(marker in lowered for marker in PORTION_DEMOTE):
        return len(PORTION_PREFERENCE) + 1

    for index, prefix in enumerate(PORTION_PREFERENCE):
        if lowered.startswith(prefix):
            return index
    return len(PORTION_PREFERENCE)


def _pick_portions(survey_dir: Path) -> pd.DataFrame:
    """Выбрать по одной бытовой порции на блюдо."""
    portions = pd.read_csv(
        survey_dir / "food_portion.csv",
        usecols=["fdc_id", "seq_num", "portion_description", "gram_weight"],
    )
    portions["portion_description"] = portions["portion_description"].fillna("").str.strip()

    usable = portions[
        ~portions["portion_description"].str.lower().isin(PORTION_JUNK)
        & portions["gram_weight"].notna()
        & (portions["gram_weight"] > 0)
    ].copy()

    usable["rank"] = usable["portion_description"].map(_portion_rank)
    best = (
        usable.sort_values(["fdc_id", "rank", "seq_num"])
        .groupby("fdc_id", as_index=False)
        .first()
    )

    return best[["fdc_id", "portion_description", "gram_weight"]].rename(
        columns={"portion_description": "portion", "gram_weight": "portion_g"}
    )


# ────────────────────────────────────────────────────────────
# Сборка каталога
# ────────────────────────────────────────────────────────────


def build_catalog(verbose: bool = True) -> pd.DataFrame:
    """Собрать каталог блюд из CSV-выгрузки FNDDS."""
    # Импорт внутри функции — см. пояснение в data_preporation.prepare_chunks.
    # Рантайм зовёт только load_catalog, которая читает готовый parquet.
    from data_sources import download_usda

    survey_dir = download_usda()

    food = pd.read_csv(survey_dir / "food.csv", usecols=["fdc_id", "description"])
    survey = pd.read_csv(
        survey_dir / "survey_fndds_food.csv", usecols=["fdc_id", "wweia_category_number"]
    )
    categories = pd.read_csv(survey_dir / "wweia_food_category.csv")

    catalog = food.merge(survey, on="fdc_id").merge(
        categories,
        left_on="wweia_category_number",
        right_on="wweia_food_category",
        how="left",
    )
    catalog = catalog.rename(
        columns={
            "description": "name",
            "wweia_category_number": "category_id",
            "wweia_food_category_description": "category",
        }
    )[["fdc_id", "name", "category_id", "category"]]

    if verbose:
        print(f"[foods] блюд в выгрузке: {len(catalog)}")

    # ── Нутриенты: длинный формат → колонки ──────────────────
    nutrients = pd.read_csv(
        survey_dir / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"]
    )
    # ВНИМАНИЕ: nutrient_id здесь хранит nutrient_nbr (208 = ккал), см. docstring.
    nutrients = nutrients[nutrients["nutrient_id"].isin(NUTRIENTS)]
    wide = nutrients.pivot_table(
        index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first"
    ).rename(columns=NUTRIENTS)
    wide.columns.name = None

    catalog = catalog.merge(wide.reset_index(), on="fdc_id", how="left")

    before = len(catalog)
    catalog = catalog.dropna(subset=REQUIRED_NUTRIENTS)
    if verbose:
        print(f"[foods] без полного КБЖУ отброшено: {before - len(catalog)}")

    # ── Порции ───────────────────────────────────────────────
    catalog = catalog.merge(_pick_portions(survey_dir), on="fdc_id", how="left")

    before = len(catalog)
    catalog = catalog.dropna(subset=["portion_g"])
    if verbose:
        print(f"[foods] без бытовой порции отброшено: {before - len(catalog)}")

    # ── Роли ─────────────────────────────────────────────────
    catalog["role"] = catalog["category_id"].map(_role_for_category)

    before = len(catalog)
    catalog = catalog[catalog["role"].notna()].copy()
    if verbose:
        print(f"[foods] не еда для взрослого, отброшено: {before - len(catalog)}")

    # ── Основное блюдо обязано давать белок ──────────────────
    # Роль берётся из раздела справочника, а раздел про белок ничего не знает:
    # в «Rice mixed dishes» лежат и жареный рис с курицей, и «Рис с изюмом».
    # В шаблоне дня на роль main приходится 45% калорийности, и она же —
    # главный источник белка. Блюдо, дающее 5% калорий из белка, эту работу
    # не делает, а солвер потом не может набрать норму белка ничем другим.
    #
    # Наблюдалось: «Рис белый с подливкой» — 2 г белка на 100 г — оказался
    # основным блюдом на ужин.
    #
    # Такие блюда не выбрасываются: как гарнир они хороши.
    protein_share = catalog["protein_g"] * 4 / catalog["kcal"].clip(lower=1)
    demoted = (catalog["role"] == "main") & (protein_share < MAIN_MIN_PROTEIN_SHARE)
    catalog.loc[demoted, "role"] = "side"
    if verbose:
        print(f"[foods] основных блюд без белка переведено в гарниры: {demoted.sum()}")

    # ── Мейнстрим или экзотика ───────────────────────────────
    # Дичь и субпродукты остаются в каталоге (если человек спросит про печень,
    # ответить есть чем), но помечены — солвер их не предлагает сам.
    lowered = catalog["name"].str.lower()
    exotic = lowered.str.contains(EXOTIC_NAME_PATTERN, regex=True, na=False)
    service = lowered.str.contains(SERVICE_RECORD_PATTERN, regex=True, na=False)
    raw_animal = lowered.str.contains(RAW_ANIMAL_PATTERN, regex=True, na=False)

    catalog["mainstream"] = ~(exotic | service | raw_animal)
    if verbose:
        print(f"[foods] помечено как экзотика: {exotic.sum()}")
        print(f"[foods] служебные записи и общепит: {service.sum()}")
        print(f"[foods] сырое мясо, рыба и яйца: {raw_animal.sum()}")

    # ── КБЖУ на порцию ───────────────────────────────────────
    # В выгрузке нутриенты даны на 100 г, поэтому масштабируем весом порции.
    factor = catalog["portion_g"] / 100.0
    for column in ("kcal", "protein_g", "fat_g", "carb_g", "fiber_g", "sugar_g", "sodium_mg"):
        if column in catalog.columns:
            catalog[f"portion_{column}"] = (catalog[column] * factor).round(1)

    catalog = catalog.sort_values("name").reset_index(drop=True)

    if verbose:
        print(f"[foods] итоговый каталог: {len(catalog)} блюд")
        print(catalog["role"].value_counts().to_string())

    return catalog


def load_catalog(rebuild: bool = False, verbose: bool = False) -> pd.DataFrame:
    """Загрузить каталог, собрав его при первом обращении."""
    if CATALOG_PATH.exists() and not rebuild:
        return pd.read_parquet(CATALOG_PATH)

    catalog = build_catalog(verbose=verbose or rebuild)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(CATALOG_PATH, index=False)
    return catalog


# ────────────────────────────────────────────────────────────
# Проверки данных
# ────────────────────────────────────────────────────────────


def check_catalog(catalog: pd.DataFrame) -> dict[str, float]:
    """Детерминированные проверки каталога.

    Главная — согласованность по Этуотеру: заявленная калорийность должна
    примерно совпадать с 4×белки + 9×жиры + 4×углеводы. Если после обновления
    выгрузки join развалится и колонки перепутаются, эта проверка это поймает,
    а вот глазами такое не заметишь.
    """
    atwater = 4 * catalog["protein_g"] + 9 * catalog["fat_g"] + 4 * catalog["carb_g"]
    # Считаем только блюда с заметной калорийностью: на «1 ккал» любой
    # относительный порог бессмыслен.
    solid = catalog[catalog["kcal"] > 20]
    deviation = (
        (4 * solid["protein_g"] + 9 * solid["fat_g"] + 4 * solid["carb_g"] - solid["kcal"]).abs()
        / solid["kcal"]
    )

    report = {
        "foods": float(len(catalog)),
        "atwater_median_deviation": float(deviation.median()),
        "atwater_within_25pct": float((deviation <= 0.25).mean()),
        "negative_values": float((catalog[REQUIRED_NUTRIENTS] < 0).any(axis=1).sum()),
        "portion_median_g": float(catalog["portion_g"].median()),
        "roles": float(catalog["role"].nunique()),
    }

    assert report["negative_values"] == 0, "В каталоге отрицательные значения нутриентов"
    assert report["atwater_within_25pct"] > 0.9, (
        "Калорийность не сходится с составом — вероятно, join по нутриентам "
        f"снова поехал (сходится {report['atwater_within_25pct']:.1%})"
    )
    assert len(atwater) == len(catalog)

    return report


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    catalog = load_catalog(rebuild=rebuild, verbose=True)

    print()
    print("=" * 60)
    print("Проверки каталога")
    print("=" * 60)
    for key, value in check_catalog(catalog).items():
        print(f"  {key:28} {value:.3f}")

    print()
    print("Примеры блюд:")
    columns = ["name", "role", "portion", "portion_g", "portion_kcal", "portion_protein_g"]
    print(catalog.sample(8, random_state=1)[columns].to_string(index=False))


if __name__ == "__main__":
    main()
