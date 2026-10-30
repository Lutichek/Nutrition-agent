"""
Солвер рациона: собрать день из блюд каталога под целевые КБЖУ.

Третий детерминированный слой. Задача такая: есть 4782 блюда с известным
составом и есть цель на день (например, 1477 ккал и 133 г белка). Нужно
выбрать 6-7 блюд и размеры порций так, чтобы сумма сошлась с целью.

Почему это не работа для LLM: модель не умеет складывать 7 чисел так, чтобы
сумма попала в ±5% от заданной, и уж точно не умеет делать это повторяемо.
Зато это классическая задача оптимизации, и решается она за доли секунды.
Роль LLM в проекте — объяснить и обсудить готовый план, а не считать его.

Алгоритм — жадная сборка плюс локальный поиск:

1. **Шаблон дня** задаёт слоты (завтрак, обед, ужин, перекусы) и долю
   калорийности каждого.
2. **Жадный проход** подбирает в каждый слот блюдо, лучше всего попадающее
   в норму этого слота.
3. **Локальный поиск** многократно пробует заменить одно блюдо или изменить
   размер порции и оставляет замену, если общая невязка уменьшилась.

Результат воспроизводим: при одном и том же ``seed`` план всегда одинаков.

Использование::

    from foods import load_catalog
    from solver import build_day
    from targets import Profile, compute_targets

    catalog = load_catalog()
    targets = compute_targets(Profile(sex="female", age=31, height_cm=168,
                                      weight_kg=74, goal="lose"))
    plan = build_day(catalog, targets)
    print(plan.render())
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from functools import lru_cache
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, computed_field

from ingredients import ingredient_dishes
from prices import cost_per_gram
from restrictions import (
    GLUTEN_FREE,
    cautions_for,
    ingredient_terms,
    is_gluten_term,
    matches_terms,
    matches_words,
    to_category_pattern,
    to_pattern,
)
from targets import Targets

# ────────────────────────────────────────────────────────────
# Шаблон дня
# ────────────────────────────────────────────────────────────
# Слот — это (название, роль в каталоге, доля калорийности дня).
# Доли подобраны под привычную структуру питания и в сумме дают единицу.
DAY_TEMPLATE: list[tuple[str, str, float]] = [
    ("Завтрак", "breakfast", 0.25),
    ("Обед — основное", "main", 0.24),
    ("Обед — гарнир", "side", 0.09),
    ("Ужин — основное", "main", 0.21),
    ("Ужин — гарнир", "side", 0.09),
    ("Перекус", "snack", 0.07),
    ("Перекус", "snack", 0.05),
]

# Допустимые множители порции. Дробим не мельче четверти: «0.37 порции»
# человек всё равно не отмерит.
SERVING_OPTIONS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

# Блюдо не должно быть ни крошкой, ни тазом.
MIN_PORTION_G = 30.0
MAX_PORTION_G = 600.0

# Веса невязки. Калорийность важнее всего, следом белок: именно они определяют,
# сработает план или нет.
#
# Жиры и углеводы весили 0.7, пока каталог был широким. После отсева блюд,
# недоступных в СНГ, он сузился на пятую часть, и на профиле набора массы
# (3074 ккал) оптимизатор стал вылезать по жирам на +43% — при таком весе
# ему выгоднее было точно попасть в калории и белок. Наращивание итераций
# не помогало: дело не в поиске, а в том, что ему поручено искать.
#
# Замер на четырёх профилях по восьми сидам: при 1.8 провалов ноль,
# отклонение по калориям 2.1% (допуск 6%). При 2.5 провалов тоже ноль,
# но калории уплывают до 4.1% — слишком близко к границе.
WEIGHTS: dict[str, float] = {
    "kcal": 3.0,
    "protein_g": 2.0,
    "fat_g": 1.8,
    "carb_g": 1.8,
    "fiber_g": 0.8,
}

# Недобор белка и клетчатки хуже перебора, поэтому штраф несимметричный.
UNDERSHOOT_PENALTY: dict[str, float] = {"protein_g": 2.0, "fiber_g": 2.0}

# Гигиена рациона: верхние границы, за которые план заходить не должен.
SODIUM_LIMIT_MG = 2300.0
SUGAR_KCAL_SHARE_LIMIT = 0.10

NUTRIENT_COLUMNS = ["kcal", "protein_g", "fat_g", "carb_g", "fiber_g", "sugar_g", "sodium_mg"]

# Штраф за перекос отдельного приёма пищи относительно его доли в дне.
#
# Зачем: целевая функция считает дневные итоги, и по ним план мог сходиться
# идеально, а состоять при этом из 745 ккал фисташек «на перекус» — четверть
# дневной нормы в слоте, рассчитанном на 7%. Жадный проход ставит блюдо
# по норме слота, но локальный поиск потом свободно его меняет, улучшая
# дневную сумму и разрушая структуру дня.
SLOT_BALANCE_WEIGHT = 1.5

# Во сколько раз приём пищи может превысить свою долю дня. Это ЖЁСТКИЙ потолок
# при отборе кандидатов, а не штраф: штраф конкурировал с попаданием
# в калорийность и проигрывал, оставляя перекос до 6.7 раза.
#
# Замер на пяти профилях по шести сидам: при 2.0 ни один приём не выходит
# за двойную долю, провалов проверок нет, отклонение по калориям 2.7%.
# При 1.7 калории уже плывут на 6.8%: каталог не даёт достаточно блюд,
# чтобы одновременно попасть и в норму дня, и в жёсткие рамки каждого слота.
SLOT_TOLERANCE = 2.0

# Ослабленный потолок для запасного прохода.
#
# Бывает, что слот физически не может набрать свою долю: у человека с набором
# массы и исключениями «свинина, сладости, мучное, лактоза» на завтрак
# остаётся 123 блюда, и лишь 8 из них дотягивают до нужных 683 ккал.
# Строгий потолок тогда не даёт другим приёмам добрать недостачу, и день
# недотягивает по калориям на 6-10%.
#
# Наращивать iterations в такой ситуации бесполезно — проверено: 300, 600,
# 1200 и 2400 дают ровно те же два провала. Дело не в поиске, а в том,
# что ему запрещено.
#
# Поэтому потолок ослабляется только при неудаче строгого прохода. Обычный
# день собирается как раньше и остаётся сбалансированным; перекос допускается
# лишь там, где иначе плана не будет вовсе.
#
# Значение — минимальное из тех, что убирают провалы. Замер на пяти профилях
# по шести сидам (python -m experiments.measure_solver):
#
#     ослабл.   провалов   откл.ккал   худший слот
#        2.3           1        1.5%          2.2x
#        2.5           0        1.1%          2.5x
#        2.8           0        1.2%          2.7x
#        3.0           0        1.1%          2.9x
#
# Дальше 2.5 смысла нет: провалов уже ноль, а каждая лишняя десятая —
# это ещё более перекошенный день у того, кому и так тяжело подобрать.
#
# ⚠️ Значение зависит от каталога и пересматривается вместе с ним. При 2.3
# провалов не было, пока из подбора не убрали сырое мясо и рыбу: каталог
# сузился, и запасному проходу перестало хватать свободы.
SLOT_TOLERANCE_RELAXED = 2.5


class PlanNotFeasible(Exception):
    """План собрать не удалось — как правило, из-за слишком узкого каталога.

    Отдельный тип нужен, чтобы вызывающий код (агент, API) отличал «не сошлось
    по данным» от настоящей поломки и мог ответить человеку по-человечески,
    а не пятисоткой.

    Args:
        message: что именно не получилось.
        role: роль блюда, на которой всё встало (если известна).
    """

    def __init__(self, message: str, role: str | None = None) -> None:
        super().__init__(message)
        self.role = role


# ────────────────────────────────────────────────────────────
# Модели
# ────────────────────────────────────────────────────────────


class MealItem(BaseModel):
    """Одна позиция в плане дня."""

    slot: str
    fdc_id: int
    name: str          # английское название из справочника — по нему идёт отбор
    name_ru: str = ""  # русское название для показа человеку
    role: str
    portion: str          # английское описание порции из справочника
    portion_ru: str = ""  # русское описание для показа
    servings: float
    grams: float

    kcal: float
    protein_g: float
    fat_g: float
    carb_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float

    @property
    def portion_title(self) -> str:
        """Как называть порцию в тексте для человека."""
        return self.portion_ru or self.portion

    @property
    def title(self) -> str:
        """Как называть блюдо в тексте для человека."""
        return self.name_ru or self.name

    # computed_field, а не голое property: состав должен уезжать в API
    # вместе с блюдом, иначе интерфейсу пришлось бы ходить за ним отдельно.
    @computed_field
    @property
    def composition(self) -> str:
        """Из чего состоит порция: «курица 69 г, рис 46 г».

        Считается по рецептуре FNDDS — по той же, по которой считается
        и цена. Человек видит ровно то, из чего сложилась сумма.

        Пусто, если артефакт состава не собран: без него план остаётся
        полностью рабочим, просто без этой строчки.
        """
        from ingredients import describe, load_composition

        return describe(load_composition().get(self.fdc_id, []), self.grams)

    def describe(self) -> str:
        """Строка для человека.

        Мера — граммы, и только они. Справочник USDA измеряет 72% блюд
        в стаканах, включая торты и сэндвичи: «Торт Чёрный лес — 1 стакан»
        для американской базы норма, по-русски бессмыслица. Множитель порции
        (×0.5) — тем более внутренняя кухня солвера: он масштабирует
        справочную порцию, чтобы попасть в норму, и пользователю знать
        об этом незачем.

        Исходные portion и servings остаются в модели: по ним солвер считает
        и их видно в API, но в текст для человека они не идут.
        """
        return (
            f"{self.title} — {self.grams:.0f} г, "
            f"{self.kcal:.0f} ккал, Б {self.protein_g:.0f} / "
            f"Ж {self.fat_g:.0f} / У {self.carb_g:.0f}"
        )


class DayPlan(BaseModel):
    """Собранный день: позиции, итоги и отклонение от нормы."""

    items: list[MealItem]
    totals: dict[str, float]
    targets: dict[str, float]
    deviation: dict[str, float] = Field(default_factory=dict)
    loss: float = 0.0

    # Бюджет, под который день собирался. None — про деньги не говорили.
    # Хранится в плане, чтобы было с чем сравнить стоимость: без него
    # «510 ₽» это просто число, а не «уложились» или «не уложились».
    budget: float | None = None

    @computed_field
    @property
    def cost(self) -> float | None:
        """Во что обходится день, рубли. None — у какого-то блюда нет цены."""
        total = day_cost(self.items)
        return None if total is None else round(total, 1)

    @computed_field
    @property
    def within_budget(self) -> bool | None:
        """Уложились ли в бюджет. None — бюджета не задавали или стоимость
        посчитать нельзя: «не знаем» не превращается ни в «да», ни в «нет»."""
        if not self.budget or self.cost is None:
            return None
        return self.cost <= self.budget

    def render(self, with_composition: bool = False) -> str:
        """Человекочитаемый план — то, что агент показывает пользователю.

        Args:
            with_composition: печатать под каждым блюдом его состав.
                По умолчанию нет: в переписке состав удлиняет план втрое,
                а нужен он не всегда. Включается там, где место есть —
                в выгрузке и в списке покупок.
        """
        lines: list[str] = []
        current_slot = None

        for item in self.items:
            if item.slot != current_slot:
                lines.append(f"\n{item.slot}:")
                current_slot = item.slot
            lines.append(f"  • {item.describe()}")
            if with_composition and (composition := item.composition):
                lines.append(f"      состав: {composition}")

        lines.append("")
        lines.append(
            f"Итого за день: {self.totals['kcal']:.0f} ккал "
            f"(цель {self.targets['kcal']:.0f}), "
            f"Б {self.totals['protein_g']:.0f} / "
            f"Ж {self.totals['fat_g']:.0f} / "
            f"У {self.totals['carb_g']:.0f} г, "
            f"клетчатка {self.totals['fiber_g']:.0f} г"
        )
        return "\n".join(lines).strip()


# ────────────────────────────────────────────────────────────
# Подготовка кандидатов
# ────────────────────────────────────────────────────────────


def _filter_catalog(
    catalog: pd.DataFrame, exclude: list[str] | None, allow_exotic: bool = False
) -> pd.DataFrame:
    """Убрать из каталога то, что человеку не подходит или не годится в план."""
    usable = catalog[
        (catalog["portion_g"] >= MIN_PORTION_G)
        & (catalog["portion_g"] <= MAX_PORTION_G)
        & (catalog["portion_kcal"] > 0)
    ].copy()

    # Дичь и субпродукты в автоматический подбор не идут: по калориям и белку
    # опоссум в обед проходит идеально, а показывать такое пользователю нельзя.
    if not allow_exotic and "mainstream" in usable.columns:
        usable = usable[usable["mainstream"]]

    # Блюда, продукты для которых не купить в России и СНГ (гритс, окра,
    # адобо, позиции американских сетей), в рацион не идут: человек просто
    # не найдёт из чего готовить. Разметку делает review_catalog.py.
    if "available_ru" in usable.columns:
        usable = usable[usable["available_ru"].fillna(True).astype(bool)]

    # Второй отсев, с другим вопросом: не «можно ли достать», а «едят ли это
    # у нас». Достать горькую дыню, розовую фасолу и семислойный салат
    # при желании можно — предлагать их в меню всё равно бессмысленно.
    # Отсекается не иностранное: паста, пицца, суши, плов и шаурма остаются.
    # Разметку делает review_familiar.py.
    if "familiar_ru" in usable.columns:
        usable = usable[usable["familiar_ru"].fillna(True).astype(bool)]

    # Блюдо, цену которого посчитать не удалось, в план не идёт.
    #
    # Причина прямая: стоимость рациона показывается всегда, и блюдо без
    # цены занижает её молча. Человек видит «около 400 ₽» и не знает, что
    # треть корзины в эту сумму не вошла. Подставлять вместо цены медиану
    # мало: она честна в среднем и произвольна в конкретном дне.
    #
    # С появлением подбора под бюджет это стало опасно вдвойне: солвер
    # искал дешёвое и шёл ИМЕННО за блюдами без цены — они выглядели
    # бесплатными. Так в дневной рацион на 60 ₽ попало 169 г жареных раков.
    #
    # Цена отсева невелика: 77 блюд из 4782, не более 3.3% внутри любой
    # роли. Проверено, что ни одна роль не обеднела настолько, чтобы
    # подбор перестал сходиться.
    #
    # Без условия «если цены есть»: раньше пустая таблица цен молча
    # выключала этот отсев, и план собирался из блюд, стоимость которых
    # никто не знает. Нет цен — это поломка данных, о ней надо сказать.
    priced = cost_per_gram()
    if not priced:
        raise RuntimeError(
            "Цены блюд не посчитаны (prices.cost_per_gram пуст) — подбор без цен "
            "занизил бы стоимость молча. Проверьте ingredients.parquet и prices.py."
        )
    usable = usable[usable["fdc_id"].astype(int).isin(priced)]

    if exclude:
        # Раскрытие категорий и сборку выражения делает restrictions:
        # «сладости» превращаются в полсотни слов, а совпадение ищется
        # по границам слова. Наивная подстрока здесь не годилась —
        # «oil» находился внутри «broiled» и выбрасывал запечённую курицу.
        # С исключениями: «Almond milk» не молоко, «Rice noodles» и «Bread,
        # gluten free» не глютен (restrictions.without_exceptions).
        usable = usable[~matches_terms(usable["name"], exclude).to_numpy()]

        # Второй проход — по разделу справочника. Он знает тип блюда там,
        # где название молчит: в «Gyro sandwich» слова «bread» нет, но
        # раздел «Sandwiches» сразу говорит, что внутри лепёшка.
        #
        # Пометку «gluten free» раздел не видит: «Bread, gluten free» лежит
        # в «Yeast breads». Для глютена такие блюда разделом не отсекаются
        # (их состав проверяет третий проход), для «мучного» — отсекаются:
        # это всё равно хлеб.
        if "category" in usable.columns:
            sections = usable["category"].str.lower()
            other = [t for t in exclude if not is_gluten_term(t)]
            gluten = [t for t in exclude if is_gluten_term(t)]
            dropped = pd.Series(False, index=usable.index)
            if other and (pattern := to_category_pattern(other)):
                dropped |= sections.str.contains(pattern, regex=True, na=False)
            if gluten and (pattern := to_category_pattern(gluten)):
                gluten_free = usable["name"].str.contains(GLUTEN_FREE, case=False, regex=True,
                                                          na=False)
                dropped |= sections.str.contains(pattern, regex=True, na=False) & ~gluten_free
            usable = usable[~dropped]

        # Третий проход — по составу. Название и раздел говорят, ЧТО это
        # за блюдо, а не что внутри: в «Club sandwich» ветчина и бекон,
        # в «Spanish rice» бекон, в «Bao bun» сало — и ни слова о свинине.
        # Без этого прохода после exclude=["pork"] в подборе оставалось
        # 123 блюда со свининой в рецептуре.
        #
        # Только ограничения-ИНГРЕДИЕНТЫ: «мучное», «сладости» и «жареное» —
        # типы блюд, их знают название и раздел (restrictions.TYPE_CATEGORIES).
        by_ingredient = ingredient_terms(exclude)
        if by_ingredient:
            with_excluded = _dishes_with_ingredient(tuple(sorted(by_ingredient)))
            if with_excluded:
                usable = usable[~usable["fdc_id"].astype(int).isin(with_excluded)]

    return usable


@lru_cache(maxsize=256)
def _dishes_with_ingredient(terms: tuple[str, ...]) -> frozenset[int]:
    """Блюда, в полном составе которых есть ингредиент из ограничений.

    Состав — полный (ingredients.build_leaves): с раскрытыми вложенными
    рецептурами и следовыми долями, а не тот, что показывается человеку.

    ``skip_absent=True`` — в отличие от названий блюд. Ингредиент — один
    продукт, и «Potatoes, mashed, flakes without milk» или «Ice cream,
    no sugar added» честно говорят об отсутствии: спрятать под хвостом
    там нечего. В названии блюда «…; no potatoes, gravy» подлива есть.

    Исключения — restrictions.without_exceptions: растительное «молоко»
    не лактоза (NOT_DAIRY), рисовая лапша и «gluten free» не глютен
    (NOT_GLUTEN, GLUTEN_FREE).
    Вырезается только само словосочетание, остаток названия проверяется:
    снимать название целиком — значит пропустить «Granola bar, milk
    chocolate coated, peanut butter» из-за арахисовой пасты.

    Кэш по набору ограничений: состав неизменен, а фильтр зовут на каждый
    день меню.
    """
    index = ingredient_dishes()
    if not index:
        return frozenset()
    names = pd.Series(list(index), dtype=object)
    mask = matches_terms(names, list(terms), skip_absent=True).to_numpy()
    return frozenset().union(*(index[name] for name in names[mask]))


def _candidates_by_role(catalog: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Разложить каталог по ролям — солверу так удобнее выбирать."""
    columns = ["fdc_id", "name", "name_ru", "role", "portion", "portion_ru", "portion_g"] + [
        f"portion_{name}" for name in NUTRIENT_COLUMNS
    ]
    available = [column for column in columns if column in catalog.columns]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for role, chunk in catalog[available].groupby("role"):
        grouped[str(role)] = chunk.to_dict("records")
    return grouped


def _make_item(slot: str, record: dict[str, Any], servings: float) -> MealItem:
    """Собрать позицию плана: масштабировать порцию на множитель."""
    return MealItem(
        slot=slot,
        fdc_id=int(record["fdc_id"]),
        name=str(record["name"]),
        name_ru=str(record.get("name_ru") or ""),
        role=str(record["role"]),
        portion=str(record["portion"]),
        portion_ru=str(record.get("portion_ru") or ""),
        servings=servings,
        grams=float(record["portion_g"]) * servings,
        kcal=float(record.get("portion_kcal", 0.0)) * servings,
        protein_g=float(record.get("portion_protein_g", 0.0)) * servings,
        fat_g=float(record.get("portion_fat_g", 0.0)) * servings,
        carb_g=float(record.get("portion_carb_g", 0.0)) * servings,
        fiber_g=float(record.get("portion_fiber_g", 0.0) or 0.0) * servings,
        sugar_g=float(record.get("portion_sugar_g", 0.0) or 0.0) * servings,
        sodium_mg=float(record.get("portion_sodium_mg", 0.0) or 0.0) * servings,
    )


# ────────────────────────────────────────────────────────────
# Целевая функция
# ────────────────────────────────────────────────────────────


def _totals(items: list[MealItem]) -> dict[str, float]:
    """Просуммировать нутриенты по всем позициям."""
    return {
        name: float(sum(getattr(item, name) for item in items)) for name in NUTRIENT_COLUMNS
    }


def _is_feasible(totals: dict[str, float], sodium_limit: float = SODIUM_LIMIT_MG) -> bool:
    """Не нарушены ли жёсткие ограничения рациона.

    Натрий — единственная граница, которая в приёмке проверяется точно
    («не выше 2300 мг»), а в целевой функции была лишь штрафом. Из-за этого
    оптимизатор охотно менял пару миллиграммов натрия на улучшение по белку
    и выдавал план на 2310 мг — нутрициологически неотличимый от 2295,
    но проверку не проходящий.

    ``sodium_limit`` ниже обычного — у остатка дня после съеденного
    (``rebuild_rest``): часть суточного предела уже израсходована.
    """
    return totals.get("sodium_mg", 0.0) <= sodium_limit


def _slot_penalty(items: list[MealItem], template: list[tuple[str, str, float]],
                  target_kcal: float) -> float:
    """Штраф за приёмы пищи, выбивающиеся из своей доли дня.

    Считается по абсолютной норме слота (доля × целевая калорийность),
    а не по фактической сумме дня: иначе перекос одного блюда «размазывался»
    бы по знаменателю и сам себя маскировал.
    """
    if target_kcal <= 0:
        return 0.0

    penalty = 0.0
    for item, (_slot, _role, share) in zip(items, template, strict=True):
        allowed = share * target_kcal * SLOT_TOLERANCE
        if item.kcal > allowed:
            penalty += (item.kcal - allowed) / target_kcal
    return penalty * SLOT_BALANCE_WEIGHT


# Доля приёмов пищи, которую предпочтения занимают «досыта». Дальше
# штраф не растёт: набить предпочтениями весь день — значит превратить
# неделю в три блюда, а разнообразие человеку нужно не меньше вкуса.
#
# 0.4 — это примерно три слота из семи: завтрак и два любимых блюда
# в день. Значение подобрано замером, см. EVOLUTION.md.
PREFERENCE_SATURATION = 0.4

# Вес предпочтений в общей невязке.
#
# Сознательно ниже веса любого нутриента: «хочу видеть гречку» не должно
# перевешивать попадание в норму по белку. Предпочтение — это тай-брейк
# между блюдами, одинаково подходящими по составу, а не цель сама по себе.
PREFERENCE_WEIGHT = 0.9


def preferred_ids(catalog: pd.DataFrame, include: list[str] | None) -> set[int]:
    """Какие блюда каталога человек хотел бы видеть.

    Сопоставление — той же машинкой, что и у исключений: категории
    раскрываются («овощи» → полсотни слов), совпадение ищется по границам
    слова. Разница только в знаке: там выбрасываем, здесь предпочитаем.

    Ищем и по английскому названию, и по русскому. У исключений хватало
    английского, потому что модель обязана возвращать английские слова.
    Здесь человек может назвать блюдо и по-русски («хочу творог»),
    а цена промаха разная: пропущенное исключение кладёт в план запрещённое,
    пропущенное предпочтение всего лишь не сработает.

    Хвосты «excluding …», «без …» пропускаются: блюдо «…excluding broccoli»
    не может засчитываться за пожелание «брокколи».
    """
    if not include:
        return set()

    pattern = to_pattern(include)
    if not pattern:
        return set()

    matched = matches_words(catalog["name"], pattern, skip_absent=True)
    if "name_ru" in catalog.columns:
        matched |= matches_words(catalog["name_ru"], pattern, skip_absent=True)

    by_category = to_category_pattern(include)
    if by_category and "category" in catalog.columns:
        matched |= catalog["category"].str.lower().str.contains(by_category, regex=True, na=False)

    return set(catalog.loc[matched, "fdc_id"].astype(int))


def _preference_penalty(items: list[MealItem], preferred: set[int]) -> float:
    """Штраф за то, что любимых блюд в дне меньше, чем хотелось бы.

    Ноль, когда предпочтений не задано вовсе, — тогда слагаемое не влияет
    ни на что и старые планы не меняются.

    Штраф насыщается на ``PREFERENCE_SATURATION``: дальше солверу незачем
    вытеснять остальные блюда, и день не схлопывается в одно и то же.
    """
    if not preferred or not items:
        return 0.0

    hits = sum(1 for item in items if item.fdc_id in preferred)
    share = hits / len(items)
    shortfall = max(0.0, PREFERENCE_SATURATION - share) / PREFERENCE_SATURATION
    return shortfall * PREFERENCE_WEIGHT


# Вес перерасхода бюджета в общей невязке.
#
# Ниже калорий (3.0) и белка (2.0) намеренно и по той же причине, что
# у предпочтений: уложиться в деньги ценой провала по белку — это не
# исполнение просьбы, а её извращение. Человек, попросивший 500 ₽ в день,
# не просил рацион из макарон.
#
# Значение подобрано замером, см. EVOLUTION.md.
BUDGET_WEIGHT = 2.2


def _budget_penalty(items: list[MealItem], budget: float | None,
                    costs: dict[int, float]) -> float:
    """Штраф за выход за дневной бюджет.

    Это ОГРАНИЧЕНИЕ, а не минимизация: пока план укладывается в сумму,
    штраф ровно ноль, и солверу незачем экономить дальше.

    Разница принципиальная. Минимизация цены схлопнула бы неделю
    в одно и то же самое дешёвое блюдо — а дешевле всего повторять.
    Ограничение же не трогает ничего, пока просьба выполняется.

    Ноль и при отсутствии бюджета: планы тех, кто про деньги не говорил,
    обязаны остаться прежними до последнего грамма.
    """
    if not budget or budget <= 0 or not items:
        return 0.0

    # Блюдо без цены сюда не доходит: `_filter_catalog` его не пускает.
    # Раньше на этот случай стояла медиана — и блюдо без цены тихо
    # получало «среднюю» стоимость. Теперь это ошибка: значит, план
    # собран в обход фильтра, и молчать о таком нельзя. Ноль тем более
    # нельзя: солвер немедленно набил бы день «бесплатным» (так в рацион
    # на 60 ₽ однажды попало 169 г жареных раков без цены).
    unpriced = [item.name for item in items if item.fdc_id not in costs]
    if unpriced:
        raise ValueError(f"Блюда без цены в подборе под бюджет: {unpriced[:3]}")
    total = sum(costs[item.fdc_id] * item.grams for item in items)
    if total <= budget:
        return 0.0

    return (total - budget) / budget * BUDGET_WEIGHT


def day_cost(items: list[MealItem], costs: dict[int, float] | None = None) -> float | None:
    """Во что обходится день, рубли. None — посчитать нельзя.

    Если хоть у одного блюда нет цены, стоимость дня НЕ называется:
    ни нулём (день выглядел бы дешевле), ни медианой (число выглядело бы
    посчитанным, а было бы догадкой). В план такие блюда не идут —
    `_filter_catalog` их отсекает, — так что None означает план,
    собранный в обход фильтра, и агент говорит об этом прямо.
    """
    costs = costs if costs is not None else cost_per_gram()
    if any(item.fdc_id not in costs for item in items):
        return None
    return sum(costs[item.fdc_id] * item.grams for item in items)


def _loss(totals: dict[str, float], targets: dict[str, float],
          sodium_limit: float = SODIUM_LIMIT_MG) -> float:
    """Насколько план не попал в норму. Меньше — лучше, ноль — идеально.

    Считаем относительные отклонения, а не абсолютные: промах на 100 ккал
    при цели 1400 и при цели 3000 — это разные по тяжести промахи.
    """
    loss = 0.0

    for name, weight in WEIGHTS.items():
        target = targets.get(name, 0.0)
        if target <= 0:
            continue

        relative = (totals.get(name, 0.0) - target) / target
        penalty = weight * abs(relative)

        # Недобор белка и клетчатки штрафуем дополнительно.
        if relative < 0 and name in UNDERSHOOT_PENALTY:
            penalty *= UNDERSHOOT_PENALTY[name]

        loss += penalty

    # Гигиена рациона: соль и добавленный сахар. Штраф за натрий намеренно
    # крупный — в FNDDS много ресторанной и консервированной еды, и без веса
    # солвер спокойно собирал день на 3500 мг натрия.
    if totals.get("sodium_mg", 0.0) > sodium_limit:
        loss += 3.0 * (totals["sodium_mg"] - sodium_limit) / SODIUM_LIMIT_MG

    sugar_kcal = totals.get("sugar_g", 0.0) * 4.0
    kcal = max(totals.get("kcal", 0.0), 1.0)
    sugar_share = sugar_kcal / kcal
    if sugar_share > SUGAR_KCAL_SHARE_LIMIT:
        loss += (sugar_share - SUGAR_KCAL_SHARE_LIMIT) * 3.0

    return loss


# ────────────────────────────────────────────────────────────
# Сборка
# ────────────────────────────────────────────────────────────


def _greedy_pick(
    slot: str,
    role: str,
    slot_kcal: float,
    pool: list[dict[str, Any]],
    used_ids: set[int],
    rng: random.Random,
    sample_size: int = 120,
    tolerance: float = SLOT_TOLERANCE,
) -> MealItem | None:
    """Выбрать блюдо в слот: ближе всего к норме слота по калорийности.

    Смотрим не весь пул, а случайную выборку — иначе каждый запуск давал бы
    один и тот же «оптимальный» завтрак, и планы на разные дни не отличались бы.
    """
    available = [record for record in pool if int(record["fdc_id"]) not in used_ids]
    if not available:
        return None

    sample = rng.sample(available, min(sample_size, len(available)))

    best_item: MealItem | None = None
    best_error = float("inf")
    cap = slot_kcal * tolerance

    for record in sample:
        for servings in SERVING_OPTIONS:
            item = _make_item(slot, record, servings)
            if not (MIN_PORTION_G <= item.grams <= MAX_PORTION_G):
                continue
            if item.kcal > cap:
                continue

            error = abs(item.kcal - slot_kcal)
            if error < best_error:
                best_error = error
                best_item = item

    return best_item


def _local_search(
    items: list[MealItem],
    template: list[tuple[str, str, float]],
    candidates: dict[str, list[dict[str, Any]]],
    targets: dict[str, float],
    rng: random.Random,
    iterations: int,
    sample_size: int = 40,
    tolerance: float = SLOT_TOLERANCE,
    preferred: set[int] | None = None,
    budget: float | None = None,
    costs: dict[int, float] | None = None,
    sodium_limit: float = SODIUM_LIMIT_MG,
) -> list[MealItem]:
    """Улучшать план по одной замене, пока невязка падает."""
    target_kcal = targets.get("kcal", 0.0)
    preferred = preferred or set()
    costs = costs if costs is not None else {}

    best = list(items)
    best_totals = _totals(best)
    best_loss = (_loss(best_totals, targets, sodium_limit)
                 + _slot_penalty(best, template, target_kcal)
                 + _preference_penalty(best, preferred)
                 + _budget_penalty(best, budget, costs))
    best_feasible = _is_feasible(best_totals, sodium_limit)

    for _ in range(iterations):
        index = rng.randrange(len(best))
        slot, role, share = template[index]
        pool = candidates.get(role, [])
        if not pool:
            continue

        used_ids = {item.fdc_id for position, item in enumerate(best) if position != index}

        # Половину попыток тратим на смену блюда, половину — на размер порции
        # уже выбранного. Второе дешевле и часто достаточно.
        if rng.random() < 0.5:
            trial_records = [
                record
                for record in rng.sample(pool, min(sample_size, len(pool)))
                if int(record["fdc_id"]) not in used_ids
            ]
        else:
            current = best[index]
            trial_records = [
                record for record in pool if int(record["fdc_id"]) == current.fdc_id
            ]

        slot_cap = share * target_kcal * tolerance

        for record in trial_records:
            for servings in SERVING_OPTIONS:
                item = _make_item(slot, record, servings)
                if not (MIN_PORTION_G <= item.grams <= MAX_PORTION_G):
                    continue
                # Приём пищи не может быть в разы больше своей доли дня.
                # Проверяем здесь, а не штрафом в целевой функции: штраф
                # конкурировал с попаданием в калории и проигрывал ему,
                # а перекос слота — это не «чуть хуже», это другой рацион.
                if item.kcal > slot_cap:
                    continue

                trial = list(best)
                trial[index] = item
                trial_totals = _totals(trial)
                trial_loss = (
                    _loss(trial_totals, targets, sodium_limit)
                    + _slot_penalty(trial, template, target_kcal)
                    + _preference_penalty(trial, preferred)
                    + _budget_penalty(trial, budget, costs)
                )
                trial_feasible = _is_feasible(trial_totals, sodium_limit)

                # Допустимое решение всегда лучше недопустимого, даже если
                # у последнего невязка меньше. Внутри одной категории
                # сравниваем по невязке, как раньше.
                better = (
                    (trial_feasible, -trial_loss) > (best_feasible, -best_loss)
                )
                if better:
                    best = trial
                    best_loss = trial_loss
                    best_feasible = trial_feasible

    return best


def _build_day_once(
    catalog: pd.DataFrame,
    targets: Targets,
    exclude: list[str] | None = None,
    seed: int = 0,
    # 150, а не 1200: жадный проход уже подходит близко, и локальный поиск
    # сходится за первые сотни замен.
    #
    # Значение снижали дважды. Сначала 1200 → 300: та же доля пройденных
    # проверок вчетверо быстрее. Потом 300 → 150, после того как из роли
    # «основное блюдо» убрали позиции без белка: на более структурированном
    # каталоге поиску просто нечего долго перебирать.
    #
    # Замер на пяти профилях по шести сидам:
    #
    #     итераций   провалов   откл.ккал   худший слот   сек/день
    #        150            0        1.2%          2.3x       0.48
    #        200            0        1.2%          2.3x       0.60
    #        300            0        1.2%          2.3x       0.88
    #
    # Качество одинаковое, поэтому берём самое быстрое.
    iterations: int = 150,
    allow_exotic: bool = False,
    template: list[tuple[str, str, float]] | None = None,
    tolerance: float = SLOT_TOLERANCE,
    include: list[str] | None = None,
    budget: float | None = None,
    avoid_ids: set[int] | None = None,
    sodium_limit: float = SODIUM_LIMIT_MG,
) -> DayPlan:
    """Собрать план на день под целевые КБЖУ.

    Args:
        catalog: каталог блюд из ``foods.load_catalog()``.
        targets: нормы из ``targets.compute_targets()``.
        exclude: подстроки в названиях блюд, которые нужно исключить.
        seed: фиксирует случайность — один seed даёт один и тот же план.
        iterations: сколько замен пробует локальный поиск (см. комментарий
            к значению по умолчанию).
        allow_exotic: разрешить дичь и субпродукты (по умолчанию нет).
    """
    rng = random.Random(seed)
    template = template or DAY_TEMPLATE

    usable = _filter_catalog(catalog, exclude, allow_exotic=allow_exotic)
    if avoid_ids:
        usable = usable[~usable["fdc_id"].astype(int).isin(avoid_ids)]
    candidates = _candidates_by_role(usable)

    # Предпочтения ищем УЖЕ в отфильтрованном каталоге: если человек
    # одновременно просит молоко и исключает лактозу, побеждает исключение.
    # Так безопаснее — нарушенное ограничение может быть аллергией,
    # а несработавшее предпочтение всего лишь огорчает.
    preferred = preferred_ids(usable, include)

    # Карта цен строится один раз на день и передаётся в локальный поиск:
    # он обращается к ней на каждой из сотен проб. Сама она кэширована
    # глобально, так что второй день берёт её даром.
    costs = cost_per_gram() if budget else {}

    # Проверяем не только наличие роли, но и запас по количеству: в шаблоне
    # роль «main» встречается дважды, а блюда в дне не повторяются, поэтому
    # одного подходящего блюда на две позиции не хватит.
    needed_per_role: dict[str, int] = {}
    for _slot, role, _share in template:
        needed_per_role[role] = needed_per_role.get(role, 0) + 1

    for role, needed in needed_per_role.items():
        available = len(candidates.get(role, []))
        if available < needed:
            raise PlanNotFeasible(
                f"После фильтрации осталось {available} блюд для роли «{role}», "
                f"а нужно минимум {needed}. Вероятно, список исключений слишком широкий.",
                role=role,
            )

    target_values = targets.as_dict()

    # 1. Жадная сборка по слотам.
    items: list[MealItem] = []
    used_ids: set[int] = set()

    for slot, role, share in template:
        item = _greedy_pick(slot, role, targets.kcal * share, candidates[role],
                            used_ids, rng, tolerance=tolerance)
        if item is None:
            raise PlanNotFeasible(
                f"Не удалось подобрать блюдо в слот «{slot}» (роль {role}): "
                f"подходящие блюда закончились.",
                role=role,
            )
        items.append(item)
        used_ids.add(item.fdc_id)

    # 2. Локальный поиск.
    items = _local_search(items, template, candidates, target_values, rng,
                          iterations, tolerance=tolerance, preferred=preferred,
                          budget=budget, costs=costs, sodium_limit=sodium_limit)

    totals = _totals(items)
    deviation = {
        name: round((totals[name] - value) / value, 4)
        for name, value in target_values.items()
        if value > 0
    }

    return DayPlan(
        items=items,
        totals={name: round(value, 1) for name, value in totals.items()},
        targets=target_values,
        deviation=deviation,
        budget=budget,
        loss=round(_loss(totals, target_values, sodium_limit)
                   + _slot_penalty(items, template, targets.kcal)
                   + _preference_penalty(items, preferred)
                   + _budget_penalty(items, budget, costs), 4),
    )


def build_day(
    catalog: pd.DataFrame,
    targets: Targets,
    exclude: list[str] | None = None,
    seed: int = 0,
    iterations: int = 150,
    allow_exotic: bool = False,
    template: list[tuple[str, str, float]] | None = None,
    include: list[str] | None = None,
    budget: float | None = None,
    avoid_ids: set[int] | None = None,
    sodium_limit: float = SODIUM_LIMIT_MG,
) -> DayPlan:
    """Собрать план на день под целевые КБЖУ.

    Сначала — строгий проход: ни один приём пищи не больше двойной своей доли.
    Если день не сошёлся по калориям, потолок ослабляется и подбор повторяется.

    Порядок именно такой, а не наоборот: обычный день должен быть
    сбалансированным, и перекос разрешается только там, где иначе плана
    не будет вовсе. См. SLOT_TOLERANCE_RELAXED.

    Args:
        catalog: каталог блюд из ``foods.load_catalog()``.
        targets: нормы из ``targets.compute_targets()``.
        exclude: ограничения по еде, см. ``restrictions``.
        seed: фиксирует случайность — один seed даёт один и тот же план.
        iterations: сколько замен пробует локальный поиск.
        allow_exotic: разрешить дичь и субпродукты (по умолчанию нет).
        avoid_ids: блюда, которых в этот день быть не должно (окно
            разнообразия меню, см. ``build_menu``).
        sodium_limit: предел натрия на ЭТИ слоты. Ниже суточного — у остатка
            дня после съеденного (``rebuild_rest``).
    """
    args = (catalog, targets, exclude, iterations, allow_exotic, template,
            include, budget, avoid_ids, sodium_limit)
    plan = _build_day_seeded(*args, seed=seed)
    if all(_checks(plan, sodium_limit).values()):
        return plan

    # День не прошёл проверки — пересобрать с производными сидами и взять
    # лучший. На узком каталоге солвер чувствителен к пути поиска: профиль
    # без свинины, сладкого, мучного и лактозы проваливал 1 сид из 40
    # (худшее отклонение 6.9%), а после сужения каталога по составу —
    # 2 из 40 с отклонением 19%, хотя годные дни в том же пуле были.
    # Это не подстройка допусков: проверки те же, ищем дольше только там,
    # где первый поиск не нашёл. Детерминизм сохраняется — производные
    # сиды выводятся из исходного.
    best = plan
    for attempt in range(1, RETRY_SEEDS + 1):
        alternative = _build_day_seeded(*args, seed=_retry_seed(seed, attempt))
        if (_plan_rank(alternative, targets, sodium_limit)
                < _plan_rank(best, targets, sodium_limit)):
            best = alternative
        if all(_checks(best, sodium_limit).values()):
            break
    return best


def _checks(plan: DayPlan, sodium_limit: float) -> dict[str, bool]:
    """check_plan с пределом натрия этих слотов, а не всего дня."""
    checks = check_plan(plan)
    checks["sodium_within_limit"] = plan.totals.get("sodium_mg", 0.0) <= sodium_limit
    return checks


# Сколько раз пересобирать день, не прошедший проверки. Подобрано замером
# (experiments.measure_solver, «с ограничениями» × 40 сидов).
RETRY_SEEDS = 4


def _retry_seed(seed: int, attempt: int) -> int:
    """Производный сид: свой для каждой попытки и не совпадающий с соседними
    днями меню (те идут подряд: seed, seed + 1, ...)."""
    return seed * 1_009 + attempt * 7_919 + 100_003


def _plan_rank(plan: DayPlan, targets: Targets,
               sodium_limit: float = SODIUM_LIMIT_MG) -> tuple[int, float]:
    """Чем меньше, тем лучше: сначала число непройденных проверок, потом
    отклонение по калориям."""
    failed = sum(not passed for passed in _checks(plan, sodium_limit).values())
    return failed, abs(plan.totals["kcal"] - targets.kcal) / targets.kcal


def _build_day_seeded(
    catalog: pd.DataFrame,
    targets: Targets,
    exclude: list[str] | None,
    iterations: int,
    allow_exotic: bool,
    template: list[tuple[str, str, float]] | None,
    include: list[str] | None,
    budget: float | None,
    avoid_ids: set[int] | None,
    sodium_limit: float,
    seed: int,
) -> DayPlan:
    """Одна попытка build_day: строгий проход, при неудаче — ослабленный."""
    plan = _build_day_once(catalog, targets, exclude, seed, iterations,
                           allow_exotic, template, SLOT_TOLERANCE, include, budget,
                           avoid_ids, sodium_limit)

    if check_plan(plan)["kcal_in_range"]:
        # Через эту ветку тоже надо проверить бюджет против нормы: калории
        # могут сойтись, а белок — нет. Ранний выход однажды уже пропустил
        # такой день мимо проверки (seed 4: 2023 ккал в допуске, белок
        # 92 г против 112 — и план уехал человеку).
        return _nutrition_wins_over_budget(plan, catalog, targets, exclude, seed,
                                           iterations, allow_exotic, template, include,
                                           avoid_ids, sodium_limit)

    relaxed = _build_day_once(catalog, targets, exclude, seed, iterations,
                              allow_exotic, template, SLOT_TOLERANCE_RELAXED,
                              include, budget, avoid_ids, sodium_limit)

    # Возвращаем ослабленный только если он и правда лучше: иначе человек
    # получил бы менее сбалансированный день без выигрыша по калориям.
    strict_gap = abs(plan.totals["kcal"] - targets.kcal)
    relaxed_gap = abs(relaxed.totals["kcal"] - targets.kcal)
    best = relaxed if relaxed_gap < strict_gap else plan

    return _nutrition_wins_over_budget(best, catalog, targets, exclude, seed,
                                       iterations, allow_exotic, template, include,
                                       avoid_ids, sodium_limit)


def _nutrition_wins_over_budget(
    plan: DayPlan,
    catalog: pd.DataFrame,
    targets: Targets,
    exclude: Sequence[str] | None,
    seed: int,
    iterations: int,
    allow_exotic: bool,
    template: list[tuple[str, str, float]] | None,
    include: list[str] | None,
    avoid_ids: set[int] | None = None,
    sodium_limit: float = SODIUM_LIMIT_MG,
) -> DayPlan:
    """Если ради бюджета сломалась норма — вернуть план без бюджета.

    Наблюдавшийся случай, и он опасный. На норме 3755 ккал с бюджетом
    60 ₽ в день солвер честно уложился в деньги и выдал план, проваливший
    проверки И по калориям, И по белку. Формально просьба выполнена,
    по существу человек получил недоедание.

    Деньги — это пожелание, норма — то, ради чего он пришёл. Поэтому
    при конфликте побеждает норма, а про невыполненный бюджет человеку
    говорится прямо (``_budget_note`` в agent.py).

    Возвращается план БЕЗ бюджета — без сравнения с бюджетным: тот уже
    провалил норму, а при конфликте побеждает она. Сам план без бюджета
    собирается как у SLOT_TOLERANCE этажом выше: строгий проход, при
    промахе по калориям — ослабленный. Отдельно белок у него не
    перепроверяется: без денежного штрафа солвер собирает обычный день,
    и его проверки — те же, что у дня без бюджета.
    """
    if not plan.budget:
        return plan

    checks = check_plan(plan)
    if checks["kcal_in_range"] and checks["protein_g_in_range"]:
        return plan

    # Тот же день без денежного ограничения.
    unlimited = _build_day_once(catalog, targets, exclude, seed, iterations,
                                allow_exotic, template, SLOT_TOLERANCE, include, None,
                                avoid_ids, sodium_limit)
    if not check_plan(unlimited)["kcal_in_range"]:
        unlimited = _build_day_once(catalog, targets, exclude, seed, iterations,
                                    allow_exotic, template, SLOT_TOLERANCE_RELAXED,
                                    include, None, avoid_ids, sodium_limit)

    # Бюджет сохраняем в плане: он не выполнен, но назван — и агент
    # обязан сказать об этом, а не промолчать.
    return unlimited.model_copy(update={"budget": plan.budget})


# ────────────────────────────────────────────────────────────
# «Съел не по плану — пересчитай день»
# ────────────────────────────────────────────────────────────

# Ниже этого остатка калорий пересобирать нечего: оставшиеся приёмы пищи
# пришлось бы собирать из крошечных порций или выходить за норму.
# 150 ккал — это один лёгкий перекус; честнее сказать «на сегодня всё»,
# чем натягивать три приёма пищи на остаток в 120 ккал.
MIN_REST_KCAL = 150.0


def eaten_item(catalog: pd.DataFrame, fdc_id: int, grams: float, slot: str) -> MealItem:
    """Съеденное как позиция дня: числа — из каталога по граммам, не от модели."""
    record = _records_by_id(catalog)[int(fdc_id)]
    servings = grams / float(record["portion_g"])
    return _make_item(f"{slot} (съедено)", record, servings)


def rebuild_rest(
    catalog: pd.DataFrame,
    targets: Targets,
    eaten: list[MealItem],
    covered: set[int],
    exclude: list[str] | None = None,
    include: list[str] | None = None,
    budget: float | None = None,
    seed: int = 0,
    kept: dict[int, MealItem] | None = None,
) -> DayPlan:
    """День, где съеденное зафиксировано, а остальные слоты собраны под остаток.

    ``covered`` — какие слоты DAY_TEMPLATE уже закрыты съеденным. ``kept`` —
    блюда прежнего плана, которые остаются как были ({слот: блюдо}): приёмы
    пищи ДО съеденного уже позади, и пересобирать завтрак, когда человек
    рассказывает про обед, — менять то, что уже съедено. Оставшиеся слоты
    собираются обычным ``build_day``: под остаток нормы, с долями слотов,
    пересчитанными на этот остаток. Проверки — по всему дню: план обязан
    быть честным про весь день, а не про его хвост.

    Остатка меньше MIN_REST_KCAL — ``PlanNotFeasible`` с числами: натягивать
    оставшиеся приёмы пищи на крохи или выходить за норму нельзя, человек
    должен услышать это прямо.
    """
    kept = {i: item for i, item in (kept or {}).items() if i not in covered}
    fixed = [*eaten, *kept.values()]
    full = targets.as_dict()
    fixed_totals = _totals(fixed)
    rest = [(slot, role, share) for i, (slot, role, share) in enumerate(DAY_TEMPLATE)
            if i not in covered and i not in kept]

    if rest:
        residual = {k: max(0.0, v - fixed_totals.get(k, 0.0)) for k, v in full.items()}
        if residual["kcal"] < MIN_REST_KCAL:
            raise PlanNotFeasible(
                f"За день уже набрано {fixed_totals['kcal']:.0f} ккал из {full['kcal']:.0f}: "
                f"на оставшиеся приёмы пищи остаётся {residual['kcal']:.0f} ккал — "
                "собирать из этого ещё приёмы пищи значит выйти за норму.")
        total_share = sum(share for *_, share in rest)
        template = [(slot, role, share / total_share) for slot, role, share in rest]
        rest_targets = targets.model_copy(update=residual)
        # Натрий съеденного тоже уже в дне. Без этого остаток собирался под
        # весь суточный предел, и после 200 г блинов (1000 мг натрия) день
        # проваливал проверку соли в 18 случаях из 18.
        sodium_left = max(0.0, SODIUM_LIMIT_MG - fixed_totals.get("sodium_mg", 0.0))
        rest_plan = build_day(catalog, rest_targets, exclude=exclude, seed=seed,
                              template=template, include=include, budget=budget,
                              avoid_ids={item.fdc_id for item in fixed},
                              sodium_limit=sodium_left)
        rest_items = iter(rest_plan.items)
    else:
        rest_items = iter(())

    # Позиции — в порядке дня: съеденное на месте своих слотов.
    by_slot: dict[int, list[MealItem]] = {}
    for item in eaten:
        base = item.slot.removesuffix(" (съедено)")
        index = next((i for i, (slot, *_) in enumerate(DAY_TEMPLATE)
                      if slot == base and i in covered), min(covered, default=0))
        by_slot.setdefault(index, []).append(item)
    items: list[MealItem] = []
    for i in range(len(DAY_TEMPLATE)):
        if i in covered:
            items += by_slot.get(i, [])
        else:
            items.append(kept[i] if i in kept else next(rest_items))
    return _assemble(items, full, budget, template=None)


# ────────────────────────────────────────────────────────────
# Замена одного блюда (ревью R8)
# ────────────────────────────────────────────────────────────

# Из скольких лучших замен выбирать по seed. Одна лучшая на каждый план
# давала бы на «замени ещё раз» то же блюдо, а случайная из всех — блюдо,
# которое еле проходит проверки. Пять — разнообразие без потери качества.
SWAP_TOP = 5

# Насколько замена может быть хуже лучшей (во сколько раз по невязке),
# чтобы попасть в те пять. Без порога случайный выбор из пяти брал и
# четвёртую: живой прогон — «замени завтрак» вместо греческого йогурта
# дало швейцарский сыр (невязка 1.50 против 0.95 у лучшей, жиры +32%).
SWAP_LOSS_MARGIN: float | None = 1.2

# Во сколько раз лучшая замена БЕЗ подстройки порций может быть хуже
# исходного дня, прежде чем подстроить порции остальных блюд. Раньше
# подстраивали, только если иначе проверки не проходят, — а проверки
# широкие (жиры ±35%), и каждая десятая замена делала день в 3.8 раза хуже.
#
# Замер experiments/swap_quality.py (252 замены, 6 профилей × 2 дня):
#
#     вариант                          в ±10%   рост невязки (медиана / 90%)
#     без порога и подстройки (было)     88%        ×1.14 / ×3.76
#     порог ×1.2                         90%        ×1.14 / ×3.75
#     порог ×1.2, порции при ×1.5        94%        ×1.11 / ×2.56
#     порог ×1.2, порции при ×1.2        95%        ×1.09 / ×2.56
#
# Проверки проходят все, разнообразие «замени ещё раз» то же (2.99 разных
# из 3), время 0.20 → 0.21 с.
SWAP_TUNE_ABOVE: float | None = 1.2


def _records_by_id(catalog: pd.DataFrame) -> dict[int, dict[str, Any]]:
    columns = [c for c in ["fdc_id", "name", "name_ru", "role", "portion", "portion_ru",
                           "portion_g"] + [f"portion_{n}" for n in NUTRIENT_COLUMNS]
               if c in catalog.columns]
    return {int(r["fdc_id"]): r for r in catalog[columns].to_dict("records")}


def _assemble(items: list[MealItem], targets: dict[str, float], budget: float | None,
              template: list[tuple[str, str, float]] | None) -> DayPlan:
    """День из готовых позиций: итоги, отклонения и невязка — как у build_day."""
    totals = _totals(items)
    loss = _loss(totals, targets)
    if template is not None:
        loss += _slot_penalty(items, template, targets.get("kcal", 0.0))
    return DayPlan(
        items=items,
        totals={name: round(value, 1) for name, value in totals.items()},
        targets=targets,
        deviation={name: round((totals[name] - value) / value, 4)
                   for name, value in targets.items() if value > 0},
        budget=budget,
        loss=round(loss, 4),
    )


def _tune_portions(items: list[MealItem], keep: int, records: dict[int, dict[str, Any]],
                   targets: dict[str, float], template: list[tuple[str, str, float]] | None,
                   passes: int = 3) -> list[MealItem]:
    """Подправить ПОРЦИИ остальных блюд (сами блюда не меняются), чтобы
    день с новым блюдом снова сошёлся. Покоординатный спуск — детерминирован."""
    best = list(items)
    best_loss = _assemble(best, targets, None, template).loss
    for _ in range(passes):
        improved = False
        for index, item in enumerate(best):
            if index == keep:
                continue
            for servings in SERVING_OPTIONS:
                if servings == item.servings:
                    continue
                candidate = _make_item(item.slot, records[item.fdc_id], servings)
                if not (MIN_PORTION_G <= candidate.grams <= MAX_PORTION_G):
                    continue
                trial = list(best)
                trial[index] = candidate
                loss = _assemble(trial, targets, None, template).loss
                if loss < best_loss:
                    best, best_loss, improved = trial, loss, True
            item = best[index]
        if not improved:
            break
    return best


def swap_item(
    catalog: pd.DataFrame,
    plan: DayPlan,
    index: int,
    exclude: list[str] | None = None,
    seed: int = 0,
    avoid_ids: set[int] | None = None,
) -> DayPlan:
    """Заменить блюдо на позиции ``index`` другим той же роли.

    Результат обязан пройти все проверки ``check_plan`` — иначе
    ``PlanNotFeasible``: человек должен услышать «не получилось», а не
    получить день, который тихо вышел из нормы.

    Числа блюд берутся из каталога по fdc_id и множителю порции, а не из
    присланного плана: ни одно число не приходит снаружи (через API план
    присылает клиент). При одном seed — одна и та же замена.

    Сначала остальные блюда не трогаются вовсе; если ни одна замена так
    не проходит проверки — подправляются их порции (но не сами блюда).

    Args:
        avoid_ids: какие блюда в этот слот уже предлагались и отвергнуты —
            чтобы «замени ещё раз» не вернуло прежнее.
    """
    records = _records_by_id(catalog)
    try:
        items = [_make_item(it.slot, records[it.fdc_id], it.servings) for it in plan.items]
    except KeyError as error:
        raise PlanNotFeasible(f"В справочнике нет блюда {error} из плана") from error
    if not 0 <= index < len(items):
        raise PlanNotFeasible(f"В плане нет позиции {index}")

    old = items[index]
    template = DAY_TEMPLATE if len(items) == len(DAY_TEMPLATE) else None
    share = template[index][2] if template else old.kcal / max(plan.targets.get("kcal", 1), 1)
    cap = share * plan.targets.get("kcal", 0.0) * SLOT_TOLERANCE_RELAXED
    taken = {it.fdc_id for it in items} | set(avoid_ids or ())
    usable = _filter_catalog(catalog, exclude)
    pool = [r for r in usable[usable["role"] == old.role].to_dict("records")
            if int(r["fdc_id"]) not in taken]

    def ranked(day_items: list[list[MealItem]]) -> list[DayPlan]:
        plans = [_assemble(d, plan.targets, plan.budget, template) for d in day_items]
        return sorted((p for p in plans if all(check_plan(p).values())),
                      key=lambda p: (p.loss, p.items[index].fdc_id))

    trials = []
    for record in pool:
        for servings in SERVING_OPTIONS:
            item = _make_item(old.slot, record, servings)
            if MIN_PORTION_G <= item.grams <= MAX_PORTION_G and item.kcal <= cap:
                trial = list(items)
                trial[index] = item
                trials.append(trial)

    passing = ranked(trials)
    worse = (SWAP_TUNE_ABOVE is not None and passing
             and passing[0].loss > plan.loss * SWAP_TUNE_ABOVE)
    if not passing or worse:
        # Блюдо меняется, порции остальных подстраиваются. Только для самых
        # близких к норме кандидатов: спуск по всем — секунды на запрос.
        closest = sorted(trials, key=lambda d: _assemble(d, plan.targets, None, template).loss)
        tuned = ranked([_tune_portions(d, index, records, plan.targets, template)
                        for d in closest[:30]])
        passing = sorted(passing + tuned, key=lambda p: (p.loss, p.items[index].fdc_id))
    if not passing:
        raise PlanNotFeasible(
            "Заменить это блюдо так, чтобы день остался в норме, не получилось: "
            "подходящих по составу блюд этой роли не нашлось.", role=old.role)

    # Одно блюдо — одна лучшая порция: иначе «пять лучших» были бы одним
    # блюдом в пяти размерах.
    unique: list[DayPlan] = []
    for candidate in passing:
        if SWAP_LOSS_MARGIN is not None and candidate.loss > passing[0].loss * SWAP_LOSS_MARGIN:
            break
        if candidate.items[index].fdc_id not in {p.items[index].fdc_id for p in unique}:
            unique.append(candidate)
        if len(unique) == SWAP_TOP:
            break
    return random.Random(seed).choice(unique)


# ────────────────────────────────────────────────────────────
# Меню на несколько дней
# ────────────────────────────────────────────────────────────


class Menu(BaseModel):
    """Меню на несколько дней."""

    days: list[DayPlan]
    targets: dict[str, float]

    @property
    def unique_dishes(self) -> int:
        return len({item.fdc_id for day in self.days for item in day.items})

    def render(self, with_composition: bool = True) -> str:
        """Человекочитаемое меню — для скачивания и для терминала.

        Состав здесь печатается ПО УМОЛЧАНИЮ, в отличие от плана на день:
        меню скачивают файлом, место в нём есть, а «из чего это блюдо» —
        первый вопрос, который возникает у плиты.
        """
        lines = [f"МЕНЮ НА {len(self.days)} " + day_word(len(self.days)).upper(), ""]
        lines.append(
            f"Норма на день: {self.targets['kcal']:.0f} ккал · "
            f"Б {self.targets['protein_g']:.0f} / "
            f"Ж {self.targets['fat_g']:.0f} / "
            f"У {self.targets['carb_g']:.0f} г"
        )
        lines.append("")

        for number, day in enumerate(self.days, start=1):
            lines.append("=" * 60)
            lines.append(f"ДЕНЬ {number}")
            lines.append("=" * 60)
            lines.append(day.render(with_composition=with_composition))
            lines.append("")

        return "\n".join(lines)


def day_word(count: int) -> str:
    """«1 день», «7 дней», «22 дня» — русские окончания.

    Единственная реализация на весь Python-код: раньше их было две,
    в солвере и в агенте, и отличались они только регистром. Правило
    склонения одно, а мест, где его можно забыть поправить, было три.

    Третья копия — ``dayWord`` в static/index.html. Её не убрать:
    другой рантайм.
    """
    if 11 <= count % 100 <= 14:
        return "дней"
    return {1: "день", 2: "дня", 3: "дня", 4: "дня"}.get(count % 10, "дней")


def build_menu(
    catalog: pd.DataFrame,
    targets: Targets | Sequence[Targets],
    days: int = 1,
    exclude: list[str] | None = None,
    seed: int = 0,
    iterations: int | None = None,
    variety_window: int = 2,
    include: list[str] | None = None,
    budget: float | None = None,
) -> Menu:
    """Собрать меню на несколько дней.

    Каждый день собирается отдельно со своим seed, поэтому дни получаются
    разными. Дополнительно работает окно разнообразия: блюда из последних
    ``variety_window`` дней временно исключаются из подбора, чтобы курица
    не стояла в плане три дня подряд.

    Args:
        days: сколько дней собрать.
        variety_window: за сколько предыдущих дней помнить блюда.
            0 — не следить за повторами между днями.

    Почему окно, а не запрет повторов на всё меню: на месяц потребовалось бы
    210 разных блюд, и под жёсткие ограничения по КБЖУ каталог бы не вытянул.
    Окно в два дня даёт разнообразие там, где оно заметно, не сужая выбор.

    Норма может быть одна на всё меню, а может быть своя на каждый день —
    тогда передаётся последовательность длиной в ``days``. Так работает
    циклирование: у тренировочного дня и дня отдыха разная калорийность,
    и собирать их по одной норме бессмысленно.
    """
    per_day = menu_targets(targets, days)
    plans = list(menu_days(
        catalog, per_day, exclude=exclude, seed=seed, iterations=iterations,
        variety_window=variety_window, include=include, budget=budget,
    ))
    return Menu(days=plans, targets=mean_targets(per_day))


def menu_targets(targets: Targets | Sequence[Targets], days: int) -> list[Targets]:
    """Норма на каждый день меню: одна на все дни или своя у каждого."""
    if days < 1:
        raise ValueError("Меню нельзя собрать меньше чем на один день")

    per_day = list(targets) if isinstance(targets, Sequence) else [targets] * days
    if len(per_day) != days:
        raise ValueError(
            f"Норм передано {len(per_day)}, а дней {days} — должно совпадать"
        )
    return per_day


def mean_targets(per_day: Sequence[Targets]) -> dict[str, float]:
    """Средняя норма меню — то, с чем сравнивается неделя целиком."""
    return {
        key: sum(t.as_dict()[key] for t in per_day) / len(per_day)
        for key in per_day[0].as_dict()
    }


def menu_days(
    catalog: pd.DataFrame,
    per_day: Sequence[Targets],
    exclude: list[str] | None = None,
    seed: int = 0,
    iterations: int | None = None,
    variety_window: int = 2,
    include: list[str] | None = None,
    budget: float | None = None,
) -> Iterator[DayPlan]:
    """Дни меню по одному — те же, что собирает ``build_menu``, и в том же порядке.

    Генератор, а не список, ради агента: объяснение первого дня моделью
    идёт параллельно со сборкой остальных (``NutritionAgent._node_make_plan``).
    Меню на месяц собирается 16 с, а объясняется только первый день —
    ждать остальные 29 дней перед вызовом модели незачем. Состояние окна
    разнообразия живёт внутри генератора, поэтому дни выходят те же до
    блюда, что и при сборке целиком.
    """
    days = len(per_day)
    recent: list[set[int]] = []

    # Любимые блюда окно разнообразия не трогает, иначе предпочтение
    # отменяло бы само себя: блюдо появлялось один раз и на следующие
    # два дня выпадало из подбора. Человек, попросивший овсянку
    # на завтрак, хочет её именно каждый день.
    liked = preferred_ids(catalog, include)

    for day_number in range(days):
        # Окно — по fdc_id, а не по названию через exclude. Наблюдавшийся
        # случай: названия шли в to_pattern уже экранированными, экранировались
        # второй раз, и шаблон искал обратную косую черту. Окно срабатывало
        # для 102 названий из 4782 — на семи днях 123 повтора внутри окна
        # при нуле ошибок и проходящих тестах.
        #
        # Сравнивались три починки (7 дней × 6 профилей × 8 сидов):
        #
        #     способ                    повторов  уникальных  провал.дней
        #     как было                     123       0.888          0
        #     fdc_id, жёсткий фильтр         0       0.922          0   ← этот
        #     названия без двойного         0       0.932          2
        #       экранирования (4 сида)
        #     штраф за повтор, вес 4         1       0.930          0
        #
        # Названия выбрасывали и похожие блюда («Rice, white» уносил все
        # «Rice, white, with …») и ломали норму. Штраф почти не уступает,
        # но повтор не исключает и добавляет ещё один подбираемый вес.
        avoid: set[int] = set()
        for used in recent[-variety_window:] if variety_window else []:
            avoid |= used
        avoid -= liked

        day_targets = per_day[day_number]

        try:
            plan = build_day(
                catalog,
                day_targets,
                exclude=list(exclude or []),
                seed=seed + day_number,
                iterations=iterations if iterations is not None else 300,
                include=include,
                budget=budget,
                avoid_ids=avoid,
            )
        except PlanNotFeasible:
            # Окно разнообразия сузило каталог слишком сильно — собираем день
            # без него. Лучше повтор блюда, чем отсутствие дня в меню.
            plan = build_day(
                catalog,
                day_targets,
                exclude=list(exclude or []),
                seed=seed + day_number,
                iterations=iterations if iterations is not None else 300,
                include=include,
                budget=budget,
            )

        recent.append({item.fdc_id for item in plan.items})
        yield plan


# ────────────────────────────────────────────────────────────
# Проверки плана
# ────────────────────────────────────────────────────────────

# Допуски приёмки. Это и есть тот детерминированный контроль, ради которого
# в проекте вообще считается КБЖУ: свойство плана проверяется арифметикой,
# а не мнением второй языковой модели.
TOLERANCES: dict[str, tuple[float, float]] = {
    "kcal": (-0.06, 0.06),        # ±6% от нормы
    "protein_g": (-0.12, 0.60),   # белок: недобор строго, перебор допустим
    "fat_g": (-0.35, 0.35),
    "carb_g": (-0.35, 0.35),
}

MIN_FIBER_SHARE = 0.60


@lru_cache(maxsize=16)
def _dishes_with_leaf(regex: str) -> frozenset[int]:
    """Блюда, в полном составе которых есть ингредиент, совпавший с выражением."""
    index = ingredient_dishes()
    names = pd.Series(list(index), dtype=object)
    hit = names[names.str.contains(regex, case=False, regex=True, na=False).to_numpy()]
    return frozenset().union(*(index[name] for name in hit))


def plan_cautions(items: Sequence[MealItem], exclude: list[str] | None
                  ) -> list[tuple[str, str, list[str]]]:
    """Спорные продукты в плане: (название, что сказать, блюда по-русски).

    Не фильтр: эти блюда в план идут (restrictions.CAUTIONS — решение
    владельца продукта), но человек должен услышать о них, а не найти сам.
    """
    found = []
    for regex, label, text in cautions_for(exclude):
        dishes = _dishes_with_leaf(regex)
        names = list(dict.fromkeys(
            item.name_ru or item.name for item in items if item.fdc_id in dishes))
        if names:
            found.append((label, text, names))
    return found


def check_plan(plan: DayPlan) -> dict[str, bool]:
    """Проверить план на попадание в норму. Возвращает карту «проверка → прошла».

    Ничего не бросает: агенту нужно знать, что именно не сошлось, чтобы
    честно сказать об этом пользователю.
    """
    results: dict[str, bool] = {}

    for name, (low, high) in TOLERANCES.items():
        relative = plan.deviation.get(name)
        results[f"{name}_in_range"] = relative is not None and low <= relative <= high

    fiber_target = plan.targets.get("fiber_g", 0.0)
    results["fiber_sufficient"] = (
        fiber_target <= 0 or plan.totals.get("fiber_g", 0.0) >= fiber_target * MIN_FIBER_SHARE
    )

    results["sodium_within_limit"] = plan.totals.get("sodium_mg", 0.0) <= SODIUM_LIMIT_MG
    results["no_duplicates"] = len({item.fdc_id for item in plan.items}) == len(plan.items)
    results["portions_sane"] = all(
        MIN_PORTION_G <= item.grams <= MAX_PORTION_G for item in plan.items
    )

    return results


def main() -> None:
    """Прогнать солвер на нескольких профилях и показать, что сошлось."""
    from foods import load_catalog
    from targets import Profile, compute_targets

    catalog = load_catalog()
    print(f"Каталог: {len(catalog)} блюд\n")

    profiles = [
        ("Похудение, женщина 31", Profile(sex="female", age=31, height_cm=168,
                                          weight_kg=74, activity="light", goal="lose")),
        ("Похудение, мужчина 40", Profile(sex="male", age=40, height_cm=182,
                                          weight_kg=95, activity="moderate", goal="lose")),
        ("Набор массы, мужчина 25", Profile(sex="male", age=25, height_cm=178,
                                            weight_kg=63, activity="high", goal="gain")),
        ("Без свинины и молочного", Profile(sex="female", age=45, height_cm=165,
                                            weight_kg=68, activity="moderate", goal="lose",
                                            exclude=["pork", "bacon", "ham", "milk", "cheese"])),
    ]

    for label, profile in profiles:
        targets = compute_targets(profile)
        plan = build_day(catalog, targets, exclude=profile.exclude, seed=7)
        checks = check_plan(plan)

        print("=" * 70)
        print(label)
        print("=" * 70)
        print(plan.render())
        print()
        failed = [name for name, passed in checks.items() if not passed]
        status = "все проверки пройдены" if not failed else f"не прошли: {', '.join(failed)}"
        print(f"Невязка {plan.loss:.3f} — {status}")
        print()


if __name__ == "__main__":
    main()
