"""
Список покупок из меню: что и сколько купить на весь период.

Зачем. Меню на неделю — это 49 строк, разбросанных по семи дням, и одно
и то же блюдо встречается в разных днях. Человеку в магазине нужен другой
срез: не «что я ем в среду», а «сколько всего курицы взять».

Данные для этого уже есть целиком — считать нечего, нужно только свернуть
меню по блюдам и разложить по отделам магазина.

## Что здесь НЕ делается

**Блюда не раскладываются на ингредиенты.** В справочнике USDA единица —
готовое блюдо («Куриное бедро, запечённое»), а не рецепт. Превращать его
в «курица 180 г, масло 5 г, соль» нечем: состава рецептуры в данных нет,
а просить модель — значит получить выдуманные граммовки там, где весь
проект построен на проверяемых числах.

Поэтому список покупок — это список блюд с суммарным весом, а не список
продуктов. Для магазина этого достаточно: «куриное бедро, 1.2 кг» — вполне
рабочая строка.

## Отделы магазина

Категории справочника (133 штуки в отобранном каталоге) слишком дробные:
«Chicken, whole pieces», «Poultry mixed dishes» и «Turkey, whole pieces» —
это один отдел. Сворачиваем их в десяток групп, по которым человек
действительно ходит.

Сопоставление идёт по шаблону, а не моделью: названия категорий USDA
стабильны и конечны, и это ровно тот случай, где регулярка надёжнее
и бесплатнее.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, computed_field

from prices import cost_of

# Отделы магазина в порядке обхода: сначала то, что берут в глубине зала,
# бакалея последней. Порядок влияет только на вид списка.
#
# Каждый отдел — набор шаблонов по названию категории USDA. Проверяются
# по порядку, первое совпадение выигрывает, поэтому более узкие правила
# стоят выше более широких.
# Порядок проверки — от узкого к широкому, и он несколько раз существен.
# Три ловушки, на которых первая версия ошиблась, и все три — подстрока
# в чужом смысле:
#
#   «Soy and meat-alternative products»          содержит meat → уезжало в мясо
#   «Other fruits and fruit salads»              содержит salad → уезжало в овощи
#   «Pasta mixed dishes, excludes macaroni and cheese»
#                                                содержит cheese → уезжало в молочное
#
# Первые две лечатся порядком: соя проверяется раньше мяса, фрукты раньше
# овощей. Третья порядком не лечится вовсе — слово стоит внутри фразы
# «excludes ...», то есть означает прямо противоположное. Такие хвосты
# отрезаются до сопоставления.
DEPARTMENTS: list[tuple[str, str]] = [
    # Соя и заменители мяса — строго перед мясом.
    ("Соя и растительный белок", r"soy|meat-alternative|tofu"),
    ("Рыба и морепродукты", r"fish|seafood|shrimp|tuna|salmon|shellfish"),
    ("Мясо и птица", r"beef|pork|lamb|veal|chicken|turkey|poultry|meat|sausage|frankfurter|bacon|ham\b|organ"),
    ("Молочное и яйца", r"milk|cheese|yogurt|dairy|cream|egg"),
    # Снеки и сладкое — перед крупами и овощами: «Tortilla, corn, other chips»
    # иначе уходит то в хлеб (tortilla), то в овощи (corn).
    ("Снеки и сладкое", r"chips|cookie|cake|pie\b|pastry|dessert|candy|snack|sweet|ice cream|"
                        r"doughnut|gelatin|sorbet|ices\b|pudding"),
    # Фрукты — перед овощами: «fruit salads» иначе ловится словом salad.
    ("Фрукты и ягоды", r"fruit|apple|banana|berr|citrus|melon|grape|peach|pear|mango|papaya"),
    ("Овощи и зелень", r"vegetable|potato|tomato|carrot|lettuce|greens|onion|pepper|squash|corn\b|"
                       r"bean|pea\b|peas\b|legume|salad|broccoli|cabbage|spinach|coleslaw"),
    ("Крупы, хлеб, макароны", r"grain|rice|pasta|noodle|bread|cereal|oat|cracker|tortilla|flour|"
                              r"macaroni|bagel|muffin|roll|bun\b"),
    ("Орехи и семечки", r"nuts|seeds|peanut"),
    ("Масла, соусы, специи", r"oil|fat\b|butter|dressing|sauce|gravy|condiment|spice|sugar|syrup|jam"),
    ("Напитки", r"coffee|tea\b|juice|drink|beverage|water|soda|smoothie"),
    ("Готовые блюда", r"soup|pizza|sandwich|burger|stew|casserole|burrito|taco|mexican"),
]

DEFAULT_DEPARTMENT = "Прочее"

# Хвосты-исключения в названиях категорий USDA: «..., excludes macaroni
# and cheese», «..., not baby food». Слова после них означают, чего в
# категории НЕТ, и сопоставлять по ним — прямая ошибка.
_EXCLUSION_TAIL = re.compile(r",\s*(excludes|excluding|not)\b.*$", re.IGNORECASE)


def department_for(category: str) -> str:
    """Отдел магазина по категории справочника."""
    lowered = _EXCLUSION_TAIL.sub("", (category or "")).lower()
    for name, pattern in DEPARTMENTS:
        if re.search(pattern, lowered):
            return name
    return DEFAULT_DEPARTMENT


class ShoppingItem(BaseModel):
    """Одна строка списка покупок."""

    fdc_id: int
    name: str
    department: str
    grams: float
    times: int          # в скольких приёмах пищи встречается за период
    kcal: float         # суммарная калорийность — видно, на что уходит бюджет дня

    # Примерная стоимость позиции, рубли. None — цены нет, и это НЕ ноль:
    # неопознанный продукт не бесплатный, он неизвестной стоимости.
    # Из-за этого различия сумма всегда идёт вместе с покрытием.
    cost: float | None = None

    @property
    def display_cost(self) -> str:
        """Цена строки. Прочерк, если оценить не удалось."""
        return "—" if self.cost is None else f"{self.cost:.0f} ₽"

    @property
    def display_amount(self) -> str:
        """Вес в том виде, в каком его читают в магазине."""
        if self.grams >= 1000:
            return f"{self.grams / 1000:.1f} кг".replace(".0 кг", " кг")
        return f"{self.grams:.0f} г"


class ShoppingList(BaseModel):
    """Список покупок на весь период меню."""

    days: int
    items: list[ShoppingItem]

    @property
    def total_grams(self) -> float:
        return sum(item.grams for item in self.items)

    @property
    def total_kcal(self) -> float:
        return sum(item.kcal for item in self.items)

    # computed_field, а не голое property: иначе стоимость не попадёт
    # в JSON, и интерфейс получит список покупок без единственного числа,
    # ради которого он затевался.
    @computed_field
    @property
    def total_cost(self) -> float:
        """Сумма по тем позициям, которые удалось оценить.

        Смотреть на неё в отрыве от ``priced_share`` нельзя: при покрытии
        в половину веса это сумма за половину корзины, а выглядит как
        за всю.
        """
        return sum(item.cost for item in self.items if item.cost is not None)

    @computed_field
    @property
    def priced_share(self) -> float:
        """Какая доля ВЕСА корзины оценена. Доля веса, а не позиций.

        Позиции считать здесь бессмысленно: неоценёнными чаще остаются
        мелочи вроде специй, и «оценено 90% позиций» скрывало бы, что
        без цены осталось килограмм мяса.
        """
        total = self.total_grams
        if not total:
            return 0.0
        priced = sum(item.grams for item in self.items if item.cost is not None)
        return priced / total

    def cost_estimate(self) -> str:
        """Стоимость словами — с округлением и оговоркой.

        Округление до десятков рублей намеренное. Источник — средние цены
        по стране, разброс по регионам достигает трети, а овощи ходят
        в разы. Печатать «1 847 ₽» значит обещать точность, которой нет.
        """
        from prices import PRICES_SOURCE

        if self.priced_share < 0.5:
            return "Стоимость оценить не удалось: слишком мало позиций с известной ценой."

        rounded = round(self.total_cost / 10) * 10
        text = f"Примерная стоимость: около {rounded:,.0f} ₽".replace(",", " ")

        if self.days > 1:
            per_day = round(self.total_cost / self.days / 10) * 10
            text += f" (≈{per_day:,.0f} ₽ в день)".replace(",", " ")

        text += f".\nИсточник — {PRICES_SOURCE}; это оценка, а не чек."

        if self.priced_share < 0.95:
            text += f" Оценено {self.priced_share:.0%} веса корзины."

        return text

    def by_department(self) -> list[tuple[str, list[ShoppingItem]]]:
        """Сгруппировать по отделам в порядке обхода магазина."""
        order = [name for name, _ in DEPARTMENTS] + [DEFAULT_DEPARTMENT]
        groups: dict[str, list[ShoppingItem]] = {}
        for item in self.items:
            groups.setdefault(item.department, []).append(item)
        return [(name, groups[name]) for name in order if name in groups]

    def render(self) -> str:
        """Человекочитаемый список — для терминала и для выгрузки."""
        from solver import day_word

        lines = [f"СПИСОК ПОКУПОК НА {self.days} {day_word(self.days).upper()}", ""]
        for department, items in self.by_department():
            lines.append(department.upper())
            for item in items:
                mark = f" ×{item.times}" if item.times > 1 else "  "
                lines.append(f"  {item.name:<48} {item.display_amount:>8}{mark:<4}"
                             f"{item.display_cost:>9}")
            lines.append("")
        lines.append(f"Всего позиций: {len(self.items)}, "
                     f"общий вес {self.total_grams / 1000:.1f} кг")
        lines.append("")
        lines.append(self.cost_estimate())
        return "\n".join(lines)


def build_shopping_list(menu, catalog=None) -> ShoppingList:
    """Свернуть меню в список покупок.

    Одно и то же блюдо в разных днях складывается в одну строку: в магазине
    нужен суммарный вес, а не расписание.

    Args:
        menu: ``Menu`` или ``DayPlan`` — на один день тоже осмысленно.
        catalog: каталог блюд; нужен только чтобы узнать категорию для отдела.
            Без него все позиции попадут в «Прочее» — список останется
            рабочим, просто не разложенным по залу.
    """
    days = getattr(menu, "days", None)
    plans = days if isinstance(days, list) else [menu]

    categories: dict[int, str] = {}
    if catalog is not None:
        categories = dict(zip(catalog["fdc_id"].astype(int), catalog["category"].astype(str)))

    merged: dict[int, dict] = {}
    for plan in plans:
        for item in plan.items:
            entry = merged.setdefault(item.fdc_id, {
                "fdc_id": item.fdc_id,
                "name": item.name_ru or item.name,
                "department": department_for(categories.get(item.fdc_id, "")),
                "grams": 0.0,
                "times": 0,
                "kcal": 0.0,
                # Английское название храним отдельно: в строку списка идёт
                # русское, а цены сопоставляются по английскому — оно из USDA
                # и не зависит от качества перевода.
                "_en": item.name,
            })
            entry["grams"] += item.grams
            entry["kcal"] += item.kcal
            entry["times"] += 1

    # Цену считаем после свёртки, по суммарному весу: округление на каждом
    # дне копило бы ошибку.
    for entry in merged.values():
        entry["cost"] = cost_of(
            entry["grams"], entry.pop("_en"), categories.get(entry["fdc_id"], "")
        )

    # Внутри отдела — от тяжёлого к лёгкому: крупные позиции ищут первыми,
    # а мелочь вроде специй набирают заодно.
    items = sorted(
        (ShoppingItem(**entry) for entry in merged.values()),
        key=lambda item: (item.department, -item.grams),
    )
    return ShoppingList(days=len(plans), items=items)
