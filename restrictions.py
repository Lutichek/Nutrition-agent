"""
Пищевые ограничения: из слов человека — в отбор по каталогу.

Человек говорит «не ем сладости и мучное», а каталог англоязычный и состоит
из конкретных блюд: «Doughnut, cake type, plain». Между этими двумя языками
нужен переводчик, и держать его на стороне модели нельзя.

Почему нельзя. Модель отлично справляется с «свинина → pork», но категорию
раскрыть не может: чтобы отсечь сладости, надо перечислить полсотни слов —
печенье, пирожные, пончики, мороженое, сиропы, — и модель либо перечислит
половину, либо не перечислит ничего. В наблюдавшемся случае «не ем сладости,
мучное» дало ``["pork", "milk", "cheese", "yogurt", "cream"]``: обе категории
потерялись молча, и в плане оказались два пончика.

Поэтому договор такой: модель называет категорию одним словом из
``CATEGORIES``, а раскрывает её этот модуль — детерминированно и под тестами.
Незнакомое слово тоже работает: оно просто ищется как есть.

Второе, что здесь решается, — как именно искать. Наивное вхождение подстроки
даёт тихие ложные срабатывания: «oil» находится внутри «broiled» и выбрасывает
82 блюда, включая обычную запечённую курицу, а «ham» находится внутри «graham
cracker». Поэтому совпадение только по границам слова.
"""

from __future__ import annotations

import re

# ── Категории ────────────────────────────────────────────────
# Ключ — то, что называет модель; значение — слова, которые ищутся
# в англоязычных названиях каталога.
#
# Списки намеренно многословны: пропущенное слово означает блюдо в тарелке
# у человека, который просил его не давать. Лишнее слово означает чуть более
# бедный выбор — цена несопоставима.
CATEGORIES: dict[str, list[str]] = {
    # «Сладости», «сладкое», «десерты»
    "sweets": [
        "candy", "candies", "chocolate", "cookie", "cookies", "biscuit",
        "cake", "cakes", "cupcake", "brownie", "brownies", "pie", "pies",
        "pastry", "pastries", "doughnut", "doughnuts", "donut", "donuts",
        "muffin", "muffins", "danish", "eclair", "tart", "tarts",
        "ice cream", "sherbet", "sorbet", "pudding", "custard", "mousse",
        "syrup", "jam", "jelly", "marmalade", "honey", "caramel", "fudge",
        "marshmallow", "marshmallows", "sweet roll", "sweet rolls",
        "frosting", "icing", "dessert", "desserts", "gelatin",
        "cheesecake", "cobbler", "turnover", "strudel", "baklava",
    ],
    # «Мучное», «выпечка», «хлебобулочное»
    "flour": [
        "bread", "breads", "roll", "rolls", "bun", "buns", "bagel", "bagels",
        "biscuit", "biscuits", "croissant", "croissants", "muffin", "muffins",
        "pancake", "pancakes", "waffle", "waffles", "crepe", "crepes",
        "tortilla", "tortillas", "pita", "cracker", "crackers",
        "doughnut", "doughnuts", "donut", "donuts", "pastry", "pastries",
        "cake", "cakes", "pie", "pies", "toast", "crouton", "croutons",
        "breadstick", "breadsticks", "dumpling", "dumplings", "pizza",
        "breaded", "breading", "batter", "battered", "flour", "dough",
    ],
    # «Аллергия на лактозу», «без молочного»
    "lactose": [
        "milk", "cheese", "yogurt", "yoghurt", "cream", "butter",
        "ice cream", "whey", "custard", "kefir", "ricotta", "mozzarella",
        "cheddar", "parmesan", "latte", "milkshake", "dairy",
    ],
    # Глютен
    "gluten": [
        "wheat", "bread", "breads", "pasta", "spaghetti", "macaroni",
        "noodle", "noodles", "barley", "rye", "couscous", "cracker",
        "crackers", "cake", "cakes", "cookie", "cookies", "pastry",
        "pastries", "tortilla", "tortillas", "pancake", "pancakes",
        "waffle", "waffles", "bagel", "bagels", "muffin", "muffins",
        "pizza", "dumpling", "dumplings", "breaded",
    ],
    # Вегетарианство: без мяса и рыбы
    "vegetarian": [
        "beef", "pork", "chicken", "turkey", "lamb", "veal", "duck",
        "goose", "bacon", "ham", "sausage", "salami", "pepperoni",
        "meat", "meatball", "meatballs", "meatloaf", "steak", "brisket",
        "liver", "gizzard", "fish", "salmon", "tuna", "cod", "tilapia",
        "trout", "sardine", "sardines", "anchovy", "anchovies", "herring",
        "shrimp", "crab", "lobster", "clam", "clams", "oyster", "oysters",
        "mussel", "mussels", "squid", "octopus", "scallop", "scallops",
        "seafood", "gelatin",
    ],
    # Веганство: вегетарианство плюс всё животное
    "vegan": [],   # заполняется ниже
    # «Не ем жареное»
    "fried": [
        "fried", "deep-fried", "deep fried", "batter-fried", "tempura",
        "french fries", "hash brown", "hash browns", "fritter", "fritters",
        "doughnut", "doughnuts", "donut", "donuts",
    ],
    "alcohol": [
        "beer", "wine", "vodka", "whiskey", "whisky", "rum", "liqueur",
        "brandy", "cocktail", "cocktails", "ale", "cider", "champagne",
    ],
    "nuts": [
        "nut", "nuts", "peanut", "peanuts", "almond", "almonds", "walnut",
        "walnuts", "cashew", "cashews", "pistachio", "pistachios", "pecan",
        "pecans", "hazelnut", "hazelnuts", "macadamia", "praline",
    ],
    "seafood": [
        "fish", "salmon", "tuna", "cod", "tilapia", "trout", "sardine",
        "sardines", "anchovy", "anchovies", "herring", "mackerel", "halibut",
        "shrimp", "crab", "lobster", "clam", "clams", "oyster", "oysters",
        "mussel", "mussels", "squid", "octopus", "scallop", "scallops",
        "seafood", "caviar", "shark", "eel",
    ],
    "eggs": ["egg", "eggs", "omelet", "omelette", "frittata", "quiche",
             "meringue", "mayonnaise"],
    "sugar": ["sugar", "sugars", "sweetened", "syrup", "candy", "candies",
              "soda", "cola", "soft drink"],
}

# Веган — это вегетарианец плюс молочное, яйца и мёд. Собирается из готовых
# списков, чтобы не поддерживать одни и те же слова в двух местах.
CATEGORIES["vegan"] = sorted(set(
    CATEGORIES["vegetarian"] + CATEGORIES["lactose"] + CATEGORIES["eggs"]
    + ["honey"]
))

# ── Отбор по разделу справочника ─────────────────────────────
# У каждого блюда FNDDS есть раздел («Cookies and brownies», «Yeast breads»),
# и это куда надёжнее названия. По названию «Gyro sandwich» не понять, что
# внутри лепёшка: слова «bread» в нём нет — а раздел «Sandwiches» знает.
# Так через фильтр «мучное» проходило 211 сэндвичей и «Italian Ice».
#
# Два механизма работают вместе и ловят разное: раздел знает ТИП блюда,
# название — его СОСТАВ. Свинину внутри «Meat mixed dishes» видно только
# по названию, а лепёшку внутри гироса — только по разделу.
#
# Значения — подстроки, ищутся в названии раздела без учёта регистра.
CATEGORY_PATTERNS: dict[str, list[str]] = {
    "sweets": [
        "cookies", "cakes and pies", "doughnut", "candy", "ice cream",
        "pudding", "gelatins, ices", "cereal bars", "higher sugar",
        "sugars and", "jams", "syrup", "frozen dairy", "sweet roll",
        "milk shakes", "smoothies",
    ],
    "flour": [
        "bread", "rolls and buns", "bagels", "biscuits", "muffins",
        "pancakes", "cracker", "tortilla", "pizza", "pasta",
        "grain-based", "cookies", "cakes and pies", "doughnut",
        "sandwich", "burger", "burritos", "pretzel", "dumplings",
        "macaroni", "noodles",
    ],
    "lactose": [
        "cheese", "milk", "yogurt", "ice cream", "frozen dairy",
        "dairy drinks", "cream-based", "butter",
    ],
    "gluten": [
        "bread", "rolls and buns", "bagels", "biscuits", "muffins",
        "pancakes", "cracker", "tortilla", "pizza", "pasta", "noodles",
        "grain-based", "cookies", "cakes and pies", "doughnut",
        "sandwich", "burger", "burritos", "pretzel", "macaroni",
    ],
    "vegetarian": [
        "meat", "poultry", "chicken", "turkey", "beef", "pork", "lamb",
        "bacon", "sausage", "fish", "seafood", "shellfish", "cold cuts",
        "frankfurter", "burgers", "liver and organ",
    ],
    "fried": ["fried", "french fries", "chicken patties", "doughnut"],
    "nuts": ["nuts and seeds", "peanut butter"],
    "seafood": ["fish", "seafood", "shellfish"],
    "eggs": ["eggs and omelets", "egg/breakfast", "egg rolls"],
    "sugar": ["candy", "soft drinks", "fruit drinks", "sport and energy",
              "higher sugar", "sugars and", "syrup", "jams"],
    "alcohol": ["alcoholic", "beer", "wine", "liquor"],
}

CATEGORY_PATTERNS["vegan"] = sorted(set(
    CATEGORY_PATTERNS["vegetarian"] + CATEGORY_PATTERNS["lactose"]
    + CATEGORY_PATTERNS["eggs"]
))

# Синонимы: модель называет категорию не всегда тем словом, что в ключе.
ALIASES = {
    "sweet": "sweets", "candy": "sweets", "desserts": "sweets",
    "dessert": "sweets", "sugary": "sweets",
    "flour products": "flour", "baked goods": "flour", "bakery": "flour",
    "pastry": "flour", "bread products": "flour",
    "dairy": "lactose", "milk products": "lactose", "lactose-free": "lactose",
    "meat": "vegetarian", "no meat": "vegetarian",
    "frying": "fried", "deep-fried": "fried",
    "nut": "nuts", "tree nuts": "nuts",
    "fish": "seafood", "shellfish": "seafood",
    "egg": "eggs",
}


def expand(terms: list[str] | None) -> list[str]:
    """Раскрыть ограничения человека в список слов для поиска.

    Категории разворачиваются в свои списки, остальные слова проходят
    как есть — незнакомое ограничение лучше применить буквально,
    чем молча выбросить.

    >>> expand(["pork", "sweets"])[:3]
    ['pork', 'candy', 'candies']
    """
    if not terms:
        return []

    words: list[str] = []
    seen: set[str] = set()

    for raw in terms:
        term = raw.strip().lower()
        if not term:
            continue

        category = ALIASES.get(term, term)
        for word in CATEGORIES.get(category, [term]):
            if word not in seen:
                seen.add(word)
                words.append(word)

    return words


def to_pattern(terms: list[str] | None) -> str:
    """Собрать регулярное выражение для отбора по названию блюда.

    Совпадение только по целому слову (плюс английское множественное
    число). Без этого «oil» находится внутри «broiled» и выбрасывает
    запечённую курицу, а «ham» — внутри «graham cracker».

    Пустая строка означает «ничего не исключаем».
    """
    words = expand(terms)
    if not words:
        return ""

    # re.escape обязателен: слова приходят из ответа модели, а через неё —
    # от пользователя. Скобка или звёздочка в ограничении не должна
    # уронить подбор рациона.
    alternatives = "|".join(re.escape(word) for word in words)
    return rf"\b(?:{alternatives})(?:s|es)?\b"


def to_category_pattern(terms: list[str] | None) -> str:
    """Выражение для отбора по разделу справочника.

    В отличие от ``to_pattern``, границы слова здесь не нужны: названия
    разделов — короткий закрытый список, составленный вручную, и ловить
    «Yeast breads» подстрокой «bread» тут правильно.

    Пустая строка означает, что среди ограничений нет ни одной категории.
    """
    if not terms:
        return ""

    patterns: list[str] = []
    for raw in terms:
        term = raw.strip().lower()
        category = ALIASES.get(term, term)
        patterns.extend(CATEGORY_PATTERNS.get(category, []))

    if not patterns:
        return ""
    return "|".join(re.escape(p) for p in dict.fromkeys(patterns))


def describe(terms: list[str] | None) -> str:
    """Ограничения по-русски — для показа человеку."""
    if not terms:
        return ""
    return ", ".join(RUSSIAN.get(term.strip().lower(), term) for term in terms)


# Названия ограничений по-русски. Человек писал «не ем свинину»,
# и показывать ему обратно «pork» неправильно.
RUSSIAN = {
    "sweets": "сладости", "flour": "мучное", "lactose": "лактоза",
    "gluten": "глютен", "vegetarian": "мясо и рыба", "vegan": "всё животное",
    "fried": "жареное", "alcohol": "алкоголь", "nuts": "орехи",
    "seafood": "рыба и морепродукты", "eggs": "яйца", "sugar": "сахар",

    "pork": "свинина", "beef": "говядина", "chicken": "курица",
    "turkey": "индейка", "lamb": "баранина", "veal": "телятина",
    "bacon": "бекон", "ham": "ветчина", "sausage": "колбаса",
    "fish": "рыба", "shrimp": "креветки", "milk": "молоко",
    "cheese": "сыр", "yogurt": "йогурт", "cream": "сливки",
    "butter": "сливочное масло", "egg": "яйца", "mushroom": "грибы",
    "mushrooms": "грибы", "white bread": "белый хлеб", "bread": "хлеб",
    "chips": "чипсы", "crackers": "сухарики", "soy": "соя",
    "wheat": "пшеница", "peanut": "арахис", "honey": "мёд",
}
