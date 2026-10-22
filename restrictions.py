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
from itertools import permutations

import pandas as pd

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
        # Аудит глютена (навык nutrition-dev): 108 блюд с пшеницей или ячменём
        # проходили exclude=["gluten"], в планах — 26–30% дней. Пшеница
        # там, где слова «wheat» нет:
        # — зерно и изделия под своими именами;
        "seitan", "bulgur", "semolina", "farina", "spelt", "graham", "panko",
        "orzo", "lasagna", "ravioli", "tortellini", "gnocchi", "ramen", "udon",
        "wonton", "wontons", "egg roll", "egg rolls", "potsticker", "potstickers",
        "biscuit", "biscuits", "croissant", "croissants", "doughnut", "doughnuts",
        "donut", "donuts", "bun", "buns", "pita", "crepe", "crepes", "pie crust",
        "pretzel", "pretzels", "crouton", "croutons", "breadstick", "breadsticks",
        "breading", "batter", "battered", "stuffing", "ice cream cone",
        "chicken nugget", "chicken nuggets", "kiev",
        # — солод и пиво (ячмень);
        "malt", "malted", "beer", "coffee substitute", "cereal grain beverage",
        "milky way",
        # — продукты справочника SR без рецептуры: мука или соевый соус
        #   на пшенице в них по определению, а раскрыть их нечем
        #   (тот же довод, что у «pizza meat topping» для свинины).
        #   «Gravy, canned» стоял в 33 блюдах, грибной суп-концентрат — в 18;
        "gravy", "cream of mushroom", "cheese sauce prepared recipe",
        "soy sauce", "teriyaki", "hoisin", "lo mein", "chow mein",
        "fried rice", "hot and sour soup", "surimi", "vermicelli mix",
        "crunch coating",
        # Голое «flour» НЕ вписывать: уносит рисовую, кукурузную и
        # гречневую муку, а пшеничную и так ловит «wheat». «nugget» без
        # «chicken» уносил «Ice cream bar, stick or nugget».
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
        # «hard cider», а не «cider»: в USDA «Apple cider» — яблочный сок
        # (раздел «Apple juice»), и при халяле он уходил как алкоголь.
        "brandy", "cocktail", "cocktails", "ale", "hard cider", "champagne",
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
    # «Не ем свинину». Своя категория, а не слово «pork» как есть: в USDA
    # свинина чаще названа без него — «Ham», «Bacon», «Hog maws», «Salami».
    # Пока категории не было, 91 блюдо со свиным словом в названии
    # проходило мимо exclude=["pork"], и в план попали свиные желудки.
    # «Turkey bacon» и «Beef sausage» уходят вместе со свининой: беднее
    # выбор, а не запрет в тарелке.
    "pork": [
        "pork", "ham", "hams", "bacon", "hog", "hogs", "pig", "pigs",
        "lard", "prosciutto", "pancetta", "salami", "pepperoni", "chorizo",
        "kielbasa", "bratwurst", "mortadella", "chitterlings", "cracklings",
        "scrapple", "spareribs", "pupusas del cerdo",
        "boar",
        # Мясная начинка замороженной пиццы — продукт справочника SR, а не
        # блюдо FNDDS: своей рецептуры у неё нет, и раскрыть её нечем
        # (ingredients.build_leaves). На деле это пепперони и колбаски.
        #
        # Других слов «мясо неизвестно какое» тут нет НАМЕРЕННО. Их заменила
        # рецептура: «Breakfast meat» раскрывается в бекон и ветчину сам,
        # а угаданный «hot dog, NFS» по рецептуре оказался говяжьей
        # сосиской — догадка уносила блюда без свинины.
        "pizza meat topping",
    ],
}

# Вегетарианец не ест и свинину — включая сало и прошутто, которых
# в его списке раньше не было.
CATEGORIES["vegetarian"] = sorted(set(CATEGORIES["vegetarian"] + CATEGORIES["pork"]))

# Халяль: без свинины и без алкоголя. Желатин не исключается, а
# подсвечивается (CAUTIONS): он бывает и свиным, и говяжьим, и рыбным.
# Без этой категории «halal» искался буквально и не отсекал ничего.
CATEGORIES["halal"] = sorted(set(CATEGORIES["pork"] + CATEGORIES["alcohol"]))

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
    # Подстроки «pasta», «noodles», «tortilla» уносили разделы целиком:
    # «Pasta, noodles, cooked grains» — с единственной в каталоге гречкой,
    # пшеном и киноа, «Tortilla, corn, other chips» — с кукурузными чипсами.
    # Пшеничное в этих разделах ловится по названию (pasta, noodle, bulgur,
    # couscous, barley), поэтому раздел сужен до тех, где глютен — всё.
    "gluten": [
        "bread", "rolls and buns", "bagels", "biscuits", "muffins",
        "pancakes", "cracker", "tortillas", "pizza", "pasta mixed",
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
    "pork products": "pork", "swine": "pork", "pig meat": "pork",
    # Формулировки диагноза и «X-free». Незнакомое слово ищется буквально,
    # и это опасно в обе стороны: «gluten-free» буквально уносил именно
    # безглютеновое (11 блюд) и оставлял всё остальное, а «celiac»,
    # «no gluten», «halal», «dairy-free», «nut allergy» не отсекали ничего.
    "celiac": "gluten", "coeliac": "gluten", "celiac disease": "gluten",
    "gluten-free": "gluten", "gluten free": "gluten", "no gluten": "gluten",
    "without gluten": "gluten", "wheat allergy": "gluten",
    "dairy-free": "lactose", "dairy free": "lactose", "milk allergy": "lactose",
    "lactose intolerance": "lactose", "no dairy": "lactose",
    "nut allergy": "nuts", "tree nut allergy": "nuts", "peanut allergy": "peanut",
    "egg allergy": "eggs", "fish allergy": "seafood", "shellfish allergy": "seafood",
    "no pork": "pork", "vegetarian diet": "vegetarian", "vegan diet": "vegan",
    "no alcohol": "alcohol", "alcohol-free": "alcohol",
}

# ── Исключения из совпадений ─────────────────────────────────
# Слово категории есть, продукта нет. Вырезается только словосочетание,
# остаток названия проверяется как обычно (см. NOT_DAIRY ниже).
#
# Глютен: рисовая лапша и бумага, кукурузная тортилья — без пшеницы.
# Лист «Rice noodles» по слову «noodles» уносил 7 блюд пад-тай и фо.
NOT_GLUTEN = (
    r"\brice\s+(?:noodles?|paper|vermicelli|crackers?|cakes?)\b"
    r"|\bcorn\s+tortillas?\b|\btortillas?,\s+corn\b"
)
# ── Предупреждения: не исключать, а сказать вслух ────────────
# Продукт, который при ограничении СПОРЕН, а не запрещён: решает человек,
# но знать о нём он обязан. Выбор владельца продукта (не кода):
#
# * овёс при глютене: сам по себе без глютена, но почти всегда загрязнён
#   пшеницей; исключать — значит отнять 60 блюд, в основном завтраки;
# * желатин при отказе от свинины: бывает свиным, говяжьим, рыбным —
#   по названию не различить.
#
# Ключ — категории ограничения (после ALIASES), значение — что искать
# в полном составе блюда, как назвать по-русски и что сказать.
CAUTIONS: list[tuple[frozenset[str], str, str, str]] = [
    (frozenset({"gluten"}), r"\boats?\b|\boatmeal\b|\bgranola\b|\bmuesli\b", "овёс",
     "Обычный овёс почти всегда загрязнён пшеницей, поэтому при целиакии "
     "берите овёс и хлопья с маркировкой «без глютена»"),
    (frozenset({"pork", "halal"}), r"\bgelatin", "желатин",
     "Желатин бывает свиным или содержит следы свинины. Если это для вас "
     "важно, выбирайте продукт с говяжьим или рыбным желатином либо "
     "с маркировкой «халяль»"),
]


def cautions_for(terms: list[str] | None) -> list[tuple[str, str, str]]:
    """Предупреждения, относящиеся к ограничениям: (что искать, название, текст)."""
    categories = {ALIASES.get(t.strip().lower(), t.strip().lower()) for t in terms or []}
    return [(regex, label, text) for keys, regex, label, text in CAUTIONS
            if keys & categories]


# Пометка «gluten free» снимает совпадение со всего названия: «Bread,
# gluten free» — хлеб, но без глютена. 12 таких блюд уходили по словам
# bread, pasta, cracker. Только для глютена: для «мучного» это всё равно хлеб.
GLUTEN_FREE = r"\bgluten[-\s]*free\b"


# Категории-ТИПЫ блюд: по составу они не проверяются.
#
# «Не ем мучное» — это «не ем хлеб и выпечку», а не «ни ложки муки
# в соусе»; «сладости» — десерты, а не сахар в маринаде; «жареное» —
# способ приготовления, его в рецептуре нет вовсе. Тип блюда знают
# название и раздел справочника. Проверка по составу у них уносила
# тушёную курицу по-мексикански из-за загустителя, а профиль с четырьмя
# ограничениями на 40 сидах проваливал 2 дня с отклонением до 19%.
#
# Остальное — ингредиенты (свинина, лактоза, глютен, орехи, яйца,
# морепродукты, отдельные продукты): для них важно именно что ВНУТРИ,
# вплоть до следовых количеств.
TYPE_CATEGORIES = frozenset({"sweets", "flour", "fried"})

# Растительное «молоко», «масло», «сливки» — не молочное. При проверке
# по составу слово «milk» находило «Nuts, coconut milk» в карри,
# «butter» — «Peanut butter», «cream» — «Cream of wheat».
# Слова строго СОСЕДНИЕ: «rice … milk» через запятую или «made with»
# снял бы «Rice pudding, made with milk» — аллерген в тарелке.
NOT_DAIRY = (
    r"\b(?:coconut|almond|soy|soya|oat|rice|cashew|hemp)\s+"
    r"(?:milk|milks|cream|yogurt|beverage)\b|\bsoymilk\b"
    # Ореховая паста — не молочное, только когда название ЕЮ И ЯВЛЯЕТСЯ:
    # «Peanut butter, smooth», «Nuts, cashew butter». В «Cookie, peanut
    # butter» и «Chocolate candy, peanut butter filled» вырезанная паста
    # прятала сливочное масло и молочный шоколад — 8 блюд возвращались
    # человеку без лактозы.
    r"|^(?:peanut|almond|cashew|nut|seed|sunflower seed)\s+butter\b"
    r"|^nuts,\s+\w+\s+butter\b|\b(?:cocoa|shea|apple)\s+butter\b"
    r"|\bcream\s+(?:substitute|of\s+(?:wheat|rice|tartar))\b"
    r"|\bnon-?dairy\b"
)


def ingredient_terms(terms: list[str] | None) -> list[str]:
    """Ограничения, которые проверяются по СОСТАВУ блюда — без типов блюд."""
    return [
        term for term in (terms or [])
        if ALIASES.get(term.strip().lower(), term.strip().lower()) not in TYPE_CATEGORIES
    ]


def is_dairy_term(term: str) -> bool:
    """Раскрывается ли ограничение в молочные слова (lactose, vegan, «milk»…)."""
    return bool(set(expand([term])) & set(CATEGORIES["lactose"]))


def is_gluten_term(term: str) -> bool:
    """Ограничение — глютен (в том числе синонимом), а не «мучное»."""
    key = term.strip().lower()
    return ALIASES.get(key, key) == "gluten"


def has_exceptions(term: str) -> bool:
    return is_dairy_term(term) or is_gluten_term(term)


def without_exceptions(names: pd.Series, term: str) -> pd.Series:
    """Названия, в которых вырезаны ложные совпадения для этого ограничения.

    Применяется и к названиям блюд, и к ингредиентам полного состава —
    в обоих местах «coconut milk» не молоко, а «rice noodles» не пшеница.
    """
    out = names.astype(object)
    if is_dairy_term(term):
        out = out.str.replace(NOT_DAIRY, " ", case=False, regex=True)
    if is_gluten_term(term):
        out = out.where(~out.str.contains(GLUTEN_FREE, case=False, regex=True, na=False), "")
        out = out.str.replace(NOT_GLUTEN, " ", case=False, regex=True)
    return out


def matches_terms(names: pd.Series, terms: list[str], skip_absent: bool = False) -> pd.Series:
    """Совпадение с любым из ограничений — с учётом исключений каждого.

    Ограничения без исключений ищутся одним выражением, как раньше,
    — это быстрее и даёт тот же результат.
    """
    hit = pd.Series(False, index=names.index)
    plain = [t for t in terms if not has_exceptions(t)]
    if plain and (pattern := to_pattern(plain)):
        hit |= matches_words(names, pattern, skip_absent=skip_absent)
    for term in terms:
        if has_exceptions(term) and (pattern := to_pattern([term])):
            hit |= matches_words(without_exceptions(names, term), pattern,
                                 skip_absent=skip_absent)
    return hit


def expand(terms: list[str] | None) -> list[str]:
    """Раскрыть ограничения человека в список слов для поиска.

    Категории разворачиваются в свои списки, остальные слова проходят
    как есть — незнакомое ограничение лучше применить буквально,
    чем молча выбросить.

    >>> expand(["pork", "sweets"])[:3]
    ['pork', 'ham', 'hams']
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
        # [category], а не [term]: синоним может вести и на отдельный продукт
        # («peanut allergy» → «peanut»), а не только на категорию.
        for word in CATEGORIES.get(category, [category]):
            if word not in seen:
                seen.add(word)
                words.append(word)

    return words


def to_pattern(terms: list[str] | None) -> str:
    """Собрать регулярное выражение для отбора по названию блюда.

    Совпадение только по целому слову (плюс английское множественное
    число). Без этого «oil» находится внутри «broiled» и выбрасывает
    запечённую курицу, а «ham» — внутри «graham cracker».

    Число слова не важно в обе стороны: модель пишет то «plums», то «plum»,
    а USDA — «Plums, raw» и «Plum, canned». Прежний шаблон умел только
    ДОПИСАТЬ окончание, и «cucumbers», «raspberry», «tomatoes» не находили
    ничего — 9 промахов из 132 пожеланий, при том что блюда в каталоге есть.

    Русское слово ищется по основе (см. ``_russian_forms``): оно приходит
    запасным путём для пожеланий, у которых не нашлось английского слова.

    Название из нескольких слов совпадает, когда в блюде есть ВСЕ его слова,
    в любом порядке (см. ``_term_pattern``).

    Пустая строка означает «ничего не исключаем».
    """
    words = expand(terms)
    if not words:
        return ""

    return "(?:" + "|".join(_term_pattern(word) for word in words) + ")"


_CYRILLIC = re.compile(r"[а-яё]")
_PLAIN_WORD = re.compile(r"[a-zа-яё]+(?:['-][a-zа-яё]+)*")
_RUSSIAN_ENDING = re.compile(r"[аяоеиыуюйь]+$")


def _term_pattern(term: str) -> str:
    """Выражение для одного продукта: одно слово — по границам слова,
    несколько — все слова в любом порядке.

    Порядок слов в USDA обратный разговорному: «Cheese, Feta», «Yogurt,
    Greek», «Rice, brown and wild». Фраза целиком не находила их вовсе,
    и это било в обе стороны. Исключение «feta cheese» и «greek yogurt»
    не отсекало НИЧЕГО — аллерген оставался в подборе. А пожелание «сыр
    фета» уходило в русский запасной путь, где по основе первого слова
    «сыр» находило 505 блюд: «Сырое яблоко», «Спаржа, сырая».

    Цена — лишние совпадения: «cream cheese» уносит и «Ravioli,
    cheese-filled, with cream sauce». Для исключения это беднее выбор,
    а не аллерген в тарелке; перестало отсекаться ноль блюд.

    «В любом порядке» — перестановками слов, а не просмотром вперёд
    (``^(?=.*a)(?=.*b)``). Результат тот же до блюда, а фильтр каталога
    с исключениями «свинина, сладкое, мучное, лактоза» — 15 мс против 211:
    просмотр вперёд замедлял подбор дня на 8%. Перестановок n!, поэтому
    для фраз длиннее ``_MAX_PERMUTED`` слов — всё-таки просмотр.
    """
    tokens = term.split()
    if len(tokens) == 1 or not all(_PLAIN_WORD.fullmatch(token) for token in tokens):
        return rf"\b(?:{_word_forms(term)})\b"

    forms = [rf"\b(?:{_word_forms(token)})\b" for token in tokens]
    if len(tokens) > _MAX_PERMUTED:
        return "^" + "".join(f"(?=.*{form})" for form in forms)
    return "(?:" + "|".join(".*".join(order) for order in permutations(forms)) + ")"


_MAX_PERMUTED = 4


def _word_forms(word: str) -> str:
    """Выражение для одного слова: все формы числа, без границ слова."""
    if _CYRILLIC.search(word):
        return _russian_forms(word)

    # re.escape обязателен: слова приходят из ответа модели, а через неё —
    # от пользователя. Скобка или звёздочка в ограничении не должна
    # уронить подбор рациона. Всё, что не похоже на обычное слово,
    # ищется буквально, как и раньше.
    if not _PLAIN_WORD.fullmatch(word):
        return re.escape(word) + "(?:s|es)?"

    return _english_forms(word)


def _english_forms(word: str) -> str:
    """Единственное и множественное число английского слова.

    «-ies» даёт и «-y», и «-ie»: berries → berry, но cookies → cookie.
    """
    if re.search(r"[^aeiou]ies$", word):
        return re.escape(word[:-3]) + "(?:y|ies?)"
    if re.search(r"[^aeiou]y$", word):
        return re.escape(word[:-1]) + "(?:y|ies?)"
    if re.search(r"(?:oes|ches|shes|xes|sses)$", word):
        return re.escape(word[:-2]) + "(?:es)?"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return re.escape(word[:-1]) + "(?:s|es)?"
    return re.escape(word) + "(?:s|es)?"


def _russian_forms(word: str) -> str:
    """Русское слово по основе: «свёкла» находит «Свекла, вареная».

    Чередование основы приём не ловит: «свекольный» не найдётся.

    Отрезается окончание из гласных, ё приравнивается к е (в каталоге
    пишут по-разному, сравнение в ``matches_words`` тоже без ё). Основа
    короче трёх букв ищется целым словом, иначе «щи» нашли бы всё на «щ».

    Слово здесь ОДНО: фразу разбирает ``_term_pattern``. Раньше из фразы
    бралось только первое слово, и «сыр фета» искался как «сыр…» —
    вместе со всем «сырым».
    """
    word = word.replace("ё", "е")
    stem = _RUSSIAN_ENDING.sub("", word)
    if len(stem) < 3:
        return re.escape(word)
    return re.escape(stem) + r"\w*"


# Хвосты названий, где перечислено то, чего в блюде НЕТ.
#
# Наблюдавшийся случай: пожелание «брокколи» засчитывало «Beef and vegetables
# excluding carrots, broccoli, and dark-green leafy» — блюдо, в котором
# брокколи нет по определению.
#
# Два вида хвоста, и путать их нельзя. «excluding»/«кроме» открывает ПЕРЕЧЕНЬ
# до точки с запятой. «no»/«without»/«без» отрицает ОДНУ позицию до запятой:
# в «…; no potatoes, gravy» картошки нет, а подлива есть.
_ABSENT_LIST = re.compile(r"\b(?:excluding|excludes?|исключая|кроме)\b[^;]*")
_ABSENT_ITEM = re.compile(r"\b(?:no|without|без)\s+[^,;]*")


def matches_words(names: pd.Series, pattern: str, skip_absent: bool = False) -> pd.Series:
    """Какие названия совпали с выражением из ``to_pattern``.

    Только через эту функцию, а не через ``names.str.contains`` напрямую.
    В pandas 3 строковая колонка хранится в pyarrow, и регулярка уходит
    в RE2, где ``\\b`` понимает границу слова ТОЛЬКО для латиницы.
    «гречка» не находилась в «Гречка варёная» без единой ошибки: пожелание
    по-русски молча не срабатывало, и агент объяснял человеку, что гречки
    в справочнике нет. В ``object`` поиск идёт через ``re`` из стандартной
    библиотеки, где граница слова юникодная.

    ``skip_absent`` — не искать в хвостах «excluding …», «без …». Только
    для ПОЖЕЛАНИЙ. Исключения ищут по всему названию: хвост может спрятать
    продукт, который в блюде есть, а цена такой ошибки — аллерген в тарелке.
    """
    lowered = names.astype(object).str.lower().str.replace("ё", "е", regex=False)
    if skip_absent:
        lowered = (lowered.str.replace(_ABSENT_LIST, "", regex=True)
                          .str.replace(_ABSENT_ITEM, "", regex=True))
    return lowered.str.contains(pattern, regex=True, na=False)


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
