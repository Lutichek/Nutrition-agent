"""
Тесты пищевых ограничений.

Оба дефекта, ради которых написан модуль, были тихими: рацион собирался,
проверки проходили, и только в тарелке оказывалось не то. Поэтому проверки
здесь идут не по коду, а по наблюдавшимся случаям.
"""

from __future__ import annotations

import re

import pytest

from restrictions import (
    CATEGORIES,
    describe,
    expand,
    to_category_pattern,
    to_pattern,
)
from solver import _filter_catalog


def matches(term: str, name: str) -> bool:
    return bool(re.search(to_pattern([term]), name.lower()))


class TestCategories:
    def test_category_expands_to_many_words(self):
        """«Сладости» — это не одно слово, а полсотни конкретных блюд."""
        assert len(expand(["sweets"])) > 30

    def test_plain_term_passes_through(self):
        assert expand(["mushroom"]) == ["mushroom"]

    def test_unknown_term_is_kept_not_dropped(self):
        """Незнакомое ограничение применяем буквально, а не выбрасываем."""
        assert expand(["unobtainium"]) == ["unobtainium"]

    def test_aliases_reach_the_same_category(self):
        assert expand(["dairy"]) == expand(["lactose"])

    def test_vegan_covers_vegetarian(self):
        assert set(expand(["vegetarian"])) <= set(expand(["vegan"]))

    def test_no_category_is_empty(self):
        assert all(words for words in CATEGORIES.values())


class TestWordBoundaries:
    """Наивная подстрока давала тихие ложные срабатывания.

    «oil» находился внутри «broiled» и выбрасывал 82 блюда, включая обычную
    запечённую курицу; «ham» — внутри «graham cracker». Человек этого не видел:
    рацион просто становился беднее без объяснения.
    """

    @pytest.mark.parametrize("term,name", [
        ("oil", "Chicken breast, baked or broiled"),
        ("ham", "Cookie, graham cracker"),
        ("egg", "Eggplant, cooked"),
        ("rice", "Licorice candy"),
    ])
    def test_does_not_match_inside_another_word(self, term, name):
        assert not matches(term, name)

    @pytest.mark.parametrize("term,name", [
        ("oil", "Vegetable oil"),
        ("ham", "Ham, sliced"),
        ("egg", "Egg, fried"),
        ("pork", "Pork chop, fried"),
    ])
    def test_matches_the_whole_word(self, term, name):
        assert matches(term, name)

    def test_matches_english_plural(self):
        assert matches("doughnut", "Doughnuts, glazed")

    def test_special_characters_do_not_break_the_pattern(self):
        """Слова приходят от модели, а через неё — от человека."""
        assert to_pattern(["a(b"]) is not None
        assert not matches("a(b", "Chicken, roasted")


class TestAgainstTheCatalog:
    """Наблюдавшийся случай целиком.

    Запрос: «не ем свинину, сладости, мучное и аллергия на лактозу».
    В плане оказались «Пончик, бисквитный, простой» и «Пончик с сахаром»:
    модель вернула ["pork","milk","cheese","yogurt","cream"], а обе категории
    потеряла молча — и ничто в списке слову «doughnut» не соответствовало.
    """

    EXCLUDE = ["pork", "sweets", "flour", "lactose"]

    def test_doughnuts_are_gone(self, catalog):
        left = _filter_catalog(catalog, self.EXCLUDE)
        names = left["name"].str.lower()
        assert not names.str.contains("doughnut|donut", regex=True, na=False).any()

    def test_sweets_and_pastry_are_gone(self, catalog):
        left = _filter_catalog(catalog, self.EXCLUDE)
        names = left["name"].str.lower()
        for word in ("cookie", "cake", "candy", "bread", "muffin", "breaded"):
            pattern = to_pattern([word])
            assert not names.str.contains(pattern, regex=True, na=False).any(), word

    def test_breadfruit_is_not_bread(self, catalog):
        """Совпадение по целому слову: «breadfruit» — это фрукт, а не хлеб."""
        assert matches("bread", "White bread")
        assert not matches("bread", "Breadfruit, cooked")

    def test_enough_dishes_remain_to_build_a_day(self, catalog):
        """Ограничения не должны выкашивать каталог до непригодности."""
        left = _filter_catalog(catalog, self.EXCLUDE)
        assert len(left) > 1000
        assert set(left["role"]) >= {"main", "side", "snack", "breakfast"}

    def test_baked_chicken_survives_an_oil_restriction(self, catalog):
        """Раньше «не ем масло» уносило и запечённую курицу тоже."""
        left = _filter_catalog(catalog, ["oil"])
        assert left["name"].str.lower().str.contains("broiled", na=False).any()


class TestPluralExclusionsDoNotLeak:
    """Исключение во множественном числе отсекает и единственное.

    Наблюдавшийся случай: «не ем арахис» → exclude ["peanuts"], а шаблон
    умел только ДОПИСАТЬ окончание. «Candy, peanut brittle» и ещё 40 блюд
    с арахисом оставались в подборе — аллерген в тарелке. Так же протекали
    «tomatoes» (163 блюда), «potatoes» (200), «cookies» (95).
    """

    @pytest.mark.parametrize("term, dish", [
        ("peanuts", "Candy, peanut brittle"),
        ("cookies", "Cookie, NFS"),
        ("strawberries", "Pie, strawberry"),
    ])
    def test_singular_dish_is_excluded(self, catalog, term, dish):
        left = _filter_catalog(catalog, [term])
        assert dish in set(catalog["name"])
        assert dish not in set(left["name"])

    def test_absent_tail_still_counts_for_exclusion(self, catalog):
        """Хвост «excluding …» пропускают только ПОЖЕЛАНИЯ.

        В «…; no potatoes, gravy» подлива в блюде есть, и исключение
        обязано видеть всё название: цена промаха — аллерген.
        """
        left = _filter_catalog(catalog, ["gravy"])
        assert not left["name"].str.contains("no potatoes, gravy", regex=False).any()


class TestMultiWordNamesInAnyOrder:
    """Название из нескольких слов совпадает в любом порядке слов.

    Порядок в USDA обратный разговорному: «Cheese, Feta», «Yogurt, Greek».
    Фразой целиком исключение «feta cheese» не отсекало ни одного блюда —
    аллерген оставался в подборе.
    """

    @pytest.mark.parametrize("term, dish", [
        ("feta cheese", "Cheese, Feta"),
        ("greek yogurt", "Yogurt, Greek, NS as to type of milk or flavor"),
        ("brown rice", "Rice, brown and wild, cooked, NS as to fat"),
    ])
    def test_reversed_order_is_excluded(self, catalog, term, dish):
        assert dish in set(catalog["name"])
        assert dish not in set(_filter_catalog(catalog, [term])["name"])

    def test_words_must_all_be_present(self):
        """Одного слова мало: «cream cheese» — не любой «cheese»."""
        assert matches("cream cheese", "Cheese, cream, low fat")
        assert not matches("cream cheese", "Cheese, cheddar")


class TestFilteringByCatalogSection:
    """Раздел справочника знает то, чего не знает название.

    В «Gyro sandwich» нет слова «bread», в «Italian Ice» — ни «candy»,
    ни «ice cream». По названию через фильтр «мучное» проходил 211 сэндвич,
    и оба этих блюда попали в реальный план человека, который просил
    без мучного и без сладкого.
    """

    @pytest.mark.parametrize("dish", ["gyro", "sandwich", "burger", "pizza"])
    def test_flour_restriction_removes_bread_shaped_dishes(self, catalog, dish):
        left = _filter_catalog(catalog, ["flour"])
        # Овощ «для сэндвича» — не сэндвич, поэтому смотрим на раздел.
        assert not left["category"].str.lower().str.contains(dish, na=False).any()

    def test_sweets_restriction_removes_italian_ice(self, catalog):
        left = _filter_catalog(catalog, ["sweets"])
        assert not left["name"].str.lower().str.contains("italian ice", na=False).any()

    def test_both_mechanisms_are_needed(self, catalog):
        """Раздел знает тип блюда, название — состав. Порознь дырявы оба."""
        # Свинина прячется внутри раздела «Meat mixed dishes» — её видно
        # только по названию.
        by_name = _filter_catalog(catalog, ["pork"])
        assert not by_name["name"].str.lower().str.contains(r"\bpork\b", regex=True, na=False).any()

        # А лепёшка в гиросе — только по разделу.
        assert to_category_pattern(["flour"]) != ""

    def test_unknown_term_has_no_category_rule(self):
        assert to_category_pattern(["quinoa"]) == ""


class TestPorkIsCaughtUnderOtherNames:
    """Свинина в USDA чаще названа без слова «pork».

    Наблюдалось в журнале чата: профиль ``exclude=["pork"]``, в плане —
    «Hog maws» (свиные желудки). Замер по каталогу: мимо фильтра проходило
    91 блюдо со свиным словом в названии и ещё больше — со свининой только
    в рецептуре. Человеку, который не ест свинину по вере, это хуже, чем
    любая неточность по калориям.
    """

    @pytest.mark.parametrize("dish", [
        "Hog maws", "Ham", "Bacon, NS as to type of meat, cooked", "Ham, prosciutto",
        "Pepperoni, NFS", "Salami, NFS", "Chorizo", "Bratwurst",
    ])
    def test_pork_by_name(self, catalog, dish):
        assert dish in set(catalog["name"]), "блюдо пропало из каталога — тест устарел"
        assert dish not in set(_filter_catalog(catalog, ["pork"])["name"])

    @pytest.mark.parametrize("dish", [
        # Ни слова о свинине в названии — только в рецептуре.
        "Club sandwich on white",        # ветчина и бекон
        "Egg, Benedict",                 # канадский бекон
        "Spanish rice, fat added",       # бекон
        "Bao bun",                       # свинина и сало
        "Soup, split pea, with meat",    # свиная лопатка
        # «Breakfast meat as ingredient in omelet» — вложенная рецептура:
        # бекон, ветчина, колбаски. Наблюдалось на живом плане без свинины.
        "Egg omelet or scrambled egg, with meat, made with margarine",
        # Начинка — продукт SR без рецептуры, ловится словом.
        "Pizza with meat and vegetables, from frozen, thin crust",
    ])
    def test_pork_by_composition(self, catalog, dish):
        assert dish in set(catalog["name"]), "блюдо пропало из каталога — тест устарел"
        assert dish not in set(_filter_catalog(catalog, ["pork"])["name"])

    @pytest.mark.parametrize("dish", [
        "Roll, white, hot dog bun",
        # «Hot dog, NFS» по рецептуре — «Frankfurter, beef». Угаданное
        # «сосиска неизвестного мяса = свинина» уносило его зря.
        "Frankfurters or hot dogs and sauerkraut",
        "Beef sausage",
    ])
    def test_no_pork_stays(self, catalog, dish):
        """Беднее выбор — это тоже цена: без свинины блюдо остаётся."""
        assert dish in set(_filter_catalog(catalog, ["pork"])["name"])

    def test_vegetarian_also_excludes_pork_forms(self):
        assert {"lard", "prosciutto", "hog"} <= set(expand(["vegetarian"]))


class TestExclusionByComposition:
    """Проверка по составу — для ВСЕХ исключений, не только мяса."""

    def test_milk_inside_mashed_potatoes(self, catalog):
        """Пюре с молоком: в названии молока нет, в рецептуре — есть."""
        left = set(_filter_catalog(catalog, ["lactose"])["fdc_id"])
        assert 2709502 in set(catalog["fdc_id"])
        assert 2709502 not in left

    def test_absent_ingredient_is_not_a_match(self):
        """«flakes without milk» — честное отсутствие: у ингредиента нет
        хвоста, под которым что-то спрятано, в отличие от названия блюда.

        2709503 «Potato, mashed, from dry mix»: молоко в составе упомянуто
        только как «without milk» (сливки там есть, но это не «milk»).
        """
        from solver import _dishes_with_ingredient

        assert 2709503 not in _dishes_with_ingredient(("milk",))
        # А сливки находятся — проверка по составу работает и тут.
        assert 2709503 in _dishes_with_ingredient(("cream",))

    def test_trace_amount_is_found(self, catalog):
        """Сухое молоко в панировке индейки — 0.2% массы.

        Показываемый состав отбрасывает доли меньше 1%: цене они не важны.
        Аллергику важны: у 42 блюд молочное есть ТОЛЬКО в таких долях.
        """
        assert 2706115 in set(catalog["fdc_id"])
        assert 2706115 not in set(_filter_catalog(catalog, ["lactose"])["fdc_id"])

    def test_cooking_fat_choice_is_not_butter(self, catalog):
        """«Fat added» — жир на выбор готовящего, а не сливочное масло.

        «Table fat, NFS» у USDA — смесь сливочного масла и маргарина.
        Раскрыв её, у человека без лактозы теряли 300+ блюд, которые
        он и так приготовит на растительном масле.
        """
        dish = "Potato, roasted, from fresh, peel eaten, fat added, NS as to fat type"
        assert dish in set(catalog["name"]), "блюдо пропало из каталога — тест устарел"
        assert dish in set(_filter_catalog(catalog, ["lactose"])["name"])


class TestTypeCategoriesAreNotCheckedByComposition:
    """«Не ем мучное» — это «не ем хлеб и выпечку», а не «ни ложки муки».

    Проверка по составу у «мучного» уносила тушёную курицу по-мексикански
    из-за загустителя в соусе, а профиль с четырьмя ограничениями на 40 сидах
    проваливал 2 дня с отклонением калорий до 19%.
    """

    def test_flour_thickened_stew_stays(self, catalog):
        assert "Stewed chicken with tomato-based sauce, Mexican style" in set(
            _filter_catalog(catalog, ["flour"])["name"])

    def test_bread_still_goes(self, catalog):
        left = _filter_catalog(catalog, ["flour"])
        assert not left["name"].str.lower().str.contains(r"\bbread\b", regex=True).any()

    def test_gluten_is_an_ingredient(self, catalog):
        """Глютен — ингредиент: та же курица с мукой уходит."""
        assert "Stewed chicken with tomato-based sauce, Mexican style" not in set(
            _filter_catalog(catalog, ["gluten"])["name"])


class TestPlantMilkIsNotDairy:
    def test_coconut_milk_is_not_lactose(self):
        """Хаупия — кокосовый пудинг: «milk» в составе только в «coconut milk»."""
        from solver import _dishes_with_ingredient

        assert 2710316 not in _dishes_with_ingredient(("lactose",))

    def test_dairy_beside_plant_milk_is_still_found(self):
        """Вырезается словосочетание, а не название целиком: в «Granola bar,
        milk chocolate coated, peanut butter» молочный шоколад есть."""
        import pandas as pd

        from restrictions import NOT_DAIRY, matches_words

        name = pd.Series(["Granola bar, soft, milk chocolate coated, peanut butter"],
                         dtype=object)
        cleaned = name.str.replace(NOT_DAIRY, " ", case=False, regex=True)
        assert matches_words(cleaned, to_pattern(["lactose"]), skip_absent=True).iloc[0]

    def test_rice_pudding_with_milk_is_dairy(self):
        """Слова только соседние: «rice … milk» через запятую — это молоко."""
        import pandas as pd

        from restrictions import NOT_DAIRY

        assert not pd.Series(["Rice pudding, made with milk"], dtype=object).str.contains(
            NOT_DAIRY, case=False, regex=True).iloc[0]


class TestLeavesAreBuiltFromRawRecipes:
    """Сборка полного состава на крошечном сырье — без сети и без выгрузки."""

    @pytest.fixture
    def raw(self, tmp_path):
        (tmp_path / "input_food.csv").write_text(
            "fdc_id,sr_description\n"
            "1,Egg\n1,Breakfast meat\n1,Oil or table fat\n"   # омлет
            "2,Bacon\n2,Turkey sausage\n"                      # «мясо к завтраку»
            "3,Butter\n3,Margarine\n"                          # жир на выбор
            "4,Loop\n5,Loop back\n",                           # цикл 4 → 5 → 4
            encoding="utf-8",
        )
        (tmp_path / "food.csv").write_text(
            "fdc_id,description\n1,Omelet\n2,Breakfast meat\n3,Oil or table fat\n"
            "4,Loop back\n5,Loop\n",
            encoding="utf-8",
        )
        return tmp_path

    def _leaves(self, raw, monkeypatch, fat_choices=frozenset()):
        import ingredients

        monkeypatch.setattr(ingredients, "COOKING_FAT_CHOICES", fat_choices)
        frame = ingredients.build_leaves(raw / "input_food.csv", raw / "food.csv")
        return {k: set(g["name_en"]) for k, g in frame.groupby("fdc_id")}

    def test_nested_recipe_is_expanded(self, raw, monkeypatch):
        omelet = self._leaves(raw, monkeypatch)[1]
        assert {"Breakfast meat", "Bacon", "Turkey sausage"} <= omelet

    def test_fat_choice_is_not_expanded(self, raw, monkeypatch):
        omelet = self._leaves(raw, monkeypatch, frozenset({"Oil or table fat"}))[1]
        assert "Oil or table fat" in omelet
        assert "Butter" not in omelet

    def test_cycle_terminates(self, raw, monkeypatch):
        loop = self._leaves(raw, monkeypatch)[4]
        assert loop == {"Loop", "Loop back"}


class TestShippedLeavesMatchComposition:
    """Полный состав в репозитории не разошёлся с показываемым.

    Собирать его заново тесту нечем — сырья в репозитории нет. Но он
    обязан быть НАДМНОЖЕСТВОМ показываемого: иначе один из двух
    артефактов собран по другой версии каталога.
    """

    def test_leaves_cover_shown_composition(self):
        import pandas as pd

        from ingredients import COMPOSITION_PATH, LEAVES_PATH

        if not LEAVES_PATH.exists():
            pytest.fail(f"нет {LEAVES_PATH.name}: python ingredients.py --rebuild-leaves")
        leaves = pd.read_parquet(LEAVES_PATH)
        shown = pd.read_parquet(COMPOSITION_PATH, columns=["fdc_id", "name_en"])
        pairs = set(zip(leaves["fdc_id"], leaves["name_en"], strict=True))
        missing = [p for p in zip(shown["fdc_id"], shown["name_en"], strict=True)
                   if p not in pairs]
        assert not missing, f"расхождение: {missing[:3]}"


class TestDescription:
    def test_categories_are_shown_in_russian(self):
        assert describe(["pork", "sweets", "flour", "lactose"]) == \
            "свинина, сладости, мучное, лактоза"

    def test_unknown_term_shown_as_is(self):
        assert describe(["quinoa"]) == "quinoa"

    def test_empty_is_empty(self):
        assert describe(None) == "" and describe([]) == ""


class TestGlutenHiddenSources:
    """Аудит глютена: 108 блюд с пшеницей или ячменём проходили фильтр.

    Слова «wheat» в их составе нет: подлива из банки, грибной суп-
    концентрат, ресторанный жареный рис и ло мейн на соевом соусе,
    панировка, солод, гренки. В планах с exclude=["gluten"] такое блюдо
    стояло в 26–30% дней.
    """

    @pytest.mark.parametrize("dish", [
        "Rice, white, with gravy, NS as to fat",
        "Beef with mushroom sauce",
        "Lo mein, meatless",
        "Rice, fried, with pork",
        "Chicken kiev",
        "Caesar salad, with romaine, no dressing",
        "Milk, malted",
        "Beer, nonalcoholic",
        "Light ice cream cone, vanilla, prepackaged",
    ])
    def test_hidden_gluten_is_excluded(self, catalog, dish):
        assert dish in set(catalog["name"]), "блюдо пропало из каталога — тест устарел"
        assert dish not in set(_filter_catalog(catalog, ["gluten"])["name"])


class TestGlutenFreeStays:
    """Обратная сторона: фильтр глютена уносил безглютеновое.

    Подстрока «noodles» в разделе «Pasta, noodles, cooked grains» — единственную
    гречку, пшено и киноа; лист «Rice noodles» — пад-тай; «tortilla» —
    кукурузные чипсы; пометку «gluten free» не видел никто.
    """

    @pytest.mark.parametrize("dish", [
        "Buckwheat groats", "Millet", "Quinoa, no added fat", "Rice noodles, cooked",
        "Pad Thai with chicken", "Corn chips, plain", "Bread, gluten free",
    ])
    def test_gluten_free_dish_stays(self, catalog, dish):
        assert dish in set(_filter_catalog(catalog, ["gluten"])["name"])

    def test_gluten_free_bread_is_still_flour(self, catalog):
        """«Не ем мучное» — хлеб остаётся хлебом и без глютена."""
        assert "Bread, gluten free" not in set(_filter_catalog(catalog, ["flour"])["name"])


class TestRestrictionSynonyms:
    """Незнакомое слово ищется буквально — и это опасно в обе стороны.

    «gluten-free» уносил именно безглютеновое и оставлял всё остальное;
    «celiac», «halal», «dairy-free», «nut allergy» не отсекали ничего.
    """

    @pytest.mark.parametrize(("synonym", "category"), [
        ("celiac", "gluten"), ("gluten-free", "gluten"), ("no gluten", "gluten"),
        ("dairy-free", "lactose"), ("nut allergy", "nuts"),
    ])
    def test_synonym_equals_category(self, catalog, synonym, category):
        assert set(_filter_catalog(catalog, [synonym])["fdc_id"]) == \
            set(_filter_catalog(catalog, [category])["fdc_id"])

    def test_halal_excludes_pork_and_alcohol(self, catalog):
        left = set(_filter_catalog(catalog, ["halal"])["name"])
        assert "Bacon, NS as to type of meat, cooked" not in left
        assert "Beer, nonalcoholic" not in left
        # «Apple cider» в USDA — сок, а не алкоголь.
        assert "Apple cider" in left

    def test_synonym_to_single_product(self):
        """Синоним может вести на продукт, а не только на категорию."""
        assert expand(["peanut allergy"]) == ["peanut"]


class TestNutButterIsNotDairyOnlyByItself:
    """Ореховая паста — не молочное, только когда блюдо ею и является.

    Неякорное исключение вырезало «peanut butter» из «Cookie, peanut butter»
    и «Chocolate candy, peanut butter filled»: печенье на сливочном масле
    и молочный шоколад возвращались человеку без лактозы.
    """

    def test_peanut_butter_cookie_is_dairy(self, catalog):
        assert "Cookie, peanut butter" not in set(_filter_catalog(catalog, ["lactose"])["name"])

    def test_plant_milk_is_not_milk(self, catalog):
        assert "Almond milk, unsweetened" in set(_filter_catalog(catalog, ["milk"])["name"])


class TestCautions:
    """Спорное не исключается, а называется вслух (решение владельца)."""

    def _item(self, catalog, name):
        from solver import _make_item

        record = catalog[catalog["name"] == name].iloc[0].to_dict()
        return _make_item("breakfast", record, 1.0)

    def test_oats_named_for_gluten(self, catalog):
        from solver import plan_cautions

        oatmeal = self._item(catalog, "Oatmeal, regular or quick, made with water, no added fat")
        found = plan_cautions([oatmeal], ["celiac"])
        assert [label for label, _, _ in found] == ["овёс"]
        assert "без глютена" in found[0][1]

    def test_gelatin_named_for_pork(self, catalog):
        from solver import plan_cautions

        dessert = self._item(catalog, "Gelatin dessert")
        assert [label for label, _, _ in plan_cautions([dessert], ["pork"])] == ["желатин"]
        assert plan_cautions([dessert], ["gluten"]) == []

    def test_oats_are_not_excluded(self, catalog):
        """Овёс остаётся в подборе: предупреждение, а не запрет."""
        assert "Oatmeal, regular or quick, made with water, no added fat" in set(
            _filter_catalog(catalog, ["gluten"])["name"])
