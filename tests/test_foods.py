"""
Тесты каталога блюд.

Проверяют не «код работает», а «данные не поехали». Главная из них —
согласованность по Этуотеру: именно она ловит тихий отказ, когда join
по нутриентам разваливается и колонки молча становятся пустыми.
"""

from __future__ import annotations

import pytest

from foods import (
    EXOTIC_NAME_PATTERN,
    REQUIRED_NUTRIENTS,
    _portion_rank,
    _role_for_category,
    check_catalog,
)


class TestCatalogIntegrity:
    def test_catalog_is_not_empty(self, catalog):
        assert len(catalog) > 4000

    def test_required_nutrients_present_for_every_dish(self, catalog):
        assert catalog[REQUIRED_NUTRIENTS].notna().all().all()

    def test_atwater_consistency(self, catalog):
        """Ловушка выгрузки: nutrient_id хранит nutrient_nbr, а не FDC id.

        При неверном join нутриенты молча обнуляются. Эта проверка — сторож.
        """
        report = check_catalog(catalog)
        assert report["atwater_within_25pct"] > 0.9
        assert report["atwater_median_deviation"] < 0.1

    def test_no_negative_nutrients(self, catalog):
        assert (catalog[REQUIRED_NUTRIENTS] >= 0).all().all()

    def test_every_dish_has_a_portion(self, catalog):
        assert catalog["portion_g"].notna().all()
        assert (catalog["portion_g"] > 0).all()

    def test_portion_values_scale_from_per_100g(self, catalog):
        """portion_kcal обязан быть kcal на 100 г, умноженным на вес порции."""
        sample = catalog.head(200)
        expected = sample["kcal"] * sample["portion_g"] / 100
        assert (expected - sample["portion_kcal"]).abs().max() < 0.5


class TestRoles:
    def test_all_dishes_have_a_role(self, catalog):
        assert catalog["role"].notna().all()

    def test_roles_are_from_known_set(self, catalog):
        assert set(catalog["role"]) <= {"breakfast", "main", "side", "snack", "drink"}

    def test_every_role_has_enough_dishes_for_a_day(self, catalog):
        """В шаблоне дня роль main встречается дважды, side тоже."""
        counts = catalog["role"].value_counts()
        for role in ("breakfast", "main", "side", "snack"):
            assert counts.get(role, 0) >= 100

    @pytest.mark.parametrize(
        "category_id,expected",
        [
            (2502, "breakfast"),  # яйца и омлеты
            (1822, "breakfast"),  # греческий йогурт
            (3002, "main"),       # мясные составные блюда
            (3602, "main"),       # пицца
            (6404, "side"),       # морковь
            (6002, "snack"),      # яблоки
            (2804, "snack"),      # орехи и семена
            (7302, "drink"),      # кофе
        ],
    )
    def test_category_maps_to_expected_role(self, category_id, expected):
        assert _role_for_category(category_id) == expected

    @pytest.mark.parametrize(
        "category_id",
        [9002, 9404, 9602, 7504, 8804, 9999],
        ids=["детская каша", "смесь", "грудное молоко", "вино", "сахзам", "без категории"],
    )
    def test_non_food_categories_excluded(self, category_id):
        assert _role_for_category(category_id) is None


class TestExoticFilter:
    def test_game_and_offal_are_flagged(self, catalog):
        """Солвер ставил Opossum в обед: по КБЖУ идеально, показывать нельзя."""
        flagged = set(catalog.loc[~catalog["mainstream"], "name"])
        for name in ("Opossum", "Beaver", "Brains", "Liver, beef"):
            assert name in flagged

    def test_no_false_positives_on_lookalikes(self, catalog):
        """Якорь ^ и lookahead: kidney не должен цеплять Kidney beans.

        Смотрим на сам шаблон экзотики, а не на колонку mainstream:
        в неё теперь входит и отсев служебных записей, где «baked beans
        from restaurant» отсекается совсем по другой причине.
        """
        names = catalog["name"].str.lower()
        flagged = names[names.str.contains(EXOTIC_NAME_PATTERN, regex=True, na=False)]
        assert not flagged.str.contains("bean").any()
        assert not flagged.str.contains("cheese").any()

    def test_exotic_is_a_small_minority(self, catalog):
        names = catalog["name"].str.lower()
        exotic = names.str.contains(EXOTIC_NAME_PATTERN, regex=True, na=False).sum()
        assert 0 < exotic < 100

    def test_pattern_is_anchored(self):
        assert EXOTIC_NAME_PATTERN.startswith("^")


class TestPortionSelection:
    @pytest.mark.parametrize(
        "junk",
        ["1 cup, crumbs", "1 cubic inch", "1 surface inch", "1 slice, snack-size"],
    )
    def test_junk_portions_lose_to_normal_ones(self, junk):
        """«1 cup, crumbs» — стакан хлебных крошек, а не порция хлеба."""
        assert _portion_rank(junk) > _portion_rank("1 medium or regular slice")

    def test_cup_preferred_over_vague_options(self):
        assert _portion_rank("1 cup") < _portion_rank("1 small")

    def test_portions_are_within_edible_range(self, catalog):
        """Ни крошек, ни тазов: медиана должна быть похожа на еду."""
        assert 80 < catalog["portion_g"].median() < 300


class TestServiceRecords:
    """FNDDS — инструмент учёта съеденного, а не сборник рецептов.

    В нём есть строки, которые не блюда: «Avocado, for use on a sandwich» —
    это часть чужого блюда, а «Carrots, cooked, from restaurant» — та же
    морковь, что и обычная, только съеденная в кафе. Предлагать такое
    в меню бессмысленно.

    Отсев регуляркой, а не моделью, — намеренно: на выборке модель
    пропускала «Обёртку для сэндвича» даже тогда, когда та стояла
    в промпте прямым примером.
    """

    def test_sandwich_ingredients_are_not_offered(self, catalog):
        offered = catalog[catalog["mainstream"]]
        names = offered["name"].str.lower()
        assert not names.str.contains("for use on a sandwich", na=False).any()

    def test_restaurant_variants_are_not_offered(self, catalog):
        offered = catalog[catalog["mainstream"]]
        names = offered["name"].str.lower()
        for marker in ("from restaurant", "from fast food", "school lunch"):
            assert not names.str.contains(marker, na=False).any(), marker

    def test_ordinary_dishes_survive(self, catalog):
        """Отсев не должен задевать обычную еду."""
        offered = catalog[catalog["mainstream"]]
        names = offered["name"].str.lower()
        for dish in ("oatmeal", "chicken breast", "rice", "pasta"):
            assert names.str.contains(dish, na=False).any(), dish


class TestTranslationCompleteness:
    """Название, показываемое человеку, должно быть на русском.

    Латиница в скобках — это бренд («Крекеры (Ritz)»), и она законна.
    А вот «Омлет или scrambled eggs» — обрывок перевода: пачка не сошлась
    по размеру, сработал запасной путь, и половина названия осталась
    английской. Такие строки чинятся словарём, а не новым проходом модели.
    """

    def test_no_english_left_outside_brand_names(self, catalog):
        offered = catalog[catalog["mainstream"]]
        without_brands = offered["name_ru"].str.replace(r"\([^)]*\)", "", regex=True)
        leftovers = without_brands.str.contains(r"[A-Za-z]{3,}", regex=True, na=False)
        assert leftovers.sum() < 40, \
            f"непереведённых фрагментов: {leftovers.sum()}"

    def test_every_dish_has_a_russian_name(self, catalog):
        assert catalog["name_ru"].notna().all()
        assert (catalog["name_ru"].str.strip() != "").all()


class TestFamiliarFilter:
    """Отсев блюд, которые у нас не едят.

    Критерий отличается от доступности: горькую дыню и розовую фасоль купить
    можно, предлагать их в меню всё равно незачем. При этом отсекается
    не иностранное — мировые кухни давно свои.
    """

    def test_base_foods_never_depend_on_model_judgement(self, catalog):
        """Банан, творог и сыр не должны зависеть от настроения модели.

        В первом прогоне без этой защиты выпали «Банан, сырой»,
        «Творог, низкожирный» и 42 сыра из 57: модель, увидев список
        непривычного, начинает считать подозрительным любое уточнение.
        """
        from review_familiar import _is_base_food

        base = _is_base_food(catalog)
        assert catalog.loc[base, "familiar_ru"].fillna(True).astype(bool).all()

    def test_world_cuisines_survive(self, catalog):
        """Паста, пицца, суши и плов — это привычная у нас еда."""
        offered = catalog[
            catalog["mainstream"]
            & catalog["available_ru"].fillna(True).astype(bool)
            & catalog["familiar_ru"].fillna(True).astype(bool)
        ]
        names = offered["name_ru"].str.lower()
        for dish in ("паста", "пицц", "суши", "рис", "сыр", "творог"):
            assert names.str.contains(dish, na=False).any(), dish

    def test_enough_dishes_in_every_role(self, catalog):
        """После трёх отсевов солверу должно остаться из чего собирать день."""
        offered = catalog[
            catalog["mainstream"]
            & catalog["available_ru"].fillna(True).astype(bool)
            & catalog["familiar_ru"].fillna(True).astype(bool)
        ]
        counts = offered["role"].value_counts()
        for role in ("main", "side", "snack", "breakfast"):
            assert counts.get(role, 0) > 100, f"{role}: {counts.get(role, 0)}"


class TestMainDishesCarryProtein:
    """Основное блюдо обязано давать белок.

    Роль берётся из раздела справочника, а раздел про белок ничего не знает:
    в «Rice mixed dishes» лежат и жареный рис с курицей, и «Рис с изюмом».
    Наблюдалось: «Рис белый с подливкой» — 2 г белка на 100 г — попал
    в план ужина как основное блюдо, и норму белка добирать стало нечем.
    """

    def test_no_main_is_a_side_in_disguise(self, catalog):
        from foods import MAIN_MIN_PROTEIN_SHARE

        mains = catalog[catalog["role"] == "main"]
        share = mains["protein_g"] * 4 / mains["kcal"].clip(lower=1)
        assert (share >= MAIN_MIN_PROTEIN_SHARE).all()

    def test_demoted_dishes_are_kept_as_sides(self, catalog):
        """Блюда не выбрасываются: как гарнир рис с изюмом хорош."""
        names = catalog[catalog["role"] == "side"]["name_ru"].str.lower()
        assert names.str.contains("рис", na=False).any()

    def test_enough_mains_remain(self, catalog):
        assert (catalog["role"] == "main").sum() > 800


class TestCondimentsAreNotFood:
    """Приправами и кислыми цитрусами порции не набирают.

    Наблюдалось: солвер добирал калорийность 400 граммами сырого лайма.
    Формально это 120 ккал и попадание в норму, практически — есть невозможно.
    """

    @pytest.mark.parametrize("name", ["lime, raw", "lemon, raw", "mustard greens"])
    def test_not_offered_in_plans(self, catalog, name):
        offered = catalog[
            catalog["mainstream"]
            & catalog["available_ru"].fillna(True).astype(bool)
            & catalog["familiar_ru"].fillna(True).astype(bool)
        ]
        assert not offered["name"].str.lower().str.contains(name, na=False).any()

    def test_real_citrus_food_survives(self, catalog):
        """Отсекается лайм, а не всё цитрусовое: апельсины и соки остаются."""
        offered = catalog[catalog["mainstream"]]
        names = offered["name"].str.lower()
        assert names.str.contains("orange", na=False).any()


class TestRawAnimalProductsAreNotOffered:
    """Сырое мясо, рыба и яйца в план не идут.

    В FNDDS они законны — человек мог съесть тартар, и учесть это надо.
    Предлагать такое в рационе нельзя: это вопрос безопасности, а не вкуса.

    Наблюдалось: «Клэмы, сырые» как основное блюдо на ужин. Рядом в каталоге
    лежали «Говядина, фарш, сырая» и «Яйцо, целое, сырое».
    """

    @pytest.mark.parametrize("name", [
        "beef, ground, raw", "clams, raw", "egg, whole, raw",
        "fish, tuna, raw", "oysters, raw",
    ])
    def test_not_in_the_pool(self, catalog, name):
        offered = catalog[catalog["mainstream"]]
        assert not (offered["name"].str.lower() == name).any()

    def test_cooked_from_raw_survives(self, catalog):
        """«from raw» в середине названия значит ОБРАТНОЕ — блюдо приготовлено."""
        offered = catalog[catalog["mainstream"]]
        assert offered["name"].str.lower().str.contains("from raw", na=False).any()

    def test_raw_vegetables_are_fine(self, catalog):
        offered = catalog[catalog["mainstream"]]
        names = offered["name"].str.lower()
        assert names.str.contains("eggplant, raw", na=False).any()
        assert names.str.contains(", raw", na=False).sum() > 50


class TestTranslationReadsLikeFood:
    """Название блюда должно читаться как еда, а не как отчёт опроса.

    FNDDS помечает куриные позиции тем, съедена ли кожа: от этого зависит
    состав. Модель перевела 134 такие позиции как «с кожей» / «без кожи»,
    а 15 — как «шкура съедена». У птицы по-русски кожа, а не шкура,
    и «съедена» в названии блюда звучит как протокол, а не как ужин.

    Наблюдалось в реальном плане: «Куриное бедро, жареное на гриле
    без соуса, шкура съедена».
    """

    def test_no_hide_in_dish_names(self, catalog):
        assert not catalog["name_ru"].str.contains("шкур", case=False, na=False).any()

    def test_no_survey_phrasing_left(self, catalog):
        for phrase in ("съедена", "не съедена"):
            assert not catalog["name_ru"].str.contains(phrase, case=False, na=False).any()

    def test_skin_qualifier_is_kept_as_description(self, catalog):
        """Пометка не выброшена — она влияет на состав, только переведена."""
        thighs = catalog[catalog["name"].str.startswith("Chicken thigh", na=False)]
        names = thighs["name_ru"]
        assert names.str.contains("с кожей", na=False).any()
        assert names.str.contains("без кожи", na=False).any()

    def test_indian_flatbread_is_not_a_rotisserie(self, catalog):
        """«Роти» — лепёшка; замена rotisserie не должна была её задеть."""
        assert catalog["name_ru"].str.contains("чапатти или роти", na=False).any()

