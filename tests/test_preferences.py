"""
Тесты пожеланий по продуктам: «хочу видеть в рационе гречку и творог».

Пожелание — противоположность ограничения по смыслу, но НЕ по механике,
и вся тонкость здесь в том, чем они отличаются:

* exclude — жёсткий фильтр каталога: нарушить нельзя, там аллергены;
* include — слагаемое невязки: солвер старается, но норму не ломает.

Отсюда все проверяемые случаи. Главный — конфликт: человек может
попросить молоко и тут же сказать про непереносимость лактозы, потому
что говорит в разных репликах и не держит в голове связь. Побеждать
обязано ограничение, молча и всегда.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agent import (
    PREFERENCES_OFFER,
    NutritionAgent,
    _resolve_preferences,
    _with_russian_fallback,
)
from solver import (
    PREFERENCE_SATURATION,
    _preference_penalty,
    preferred_ids,
    build_day,
)
from targets import Profile, compute_targets


def _profile(**kwargs) -> Profile:
    base = dict(sex="male", age=30, height_cm=180, weight_kg=80)
    base.update(kwargs)
    return Profile(**base)


class TestConflictWithRestrictions:
    """Пожелание против ограничения: ограничение обязано побеждать."""

    def test_direct_conflict_drops_preference(self) -> None:
        profile = _profile(include=["pork"], exclude=["pork"])
        assert _resolve_preferences(profile) == []

    def test_conflict_through_category(self) -> None:
        """«Хочу молоко» + «аллергия на лактозу».

        Наблюдаемая формулировка человека. Слова разные — «milk»
        и «lactose», — и без раскрытия категории конфликт не виден:
        категория lactose разворачивается в milk, cheese, yogurt.
        """
        profile = _profile(include=["milk"], exclude=["lactose"])
        assert _resolve_preferences(profile) == []

    def test_partial_conflict_keeps_the_rest(self) -> None:
        """Снимается только конфликтующее пожелание, остальные живут."""
        profile = _profile(include=["milk", "buckwheat"], exclude=["lactose"])
        assert _resolve_preferences(profile) == ["buckwheat"]

    def test_substring_is_not_a_conflict(self) -> None:
        """«ham» внутри «graham cracker» — не конфликт.

        Та же ловушка, на которой уже обожглись в фильтре каталога:
        голая подстрока находит «oil» внутри «broiled». Сравнение
        обязано идти по границам слова.
        """
        profile = _profile(include=["ham"], exclude=["graham cracker"])
        assert _resolve_preferences(profile) == ["ham"]

    def test_no_restrictions_keeps_everything(self) -> None:
        profile = _profile(include=["fish", "rice"], exclude=[])
        assert _resolve_preferences(profile) == ["fish", "rice"]

    def test_empty_preferences(self) -> None:
        assert _resolve_preferences(_profile(exclude=["pork"])) == []


class TestNutrientsAreNotFoods:
    """«Добавь больше белка» — это не пожелание по продуктам.

    Наблюдавшийся случай на живой модели: «добавь больше белка» дало
    include=["protein"] в 2 прогонах из 3, хотя промпт это запрещает
    прямым текстом.

    Отказ был бы тихим и вредным: "protein" находит в каталоге
    протеиновые батончики и порошки, и человек, попросивший больше
    белка, получил бы в меню спортпит. Норму белка считает targets.py.
    """

    @pytest.mark.parametrize("word", ["protein", "fat", "carbs", "calories", "fiber"])
    def test_nutrient_is_dropped(self, word: str) -> None:
        assert _resolve_preferences(_profile(include=[word])) == []

    def test_real_food_survives_alongside_nutrient(self) -> None:
        profile = _profile(include=["protein", "chicken"])
        assert _resolve_preferences(profile) == ["chicken"]

    def test_nutrient_is_not_reported_as_conflict(self, catalog: pd.DataFrame) -> None:
        """Про отброшенный нутриент агент молчит.

        Фраза «не добавил в рацион protein — это противоречит вашим
        ограничениям» запутала бы человека: он просил не продукт,
        и никакого ограничения тут нет.
        """
        from types import SimpleNamespace

        stub = SimpleNamespace(catalog=catalog)
        profile = _profile(include=["protein"])
        assert NutritionAgent._unmatched_note(stub, profile) == ""


class TestPreferencePenalty:
    """Штраф за нехватку любимых блюд."""

    class _Item:
        def __init__(self, fdc_id: int) -> None:
            self.fdc_id = fdc_id

    def test_no_preferences_costs_nothing(self) -> None:
        """Без пожеланий слагаемое обязано быть нулём.

        Иначе планы всех, кто ничего не просил, поехали бы вместе
        с появлением фичи.
        """
        items = [self._Item(i) for i in range(5)]
        assert _preference_penalty(items, set()) == 0.0

    def test_penalty_falls_as_preferred_dishes_appear(self) -> None:
        items = [self._Item(i) for i in range(5)]
        none_present = _preference_penalty(items, {99})
        one_present = _preference_penalty(items, {1})
        assert none_present > one_present

    def test_penalty_saturates(self) -> None:
        """Выше порога насыщения штраф не падает.

        Без насыщения солверу выгодно набить любимыми блюдами весь день,
        и неделя схлопывается в три повторяющихся блюда.
        """
        items = [self._Item(i) for i in range(10)]
        enough = int(PREFERENCE_SATURATION * len(items)) + 1
        at_saturation = _preference_penalty(items, set(range(enough)))
        everything = _preference_penalty(items, set(range(10)))
        assert at_saturation == pytest.approx(everything)
        assert at_saturation == pytest.approx(0.0)

    def test_empty_day_costs_nothing(self) -> None:
        assert _preference_penalty([], {1, 2}) == 0.0


class TestPreferredIds:
    """Поиск любимых блюд в каталоге."""

    @pytest.fixture
    def catalog(self) -> pd.DataFrame:
        return pd.DataFrame({
            "fdc_id": [1, 2, 3, 4],
            "name": ["Buckwheat, cooked", "Broiled chicken", "Rice, white", "Ham, sliced"],
            "name_ru": ["Гречка варёная", "Курица запечённая", "Рис белый", "Ветчина"],
        })

    def test_matches_english_name(self, catalog: pd.DataFrame) -> None:
        assert preferred_ids(catalog, ["buckwheat"]) == {1}

    def test_matches_russian_name(self, catalog: pd.DataFrame) -> None:
        """Человек может назвать блюдо по-русски.

        У exclude хватало английского: там слова приходят от модели,
        которой велено отвечать по-английски. Пожелание же человек
        нередко пишет как есть.
        """
        assert preferred_ids(catalog, ["гречка"]) == {1}

    def test_russian_name_matches_in_shipped_catalog(self) -> None:
        """Русское пожелание находит блюда в НАСТОЯЩЕМ каталоге.

        Наблюдавшийся случай: с переходом на pandas 3 колонка ``name_ru``
        стала pyarrow-строкой, регулярка ушла в RE2, а там ``\\b`` знает
        только латиницу. «творог» находил 0 блюд из 14 без единой ошибки,
        и агент говорил человеку, что творога в справочнике нет.
        Фикстура выше ловит то же самое, только пока фикстура строит
        колонку того же типа, что и parquet.
        """
        from foods import load_catalog

        assert len(preferred_ids(load_catalog(), ["творог"])) >= 4

    def test_word_boundaries_respected(self, catalog: pd.DataFrame) -> None:
        """«oil» не должен находиться внутри «broiled»."""
        assert preferred_ids(catalog, ["oil"]) == set()

    def test_no_preferences_matches_nothing(self, catalog: pd.DataFrame) -> None:
        assert preferred_ids(catalog, None) == set()
        assert preferred_ids(catalog, []) == set()


class TestPreferenceWordForms:
    """Пожелание находит блюдо, в каком бы числе модель ни назвала продукт.

    Замер на 48 русских пожеланиях: модель отвечала «cucumbers», «plums»,
    «raspberry», «tomatoes», а USDA пишет «Cucumber salad», «Raspberries,
    raw». Шаблон умел только дописать окончание, и 9 из 132 пожеланий
    не находили ничего, хотя блюда в каталоге есть.
    """

    @pytest.mark.parametrize("term, dish", [
        ("cucumbers", "Cucumber salad made with cucumber and vinegar"),
        ("raspberry", "Raspberries, raw"),
        ("plums", "Plum, raw"),
        ("tomatoes", "Bacon, lettuce, tomato sandwich on wheat"),
        ("cookies", "Cookie, NFS"),
    ])
    def test_finds_dish_in_other_number(self, catalog: pd.DataFrame, term: str, dish: str) -> None:
        found = catalog.loc[catalog["fdc_id"].isin(preferred_ids(catalog, [term])), "name"]
        assert dish in set(found)


class TestAbsentTailIsNotAMatch:
    """«…excluding broccoli» — блюдо БЕЗ брокколи.

    Наблюдавшийся случай: пожелание «брокколи» засчитывало «Beef and
    vegetables excluding carrots, broccoli, and dark-green leafy».
    """

    def test_excluding_tail_is_skipped(self, catalog: pd.DataFrame) -> None:
        found = catalog.loc[catalog["fdc_id"].isin(preferred_ids(catalog, ["broccoli"])), "name"]
        assert not any("excluding" in name.lower() for name in found)
        assert "Beef and broccoli" in set(found)

    def test_no_negates_one_item_not_the_rest(self, catalog: pd.DataFrame) -> None:
        """В «…; no potatoes, gravy» картошки нет, а подлива есть."""
        found = catalog.loc[catalog["fdc_id"].isin(preferred_ids(catalog, ["gravy"])), "name"]
        assert any("no potatoes, gravy" in name for name in found)


class TestRussianFallback:
    """Английское слово не нашло ничего — ищем по русскому.

    Модель переводит пожелание словом, которого в USDA нет: «semolina»
    вместо «farina», «millet porridge» вместо «millet». Блюдо в каталоге
    есть, а человек слышит «не нашлось».
    """

    def test_semolina_found_through_russian(self, catalog: pd.DataFrame) -> None:
        profile = _profile(include=["semolina"], include_ru={"semolina": "манная каша"})
        terms = _with_russian_fallback(_resolve_preferences(profile), profile, catalog)
        found = catalog.loc[catalog["fdc_id"].isin(preferred_ids(catalog, terms)), "name"]
        assert any(name.startswith("Cream of wheat") for name in found)

    def test_english_wins_when_it_finds_something(self, catalog: pd.DataFrame) -> None:
        """Основа русского слова шире английского — ей не верим, если есть выбор."""
        profile = _profile(include=["buckwheat"], include_ru={"buckwheat": "гречка"})
        assert _with_russian_fallback(["buckwheat"], profile, catalog) == ["buckwheat"]

    def test_yo_is_the_same_letter(self, catalog: pd.DataFrame) -> None:
        """В каталоге «Свекла», человек пишет «свёкла»."""
        assert preferred_ids(catalog, ["свёкла"])

    def test_dropped_preference_stays_dropped(self, catalog: pd.DataFrame) -> None:
        """Снятое ограничением пожелание русская пара не воскрешает."""
        profile = _profile(include=["milk"], exclude=["lactose"],
                           include_ru={"milk": "молоко"})
        assert _with_russian_fallback(_resolve_preferences(profile), profile, catalog) == []

    def test_unmatched_note_names_it_in_russian(self, catalog: pd.DataFrame) -> None:
        """Человек просил «ряженку» — так и сказать, а не «ryazhenka»."""
        from types import SimpleNamespace

        profile = _profile(include=["ryazhenka"], include_ru={"ryazhenka": "ряженка"})
        note = NutritionAgent._unmatched_note(SimpleNamespace(catalog=catalog), profile)
        assert "ряженка" in note and "ryazhenka" not in note


class TestPreferencesOffer:
    """Предложение назвать любимые продукты — один раз и после плана."""

    def test_offered_when_nothing_asked_yet(self) -> None:
        offer = NutritionAgent._preferences_offer(_profile(), history=[])
        assert PREFERENCES_OFFER in offer

    def test_not_offered_when_preferences_already_known(self) -> None:
        profile = _profile(include=["buckwheat"])
        assert NutritionAgent._preferences_offer(profile, history=[]) == ""

    def test_not_repeated_in_the_same_conversation(self) -> None:
        """Второй план в том же разговоре предложение не повторяет.

        Иначе человек, пересчитавший план трижды, трижды получит одно
        и то же — и это выглядит поломкой, хотя формально всё работает.
        """
        history = [
            {"role": "user", "content": "составь рацион"},
            {"role": "assistant", "content": "вот план...\n\n" + PREFERENCES_OFFER},
        ]
        assert NutritionAgent._preferences_offer(_profile(), history) == ""

    def test_user_echo_does_not_count(self) -> None:
        """Совпадение ищется только в репликах агента.

        Человек может процитировать предложение в своей реплике —
        это не повод считать, что агент его уже сделал.
        """
        history = [{"role": "user", "content": PREFERENCES_OFFER}]
        assert PREFERENCES_OFFER in NutritionAgent._preferences_offer(_profile(), history)


class TestUnmatchedPreferencesAreSpokenAloud:
    """Невыполненное пожелание обязано быть названо, а не проглочено.

    Отказ здесь вероятен и не по вине человека: каталог построен на USDA
    и русские привычные продукты покрывает неровно — гречка это ОДНО
    блюдо, творог четыре, курица четыреста. Молча отдать план без гречки
    тому, кто её просил, — значит выглядеть сломанным.
    """

    @pytest.fixture
    def agent_stub(self, catalog: pd.DataFrame):
        from types import SimpleNamespace

        return SimpleNamespace(catalog=catalog)

    def test_silent_when_everything_matched(self, agent_stub) -> None:
        profile = _profile(include=["chicken"])
        assert NutritionAgent._unmatched_note(agent_stub, profile) == ""

    def test_silent_without_preferences(self, agent_stub) -> None:
        assert NutritionAgent._unmatched_note(agent_stub, _profile()) == ""

    def test_reports_dish_absent_from_catalog(self, agent_stub) -> None:
        profile = _profile(include=["zzznosuchfood"])
        note = NutritionAgent._unmatched_note(agent_stub, profile)
        assert "zzznosuchfood" in note
        assert "справочник" in note.lower()

    def test_reports_preference_blocked_by_restriction(self, agent_stub) -> None:
        """Две причины молчания названы разными словами.

        «Нет в каталоге» и «противоречит вашим ограничениям» — разные
        вещи, и человеку важно знать, какая из них: во втором случае
        он сам может снять ограничение.
        """
        profile = _profile(include=["milk"], exclude=["lactose"])
        note = NutritionAgent._unmatched_note(agent_stub, profile)
        assert "milk" in note
        assert "ограничен" in note.lower()
        assert "справочник" not in note.lower()

    def test_reports_both_reasons_separately(self, agent_stub) -> None:
        profile = _profile(include=["milk", "zzznosuchfood"], exclude=["lactose"])
        note = NutritionAgent._unmatched_note(agent_stub, profile)
        assert "ограничен" in note.lower()
        assert "справочник" in note.lower()


class TestSolverAcceptsPreferences:
    """Сквозная проверка: пожелание доходит до собранного дня."""

    def test_preferred_dish_appears_more_often(self, catalog: pd.DataFrame) -> None:
        """День с пожеланием содержит любимое блюдо заметно чаще.

        Проверяем именно ЧАСТОТУ по нескольким seed, а не наличие
        в одном дне: штраф мягкий, и гарантировать попадание в каждом
        отдельном дне он не обязан — иначе это был бы фильтр.

        Порог не символический. При замере курица без пожелания попадала
        в 1 день из 12, с пожеланием — в 11. Проверка «стало не меньше»
        прошла бы и при полностью отключённом слагаемом, поэтому требуем
        настоящего разрыва.
        """
        targets_ = compute_targets(_profile())

        def hits(include: list[str] | None) -> int:
            found = 0
            for seed in range(12):
                plan = build_day(catalog, targets_, seed=seed, include=include)
                names = " ".join(item.name for item in plan.items).lower()
                if "chicken" in names:
                    found += 1
            return found

        with_preference, without = hits(["chicken"]), hits(None)
        assert with_preference >= 8, f"пожелание почти не работает: {with_preference}/12"
        assert with_preference > without + 3, (
            f"разрыв слишком мал: {with_preference}/12 против {without}/12"
        )

    def test_plan_without_preferences_is_unchanged(self, catalog: pd.DataFrame) -> None:
        """Пустое пожелание не должно менять результат ни на грамм.

        Это страховка обратной совместимости: слагаемое обязано быть
        ровно нулём, а не «почти нулём».
        """
        targets_ = compute_targets(_profile())
        without = build_day(catalog, targets_, seed=1)
        empty = build_day(catalog, targets_, seed=1, include=[])

        assert [i.fdc_id for i in without.items] == [i.fdc_id for i in empty.items]
        assert without.loss == empty.loss
