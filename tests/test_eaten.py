"""
«Съел не по плану — пересчитай день» (этап 2).

Съеденное фиксируется, остальные приёмы пищи собираются под остаток
нормы. Проверяется то, что легко сломать молча: граммы и калории
съеденного — из каталога, а не от модели (она их только выписывает,
и то лишь названные человеком); день проверяется ЦЕЛИКОМ; при нехватке
остатка — честный отказ с числами, а не крошечные порции; съеденное
копится в пределах суток и не тянется во вчерашний день.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLMClient, FakeRetriever

from agent import NutritionAgent
from solver import DAY_TEMPLATE, PlanNotFeasible, check_plan, eaten_item, rebuild_rest
from targets import Profile, compute_targets

PROFILE = {"sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74,
           "activity": "light", "goal": "lose"}


@pytest.fixture(scope="module")
def targets():
    return compute_targets(Profile(**PROFILE))


def first_id(catalog, text: str) -> int:
    return int(catalog[catalog["name"].str.contains(text, regex=False)].iloc[0]["fdc_id"])


class TestSolver:
    def test_numbers_come_from_catalog(self, catalog):
        fdc_id = first_id(catalog, "Pizza, cheese")
        row = catalog[catalog["fdc_id"] == fdc_id].iloc[0]
        item = eaten_item(catalog, fdc_id, 300, "Обед — основное")
        assert item.grams == pytest.approx(300)
        assert item.kcal == pytest.approx(row["portion_kcal"] * 300 / row["portion_g"])
        assert item.slot == "Обед — основное (съедено)"

    def test_rest_is_built_around_what_was_eaten(self, catalog, targets):
        pizza = eaten_item(catalog, first_id(catalog, "Pizza, cheese"), 300, "Обед — основное")
        plan = rebuild_rest(catalog, targets, [pizza], {1, 2}, seed=0)

        assert plan.items[1] == pizza, "съеденное — на месте обеда"
        slots = [it.slot for it in plan.items]
        assert "Обед — гарнир" not in slots, "весь обед закрыт съеденным"
        assert len(plan.items) == len(DAY_TEMPLATE) - 1
        assert pizza.fdc_id not in [it.fdc_id for j, it in enumerate(plan.items) if j != 1]
        assert check_plan(plan)["kcal_in_range"], "остаток добирает норму дня"
        assert plan.targets == targets.as_dict(), "проверки — по всему дню"

    def test_deterministic(self, catalog, targets):
        pizza = eaten_item(catalog, first_id(catalog, "Pizza, cheese"), 300, "Обед — основное")
        a = rebuild_rest(catalog, targets, [pizza], {1, 2}, seed=4)
        b = rebuild_rest(catalog, targets, [pizza], {1, 2}, seed=4)
        assert [it.fdc_id for it in a.items] == [it.fdc_id for it in b.items]

    def test_rest_counts_the_salt_already_eaten(self, catalog, targets):
        """Остаток собирался под весь суточный предел натрия, как будто
        съеденное соли не содержит: после 200 г блинов (~1000 мг натрия)
        день проваливал проверку соли в 18 случаях из 18."""
        pancakes = eaten_item(catalog, first_id(catalog, "Pancakes, plain"), 200, "Завтрак")
        for seed in range(3):
            plan = rebuild_rest(catalog, targets, [pancakes], {0}, seed=seed)
            assert check_plan(plan)["sodium_within_limit"], f"seed {seed}"

    def test_kept_meals_stay_as_they_were(self, catalog, targets):
        """Завтрак, оставленный из прежнего плана, — тот же блюдом и граммами,
        и его калории вычтены из остатка, а не собраны ещё раз."""
        before = rebuild_rest(catalog, targets, [], set(), seed=1)
        pizza = eaten_item(catalog, first_id(catalog, "Pizza, cheese"), 300, "Обед — основное")
        plan = rebuild_rest(catalog, targets, [pizza], {1, 2}, seed=0,
                            kept={0: before.items[0]})
        assert plan.items[0] == before.items[0]
        assert before.items[0].fdc_id not in [it.fdc_id for it in plan.items[2:]]
        assert check_plan(plan)["kcal_in_range"]

    def test_rest_budget_counts_what_was_eaten(self, catalog):
        """Живой прогон: остаток собран под все 400 ₽, хотя завтрак и
        съеденное уже стоили 188, — день вышел в 653 ₽. Остатку достаётся
        бюджет за вычетом съеденного; норма при этом важнее бюджета."""
        profile = Profile(sex="male", age=35, height_cm=180, weight_kg=80,
                          activity="moderate", goal="maintain", budget_rub_per_day=400)
        day_targets = compute_targets(profile)
        before = rebuild_rest(catalog, day_targets, [], set(), seed=4, budget=400)
        pizza = eaten_item(catalog, first_id(catalog, "Pizza, cheese, stuffed crust"), 296,
                           "Обед — основное")
        plan = rebuild_rest(catalog, day_targets, [pizza], {1, 2}, seed=4, budget=400,
                            kept={0: before.items[0]})
        assert plan.within_budget, f"{plan.cost:.0f} ₽ из 400"
        assert all(check_plan(plan).values())

    def test_overspend_after_eaten_does_not_blame_restrictions(self, catalog, targets):
        """Живой прогон: после «съел пиццу» агент советовал «убрать
        ограничения» человеку, у которого их не было."""
        pizza = eaten_item(catalog, first_id(catalog, "Pizza, cheese"), 300, "Обед — основное")
        plan = rebuild_rest(catalog, targets, [pizza], {1, 2}, seed=0, budget=30)
        assert plan.cost is not None and not plan.within_budget
        note = NutritionAgent._budget_note(plan, 1, after_eaten=True)
        assert "ограничения" not in note and "съеденное" in note
        assert "убрать ограничения" in NutritionAgent._budget_note(plan, 1)

    def test_overspend_after_eaten_is_not_multiplied_by_the_week(self, catalog):
        """Живой прогон: после «съел» на меню недели агент писал «за 7 дней
        выйдет около 2910 ₽», умножая на неделю один пересобранный день."""
        agent = make_agent(catalog, [{**PIZZA, "grams": 150}], [1])
        agent.client.script["Извлеки параметры"] = json.dumps(
            {**PROFILE, "budget_rub_per_day": 30, "days": 7})
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть. Меню на неделю, 30 ₽ в день")
        answer = agent.run("На обед съел пиццу, 150 г")
        assert "перерасход" in answer.answer
        assert "За 7 дней" not in answer.answer

    def test_too_little_left_is_said_with_numbers(self, catalog, targets):
        """Не натягивать ужин и перекусы на остаток в 50 ккал."""
        pizza = eaten_item(catalog, first_id(catalog, "Pizza, cheese"), 600, "Обед — основное")
        with pytest.raises(PlanNotFeasible, match="ккал"):
            rebuild_rest(catalog, targets, [pizza], {0, 1, 2}, seed=0)


def make_agent(catalog, eaten_items: list[dict], choices: list[int]) -> NutritionAgent:
    return NutritionAgent(
        client=FakeLLMClient(script={
            "маршрутизатор": json.dumps({"intent": "plan", "red_flag": ""}),
            "Извлеки параметры": json.dumps(PROFILE),
            "Объясни человеку": "План готов.",
            "Человек рассказывает, что съел": json.dumps({"items": eaten_items}),
            "Сопоставь съеденное": json.dumps({"choices": choices}),
        }),
        model="fake", retriever=FakeRetriever([]), catalog=catalog,
    )


PIZZA = {"phrase_ru": "пиццу", "query_en": "pizza", "grams": 300, "count": None, "meal": "lunch"}


class TestInConversation:
    def test_lunch_replaced_and_rest_rebuilt(self, catalog):
        agent = make_agent(catalog, [PIZZA], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        explained = sum("Объясни человеку" in c for c in agent.client.calls)

        answer = agent.run("На обед съел пиццу, 300 г")

        assert answer.answer.startswith("Учёл съеденное")
        eaten = [it for it in answer.plan.items if it.slot.endswith("(съедено)")]
        assert len(eaten) == 1 and eaten[0].grams == pytest.approx(300)
        assert "pizza" in eaten[0].name.lower()
        assert sum("Объясни человеку" in c for c in agent.client.calls) == explained, \
            "ответ — шаблон кода, числа не пересказывает модель"

    def test_grams_not_said_come_from_usda_units(self, catalog):
        """Модель граммы не оценивает: «два куска» — два веса единицы
        из таблицы порций USDA (portions.py), а не справочной порции
        каталога: та у пиццы — целая пицца."""
        from portions import eaten_grams

        two = {**PIZZA, "grams": None, "count": 2, "unit": "slice"}
        agent = make_agent(catalog, [two], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        answer = agent.run("На обед съел два куска пиццы")
        eaten = next(it for it in answer.plan.items if it.slot.endswith("(съедено)"))
        portion = catalog.loc[catalog["fdc_id"] == eaten.fdc_id, "portion_g"].iloc[0]
        assert eaten.grams == pytest.approx(eaten_grams(eaten.fdc_id, None, 2, portion, "slice"))
        assert eaten.grams < 2 * portion or portion < 150, "не две целых пиццы"

    def test_unknown_meal_asks_instead_of_guessing(self, catalog):
        agent = make_agent(catalog, [{**PIZZA, "meal": "unknown"}], [1])
        first = agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        answer = agent.run("Съел пиццу")
        assert "на какой приём пищи" in answer.answer
        assert agent._own_state.last_plan == first.plan.model_dump(mode="json")

    def test_not_found_in_catalog_is_said(self, catalog):
        agent = make_agent(catalog, [PIZZA], [0])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        assert "Не нашёл в справочнике" in agent.run("На обед съел пиццу").answer

    def test_meals_before_the_eaten_one_stay(self, catalog):
        """Живой прогон: «на обед съел пиццу» пересобрал и завтрак — йогурт
        сменился вафлей, хотя к обеду завтрак уже съеден. Приёмы пищи ДО
        съеденного остаются как в плане, и ответ это называет."""
        agent = make_agent(catalog, [PIZZA], [1])
        first = agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        answer = agent.run("На обед съел пиццу, 300 г")
        assert answer.plan.items[0] == first.plan.items[0], "завтрак тот же"
        assert "Завтрак — как в плане" in answer.answer

    def test_after_dinner_lunch_stays_too(self, catalog):
        agent = make_agent(catalog, [{**PIZZA, "meal": "dinner", "grams": 150}], [1])
        first = agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        answer = agent.run("На ужин съел пиццу, 150 г")
        assert answer.plan.items[:3] == first.plan.items[:3]
        assert "Завтрак и обед — как в плане" in answer.answer

    def test_snack_keeps_nothing(self, catalog):
        """Когда был перекус — неизвестно: до него могло не быть и завтрака."""
        agent = make_agent(catalog, [{**PIZZA, "meal": "snack", "grams": 100}], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        answer = agent.run("Перекусил пиццей, 100 г")
        assert "как в плане" not in answer.answer

    def test_kept_dish_against_a_new_restriction_is_rebuilt(self, catalog):
        """Оставить можно только то, что проходит ограничения этой реплики."""
        agent = make_agent(catalog, [PIZZA], [1])
        first = agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        breakfast = first.plan.items[0]
        profile = PROFILE | {"exclude": [breakfast.name.split(",")[0].lower()]}
        state = {"last_plan": first.plan.model_dump(mode="json")}
        kept = agent._kept_before(state, Profile(**profile), [{"meal": "lunch"}], {1, 2})
        assert 0 not in kept

    def test_eaten_accumulates_within_the_day(self, catalog):
        # 150 г, а не 300: с оставленным завтраком 300 г пиццы (1035 ккал)
        # набирают почти всю норму, и честный ответ — «добавлять нечего».
        agent = make_agent(catalog, [{**PIZZA, "grams": 150}], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.run("На обед съел пиццу, 150 г")
        agent.client.script["Человек рассказывает, что съел"] = json.dumps({"items": [
            {"phrase_ru": "банан", "query_en": "banana", "grams": 120, "count": None,
             "meal": "snack"}]})
        answer = agent.run("Ещё перекусил бананом")
        eaten = [it for it in answer.plan.items if it.slot.endswith("(съедено)")]
        assert len(eaten) == 2, "обед никуда не делся"

    def test_yesterdays_food_does_not_count_today(self, catalog):
        agent = make_agent(catalog, [PIZZA], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.run("На обед съел пиццу, 300 г")
        agent._own_state.eaten_date = "2000-01-01"
        agent.client.script["Человек рассказывает, что съел"] = json.dumps({"items": [
            {"phrase_ru": "банан", "query_en": "banana", "grams": 120, "count": None,
             "meal": "snack"}]})
        answer = agent.run("Перекусил бананом")
        eaten = [it for it in answer.plan.items if it.slot.endswith("(съедено)")]
        assert len(eaten) == 1

    def test_meal_answer_finishes_the_recount(self, catalog):
        """Живой прогон: на уточнение человек ответил «на обед», и реплика
        ушла в обычную сборку плана — пицца и кола потерялись, а новый день
        затёр прежний вместе с заменённым завтраком. Ответ на уточнение
        дописывает приём пищи к уже выписанному, модель второй раз не зовут.
        """
        agent = make_agent(catalog, [{**PIZZA, "meal": "unknown"}], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        assert "на какой приём пищи" in agent.run("Съел пиццу, 300 г").answer

        answer = agent.run("на обед")

        assert answer.answer.startswith("Учёл съеденное")
        eaten = [it for it in answer.plan.items if it.slot.endswith("(съедено)")]
        assert len(eaten) == 1 and eaten[0].slot.startswith("Обед")
        assert sum("Человек рассказывает, что съел" in c for c in agent.client.calls) == 1
        assert agent._own_state.eaten_pending is None, "уточнение закрыто"

    def test_meal_answer_wins_over_the_router(self, catalog):
        """По одному «на ужин» модель не поймёт, что речь о рационе."""
        agent = make_agent(catalog, [{**PIZZA, "meal": "unknown"}], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.run("Съел пиццу, 300 г")
        agent.client.script["маршрутизатор"] = json.dumps({"intent": "off_topic", "red_flag": ""})
        assert agent.run("Это был ужин").answer.startswith("Учёл съеденное")

    def test_pending_lives_one_turn(self, catalog):
        """Не ответил на уточнение — через реплику «обед» уже ни к чему не относится."""
        agent = make_agent(catalog, [{**PIZZA, "meal": "unknown"}], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.run("Съел пиццу, 300 г")
        assert agent._own_state.eaten_pending is not None
        agent.run("Составь план заново")
        assert agent._own_state.eaten_pending is None

    def test_eaten_food_is_not_a_restriction_or_a_wish(self, catalog):
        """Живой прогон: на «…съел пиццу и выпил колу» модель дописала
        exclude ["sugar"], а на следующей реплике — include ["pizza","cola"].
        Исключения копятся и репликой не снимаются: сахар оставался
        исключённым навсегда, а в ответе стояло «cola противоречит sugar»."""
        agent = make_agent(catalog, [PIZZA], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.client.script["Извлеки параметры"] = json.dumps(
            {**PROFILE, "exclude": ["sugar"], "include": ["pizza", "cola"]})
        agent.run("На обед съел пиццу и выпил колу")
        profile = agent._own_state.profile
        assert not profile.get("exclude") and not profile.get("include")

    def test_restriction_said_while_reporting_food_is_kept(self, catalog):
        """Обратная сторона: названное прямо — аллерген, его терять нельзя."""
        agent = make_agent(catalog, [PIZZA], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.client.script["Извлеки параметры"] = json.dumps({**PROFILE, "exclude": ["mushroom"]})
        agent.run("На обед съел пиццу, а грибы я вообще не ем")
        assert agent._own_state.profile["exclude"] == ["mushroom"]

    def test_generic_entry_goes_first(self, catalog):
        """«На обед был суп»: двенадцать конкретных супов, и модель, не зная
        какой, отвечала «ничего не подходит». Обобщённая запись FNDDS
        («Soup, NFS») — первой, если она есть в каталоге."""
        agent = make_agent(catalog, [PIZZA], [1])
        found = [{"fdc_id": -1, "name": "Soup, chicken", "name_ru": "", "portion_g": 250.0}]
        first = agent._generic_first(found, {"query_en": "soup"})[0]
        assert first["name"] == "Soup, NFS"
        assert agent._generic_first(found, {"query_en": "pizza"}) == found, "нет «Pizza, NFS»"

    def test_new_food_is_not_an_answer(self):
        """«Выпил стакан» — ответ на «сколько выпили?», «съел ещё торт» —
        новая реплика о съеденном: в ней названо что-то кроме количества."""
        from profile_rules import pending_answer
        pending = {"items": [{"meal": "unknown"}], "need_amount": [0]}
        assert pending_answer("выпил стакан", pending) == {"amounts": [{"ml": 250.0}]}
        assert pending_answer("на обед, стакан", pending) == {
            "meal": "lunch", "amounts": [{"ml": 250.0}]}
        assert pending_answer("съел ещё торт", pending) is None
        assert pending_answer("мне 35 лет", pending) is None, "не 35 мл"

    def test_without_a_plan_it_is_a_normal_turn(self, catalog):
        """«Съел» без показанного плана пересчитывать нечего."""
        agent = make_agent(catalog, [PIZZA], [1])
        agent.run("Мне 31, женщина, 168 см, 74 кг. На обед съел пиццу")
        assert not any("Человек рассказывает, что съел" in c for c in agent.client.calls)


SMALL_PIZZA = {**PIZZA, "grams": 150}  # с завтраком 300 г пиццы — почти вся норма
# Как отдаёт живая модель: «выпил колу» — count 1, unit "cup", хотя
# количество не названо.
COLA = {"phrase_ru": "колу", "query_en": "cola", "grams": None, "count": 1, "unit": "cup",
        "meal": "lunch"}
COLA_UNKNOWN_MEAL = {**COLA, "meal": "unknown"}
PIZZA_UNKNOWN_MEAL = {**SMALL_PIZZA, "meal": "unknown"}


class TestDrinkAmount:
    """Живой прогон: «выпил колу» без количества — 512 г, «1 medium drink»
    из таблицы порций USDA, почти 220 ккал наугад. У калорийного напитка
    без количества агент спрашивает, сколько выпито, а ответ разбирает код."""

    @staticmethod
    def cola_grams(answer) -> float:
        return next(it.grams for it in answer.plan.items
                    if it.slot.endswith("(съедено)") and "cola" in it.name.lower())

    def test_amount_is_asked_not_guessed(self, catalog):
        agent = make_agent(catalog, [SMALL_PIZZA, COLA], [1, 1])
        first = agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        answer = agent.run("На обед съел пиццу, 150 г, и выпил колу")
        assert "сколько выпили: колу" in answer.answer
        assert agent._own_state.last_plan == first.plan.model_dump(mode="json"), \
            "план не тронут, пока количество неизвестно"

    @pytest.mark.parametrize(("reply", "ml"), [("330 мл", 330), ("стакан", 250),
                                               ("пол-литра", 500), ("выпил 2 стакана", 500)])
    def test_answer_finishes_the_recount(self, catalog, reply, ml):
        from portions import grams_per_ml
        agent = make_agent(catalog, [SMALL_PIZZA, COLA], [1, 1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.run("На обед съел пиццу, 150 г, и выпил колу")
        answer = agent.run(reply)
        assert answer.answer.startswith("Учёл съеденное")
        cola = next(it for it in answer.plan.items if "cola" in it.name.lower())
        assert cola.grams == pytest.approx(ml * grams_per_ml(cola.fdc_id))
        assert sum("Человек рассказывает, что съел" in c for c in agent.client.calls) == 1
        assert sum("Сопоставь съеденное" in c for c in agent.client.calls) == 1, \
            "на ответ модель второй раз не зовут"

    def test_grams_are_taken_as_they_are(self, catalog):
        agent = make_agent(catalog, [SMALL_PIZZA, COLA], [1, 1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.run("На обед съел пиццу, 150 г, и выпил колу")
        assert self.cola_grams(agent.run("200 г")) == pytest.approx(200)

    def test_meal_and_amount_in_one_question(self, catalog):
        agent = make_agent(catalog, [PIZZA_UNKNOWN_MEAL, COLA_UNKNOWN_MEAL], [1, 1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        question = agent.run("Съел пиццу, 150 г, и выпил колу").answer
        assert "на какой приём пищи" in question and "сколько выпили" in question
        answer = agent.run("на обед, банку")
        assert answer.answer.startswith("Учёл съеденное")
        assert all(it.slot.startswith("Обед") for it in answer.plan.items
                   if it.slot.endswith("(съедено)"))

    def test_half_an_answer_asks_for_the_rest(self, catalog):
        agent = make_agent(catalog, [PIZZA_UNKNOWN_MEAL, COLA_UNKNOWN_MEAL], [1, 1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.run("Съел пиццу, 150 г, и выпил колу")
        again = agent.run("на обед").answer
        assert "сколько выпили" in again and "на какой приём пищи" not in again
        assert agent.run("330 мл").answer.startswith("Учёл съеденное")

    def test_dont_remember_takes_the_usda_typical_amount(self, catalog):
        from portions import unspecified_grams
        agent = make_agent(catalog, [SMALL_PIZZA, COLA], [1, 1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        agent.run("На обед съел пиццу, 150 г, и выпил колу")
        answer = agent.run("не помню")
        cola = next(it for it in answer.plan.items if "cola" in it.name.lower())
        assert cola.grams == pytest.approx(unspecified_grams(cola.fdc_id, 0))
        assert cola.grams < 512, "не «средний напиток» фастфуда"

    @pytest.mark.parametrize(("phrase", "ml"), [("стакан колы", 250), ("банка колы", 330),
                                                ("литр колы", 1000)])
    def test_amount_said_with_the_drink_is_not_asked(self, catalog, phrase, ml):
        """Количество — из слов человека: модель на «литр сока» отдавала
        count 1000, unit cup, а «стакан колы» становился «средним
        напитком» — 512 г (строки «cup» у газировки в таблице USDA нет)."""
        from portions import grams_per_ml
        said = {**COLA, "phrase_ru": phrase, "count": 1000 if "литр" in phrase else 1}
        agent = make_agent(catalog, [SMALL_PIZZA, said], [1, 1])
        agent.run("Мне 31, женщина, 168 см, 74 кг, хочу похудеть")
        answer = agent.run(f"На обед съел пиццу, 150 г, и выпил {phrase}")
        assert answer.answer.startswith("Учёл съеденное")
        assert self.cola_grams(answer) == pytest.approx(ml * grams_per_ml(2710541), rel=0.01)

    def test_food_and_light_drinks_are_not_asked(self):
        from portions import is_drink
        assert is_drink("Soft drinks", 42)
        assert not is_drink("Diet soft drinks", 1), "диетическая — 0 ккал"
        assert not is_drink("Tea", 2) and not is_drink("Tap water", 0)
        assert not is_drink("Pizza", 270), "у еды работает таблица порций USDA"
