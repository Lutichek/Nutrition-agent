"""
Тесты агента: маршрутизация графа и разговорная память.

LLM подменена заглушкой, поэтому тесты быстрые, бесплатные и повторяемые.
Проверяется поведение графа, а не качество формулировок модели — качество
текста меряется отдельно, метриками в ``run_eval.py``.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLMClient, FakeRetriever

from agent import NutritionAgent, parse_json_response


def make_agent(catalog, script=None, chunks=None, **kwargs) -> NutritionAgent:
    """Собрать агента на заглушках."""
    client = FakeLLMClient(script=script or {})
    return NutritionAgent(
        client=client,
        model="fake-model",
        retriever=FakeRetriever(chunks or []),
        catalog=catalog,
        **kwargs,
    )


def classify_as(intent: str, red_flag: str = "") -> str:
    return json.dumps({"intent": intent, "red_flag": red_flag})


FULL_PROFILE = {
    "sex": "female", "age": 31, "height_cm": 168, "weight_kg": 74,
    "activity": "sedentary", "goal": "lose",
}


class TestJsonParsing:
    def test_plain_json(self):
        assert parse_json_response('{"intent": "plan"}') == {"intent": "plan"}

    def test_markdown_wrapped_json(self):
        """GigaChat оборачивает JSON в ```json даже при response_format."""
        raw = '```json\n{"intent": "plan"}\n```'
        assert parse_json_response(raw) == {"intent": "plan"}

    def test_json_with_surrounding_prose(self):
        raw = 'Вот ответ: {"intent": "question"} — надеюсь, помог.'
        assert parse_json_response(raw) == {"intent": "question"}

    def test_garbage_returns_empty_dict(self):
        assert parse_json_response("совсем не json") == {}

    def test_none_safe(self):
        assert parse_json_response("") == {}


class TestRouting:
    def test_off_topic_answers_without_search(self, catalog):
        agent = make_agent(catalog, script={"маршрутизатор": classify_as("off_topic")})
        answer = agent.run("Какая погода в Москве?")
        assert "только с питанием" in answer.answer
        assert agent.retriever.queries == []

    def test_red_flag_refuses_plan(self, catalog):
        agent = make_agent(
            catalog, script={"маршрутизатор": classify_as("plan", "беременность")}
        )
        answer = agent.run("Я беременна, составь рацион на 1200 ккал")
        assert "беременность" in answer.answer.lower()
        assert "врач" in answer.answer.lower()

    def test_red_flag_blocks_plan_request(self, catalog):
        """Персональное предписание при диагнозе — не наша работа."""
        agent = make_agent(catalog, script={"маршрутизатор": classify_as("plan", "диабет")})
        answer = agent.run("Составь рацион, у меня диабет")
        assert "не могу составить рацион" in answer.answer

    def test_red_flag_does_not_block_literature_question(self, catalog, fake_chunks):
        """Ключевое различие: спросить про болезнь можно, просить рацион — нет.

        Пока флаг перехватывал обе ветки, агент молча отказывался отвечать
        на 11 вопросов из 80 в прогоне метрик — просто потому, что в вопросе
        упоминалось заболевание.
        """
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("question", "диабет"),
                "has_relevant": '{"has_relevant": true, "keep": [1]}',
                "ассистент по питанию. Ответь": "Исследования показывают [PMID 111].",
            },
            chunks=fake_chunks,
        )
        answer = agent.run("Что исследования говорят о питании при диабете?")
        assert "не могу составить рацион" not in answer.answer
        assert answer.retrieved_chunks  # поиск действительно состоялся

    def test_unknown_intent_falls_back_to_question(self, catalog, fake_chunks):
        """Модель может ответить чем угодно — граф не должен падать."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": '{"intent": "нечто", "red_flag": ""}',
                "has_relevant": '{"has_relevant": true, "keep": [1]}',
            },
            chunks=fake_chunks,
        )
        answer = agent.run("Что-то непонятное")
        assert answer.answer
        assert agent.retriever.queries  # ушёл в поиск, а не упал


class TestProfileCollection:
    def test_asks_for_missing_fields(self, catalog):
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": "{}",
                "не хватает данных": "Уточните, пожалуйста, пол, возраст, рост и вес.",
            },
        )
        answer = agent.run("Хочу похудеть, составь рацион")
        assert "рост" in answer.answer.lower()

    def test_builds_plan_when_profile_complete(self, catalog):
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(FULL_PROFILE),
                "Объясни человеку": "Ваша норма посчитана, вот план.",
            },
        )
        answer = agent.run("Мне 31, женщина, 168 см, 74 кг, сидячая работа, хочу похудеть")
        assert answer.answer
        assert agent.profile["weight_kg"] == 74

    def test_profile_accumulates_across_turns(self, catalog):
        """Человек редко называет всё сразу — переспрашивать дважды нельзя."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps({"sex": "female", "age": 31}),
                "не хватает данных": "Нужны рост и вес.",
            },
        )
        agent.run("Я женщина, 31 год")
        assert agent.profile == {"sex": "female", "age": 31}

        # Вторая реплика достраивает профиль.
        agent.client.script["Извлеки параметры"] = json.dumps(
            {"height_cm": 168, "weight_kg": 74}
        )
        agent.client.script["Объясни человеку"] = "План готов."
        agent.run("Рост 168, вес 74")
        assert agent.profile["age"] == 31          # не потерялось
        assert agent.profile["height_cm"] == 168   # добавилось

    def test_reset_clears_memory(self, catalog):
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(FULL_PROFILE),
                "Объясни человеку": "План готов.",
            },
        )
        agent.run("Мне 31, женщина, 168, 74")
        assert agent.profile
        agent.reset()
        assert agent.profile == {} and agent.history == []

    def test_remember_false_isolates_runs(self, catalog):
        """На прогоне метрик профиль от одного вопроса не должен течь в другой."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(FULL_PROFILE),
                "Объясни человеку": "План готов.",
            },
        )
        agent.run("Мне 31, женщина, 168, 74", remember=False)
        assert agent.profile == {}
        assert agent.history == []


class TestProfileValidationFailures:
    def test_impossible_values_produce_readable_message(self, catalog):
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps({**FULL_PROFILE, "age": 250}),
            },
        )
        answer = agent.run("Мне 250 лет")
        assert "возраст" in answer.answer.lower()

    def test_bad_field_is_dropped_not_remembered(self, catalog):
        """Иначе агент спотыкался бы о кривое значение на каждой реплике."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps({**FULL_PROFILE, "age": 250}),
            },
        )
        agent.run("Мне 250 лет")
        assert "age" not in agent.profile

    def test_contradictory_profile_does_not_produce_empty_message(self, catalog):
        """У model_validator пустой loc — раньше пользователь получал обрубок
        «Не получилось разобрать параметры: . Уточните»."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(
                    {**FULL_PROFILE, "goal": "maintain", "rate_kg_per_week": 0.5}
                ),
            },
        )
        answer = agent.run("Хочу удерживать вес и худеть на полкило в неделю")
        assert "противоречат" in answer.answer
        assert ": ." not in answer.answer


class TestInfeasiblePlan:
    def test_narrow_restrictions_answer_gracefully(self, catalog):
        """Не пятисотка, а объяснение: ограничений слишком много.

        Каталог урезан до одной роли: реальными ограничениями его больше
        не задушить, а ветку «собрать не удалось» проверить надо.
        """
        agent = make_agent(
            catalog[catalog["role"] == "main"],
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(FULL_PROFILE),
            },
        )
        answer = agent.run("Я не ем почти ничего")
        assert "не получилось собрать рацион" in answer.answer.lower()
        # Норму при этом всё равно посчитали и показали.
        assert "ккал" in answer.answer


class TestQuestionBranch:
    def test_relevant_chunks_produce_answer(self, catalog, fake_chunks):
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("question"),
                "поисковый запрос": "protein intake weight loss",
                "has_relevant": '{"has_relevant": true, "keep": [1, 2]}',
                "ассистент по питанию. Ответь": "Белка нужно 1.6-2.4 г/кг [PMID 111].",
            },
            chunks=fake_chunks,
        )
        answer = agent.run("Сколько белка при похудении?")
        assert "PMID" in answer.answer
        assert answer.retrieved_chunks and len(answer.retrieved_chunks) == 2

    def test_irrelevant_chunks_lead_to_refusal(self, catalog, fake_chunks):
        """Лучше честный отказ, чем выдуманный ответ."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("question"),
                "has_relevant": '{"has_relevant": false, "keep": []}',
            },
            chunks=fake_chunks,
            max_refines=0,
        )
        answer = agent.run("Как фаза луны влияет на магний?")
        assert "нет информации" in answer.answer

    def test_refine_retries_search_once(self, catalog, fake_chunks):
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("question"),
                "has_relevant": '{"has_relevant": false, "keep": []}',
            },
            chunks=fake_chunks,
            max_refines=1,
        )
        agent.run("Странный вопрос")
        assert len(agent.retriever.queries) == 2  # первый поиск + повтор после refine

    def _grade_prompt(self, agent) -> str:
        prompts = [call for call in agent.client.calls if "has_relevant" in call]
        assert prompts, "узел grade не вызывался"
        return prompts[0]

    def _question_agent(self, catalog, fake_chunks, **kwargs):
        return make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("question"),
                "has_relevant": '{"has_relevant": true, "keep": [1]}',
                "ассистент по питанию. Ответь": "Ответ.",
            },
            chunks=fake_chunks,
            **kwargs,
        )

    def test_strict_grade_is_the_default(self, catalog, fake_chunks):
        """Строгий критерий выбран по замерам: +7 п.п. честности отказов
        при потере полноты в пределах шума (см. experiments_grade.py)."""
        agent = self._question_agent(catalog, fake_chunks)
        agent.run("Вопрос")
        assert "притянуть фрагмент за уши" in self._grade_prompt(agent)

    def test_lenient_grade_can_be_enabled(self, catalog, fake_chunks):
        """Мягкий режим оставлен: без него нечего сравнивать в экспериментах."""
        agent = self._question_agent(catalog, fake_chunks, strict_grade=False)
        agent.run("Вопрос")
        assert "притянуть фрагмент за уши" not in self._grade_prompt(agent)

    def test_grade_filters_out_useless_chunks(self, catalog, fake_chunks):
        """Мусор в контексте портит и ответ, и faithfulness."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("question"),
                "has_relevant": '{"has_relevant": true, "keep": [2]}',
                "ассистент по питанию. Ответь": "Ответ.",
            },
            chunks=fake_chunks,
        )
        answer = agent.run("Как быстро можно худеть?")
        assert len(answer.retrieved_chunks) == 1
        assert answer.retrieved_chunks[0].chunk_id == 2

    def test_answer_contract_carries_doc_id(self, catalog, fake_chunks, corpus_doc_ids):
        """evaluation.py ждёт doc_id в поле year — так устроен контракт ответа."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("question"),
                "has_relevant": '{"has_relevant": true, "keep": [1]}',
                "ассистент по питанию. Ответь": "Ответ.",
            },
            chunks=fake_chunks,
        )
        answer = agent.run("Вопрос", dataset_row_id="row-7")
        assert answer.dataset_row_id == "row-7"
        # В поле year лежит doc_id первого фрагмента — сверяем с фикстурой,
        # а не с литералом: схема идентификаторов уже менялась однажды.
        assert answer.retrieved_chunks[0].year == corpus_doc_ids[0]


class TestResilience:
    def test_transient_provider_failure_is_retried(self, catalog):
        """У графа до семи вызовов модели: один таймаут не должен ронять ответ."""
        client = FakeLLMClient(script={"маршрутизатор": classify_as("off_topic")}, fail_times=2)
        agent = NutritionAgent(
            client=client, model="fake", retriever=FakeRetriever(), catalog=catalog,
        )
        answer = agent.run("Погода?")
        assert "только с питанием" in answer.answer

    def test_gives_up_after_max_retries(self, catalog):
        client = FakeLLMClient(fail_times=99)
        agent = NutritionAgent(
            client=client, model="fake", retriever=FakeRetriever(),
            catalog=catalog, max_retries=2,
        )
        with pytest.raises(RuntimeError, match="не ответила"):
            agent.run("Что угодно")


class TestDeterministicLayerIsolation:
    def test_numbers_come_from_code_not_from_model(self, catalog):
        """Ключевое обещание проекта: модель не участвует в расчёте чисел.

        Заглушка отвечает мусором на всё, кроме маршрутизации и извлечения
        профиля. Если бы числа брались у модели, план бы не сошёлся.
        """
        from solver import build_day, check_plan
        from targets import Profile, compute_targets

        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(FULL_PROFILE),
                "Объясни человеку": "ерунда, которую написала модель",
            },
        )
        agent.run("Мне 31, женщина, 168, 74, сидячая, худею")

        # Тот же профиль, посчитанный напрямую, обязан дать тот же результат.
        targets = compute_targets(Profile(**FULL_PROFILE))
        plan = build_day(catalog, targets, seed=0)
        assert all(check_plan(plan).values())
        assert abs(plan.deviation["kcal"]) <= 0.06


class TestMenuPeriod:
    """Меню на неделю или месяц.

    Печатать тридцать дней в переписку невозможно, поэтому в ответ идёт
    один день как образец, а всё меню отдаётся отдельным текстом
    для скачивания. Проверяем обе половины этого договора.
    """

    def test_period_from_request_reaches_solver(self, catalog):
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps({**FULL_PROFILE, "days": 7}),
                "Объясни человеку": "План готов.",
            },
        )
        answer = agent.run("Составь меню на неделю. Женщина, 31 год, 168 см, 74 кг")
        assert answer.menu_days == 7
        assert answer.menu is not None and len(answer.menu.days) == 7

    def test_days_does_not_leak_into_profile(self, catalog):
        """days — свойство запроса, а не человека: в Profile его быть не должно."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps({**FULL_PROFILE, "days": 30}),
                "Объясни человеку": "План готов.",
            },
        )
        agent.run("Меню на месяц. Женщина, 31 год, 168 см, 74 кг")
        assert "days" not in agent.profile

    def test_single_day_offers_nothing_to_download(self, catalog):
        """Один день целиком помещается в ответ — файл не нужен."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(FULL_PROFILE),
                "Объясни человеку": "План готов.",
            },
        )
        answer = agent.run("Составь рацион. Женщина, 31 год, 168 см, 74 кг")
        assert answer.menu_days == 1

    def test_absurd_period_is_ignored(self, catalog):
        """«На год» — не повод собирать 365 дней и подвесить запрос."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps({**FULL_PROFILE, "days": 365}),
                "Объясни человеку": "План готов.",
            },
        )
        answer = agent.run("Меню на год. Женщина, 31 год, 168 см, 74 кг")
        assert answer.menu_days == 1


class TestHistoryTrimming:
    """Экономия на истории не должна стоить памяти разговора.

    Историю получают classify и extract_profile — оба на каждой реплике.
    Шесть сообщений это 3280 токенов, то есть 6560 за реплику, и больше
    половины из них — собственные развёрнутые ответы агента, которые
    ни классификации, ни извлечению профиля не нужны.

    Обрезаются ТОЛЬКО ответы агента: параметры человек называет сам,
    и потерять «рост 168» из-за экономии нельзя.
    """

    def test_user_messages_are_never_trimmed(self, catalog):
        agent = make_agent(catalog, script={})
        long_message = "Мне 31 год, " + "и ещё много подробностей. " * 40
        text = agent._format_history([{"role": "user", "content": long_message}])
        assert long_message in text

    def test_agent_answers_are_trimmed(self, catalog):
        agent = make_agent(catalog, script={})
        long_answer = "Белок нужен для мышц. " * 60
        text = agent._format_history([{"role": "assistant", "content": long_answer}])
        assert len(text) < len(long_answer) / 2
        assert text.endswith("…")

    def test_short_answers_are_left_alone(self, catalog):
        agent = make_agent(catalog, script={})
        text = agent._format_history([{"role": "assistant", "content": "План готов."}])
        assert text == "assistant: План готов."

    def test_profile_still_accumulates_across_turns(self, catalog):
        """Главная проверка: обрезка не должна ломать сбор профиля."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps({"sex": "female", "age": 31}),
                "не хватает данных": "Нужны рост и вес.",
            },
        )
        agent.run("Я женщина, 31 год")

        agent.client.script["Извлеки параметры"] = json.dumps(
            {"height_cm": 168, "weight_kg": 74}
        )
        agent.client.script["Объясни человеку"] = "План готов."
        agent.run("Рост 168, вес 74")

        assert agent.profile["age"] == 31
        assert agent.profile["height_cm"] == 168


class TestClassificationIsNotDraggedByHistory:
    """Названные параметры — всегда просьба о расчёте.

    Наблюдалось: реплика «Мне 31, женщина, рост 168, вес 74, сидячая работа,
    хочу похудеть» уходила в ветку вопросов, если до неё в разговоре был любой
    вопрос о питании. В пустом чате та же реплика классифицировалась как plan.
    Человек получал справку про похудение вместо рациона — и не понимал,
    почему агент проигнорировал его данные.

    Проверка здесь на самом промпте, а не на живой модели: заглушка отвечает
    по сценарию, поэтому смотрим, что правило вообще сформулировано.
    """

    def test_prompt_states_the_rule_explicitly(self):
        from agent import CLASSIFY_PROMPT

        assert "ЛИЧНЫЕ ПАРАМЕТРЫ" in CLASSIFY_PROMPT
        assert "независимо от того, о чём шёл разговор раньше" in CLASSIFY_PROMPT

    def test_plan_branch_runs_when_profile_follows_a_question(self, catalog):
        """Сквозная проверка ветки: план собирается, а не ответ по статьям."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(FULL_PROFILE),
                "Объясни человеку": "Ваша норма посчитана, вот план.",
            },
        )
        agent.history.extend([
            {"role": "user", "content": "Сколько белка при похудении?"},
            {"role": "assistant", "content": "Белка нужно больше нормы 0.8 г/кг."},
        ])
        answer = agent.run("Мне 31, женщина, рост 168, вес 74, сидячая работа")
        assert agent.profile["weight_kg"] == 74
        assert answer.answer


class TestPeriodParsing:
    """На сколько дней просят меню — разбором фразы, а не моделью.

    Наблюдалось: на «составь план питания на месяц» модель три раза из трёх
    не возвращала days вовсе. Инструкция «не названо — не включай поле»
    перевешивала: «на месяц» модель не считала явно названным числом.
    Человек просил месяц и получал один день без всякого объяснения.
    """

    @pytest.mark.parametrize("text,expected", [
        ("составь план питания на месяц", 30),
        ("Хочу меню на неделю", 7),
        ("меню на две недели", 14),
        ("составь на 10 дней", 10),
        ("план на 3 недели", 21),
        ("рацион на день", 1),
        ("составь мне рацион", None),
        ("на год", None),
    ])
    def test_period_is_read_from_the_phrase(self, text, expected):
        from agent import _days_from_text

        assert _days_from_text(text) == expected

    def test_real_request_from_a_user(self):
        from agent import _days_from_text

        text = ("Я мужчина, мне 24 года, рост 167 см, вес 65 кг. "
                "Хочу провести рекомпозицию тела, составь план питания на месяц")
        assert _days_from_text(text) == 30

    def test_phrase_wins_over_the_model(self, catalog):
        """Модель может не вернуть days — разбор фразы это подстрахует."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(FULL_PROFILE),   # days нет
                "Объясни человеку": "План готов.",
            },
        )
        answer = agent.run("Мне 31, женщина, 168 см, 74 кг, составь меню на неделю")
        assert answer.menu_days == 7


class TestActivityIsAsked:
    """Активность не подставляется молча.

    Наблюдалось: человек писал «хочу рекомпозицию тела» без слова о спорте,
    Profile брал значение по умолчанию «sedentary», и норма выходила
    1894 ккал вместо 2447. Разница между «сидячим» и «средним» — 550 ккал
    в день, и человеку неоткуда было понять, откуда взялась низкая цифра.
    """

    PROFILE_WITHOUT_ACTIVITY = {
        "sex": "male", "age": 24, "height_cm": 167, "weight_kg": 65,
        "goal": "recomp",
    }

    def test_plan_is_not_built_without_activity(self, catalog):
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(self.PROFILE_WITHOUT_ACTIVITY),
                "не хватает данных": "Как часто вы тренируетесь?",
            },
        )
        answer = agent.run("Мне 24, мужчина, 167 см, 65 кг, хочу рекомпозицию")
        assert "тренир" in answer.answer.lower()

    def test_activity_is_among_required_fields(self):
        from agent import REQUIRED_PROFILE_FIELDS

        assert "activity" in REQUIRED_PROFILE_FIELDS

    def test_options_are_offered_in_plain_russian(self):
        """«Moderate» человеку ничего не говорит — нужны бытовые формулировки."""
        from targets import ACTIVITY_FACTORS, ACTIVITY_LABELS

        assert set(ACTIVITY_LABELS) == set(ACTIVITY_FACTORS)
        assert "2-3 раза в неделю" in ACTIVITY_LABELS["moderate"]
        assert "тренировок нет" in ACTIVITY_LABELS["sedentary"]
        assert all("—" in label for label in ACTIVITY_LABELS.values())


class TestPeriodSurvivesTheConversation:
    """Период запоминается наравне с профилем.

    Наблюдалось: «составь план на месяц» в первой реплике, во второй человек
    дописывает про тренировки — и меню собирается на ОДИН день. Период жил
    только в состоянии графа и до следующей реплики не доживал.
    """

    def test_period_from_the_first_message_is_kept(self, catalog):
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps({
                    "sex": "male", "age": 24, "height_cm": 167, "weight_kg": 65,
                    "goal": "recomp",
                }),
                "не хватает данных": "Как часто вы тренируетесь?",
            },
        )
        agent.run("Мне 24, мужчина, 167 см, 65 кг, рекомпозиция, план на месяц")
        assert agent.days == 30

        # Вторая реплика про период молчит — он должен уцелеть.
        agent.client.script["Извлеки параметры"] = json.dumps({"activity": "moderate"})
        agent.client.script["Объясни человеку"] = "План готов."
        answer = agent.run("Тренируюсь 3 раза в неделю")

        assert agent.profile["activity"] == "moderate"
        assert answer.menu_days == 30

    def test_new_period_replaces_the_old_one(self, catalog):
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps(FULL_PROFILE),
                "Объясни человеку": "План готов.",
            },
        )
        agent.run("Мне 31, женщина, 168 см, 74 кг, сидячая работа, меню на месяц")
        answer = agent.run("А сделай на неделю")
        assert answer.menu_days == 7


class TestRecompositionIsRecognised:
    """Рекомпозицию модель путает устойчиво — поэтому её разбирает код.

    Проверено на семи формулировках: «рекомпозицию тела» и «рекомпозицию»
    модель понимает, а «мягкая рекомпозиция» отдаёт как gain, «сжечь жир
    и набрать мышцы» — тоже gain, «подсушиться, но не потерять мышцы» —
    как lose. Две последние стоят в промпте прямым примером.

    Цена ошибки видна человеку: белок 1.4 г/кг вместо 2.0, то есть обычное
    поддержание вместо смены состава тела. При весе 65 кг это 92 г вместо
    130 г — разница, ради которой рекомпозиция и затевается.
    """

    @pytest.mark.parametrize("phrase", [
        "Хочу провести рекомпозицию тела",
        "мягкая рекомпозиция",
        "хочу recomp",
        "хочу сжечь жир и набрать мышцы",
        "хочу подсушиться, но не потерять мышцы",
        "хочу похудеть и сохранить мышцы",
    ])
    def test_recomposition_is_read_from_the_phrase(self, phrase):
        from agent import _goal_from_text

        assert _goal_from_text(phrase) == "recomp"

    @pytest.mark.parametrize("phrase", [
        "хочу похудеть",
        "хочу набрать массу",
        "хочу удержать вес",
        "хочу привести себя в форму",
    ])
    def test_other_goals_are_left_to_the_model(self, phrase):
        """Остальные три цели модель различает уверенно — не отбираем."""
        from agent import _goal_from_text

        assert _goal_from_text(phrase) is None

    def test_phrase_overrides_the_model(self, catalog):
        """Модель сказала gain, а человек написал «рекомпозиция»."""
        agent = make_agent(
            catalog,
            script={
                "маршрутизатор": classify_as("plan"),
                "Извлеки параметры": json.dumps({
                    "sex": "male", "age": 24, "height_cm": 167, "weight_kg": 65,
                    "activity": "moderate", "goal": "gain",
                }),
                "Объясни человеку": "План готов.",
            },
        )
        agent.run("Мужчина, 24, 167 см, 65 кг, тренируюсь 3 раза в неделю, "
                  "мягкая рекомпозиция")
        assert agent.profile["goal"] == "recomp"
