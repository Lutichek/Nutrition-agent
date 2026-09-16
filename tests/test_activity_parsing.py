"""
Тесты разбора уровня активности из ответа человека.

Наблюдавшийся случай целиком. Агент спросил «как часто ты тренируешься»
и показал пять вариантов словами. Человек ответил подписью варианта —
«средняя активность». Модель извлекла из этого ``sedentary``, то есть
противоположный край шкалы, устойчиво: три прогона из трёх, а на полном
диалоге четыре из пяти.

Последствия тихие и тройные:

* коэффициент активности 1.2 вместо 1.55 — около 500 ккал в день;
* пропало циклирование: при ``sedentary`` тренировочных дней ноль,
  и разводить нечего;
* переспросить агент не догадался, потому что поле не пропало —
  оно пришло НЕВЕРНЫМ, а проверка обязательных полей смотрит только
  на наличие.

Отдельно стоит отметить: «высокую активность» модель понимала верно,
ошибалась только на «средней». То есть беда не в подходе, а в конкретной
формулировке — и поймать такое можно лишь замером на всех подписях,
которые агент показывает.
"""

from __future__ import annotations

import pytest

from agent import ASK_PROFILE_PROMPT, _activity_from_text
from targets import ACTIVITY_FACTORS, ACTIVITY_LABELS


class TestLabelsTheAgentOffers:
    """Подписи, которые агент показывает сам, обязаны разбираться все.

    Если агент предложил список, ответ подписью — единственное разумное
    поведение человека.
    """

    @pytest.mark.parametrize("reply,expected", [
        ("сидячий образ жизни", "sedentary"),
        ("малая активность", "light"),
        ("средняя активность", "moderate"),
        ("высокая активность", "high"),
        ("спортсмен", "athlete"),
    ])
    def test_every_offered_label_is_understood(self, reply, expected):
        assert _activity_from_text(reply) == expected

    def test_the_observed_failure(self):
        """Ровно тот ответ, на котором всё сломалось."""
        assert _activity_from_text("средняя активность") == "moderate"
        assert _activity_from_text("средняя активность") != "sedentary"

    @pytest.mark.parametrize("reply,expected", [
        ("средняя", "moderate"),
        ("малая", "light"),
    ])
    def test_shortened_label_still_works(self, reply, expected):
        """Человек часто отвечает одним словом, отбросив «активность»."""
        assert _activity_from_text(reply) == expected

    def test_label_with_its_explanation(self):
        """Скопировали строку из вопроса целиком — тоже частый случай."""
        assert _activity_from_text(
            "средняя активность — тренировки 2-3 раза в неделю") == "moderate"


class TestFrequencyPhrasing:
    """Числом тренировок модель пользуется уверенно, но разбор дешевле."""

    @pytest.mark.parametrize("reply,expected", [
        ("не тренируюсь", "sedentary"),
        ("1 раз в неделю", "light"),
        ("2-3 раза в неделю", "moderate"),
        ("4 раза в неделю", "high"),
        ("две тренировки в день", "athlete"),
    ])
    def test_frequency_maps_to_level(self, reply, expected):
        assert _activity_from_text(reply) == expected


class TestWhatIsLeftToTheModel:
    """Свободные формулировки разбору не поддаются — и не должны.

    Вернуть здесь уровень наугад было бы хуже, чем не вернуть ничего:
    модель такие фразы понимает, а выдуманный коэффициент ломает
    весь расчёт.
    """

    @pytest.mark.parametrize("reply", [
        "хожу в зал",
        "иногда бегаю по утрам",
        "мне 24 года, рост 167",
        "работаю из дома",
    ])
    def test_ambiguous_phrasing_is_not_guessed(self, reply):
        assert _activity_from_text(reply) is None


class TestAgreementWithTheRestOfTheProject:
    def test_every_level_is_reachable(self):
        """Разбор обязан покрывать все уровни, иначе часть шкалы недостижима."""
        reachable = {
            _activity_from_text(text)
            for text in ("сидячий образ жизни", "малая активность",
                         "средняя активность", "высокая активность", "спортсмен")
        }
        assert reachable == set(ACTIVITY_FACTORS)

    def test_offered_labels_match_what_the_parser_expects(self):
        """Ключевая проверка связности: если промпт переформулируют,
        а разбор не поправят, дефект вернётся молча.

        Поэтому подписи из вопроса агента прогоняются через разбор
        и обязаны дать тот же уровень, под которым объявлены.
        """
        for level, label in ACTIVITY_LABELS.items():
            head = label.split("—")[0].strip()
            assert _activity_from_text(head) == level, (
                f"подпись «{head}» для уровня {level} не разбирается"
            )

    def test_labels_are_present_in_the_question(self):
        """Вопрос агента должен показывать те же подписи, что разбираются."""
        for label in ACTIVITY_LABELS.values():
            head = label.split("—")[0].strip()
            assert head in ASK_PROFILE_PROMPT or "{activity_options}" in ASK_PROFILE_PROMPT
