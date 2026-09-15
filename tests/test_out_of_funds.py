"""
Тесты распознавания «деньги кончились» против «деньги заняты».

Наблюдавшийся случай, и он поучителен тем, что первая версия этой
проверки была НЕВЕРНОЙ. Прогон в 4 потока упал с 402 INSUFFICIENT_BALANCE,
и напрашивался вывод «кончились деньги, повторять бесполезно». Но в теле
ответа стояло:

    Требуется: 1.00 ₽, доступно: 0.87 ₽ (баланс: 3.87 ₽, зарезервировано: 3.00 ₽)

Денег хватало. polza.ai резервирует сумму под каждый запрос в полёте,
и три рубля были заняты соседними потоками. Тот же вызов, сделанный
по одному, прошёл сразу же. Быстрый отказ на 402 оборвал бы получасовой
прогон там, где достаточно подождать секунду.

Отсюда оба края, которые проверяются здесь:

* **не сдаваться рано** — 402 при живом балансе это временная нехватка;
* **не ждать напрасно** — если баланса не хватает и без резерва, или
  если OpenAI сообщил `insufficient_quota`, повторы бессмысленны.

Вторая тонкость: квоту OpenAI отдаёт кодом 429 — тем же, что и обычный
троттлинг, который повторять как раз НУЖНО. Различает их только тело ответа.
"""

from __future__ import annotations

import pytest

from agent import _is_out_of_funds


class FakeStatusError(Exception):
    """Изображает openai.APIStatusError: у него есть status_code."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TestReallyOutOfMoney:
    def test_balance_below_required(self):
        """Баланса не хватает и без учёта резерва — ждать нечего."""
        error = FakeStatusError(
            "Error code: 402 - {'error': {'code': 'INSUFFICIENT_BALANCE', "
            "'message': 'Недостаточно средств. Требуется: 1.00 ₽, доступно: 0.20 ₽ "
            "(баланс: 0.20 ₽, зарезервировано: 0.00 ₽)'}}",
            status_code=402,
        )
        assert _is_out_of_funds(error)

    def test_openai_quota_exhausted_despite_429(self):
        """Квота OpenAI кончилась — код 429, но повторять бесполезно."""
        error = FakeStatusError(
            "Error code: 429 - {'error': {'code': 'insufficient_quota', "
            "'message': 'You exceeded your current quota, please check your plan and billing details.'}}",
            status_code=429,
        )
        assert _is_out_of_funds(error)


class TestMoneyIsOnlyReserved:
    """Ровно тот случай, на котором упал прогон, — и он ВРЕМЕННЫЙ."""

    def test_reserved_by_parallel_requests_is_retryable(self):
        """Баланс 3.87 ₽, занято 3.00 ₽ соседними потоками. Через секунду
        они освободятся, и повтор пройдёт — что и случилось вживую."""
        error = FakeStatusError(
            "Error code: 402 - {'error': {'code': 'INSUFFICIENT_BALANCE', "
            "'message': 'Недостаточно средств. Требуется: 1.00 ₽, доступно: 0.87 ₽ "
            "(баланс: 3.87 ₽, зарезервировано: 3.00 ₽)'}}",
            status_code=402,
        )
        assert not _is_out_of_funds(error)

    def test_unparseable_402_is_retried(self):
        """Разобрать баланс не вышло — считаем временным.

        Ошибиться в эту сторону дешевле: три лишние попытки против
        оборванного получасового прогона.
        """
        assert not _is_out_of_funds(FakeStatusError("payment required", status_code=402))


class TestRetryableErrorsAreNotMistaken:
    """Эти повторять НУЖНО — принять их за «кончились деньги» значит уронить
    прогон там, где достаточно было подождать."""

    @pytest.mark.parametrize("error", [
        FakeStatusError("Error code: 429 - Rate limit reached for gpt-4o-mini", status_code=429),
        FakeStatusError("Error code: 503 - Service Unavailable", status_code=503),
        FakeStatusError("Error code: 500 - Internal server error", status_code=500),
        Exception("Connection reset by peer"),
        Exception("timed out"),
    ])
    def test_transient_failures_stay_retryable(self, error):
        assert not _is_out_of_funds(error)

    def test_plain_rate_limit_is_not_quota(self):
        """Главный опасный случай: 429 бывает и тем, и другим."""
        throttling = FakeStatusError(
            "Error code: 429 - {'error': {'code': 'rate_limit_exceeded', "
            "'message': 'Limit 200000 tokens per min'}}",
            status_code=429,
        )
        assert not _is_out_of_funds(throttling)
