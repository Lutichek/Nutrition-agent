"""
Ограничения частоты запросов и суточный потолок вызовов модели.

Зачем (ревью R1). Каждая реплика в чате — 4-7 платных вызовов модели,
а ручки были открыты без единого ограничения: баланс сливался бы
скриптом за ночь, а меню на месяц (15 с процессора) клало бы сервис
десятком параллельных запросов.

Два разных механизма, и путать их нельзя:

* **частота** (``RateLimiter``) — скользящее окно в памяти процесса.
  Рестарт её обнуляет, и это не страшно: окна короткие;
* **суточный потолок вызовов модели** (``LLMBudget``) — в постоянном
  хранилище (storage.py). Обнули его рестарт — потолок обходился бы
  перезапуском контейнера, который падает сам при нехватке памяти.

Потолок списывается в ``NutritionAgent._complete`` — на КАЖДЫЙ вызов,
включая вызовы наперёд (speculate) и повторы после сбоев. Считать
реплики вместо вызовов нельзя: цена реплики плавает от 4 до 7 вызовов.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any


class BudgetExhausted(RuntimeError):
    """Суточный потолок вызовов модели исчерпан.

    Подкласс RuntimeError — тот же путь, что «у провайдера кончились
    деньги» (agent._is_out_of_funds): вызов не делается, повтора нет.
    """


@dataclass
class Limits:
    """Пороги; значения по умолчанию — в config.py, переопределяются окружением."""

    chat_per_minute_session: int = 6
    chat_per_hour_session: int = 60
    chat_per_minute_ip: int = 20
    chat_per_hour_ip: int = 200
    streams_per_ip: int = 3
    calc_per_minute_ip: int = 60
    menu_per_minute_ip: int = 6
    llm_calls_per_day: int = 3000

    @classmethod
    def from_settings(cls, settings: Any) -> Limits:
        return cls(**{name: getattr(settings, name) for name in cls.__dataclass_fields__})


class RateLimiter:
    """Скользящее окно: не больше ``limit`` событий за ``window`` секунд на ключ."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._active: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def hit(self, key: str, limit: int, window: float) -> float | None:
        """Засчитать событие. None — можно; иначе — через сколько секунд повторить."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= window:
                hits.popleft()
            if len(hits) >= limit:
                return max(1.0, window - (now - hits[0]))
            hits.append(now)
            return None

    def acquire(self, key: str, limit: int) -> bool:
        """Занять одно из ``limit`` одновременных мест (поток ответа)."""
        with self._lock:
            if self._active[key] >= limit:
                return False
            self._active[key] += 1
            return True

    def release(self, key: str) -> None:
        with self._lock:
            self._active[key] = max(0, self._active[key] - 1)


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def seconds_to_utc_midnight() -> int:
    now = datetime.now(UTC)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((tomorrow - now).total_seconds()) + 1


class LLMBudget:
    """Суточный потолок вызовов модели — счётчик в постоянном хранилище."""

    def __init__(self, storage: Any, per_day: int) -> None:
        self.storage = storage
        self.per_day = per_day

    def exhausted(self) -> bool:
        return self.storage.get_usage(_today()) >= self.per_day

    def charge(self, calls: int = 1) -> None:
        """Списать вызов ДО того, как он сделан; сверх потолка — BudgetExhausted."""
        if self.exhausted():
            raise BudgetExhausted(
                f"Суточный потолок вызовов модели ({self.per_day}) исчерпан")
        self.storage.add_usage(_today(), calls)
