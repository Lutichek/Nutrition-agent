"""
Журнал чата: одна JSON-строка на реплику, файл на сутки.

Зачем. Почти все дефекты этого проекта были тихими: ответ приходит,
ошибки нет, результат неверен — «сладости» не раскрылись, пожелание
«больше белка» привело спортпит. По паре «вопрос → ответ» такое
не найти, поэтому в записи лежит ещё и то, что агент ПОНЯЛ: намерение,
профиль с ограничениями, поисковый запрос, оставленные фрагменты,
блюда плана, проверки, ответы модели по узлам (см. ``RAGAgentAnswer.trace``).

Профиль, ``seed`` и ``days`` повторяют меню до блюда без обращения
к модели — числа считает детерминированный код (``replay_plan``).
Ручки ``/plan`` и ``/menu`` для этого не годятся: пожелания, бюджет
и тренировочные дни они не учитывают, и план выйдет другим.

Чтение::

    jq 'select(.status == "error")' logs/chat-2026-10-20.jsonl
    pd.read_json("logs/chat-2026-10-20.jsonl", lines=True)

Три правила, каждое — из уже случившегося:

* **Журнал не роняет сервис.** Контейнер работает с файловой системой
  только на чтение, и ``Embedder.__init__`` однажды не дал сервису
  подняться безусловным ``mkdir``. Здесь папка создаётся при первой
  записи, а любой сбой записи — одно предупреждение в лог и выключенный
  журнал: человек получает ответ, даже если диск полон.
* **Сбой записи не уходит в stdout.** В записи — вес, возраст, диагнозы
  из ``red_flag``. Вывод контейнера ротации не имеет и живёт дольше
  30 дней, поэтому запасного пути «напечатать вместо файла» нет.
* **Срок хранения — 30 дней** (``CHAT_LOG_RETENTION_DAYS``). Старые файлы
  удаляются при смене суток.

Включение: ``CHAT_LOG_DIR`` (по умолчанию ``logs/`` рядом с кодом),
пустое значение или ``off`` выключает журнал.
"""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).parent / "logs"
DEFAULT_RETENTION_DAYS = 30

_FILE_RE = re.compile(r"^chat-(\d{4}-\d{2}-\d{2})\.jsonl$")


class ChatLog:
    """Запись реплик в ``<dir>/chat-YYYY-MM-DD.jsonl`` (сутки по UTC)."""

    def __init__(self, directory: Path | None, retention_days: int = DEFAULT_RETENTION_DAYS):
        # Ничего не создаётся и не проверяется здесь: см. правило про mkdir.
        self.directory = directory
        self.retention_days = retention_days
        self._lock = Lock()
        self._cleaned_for: date | None = None
        self._broken = False

    @classmethod
    def from_env(cls) -> ChatLog:
        raw = os.getenv("CHAT_LOG_DIR")
        if raw is None:
            directory: Path | None = DEFAULT_DIR
        elif raw.strip().lower() in ("", "off", "0", "false"):
            directory = None
        else:
            directory = Path(raw)
        retention = int(os.getenv("CHAT_LOG_RETENTION_DAYS", DEFAULT_RETENTION_DAYS))
        return cls(directory, retention_days=retention)

    @property
    def enabled(self) -> bool:
        return self.directory is not None and not self._broken

    def write(self, record: dict[str, Any]) -> None:
        """Дописать запись. Не бросает исключений никогда."""
        if not self.enabled:
            return
        now = datetime.now(UTC)
        line = json.dumps(
            {"ts": now.isoformat(timespec="seconds"), **record},
            ensure_ascii=False,
            default=str,
        )
        try:
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                if self._cleaned_for != now.date():
                    self._cleanup(now.date())
                    self._cleaned_for = now.date()
                path = self.directory / f"chat-{now.date().isoformat()}.jsonl"
                with path.open("a", encoding="utf-8") as file:
                    file.write(line + "\n")
        except Exception as error:
            # Выключаемся с первого сбоя: на read-only диске иначе каждая
            # реплика оставляла бы в логе по трассировке.
            self._broken = True
            logger.warning(
                "Журнал чата выключен: не удалось писать в %s (%s). "
                "Ответы пользователям это не затрагивает.",
                self.directory, error,
            )

    def _cleanup(self, today: date) -> None:
        """Удалить файлы старше срока хранения."""
        border = today - timedelta(days=self.retention_days)
        for path in self.directory.glob("chat-*.jsonl"):
            match = _FILE_RE.match(path.name)
            if not match:
                continue
            try:
                if date.fromisoformat(match.group(1)) < border:
                    path.unlink()
            except (ValueError, OSError):
                logger.warning("Не удалось удалить старый журнал %s", path, exc_info=True)


def replay_plan(agent: Any, trace: dict[str, Any]) -> list[list[int]] | None:
    """Пересобрать меню из записи журнала: fdc_id блюд по дням.

    Узел make_plan модель не вызывает, поэтому повтор бесплатный и точный.
    Расхождение с ``trace["plan_days"]`` значит, что с тех пор поменялся
    солвер или каталог, — а не что баг «не воспроизводится».
    """
    if agent.seed != trace.get("seed"):
        raise ValueError(f"У агента seed={agent.seed}, в записи {trace.get('seed')}")
    update = agent._node_make_plan({"profile": trace["profile"], "days": trace["days"]})
    menu = update.get("menu")
    if menu is None:
        return None
    return [[item.fdc_id for item in day.items] for day in menu.days]


def turn_record(
    *,
    session_id: str,
    turn: int,
    message: str,
    answer: Any = None,
    shown: str | None = None,
    sources: list[str] | None = None,
    elapsed_ms: int,
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Собрать запись о реплике из ответа агента (или из исключения)."""
    record: dict[str, Any] = {
        "session_id": session_id,
        "turn": turn,
        "status": "error" if error is not None else "ok",
        "elapsed_ms": elapsed_ms,
        "message": message,
    }
    if answer is not None:
        record["answer"] = answer.answer
        # Показанное отличается от ответа агента (номера PMID вырезаны),
        # и если человек жалуется на текст — смотреть надо на это поле.
        if shown is not None and shown != answer.answer:
            record["shown"] = shown
        record["sources"] = sources or []
        plan = answer.plan
        if plan is not None:
            record["plan"] = {
                "kcal": round(plan.totals.get("kcal", 0.0)),
                "target_kcal": round(plan.targets.get("kcal", 0.0)),
                "cost": plan.cost,
                "budget": plan.budget,
            }
        record["trace"] = answer.trace
    if error is not None:
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    return record
