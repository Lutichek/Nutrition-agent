"""
Общие фикстуры для тестов.

Тесты не ходят в сеть и не тратят деньги на API: LLM подменяется заглушкой,
ретривер — списком заранее заданных фрагментов. Единственная внешняя
зависимость — собранный каталог блюд (``data/processed/foods.parquet``),
который создаётся командой ``python foods.py --rebuild``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Тесты лежат в подпапке, а модули проекта — уровнем выше.
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from foods import CATALOG_PATH, load_catalog  # noqa: E402


@pytest.fixture(scope="session")
def catalog():
    """Каталог блюд. Собирается один раз на весь прогон — он неизменяемый."""
    if not CATALOG_PATH.exists():
        pytest.skip(
            "Каталог не собран. Выполните: python data_sources.py && python foods.py --rebuild"
        )
    return load_catalog()


# ────────────────────────────────────────────────────────────
# Заглушки внешних сервисов
# ────────────────────────────────────────────────────────────


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    """Отвечает по сценарию: подбирает ответ по ключевому слову в промпте."""

    def __init__(self, owner: FakeLLMClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> FakeResponse:
        prompt = kwargs["messages"][0]["content"]
        self._owner.calls.append(prompt)

        if self._owner.fail_times > 0:
            self._owner.fail_times -= 1
            raise RuntimeError("провайдер недоступен")

        for marker, reply in self._owner.script.items():
            if marker in prompt:
                return FakeResponse(reply)

        return FakeResponse(self._owner.default)


class FakeChat:
    def __init__(self, owner: FakeLLMClient) -> None:
        self.completions = FakeCompletions(owner)


class FakeLLMClient:
    """Подмена LLM-клиента.

    Args:
        script: словарь «фрагмент промпта → что ответить».
        default: ответ, если ни один фрагмент не совпал.
        fail_times: сколько первых вызовов должны упасть (проверка ретраев).
    """

    def __init__(
        self,
        script: dict[str, str] | None = None,
        default: str = "ответ модели",
        fail_times: int = 0,
    ) -> None:
        self.script = script or {}
        self.default = default
        self.fail_times = fail_times
        self.calls: list[str] = []
        self.chat = FakeChat(self)


class FakeTable:
    """Подмена таблицы LanceDB — ровно тот кусок интерфейса, который читает API."""

    def __init__(self, rows: int) -> None:
        self._rows = rows

    def count_rows(self) -> int:
        return self._rows


class FakeRetriever:
    """Подмена ретривера: всегда возвращает заданные фрагменты."""

    def __init__(self, chunks: list[dict] | None = None) -> None:
        self.chunks = chunks if chunks is not None else []
        self.queries: list[str] = []
        # Настоящий Retriever держит таблицу LanceDB; заглушка повторяет
        # интерфейс, иначе тест не поймает обращение к ней в api.health.
        self.table = FakeTable(len(self.chunks))

    def retrieve(self, query: str, k: int | None = None, **kwargs: Any) -> list[dict]:
        self.queries.append(query)
        return self.chunks[: k or len(self.chunks)]


@pytest.fixture(scope="session")
def corpus_doc_ids() -> list[int]:
    """Два настоящих doc_id из справочника корпуса.

    Раньше в фикстуре стояли выдуманные 0 и 1. Это работало, пока doc_id был
    порядковым номером, и сломалось, когда он стал PMID: агент резолвит
    источники через corpus_index, а нулевого PMID там нет. Тест это поймал —
    и правильно сделал. Чтобы фикстура не устаревала при следующей смене
    схемы, идентификаторы берутся из самого справочника.
    """
    from data_preporation import load_corpus_index

    index = load_corpus_index()
    if len(index) < 2:
        pytest.skip("Справочник корпуса пуст. Выполните: python build_index.py")
    return [int(doc_id) for doc_id in list(index)[:2]]


@pytest.fixture
def fake_chunks(corpus_doc_ids) -> list[dict]:
    return [
        {
            "chunk_id": 1,
            "doc_id": corpus_doc_ids[0],
            "text": "Higher protein intake preserves lean mass during energy restriction.",
            "title": "Protein and weight loss",
            "distance": 0.1,
        },
        {
            "chunk_id": 2,
            "doc_id": corpus_doc_ids[1],
            "text": "Weight loss exceeding 1% of body mass per week increases lean mass loss.",
            "title": "Rate of weight loss",
            "distance": 0.2,
        },
    ]
