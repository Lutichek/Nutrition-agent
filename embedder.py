"""
Эмбеддер — превращает текст в вектор чисел.

Зачем: чтобы искать «по смыслу», а не по точному совпадению слов, тексты
переводят в векторы (эмбеддинги). Похожие по смыслу тексты получают близкие
векторы, и поиск сводится к поиску ближайших векторов.

Здесь используется OpenAI-совместимый API (polza.ai) с моделью
``text-embedding-3-small`` — не требует скачивания тяжёлых моделей на диск.

Пример::

    from embedder import Embedder

    emb = Embedder(client)
    vectors = emb.embed_documents(["текст 1", "текст 2"])   # для документов
    query_vector = emb.embed_query("вопрос пользователя")   # для запроса
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Модель эмбеддингов по умолчанию и размерность её векторов.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536

# Сколько текстов отправляем в API за один запрос.
DEFAULT_BATCH_SIZE = 64


class Embedder:
    """Обёртка над API эмбеддингов с батчингом, retry и кэшем на диске.

    Args:
        client: OpenAI-совместимый клиент (``openai.OpenAI``).
        model: имя модели эмбеддингов.
        batch_size: сколько текстов слать в одном запросе.
        cache_dir: папка для кэша. Если ``None`` — кэш выключен.
    """

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_dir: str | Path | None = "lance_db/embedding_cache",
        max_retries: int = 4,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

        self.cache_dir: Path | None = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                # Кэш необязателен — см. _cache_put, там та же логика. А вот
                # падение здесь роняло бы весь сервис на старте, потому что
                # Embedder создаётся при подъёме приложения.
                #
                # Ровно это и происходило в контейнере: папка кэша исключена
                # из образа .dockerignore, файловая система смонтирована
                # только на чтение, и mkdir несуществующей папки давал
                # «Read-only file system» ещё до первого запроса.
                logger.warning(
                    "Кэш эмбеддингов отключён (%s недоступна: %s)",
                    self.cache_dir, error,
                )
                self.cache_dir = None

    # ------------------------------------------------------------------
    # Кэш: чтобы повторно не платить за эмбеддинги одного и того же текста
    # ------------------------------------------------------------------
    def _cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        # Имя файла — хэш от (модель + текст), чтобы кэши разных моделей не смешивались.
        key = hashlib.sha256(f"{self.model}::{text}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_put(self, text: str, vector: list[float]) -> None:
        path = self._cache_path(text)
        if path is None:
            return
        try:
            path.write_text(json.dumps(vector), encoding="utf-8")
        except Exception:
            # Кэш — вещь необязательная: если не записался, просто продолжаем работу.
            pass

    # ------------------------------------------------------------------
    # Вызов API
    # ------------------------------------------------------------------
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Один запрос к API с повторными попытками при сетевых сбоях."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(model=self.model, input=texts)
                # API может вернуть элементы не по порядку — сортируем по index.
                items = sorted(response.data, key=lambda d: d.index)
                return [item.embedding for item in items]
            except Exception as exc:
                last_error = exc
                # Экспоненциальная пауза: 1с, 2с, 4с, ...
                time.sleep(2**attempt)

        raise RuntimeError(f"Не удалось получить эмбеддинги после {self.max_retries} попыток: {last_error}")

    def embed_documents(self, texts: list[str], verbose: bool = False) -> list[list[float]]:
        """Векторизовать список текстов (для заливки в базу)."""
        if not texts:
            return []

        result: list[list[float] | None] = [None] * len(texts)

        # 1) Сначала забираем всё, что уже есть в кэше.
        to_compute: list[int] = []
        for i, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                result[i] = cached
            else:
                to_compute.append(i)

        if verbose:
            print(f"[Embedder] из кэша: {len(texts) - len(to_compute)}, считаем: {len(to_compute)}")

        # 2) Остальное считаем батчами через API.
        for start in range(0, len(to_compute), self.batch_size):
            batch_indices = to_compute[start : start + self.batch_size]
            batch_texts = [texts[i] for i in batch_indices]

            vectors = self._embed_batch(batch_texts)

            for idx, vector in zip(batch_indices, vectors, strict=True):
                result[idx] = vector
                self._cache_put(texts[idx], vector)

            if verbose:
                done = min(start + self.batch_size, len(to_compute))
                print(f"[Embedder] обработано {done}/{len(to_compute)}")

        return [vector for vector in result if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        """Векторизовать один поисковый запрос."""
        return self.embed_documents([text])[0]

    @property
    def dim(self) -> int:
        """Размерность вектора — нужна при создании таблицы в LanceDB."""
        if self.model == DEFAULT_EMBEDDING_MODEL:
            return DEFAULT_EMBEDDING_DIM
        # Для незнакомой модели узнаём размерность одним пробным запросом.
        return len(self.embed_query("test"))
