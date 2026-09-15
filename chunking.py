"""
Модуль стратегий чанкинга для RAG-пайплайна.

Каждая стратегия принимает текст (и метаданные документа)
и возвращает список чанков — словарей с ключами:
  - text: str          — текст фрагмента
  - metadata: dict     — метаданные (chunk_id, strategy, ...)

В проекте используется semantic_chunking; fixed_size_chunking оставлен
для сравнения стратегий в экспериментах.

"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ────────────────────────────────────────────────────────────
# 1. Fixed-Size Chunking
# ────────────────────────────────────────────────────────────


def fixed_size_chunking(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: dict | None = None,
) -> list[Document]:
    """
    Разбивает текст на фрагменты фиксированного размера.

    Простейшая стратегия: нарезаем текст на куски по ``chunk_size``
    символов с перекрытием ``chunk_overlap``.  Разделитель — двойной
    перенос строки (абзац), но если абзац длиннее chunk_size,
    текст режется принудительно.

    Args:
        text: исходный текст документа.
        chunk_size: целевой размер чанка (в символах).
        chunk_overlap: перекрытие между соседними чанками.
        metadata: метаданные документа-источника (будут скопированы
                  в каждый чанк).

    Returns:
        Список ``Document`` с полями ``page_content`` и ``metadata``.
    """
    metadata = metadata or {}

    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_text(text)

    documents: list[Document] = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                **metadata,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "strategy": "fixed-size",
            },
        )
        documents.append(doc)

    return documents


# ────────────────────────────────────────────────────────────
# 2. Semantic Chunking (Recursive)
# ────────────────────────────────────────────────────────────

# Паттерны для определения секции документа
_SECTION_PATTERNS = [
    re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE),  # Markdown-заголовки
    re.compile(r"^\*\*(.{5,80})\*\*", re.MULTILINE),  # **Жирный текст** как заголовок
    re.compile(r"^[А-ЯЁA-Z\s]{10,80}$", re.MULTILINE),  # ЗАГЛАВНЫЕ БУКВЫ
]


def _detect_section(text: str, fallback: str = "Введение") -> str:
    """Пытается определить название секции из текста чанка."""
    for pattern in _SECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1) if m.lastindex else m.group(0)
    return fallback


def semantic_chunking(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    metadata: dict | None = None,
) -> list[Document]:
    """
    Рекурсивный чанкинг с учётом структуры текста.

    В отличие от fixed-size, ``RecursiveCharacterTextSplitter``
    пробует разделители **по приоритету**:

    1. ``\\n\\n`` — граница абзаца (лучший вариант)
    2. ``\\n``   — перенос строки
    3. ``". "``  — конец предложения (с учётом русской точки)
    4. ``" "``   — пробел
    5. ``""``    — посимвольно (крайний случай)

    Если текст удаётся разбить по более «семантичному» разделителю —
    он используется;  иначе алгоритм спускается к менее осмысленному.

    Дополнительно: для каждого чанка определяется **секция** документа
    (по заголовкам), что позволяет фильтровать при поиске.

    Args:
        text: исходный текст документа.
        chunk_size: целевой размер чанка (в символах).
        chunk_overlap: перекрытие между соседними чанками.
        metadata: метаданные документа-источника.

    Returns:
        Список ``Document`` с расширенными метаданными
        (``section``, ``strategy``).
    """
    metadata = metadata or {}

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_text(text)

    documents: list[Document] = []
    current_section = "Введение"

    for i, chunk in enumerate(chunks):
        # Обновляем секцию, если в чанке есть заголовок
        detected = _detect_section(chunk, fallback=current_section)
        current_section = detected

        doc = Document(
            page_content=chunk,
            metadata={
                **metadata,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "strategy": "semantic-recursive",
                "section": current_section,
            },
        )
        documents.append(doc)

    return documents
