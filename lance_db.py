"""
Работа с векторной базой LanceDB: заливка, чтение, поиск, удаление (CRUD).

LanceDB — «лёгкая» векторная БД: хранится просто папкой на диске, не требует
поднимать сервер. Одна строка таблицы = один чанк текста + его вектор.

Типовой сценарий::

    from lance_db import build_vectorstore, read_all, vector_search

    table = build_vectorstore(chunks, embedder)   # C — create (залить корпус)
    df = read_all(table)                          # R — read   (проверить содержимое)
    hits = vector_search(table, query_vector, k=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb
import pandas as pd
from langchain_core.documents import Document

# Папка с базой и имя таблицы по умолчанию.
DEFAULT_DB_PATH = "lance_db/vectorstore"
DEFAULT_TABLE_NAME = "chunks"


# ────────────────────────────────────────────────────────────
# Подключение
# ────────────────────────────────────────────────────────────


def connect(db_path: str | Path = DEFAULT_DB_PATH):
    """Подключиться к базе (папка создастся автоматически, если её нет)."""
    Path(db_path).mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


def open_table(db_path: str | Path = DEFAULT_DB_PATH, table_name: str = DEFAULT_TABLE_NAME):
    """Открыть существующую таблицу."""
    db = connect(db_path)
    return db.open_table(table_name)


def table_exists(db_path: str | Path = DEFAULT_DB_PATH, table_name: str = DEFAULT_TABLE_NAME) -> bool:
    """Проверить, есть ли уже такая таблица в базе."""
    db = connect(db_path)
    return table_name in db.table_names()


# ────────────────────────────────────────────────────────────
# CREATE — заливка данных
# ────────────────────────────────────────────────────────────


def chunks_to_records(chunks: list[Document], embedder: Any, verbose: bool = True) -> list[dict]:
    """Превратить чанки в строки таблицы: текст + метаданные + вектор.

    Векторизация — самая долгая часть, поэтому делается одним пакетом
    (embedder внутри сам разобьёт на батчи и закэширует результат).
    """
    texts = [chunk.page_content for chunk in chunks]

    if verbose:
        print(f"Векторизуем {len(texts)} чанков...")
    vectors = embedder.embed_documents(texts, verbose=verbose)

    records: list[dict] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        meta = chunk.metadata
        records.append(
            {
                "chunk_id": int(meta.get("chunk_id", 0)),
                "text": chunk.page_content,
                # doc_id — идентификатор статьи-источника (см. комментарий в data_preporation.py)
                "doc_id": int(meta.get("doc_id", 0)),
                "title": str(meta.get("title", "")),
                "section": str(meta.get("section", "")),
                "vector": vector,
            }
        )

    return records


def build_vectorstore(
    chunks: list[Document],
    embedder: Any,
    db_path: str | Path = DEFAULT_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    overwrite: bool = True,
    verbose: bool = True,
):
    """Создать таблицу и залить в неё чанки.

    Args:
        chunks: список Document из data_preporation.prepare_chunks().
        embedder: объект Embedder.
        overwrite: True — пересоздать таблицу с нуля, False — добавить к существующей.
    """
    db = connect(db_path)
    records = chunks_to_records(chunks, embedder, verbose=verbose)

    if overwrite or table_name not in db.table_names():
        table = db.create_table(table_name, data=records, mode="overwrite")
        if verbose:
            print(f"Таблица '{table_name}' создана: {table.count_rows()} строк")
    else:
        table = db.open_table(table_name)
        table.add(records)
        if verbose:
            print(f"В таблицу '{table_name}' добавлено {len(records)} строк, всего {table.count_rows()}")

    return table


# ────────────────────────────────────────────────────────────
# READ — чтение
# ────────────────────────────────────────────────────────────


def read_all(table, with_vectors: bool = False) -> pd.DataFrame:
    """Прочитать всю таблицу целиком (нужно для проверки заливки).

    Важная деталь LanceDB: у table.to_pandas() есть лимит по умолчанию
    (вернёт только первые 10 строк!). Чтобы получить действительно ВСЕ строки,
    нужно обращаться к нижележащему датасету Lance через table.to_lance().

    Args:
        table: объект таблицы LanceDB.
        with_vectors: включать ли столбец с векторами (он большой и нечитаемый).

    Returns:
        pandas DataFrame со всеми строками таблицы.
    """
    try:
        df = table.to_lance().to_table().to_pandas()
    except Exception:
        # Запасной путь: явно просим количество строк, которое лежит в таблице.
        df = table.search().limit(table.count_rows()).to_pandas()

    if not with_vectors and "vector" in df.columns:
        df = df.drop(columns=["vector"])
    return df


def count_rows(table) -> int:
    """Сколько строк (чанков) лежит в таблице."""
    return table.count_rows()


def get_by_chunk_id(table, chunk_id: int) -> dict | None:
    """Найти один чанк по его идентификатору."""
    df = read_all(table)  # именно read_all: у to_pandas() лимит в 10 строк
    match = df[df["chunk_id"] == chunk_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


# ────────────────────────────────────────────────────────────
# SEARCH — векторный поиск
# ────────────────────────────────────────────────────────────


def vector_search(
    table,
    query_vector: list[float],
    k: int = 5,
    doc_id: int | None = None,
) -> list[dict]:
    """Найти k ближайших чанков к вектору запроса.

    Args:
        table: таблица LanceDB.
        query_vector: вектор поискового запроса (из embedder.embed_query).
        k: сколько результатов вернуть.
        doc_id: если указан — искать только внутри одной статьи (фильтр по документу).

    Returns:
        Список словарей с полями chunk_id, text, doc_id, title, section, _distance.
        Чем меньше _distance, тем ближе чанк к запросу.
    """
    query = table.search(query_vector).limit(k)

    if doc_id is not None:
        query = query.where(f"doc_id = {int(doc_id)}")

    return query.to_list()


# ────────────────────────────────────────────────────────────
# UPDATE / DELETE
# ────────────────────────────────────────────────────────────


def delete_by_doc_id(table, doc_id: int) -> None:
    """Удалить из базы все чанки одной статьи."""
    table.delete(f"doc_id = {int(doc_id)}")


def drop_table(db_path: str | Path = DEFAULT_DB_PATH, table_name: str = DEFAULT_TABLE_NAME) -> None:
    """Удалить таблицу целиком."""
    db = connect(db_path)
    if table_name in db.table_names():
        db.drop_table(table_name)


if __name__ == "__main__":
    # Быстрая проверка содержимого базы.
    if table_exists():
        tbl = open_table()
        data = read_all(tbl)
        print(f"Строк в таблице: {len(data)}")
        print(f"Колонки: {list(data.columns)}")
        print(f"Документов: {data['doc_id'].nunique()}")
        print()
        print(data.head(3).to_string())
    else:
        print("Таблица ещё не создана — запустите build_vectorstore().")
