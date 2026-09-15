"""
Сборка векторного индекса «с нуля»: корпус → чанки → эмбеддинги → LanceDB.

Запуск::

    python build_index.py

Скрипт идемпотентный: абстракты и эмбеддинги кэшируются на диске, поэтому
повторный запуск не тратит деньги на API и занимает секунды.
"""

from __future__ import annotations

from data_preporation import prepare_chunks
from embedder import Embedder
from lance_db import build_vectorstore, read_all
from providers import get_polza_client


def main() -> None:
    print("=" * 60)
    print("1. Подготовка корпуса (загрузка + очистка + чанкинг)")
    print("=" * 60)
    chunks = prepare_chunks()

    print()
    print("=" * 60)
    print("2. Векторизация и заливка в LanceDB")
    print("=" * 60)

    embedder = Embedder(get_polza_client())
    table = build_vectorstore(chunks, embedder, overwrite=True, verbose=True)

    print()
    print("=" * 60)
    print("3. Проверка: читаем всё из базы")
    print("=" * 60)
    frame = read_all(table)
    print(f"Строк в таблице: {len(frame)}")
    print(f"Колонки: {list(frame.columns)}")
    print(f"Уникальных статей: {frame['doc_id'].nunique()}")
    print()
    print(frame[["chunk_id", "doc_id", "title"]].head(5).to_string())


if __name__ == "__main__":
    main()
