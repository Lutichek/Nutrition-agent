"""
Поиск блюда каталога по смыслу — для «съел не по плану» (этап 2).

Зачем. Поиск по совпадению слов на живых фразах находил не то: на
«курицу» восемь мест занимали кожа, лапки и хвост (короткие названия),
а обычная грудка — «Chicken breast, NS as to cooking method, …» — в
список не попадала вовсе; модели оставалось выбрать котлету по-киевски.
Названия FNDDS — это американские анкеты, разговорную русскую фразу
словами с ними не сопоставить.

Замер (8 трудных фраз × 2 прогона, живая модель):

    поиск кандидатов + выбор моделью     верно   неверное блюдо
    по словам                             7/16          4
    по смыслу (здесь)                    10/16          1

Остальные промахи — честное «не нашёл, назовите проще»: отдельной
сметаны в каталоге действительно нет.

Векторы названий («русское | английское») строятся один раз, как индекс
статей, и лежат рядом с ним — в git не идут, в образ попадают::

    python food_search.py --rebuild

Нет файла — ``available()`` ложно, и агент ищет по словам, как раньше:
так работают тесты с заглушкой провайдера и запуск без ключа.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VECTORS_PATH = ROOT / "lance_db" / "catalog_names.npz"


def name_texts(catalog: pd.DataFrame) -> list[str]:
    ru = catalog["name_ru"].fillna("") if "name_ru" in catalog.columns else ""
    return [f"{r} | {e}" for r, e in zip(ru, catalog["name"], strict=True)]


def build(catalog: pd.DataFrame, embedder) -> None:
    vectors = np.asarray(embedder.embed_documents(name_texts(catalog)), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    # float16: вдвое меньше файл, а для выбора дюжины кандидатов точности хватает.
    np.savez_compressed(VECTORS_PATH, ids=catalog["fdc_id"].astype(np.int64).to_numpy(),
                        vectors=vectors.astype(np.float16))


@lru_cache(maxsize=1)
def _load() -> tuple[np.ndarray, np.ndarray] | None:
    if not VECTORS_PATH.exists():
        return None
    data = np.load(VECTORS_PATH)
    return data["ids"], data["vectors"].astype(np.float32)


def available() -> bool:
    return _load() is not None


def nearest(text: str, embedder, k: int = 12) -> list[int]:
    """fdc_id блюд, ближайших по смыслу к фразе человека."""
    loaded = _load()
    if loaded is None:
        return []
    ids, vectors = loaded
    query = np.asarray(embedder.embed_query(text), dtype=np.float32)
    query /= np.linalg.norm(query)
    return [int(i) for i in ids[np.argsort(-(vectors @ query))[:k]]]


if __name__ == "__main__":
    if "--rebuild" not in sys.argv:
        print(__doc__)
        sys.exit(0)
    from embedder import Embedder
    from foods import load_catalog
    from providers import get_clients

    _, _, embed_client = get_clients("polza")
    catalog = load_catalog()
    build(catalog, Embedder(embed_client, cache_dir=None))
    print(f"{VECTORS_PATH}: {len(catalog)} названий")
