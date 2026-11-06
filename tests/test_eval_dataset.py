"""Номер исходного чанка в датасете оценки — по тексту, а не по старой нарезке.

Наблюдалось (шаг 65): корпус перенарезали после сборки датасета, ``chunk_id``
устарел у 80 строк из 210, и разбор по номеру показал «исходный фрагмент
не находится ни одним поиском» у 40% вопросов — иллюзия, не промах поиска.
"""

from __future__ import annotations

import json

import pandas as pd

from make_eval_dataset import resync_chunk_ids, source_ids

SOURCE = ("Dietary protein intake of 1.6 g per kg body mass per day maximised "
          "gains in lean mass during resistance training in healthy adults. ") * 4


def make_corpus() -> pd.DataFrame:
    """Перенарезанный корпус: исходный текст теперь лежит под номером 7."""
    return pd.DataFrame({
        "chunk_id": [3, 7, 9],
        "doc_id": [100, 200, 300],
        "text": ["Vitamin D status and bone density in older women.",
                 "Title.\n\n" + SOURCE + " Further text.",
                 "Sodium reduction lowers blood pressure."],
    })


def make_dataset(texts: list[str], old_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "id": [f"q{i}" for i in range(len(texts))],
        "relevant_text": [json.dumps([t]) for t in texts],
        "document": [200] * len(texts),
        "chunk_id": old_ids,
    })


def test_shifted_number_is_found_by_text():
    """Старый номер 3 указывает на чужой чанк — текст нашёлся под номером 7."""
    frame = resync_chunk_ids(make_dataset([SOURCE], [3]), make_corpus())
    assert int(frame.loc[0, "chunk_id"]) == 7
    assert int(frame.loc[0, "chunk_id_old"]) == 3


def test_article_missing_from_index_gets_empty_not_old_number():
    """Статьи 31791641 в текущем корпусе нет: номер — пусто, а не прежний."""
    gone = "Dietary saturated fat intake and risk of stroke: a dose-response meta-analysis. " * 4
    frame = resync_chunk_ids(make_dataset([gone], [876]), make_corpus())
    assert pd.isna(frame.loc[0, "chunk_id"])


def test_second_run_keeps_the_original_old_number():
    once = resync_chunk_ids(make_dataset([SOURCE], [3]), make_corpus())
    twice = resync_chunk_ids(once, make_corpus())
    assert int(twice.loc[0, "chunk_id_old"]) == 3
    assert int(twice.loc[0, "chunk_id"]) == 7


def test_ambiguous_match_gets_empty_not_a_guess():
    """Текст целиком в двух чанках (дубль статьи) — номер не угадывается."""
    corpus = pd.concat([make_corpus(), pd.DataFrame(
        {"chunk_id": [11], "doc_id": [400], "text": [SOURCE]})], ignore_index=True)
    assert source_ids(SOURCE, corpus) == {7, 11}
    frame = resync_chunk_ids(make_dataset([SOURCE], [3]), corpus)
    assert pd.isna(frame.loc[0, "chunk_id"])
