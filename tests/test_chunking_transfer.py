"""
Тесты переноса разметки между нарезками.

Зачем они есть. ``qrels.csv`` хранит пары «вопрос — chunk_id». При другой
нарезке корпуса те же номера означают другие куски текста, и разметка
превращается в мусор — без единой ошибки, просто метрики начинают мерить
случайные совпадения. Это ровно тот класс тихого отказа, из-за которого
``doc_id`` в проекте сделан равным PMID, а не порядковому номеру.

Перенос по тексту — обход этой ловушки, и сам он тоже может ошибаться
молча: слишком низкий порог засчитает соседний абзац, слишком высокий
потеряет тот же кусок, разрезанный пополам. Поэтому проверяются оба края.
"""

from __future__ import annotations

from experiments.chunking import _overlap, _tokens, transfer_relevance

TITLE = "Protein intake and muscle mass in older adults"

# Кусок текста, который судья признал релевантным (нарезка 900).
GOLD = (
    f"{TITLE}\n\n"
    "Higher protein intake of 1.6 g per kg body mass was associated with "
    "greater lean mass retention during caloric restriction. Participants "
    "consuming 0.8 g per kg lost significantly more lean tissue over the "
    "twelve week intervention period."
)

DOC_ID = 37468189
RELEVANT = [(DOC_ID, _tokens(GOLD) - _tokens(TITLE))]


class TestOverlapCoefficient:
    def test_identical_sets_overlap_fully(self):
        assert _overlap({"a", "b"}, {"a", "b"}) == 1.0

    def test_subset_overlaps_fully_regardless_of_size(self):
        """Главное свойство: мера не штрафует за разницу в длине.

        Разница в длине — предмет эксперимента, и если мера будет её
        наказывать, мелкая нарезка проиграет ещё до всякого поиска.
        """
        small = {"a", "b"}
        large = {"a", "b", "c", "d", "e", "f", "g", "h"}
        assert _overlap(small, large) == 1.0

    def test_disjoint_sets_do_not_overlap(self):
        assert _overlap({"a"}, {"b"}) == 0.0

    def test_empty_set_does_not_crash(self):
        assert _overlap(set(), {"a"}) == 0.0


class TestTransferBetweenChunkings:
    def test_same_chunk_stays_relevant(self):
        """Контроль: на той же нарезке перенос обязан вернуть ту же разметку."""
        assert transfer_relevance(GOLD, DOC_ID, TITLE, RELEVANT)

    def test_finer_chunking_keeps_the_piece(self):
        """Мелкая нарезка: кусок целиком лежит внутри размеченного."""
        piece = (
            f"{TITLE}\n\nHigher protein intake of 1.6 g per kg body mass was "
            "associated with greater lean mass retention during caloric restriction."
        )
        assert transfer_relevance(piece, DOC_ID, TITLE, RELEVANT)

    def test_coarser_chunking_keeps_the_piece(self):
        """Крупная нарезка: размеченный кусок целиком лежит внутри нового."""
        wider = GOLD + (
            " Secondary outcomes included grip strength and gait speed, "
            "neither of which differed between groups."
        )
        assert transfer_relevance(wider, DOC_ID, TITLE, RELEVANT)

    def test_other_paragraph_of_the_same_article_is_not_relevant(self):
        """Верхний край: тот же документ, но текст про другое.

        Если бы порог был занижен, сюда попал бы любой фрагмент статьи —
        и число релевантных выросло бы пропорционально мелкости нарезки,
        то есть метрика мерила бы размер чанка, а не качество поиска.
        """
        other = (
            f"{TITLE}\n\nBlood pressure was measured with an automated cuff "
            "after five minutes of seated rest at each study visit."
        )
        assert not transfer_relevance(other, DOC_ID, TITLE, RELEVANT)

    def test_another_article_is_never_relevant(self):
        """Совпадение слов без совпадения статьи не считается."""
        assert not transfer_relevance(GOLD, DOC_ID + 1, TITLE, RELEVANT)

    def test_title_alone_does_not_make_a_chunk_relevant(self):
        """Заголовок дописан к КАЖДОМУ чанку и должен вычитаться.

        Без вычитания любые два фрагмента одной статьи выглядели бы
        похожими, и релевантной оказалась бы вся статья целиком.
        """
        assert not transfer_relevance(f"{TITLE}\n\nMethods", DOC_ID, TITLE, RELEVANT)
