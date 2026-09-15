"""
Retrieve: поиск релевантных чанков. Здесь сконцентрирован основной тюнинг RAG.

Что умеет ретривер (каждый приём включается отдельным флагом, чтобы можно было
честно измерить его вклад в метрики — см. EVOLUTION.md):

1. **Dense-поиск** — семантический поиск по эмбеддингам (база LanceDB).
   Включён всегда.
2. **BM25 (лексический поиск)** — классический поиск по словам. Ловит то,
   что «плывёт» у эмбеддингов: точные термины, названия препаратов, цифры.
   Включён: на запросе бесплатен, прироста по метрикам не даёт.
3. **Hybrid (RRF)** — объединение двух списков через Reciprocal Rank Fusion.
4. **LLM-реранкинг** — модель переупорядочивает кандидатов по релевантности.
   **Выключен по умолчанию.** Поиск он улучшает значимо (precision@5
   0.839 → 0.881), но до ответа прирост не доходит: correctness сдвигается
   на +0.007 при пороге различимости 0.010, а платить надо 23% времени
   ответа. Подробности и цифры — в комментарии к ``use_rerank``
   в ``agent.build_agent`` и в EVOLUTION.md, шаги 24 и 28.

Пример::

    retriever = Retriever(table, embedder, client=client)
    chunks = retriever.retrieve("Какие факторы риска у гипертонии?", k=5)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from lance_db import read_all, vector_search

# ────────────────────────────────────────────────────────────
# Лексический поиск: BM25 «своими руками»
# ────────────────────────────────────────────────────────────

# Частые слова русского языка, которые только мешают поиску.
RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою",
    "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
    "им", "более", "всегда", "конечно", "всю", "между",
}

# Корпус проекта — англоязычные абстракты PubMed, поэтому нужен и английский
# список: без него BM25 честно ранжирует документы по частоте слова "the".
EN_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "was", "were", "are", "from",
    "have", "has", "had", "not", "but", "all", "can", "may", "its",
    "our", "their", "these", "those", "there", "than", "then", "when", "which",
    "who", "whom", "been", "being", "into", "over", "under", "more", "most",
    "such", "some", "any", "each", "both", "between", "during", "after",
    "before", "while", "about", "against", "among", "also", "however",
    "although", "therefore", "thus", "study", "studies", "results", "conclusion",
    "conclusions", "background", "methods", "objective", "aim", "significant",
    "significantly", "associated", "compared", "using", "used", "based",
}

STOPWORDS = RU_STOPWORDS | EN_STOPWORDS


def tokenize(text: str) -> list[str]:
    """Разбить текст на слова: нижний регистр, только буквы/цифры, без стоп-слов.

    Дополнительно обрезаем длинные слова до 6 символов — это грубая замена
    стеммингу: в русском языке «гипертензия / гипертензии / гипертензией»
    после обрезки превращаются в одну основу «гиперт».
    """
    words = re.findall(r"[\w]+", text.lower())
    return [w[:6] for w in words if w not in STOPWORDS and len(w) > 2]


class BM25:
    """Классический алгоритм лексического поиска BM25.

    Идея простая: чанк тем релевантнее, чем чаще в нём встречаются слова
    запроса (TF), но редкие слова важнее частых (IDF), а длинные документы
    получают штраф, чтобы не выигрывать просто за счёт объёма.
    """

    def __init__(self, corpus_texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1  # насколько сильно растёт вес при повторах слова
        self.b = b  # насколько сильно штрафуем длинные документы

        self.docs_tokens = [tokenize(text) for text in corpus_texts]
        self.doc_lengths = [len(tokens) for tokens in self.docs_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_count = len(self.docs_tokens)

        # В скольких документах встречается каждое слово (document frequency).
        doc_freq: Counter[str] = Counter()
        for tokens in self.docs_tokens:
            for token in set(tokens):
                doc_freq[token] += 1

        # IDF: редкое слово = высокий вес.
        self.idf: dict[str, float] = {}
        for token, freq in doc_freq.items():
            self.idf[token] = math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))

        # Частоты слов внутри каждого документа.
        self.term_freqs = [Counter(tokens) for tokens in self.docs_tokens]

    def score(self, query: str, doc_index: int) -> float:
        """Насколько документ doc_index соответствует запросу."""
        query_tokens = tokenize(query)
        term_freq = self.term_freqs[doc_index]
        doc_length = self.doc_lengths[doc_index]

        score = 0.0
        for token in query_tokens:
            if token not in term_freq:
                continue
            freq = term_freq[token]
            idf = self.idf.get(token, 0.0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_length / (self.avg_doc_length or 1))
            score += idf * numerator / denominator

        return score

    def top_n(self, query: str, n: int = 20) -> list[tuple[int, float]]:
        """Топ-N документов по BM25: список пар (индекс документа, скор)."""
        scores = [(i, self.score(query, i)) for i in range(self.doc_count)]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:n]


# ────────────────────────────────────────────────────────────
# Объединение результатов: Reciprocal Rank Fusion
# ────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Объединить несколько ранжированных списков в один (RRF).

    Каждый список голосует за документ весом 1 / (k + позиция).
    Документ, который стоит высоко сразу в нескольких списках, получает больше
    суммарных «голосов». Метод не требует приводить к общей шкале разные
    по природе оценки (косинусная близость и BM25) — сравниваются только позиции.
    """
    scores: dict[int, float] = {}

    for ranked in ranked_lists:
        for position, doc_index in enumerate(ranked):
            scores[doc_index] = scores.get(doc_index, 0.0) + 1.0 / (k + position + 1)

    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return fused


# ────────────────────────────────────────────────────────────
# LLM-реранкинг
# ────────────────────────────────────────────────────────────

_RERANK_PROMPT = """Ты — помощник поисковой системы по научным статьям о питании.

Вопрос пользователя:
{question}

Ниже пронумерованные фрагменты документов. Отбери те, которые реально помогают
ответить на вопрос, и упорядочь их от самого полезного к наименее полезному.

Фрагменты:
{candidates}

Верни ТОЛЬКО JSON вида {{"order": [номера фрагментов по убыванию полезности]}}.
Нерелевантные фрагменты в список не включай."""


def llm_rerank(
    question: str,
    candidates: list[dict],
    client: Any,
    model: str,
    top_k: int = 5,
    max_chars: int = 1200,
) -> list[dict]:
    """Переупорядочить кандидатов с помощью LLM (одним запросом на вопрос).

    Возвращает не больше top_k чанков. Если модель ответила некорректно —
    молча возвращаем исходный порядок, чтобы не ломать пайплайн.

    Про max_chars: он должен покрывать чанк ЦЕЛИКОМ. При обрезке до 500
    символов (чанк — 1000) реранкер видел половину текста и отбрасывал
    фрагменты, ответ в которых лежал во второй половине. Увеличение до 1200
    дало +0.04 F1 и +0.08 recall — см. EVOLUTION.md.
    """
    if not candidates:
        return []

    numbered = "\n\n".join(
        f"[{i}] {c['text'][:max_chars]}" for i, c in enumerate(candidates)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": _RERANK_PROMPT.format(question=question, candidates=numbered),
                }
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        from agent import parse_json_response

        data = parse_json_response(response.choices[0].message.content or "{}")
        order = data.get("order", [])

        reranked = [candidates[i] for i in order if isinstance(i, int) and 0 <= i < len(candidates)]

        # Если модель отсеяла вообще всё — доверяем исходному порядку.
        if not reranked:
            return candidates[:top_k]

        return reranked[:top_k]

    except Exception:
        return candidates[:top_k]


# ────────────────────────────────────────────────────────────
# Главный класс ретривера
# ────────────────────────────────────────────────────────────


class Retriever:
    """Поиск релевантных чанков с настраиваемыми стратегиями.

    Args:
        table: таблица LanceDB с чанками.
        embedder: объект Embedder для векторизации запроса.
        client: LLM-клиент (нужен только для реранкинга).
        model: модель для реранкинга.
        top_k: сколько чанков вернуть в итоге.
        candidate_k: сколько кандидатов набрать до реранкинга/слияния.
        use_hybrid: включить BM25 + RRF в дополнение к векторному поиску.
        use_rerank: включить LLM-реранкинг.
    """

    def __init__(
        self,
        table,
        embedder: Any,
        client: Any = None,
        model: str = "openai/gpt-4o-mini",
        top_k: int = 5,
        candidate_k: int = 40,
        use_hybrid: bool = True,
        use_rerank: bool = False,
    ) -> None:
        self.table = table
        self.embedder = embedder
        self.client = client
        self.model = model
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.use_hybrid = use_hybrid
        self.use_rerank = use_rerank

        # Для BM25 нужен весь корпус в памяти — читаем один раз при создании.
        self._corpus_df = read_all(table)
        self._corpus_records = self._corpus_df.to_dict("records")
        self._bm25 = BM25([r["text"] for r in self._corpus_records]) if use_hybrid else None

    # ------------------------------------------------------------------
    def _dense_search(self, query: str, k: int, doc_id: int | None = None) -> list[dict]:
        """Семантический поиск по векторам."""
        query_vector = self.embedder.embed_query(query)
        hits = vector_search(self.table, query_vector, k=k, doc_id=doc_id)

        results = []
        for hit in hits:
            results.append(
                {
                    "chunk_id": int(hit["chunk_id"]),
                    "text": hit["text"],
                    "doc_id": int(hit["doc_id"]),
                    "title": hit.get("title", ""),
                    "section": hit.get("section", ""),
                    "distance": float(hit.get("_distance", 0.0)),
                }
            )
        return results

    def _bm25_search(self, query: str, k: int, doc_id: int | None = None) -> list[dict]:
        """Лексический поиск по словам."""
        if self._bm25 is None:
            return []

        # Берём с запасом, потому что часть кандидатов может отсеяться фильтром по документу.
        raw_hits = self._bm25.top_n(query, n=k * 5 if doc_id is not None else k)

        results = []
        for corpus_index, score in raw_hits:
            record = self._corpus_records[corpus_index]
            if doc_id is not None and int(record["doc_id"]) != int(doc_id):
                continue
            results.append(
                {
                    "chunk_id": int(record["chunk_id"]),
                    "text": record["text"],
                    "doc_id": int(record["doc_id"]),
                    "title": record.get("title", ""),
                    "section": record.get("section", ""),
                    # BM25 не даёт «расстояния», поэтому переводим скор в псевдо-дистанцию:
                    # чем выше скор, тем меньше значение.
                    "distance": 1.0 / (1.0 + score),
                }
            )
            if len(results) >= k:
                break

        return results

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        doc_id: int | None = None,
        verbose: bool = False,
    ) -> list[dict]:
        """Найти релевантные чанки под запрос.

        Порядок работы:
        1. Векторный поиск (всегда).
        2. BM25-поиск и слияние через RRF — если use_hybrid.
        3. LLM-реранкинг — если use_rerank.
        """
        k = k or self.top_k

        dense_hits = self._dense_search(query, k=self.candidate_k, doc_id=doc_id)

        if not self.use_hybrid:
            candidates = dense_hits
        else:
            lexical_hits = self._bm25_search(query, k=self.candidate_k, doc_id=doc_id)

            # RRF работает с позициями, поэтому собираем общий индекс по chunk_id.
            by_chunk_id: dict[int, dict] = {}
            for hit in dense_hits + lexical_hits:
                by_chunk_id.setdefault(hit["chunk_id"], hit)

            dense_ranking = [hit["chunk_id"] for hit in dense_hits]
            lexical_ranking = [hit["chunk_id"] for hit in lexical_hits]

            fused = reciprocal_rank_fusion([dense_ranking, lexical_ranking])
            candidates = [by_chunk_id[chunk_id] for chunk_id, _ in fused if chunk_id in by_chunk_id]

            if verbose:
                print(f"  [retriever] dense={len(dense_hits)}, bm25={len(lexical_hits)}, после RRF={len(candidates)}")

        if self.use_rerank and self.client is not None:
            candidates = llm_rerank(query, candidates[: self.candidate_k], self.client, self.model, top_k=k)
            if verbose:
                print(f"  [retriever] после LLM-реранкинга: {len(candidates)}")

        return candidates[:k]
