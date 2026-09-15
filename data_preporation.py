"""
Подготовка корпуса литературы: абстракты PubMed → очистка → чанки.

Корпус — 682 абстракта обзоров, мета-анализов и руководств по нутрициологии
(как он собирается — см. ``data_sources.py``). Отсюда агент берёт не цифры
рациона, а обоснования: почему при дефиците калорий белка нужно больше,
чем при удержании веса; чем опасно снижение веса быстрее килограмма в неделю.

Почему абстракты, а не полные тексты: у абстракта есть PMID, он открыт
по лицензии NCBI, и в нём уже сконцентрирован вывод статьи. Полные тексты
частью платные, частью в PDF — это отдельный проект по парсингу.

⚠️ Корпус англоязычный, а пользователь пишет по-русски. Разрыв закрывается
не здесь, а в агенте: узел ``rewrite`` переводит вопрос на английский перед
поиском. Без этого BM25 не находил бы вообще ничего (русские слова просто
не встречаются в английском тексте), а векторный поиск работал бы вполсилы.

Флоу::

    python data_preporation.py           # собрать и показать статистику

    from data_preporation import prepare_chunks
    chunks = prepare_chunks()            # список Document, готовых к заливке

Учебный проект. Ответы на его основе не заменяют консультацию врача.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document

from chunking import fixed_size_chunking, semantic_chunking

# Справочник «doc_id → выходные данные статьи». Нужен, чтобы агент мог
# сослаться на источник: в самих чанках лежит только doc_id.
CORPUS_INDEX_PATH = Path(__file__).parent / "data" / "processed" / "corpus_index.json"


def clean_text(raw_text: str) -> str:
    """Убрать из абстракта то, что мешает и поиску, и чтению."""
    text = unicodedata.normalize("NFKC", raw_text)

    # Заявления о финансировании и авторских правах: они есть почти в каждом
    # абстракте, одинаковы по формулировке и потому «склеивают» несвязанные
    # статьи при векторном поиске.
    text = re.sub(
        r"(Copyright ©|©|This article is protected by copyright|"
        r"All rights reserved\.|Published by Elsevier|"
        r"PROSPERO registration|Trial registration).*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Мягкие переносы и неразрывные пробелы.
    text = text.replace("­", "").replace(" ", " ")

    # Схлопываем пробелы, но сохраняем абзацы: по ним режет semantic-чанкинг.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_corpus(verbose: bool = True) -> list[dict]:
    """Загрузить абстракты и привести их к виду «документ корпуса».

    ``doc_id`` — это PMID статьи, а НЕ её порядковый номер в корпусе.

    Разница принципиальная. Раньше doc_id выдавался через ``enumerate``,
    то есть зависел от состава корпуса: стоило добавить статьи — и все
    идентификаторы съезжали. А doc_id записан в эталонах тестового датасета
    и используется в ``evaluation.py`` как ЖЁСТКИЙ фильтр релевантности:
    чанк засчитывается, только если его doc_id совпал с эталонным.
    После расширения корпуса все retrieval-метрики молча обнулились бы —
    ни одной ошибки, просто нули. PMID же не меняется никогда.
    """
    # Импорт внутри функции, а не наверху модуля: загрузка корпуса нужна
    # только при пересборке, а рантайму от этого модуля требуется одна
    # load_corpus_index. Наверху импорт затаскивал бы в работающий сервис
    # весь загрузчик с urllib и разбором выгрузок — код, который там
    # никогда не выполняется.
    from data_sources import load_pubmed

    articles = load_pubmed()

    corpus: list[dict] = []
    for article in articles:
        if not str(article.get("pmid", "")).isdigit():
            continue
        doc_id = int(article["pmid"])
        text = clean_text(article["abstract"])
        if len(text) < 300:
            # После вычистки копирайтов от абстракта могло почти ничего не остаться.
            continue

        corpus.append(
            {
                "doc_id": doc_id,
                "pmid": article["pmid"],
                "title": article["title"],
                "journal": article["journal"],
                "year": article["year"],
                "topics": article.get("topics", []),
                # Только сам абстракт. Заголовок дописывается к каждому чанку
                # отдельно — см. prepare_chunks.
                "text": text,
            }
        )

    if verbose:
        print(f"[корпус] статей: {len(corpus)} (из {len(articles)} загруженных)")

    return corpus


def save_corpus_index(corpus: list[dict]) -> None:
    """Сохранить справочник источников: doc_id → PMID, название, журнал, год."""
    index = {
        str(document["doc_id"]): {
            "pmid": document["pmid"],
            "title": document["title"],
            "journal": document["journal"],
            "year": document["year"],
        }
        for document in corpus
    }
    CORPUS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_corpus_index() -> dict[str, dict]:
    """Прочитать справочник источников (для оформления ссылок в ответе)."""
    if not CORPUS_INDEX_PATH.exists():
        save_corpus_index(build_corpus(verbose=False))
    return json.loads(CORPUS_INDEX_PATH.read_text(encoding="utf-8"))


# Фрагменты короче этого — обрывки: хвост абстракта или подпись. Смысла
# в них нет, а поиск они засоряют.
MIN_CHUNK_CHARS = 120


def prepare_chunks(
    strategy: str = "semantic",
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    verbose: bool = True,
) -> list[Document]:
    """Полный флоу подготовки: загрузка → очистка → чанкинг.

    Args:
        strategy: "fixed" — фиксированный размер, "semantic" — по структуре текста.
        chunk_size: целевой размер чанка в символах.
        chunk_overlap: перекрытие соседних чанков.

    Два решения здесь приняты по замерам, а не по умолчанию (цифры —
    в EVOLUTION.md):

    * **semantic, а не fixed.** ``CharacterTextSplitter`` режет только по
      абзацам, а в структурированном абстракте («Methods: ...») абзац бывает
      длиной в 4000 символов — он уезжал в один чанк целиком. Максимум по
      корпусу был 12321 символ при заданных 900.
    * **Заголовок дописывается к каждому чанку, а не режется вместе с текстом.**
      Раньше заголовок был отдельным абзацем и становился отдельным чанком
      на 7 символов; таких обрывков набиралось 527. Теперь название статьи
      есть в каждом фрагменте — это и чинит обрывки, и помогает поиску:
      чанк из середины абстракта больше не теряет тему.

    Returns:
        Список Document; в metadata лежат chunk_id, doc_id, title, section.
    """
    corpus = build_corpus(verbose=verbose)
    save_corpus_index(corpus)

    chunker = fixed_size_chunking if strategy == "fixed" else semantic_chunking

    all_chunks: list[Document] = []
    global_chunk_id = 0

    skipped = 0

    for document in corpus:
        chunks = chunker(
            document["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata={"doc_id": document["doc_id"], "title": document["title"]},
        )

        kept: list[Document] = []
        for chunk in chunks:
            body = chunk.page_content.strip()
            if len(body) < MIN_CHUNK_CHARS:
                skipped += 1
                continue
            chunk.page_content = f"{document['title']}\n\n{body}"
            kept.append(chunk)
        chunks = kept

        # Сквозная нумерация чанков по всему корпусу.
        for chunk in chunks:
            chunk.metadata["chunk_id"] = global_chunk_id
            chunk.metadata["doc_id"] = document["doc_id"]
            chunk.metadata["title"] = document["title"]
            chunk.metadata.setdefault("section", "")
            global_chunk_id += 1

        all_chunks.extend(chunks)

    if verbose and all_chunks:
        lengths = [len(chunk.page_content) for chunk in all_chunks]
        per_doc = len(all_chunks) / len(corpus)
        print(f"\nВсего чанков: {len(all_chunks)} (в среднем {per_doc:.1f} на статью)")
        print(f"Средняя длина чанка: {sum(lengths) / len(lengths):.0f} символов")
        print(f"Мин/макс длина: {min(lengths)} / {max(lengths)}")

    return all_chunks


if __name__ == "__main__":
    print("Подготовка корпуса нутрициологической литературы\n")
    prepared = prepare_chunks()
    print(f"\nГотово. Чанков подготовлено: {len(prepared)}")
    print(f"Справочник источников: {CORPUS_INDEX_PATH}")
