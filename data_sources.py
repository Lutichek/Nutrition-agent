"""
Загрузка исходных данных проекта: справочник блюд + корпус литературы.

Два независимых источника, каждый под свою задачу:

1. **USDA FNDDS (Survey Foods)** — справочник ГОТОВЫХ БЛЮД с уже посчитанным
   КБЖУ и весами бытовых порций. Не сырые продукты («лук», «куриная грудка»),
   а именно блюда: «Eggplant and meat casserole», «Fish sandwich, fried».

   Почему именно эта выгрузка, а не рецепты с ингредиентами: в рецептах КБЖУ
   пришлось бы считать самим, сопоставляя каждую строку («8 cloves chopped»,
   «¼ cup», «To taste») с записью справочника. Наивное сопоставление ошибалось
   на самых весомых ингредиентах — Chicken → «Chicken spread», Yogurt →
   «Tofu yogurt», — то есть промахивалось там, где цена ошибки максимальна.
   В FNDDS этой задачи нет: КБЖУ уже посчитан на 100 г и на порцию.

   Берём полную CSV-выгрузку, а не API: она бесплатна, не требует ключа
   (DEMO_KEY выдыхается после ~8 запросов), не имеет лимитов и делает сборку
   воспроизводимой офлайн.

2. **PubMed E-utilities** — абстракты статей по нутрициологии для RAG-корпуса.
   Отсюда агент берёт обоснования: сколько белка нужно при дефиците калорий,
   как быстро безопасно худеть, что говорят обзоры про клетчатку. У каждого
   фрагмента есть PMID, поэтому ответ можно снабдить ссылкой на источник.

Запуск::

    python data_sources.py             # скачать всё, чего ещё нет
    python data_sources.py --usda      # только справочник блюд
    python data_sources.py --pubmed    # только корпус литературы
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
USDA_DIR = RAW / "usda"
SURVEY_DIR = USDA_DIR / "survey"
PUBMED_PATH = RAW / "pubmed_abstracts.json"

USER_AGENT = "nutrition-agent/0.1"

# Выгрузка FNDDS 2019-2020. Дата в имени файла — часть URL, не опечатка.
FNDDS_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2022-10-28.zip"

# Файлы выгрузки, которые реально используются в foods.py.
FNDDS_REQUIRED = [
    "food.csv",               # блюдо: fdc_id + описание
    "food_nutrient.csv",      # нутриенты на 100 г
    "food_portion.csv",       # веса бытовых порций («1 cup» = 246 г)
    "nutrient.csv",           # справочник нутриентов
    "survey_fndds_food.csv",  # связь блюда с категорией WWEIA
    "wweia_food_category.csv",
]

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Темы корпуса. Каждая — отдельный поисковый запрос к PubMed.
# Фильтр по типу публикации намеренно жёсткий: нужны обзоры, мета-анализы
# и руководства, а не единичное исследование на двенадцати добровольцах.
PUBMED_TOPICS: dict[str, str] = {
    "energy_balance": "energy balance AND body weight regulation",
    "weight_loss_rate": "rate of weight loss AND safety AND obesity",
    "protein_requirements": "dietary protein requirement AND energy restriction",
    "protein_muscle": "protein intake AND lean body mass preservation",
    "macronutrient_composition": "macronutrient composition AND weight loss diets",
    "dietary_fiber": "dietary fiber intake AND health outcomes",
    "sugar_intake": "free sugars intake AND health",
    "dietary_fat_quality": "saturated fat replacement AND cardiovascular risk",
    "sodium": "sodium intake AND blood pressure",
    "meal_frequency": "meal frequency AND body weight",
    "satiety": "dietary protein AND satiety AND appetite regulation",
    "adherence": "diet adherence AND long term weight maintenance",
    "energy_expenditure": "resting metabolic rate AND predictive equations",
    "diet_quality": "diet quality index AND mortality",
    "micronutrient_adequacy": "micronutrient intake AND diet quality",
    "vegetarian_diets": "vegetarian diet AND nutritional adequacy",
    "glycemic_index": "glycemic index AND glycemic load AND health outcomes",
    "hydration": "water intake AND hydration AND health",

    # ── Второе поколение тем ─────────────────────────────────
    # Первые 18 тем — это макроуровень нутрициологии: энергетический баланс,
    # потребность в белке, клетчатка, натрий. Диагностика покрытия
    # (diagnose_corpus.py) показала, что живые люди спрашивают не об этом:
    # 12 из 13 отказов пришлись на конкретные продукты, добавки и бытовые
    # мифы, которых в корпусе почти не было. Хуже всего было с добавками —
    # покрытие 20%. Темы ниже добавлены точно под эти провалы.
    "supplements_omega3": "omega-3 fatty acid supplementation AND health outcomes",
    "supplements_multivitamin": "multivitamin supplementation AND health outcomes",
    "supplements_magnesium": "magnesium supplementation AND sleep OR anxiety",
    "vitamin_d_dosing": "vitamin D supplementation AND dose AND deficiency",
    "iron_deficiency": "iron deficiency anaemia AND dietary iron",
    "calcium_bone": "calcium intake AND bone health",

    "eggs_cholesterol": "egg consumption AND blood cholesterol AND cardiovascular risk",
    "coffee_health": "coffee consumption AND health outcomes",
    "dairy_health": "dairy product consumption AND health outcomes",
    "nuts_health": "nut consumption AND cardiometabolic health",
    "red_processed_meat": "red and processed meat consumption AND health risk",
    "fish_seafood": "fish consumption AND health outcomes",
    "legumes_wholegrain": "whole grain AND legume consumption AND health",

    "gluten_sensitivity": "non-coeliac gluten sensitivity AND diagnosis",
    "lactose_intolerance": "lactose intolerance AND dairy digestion",
    "fodmap": "low FODMAP diet AND irritable bowel syndrome",
    "food_allergy": "food allergy AND elimination diet",

    "alcohol_health": "alcohol consumption AND health outcomes AND body weight",
    "caffeine_effects": "caffeine intake AND performance AND sleep",
    "sweetened_beverages": "sugar sweetened beverages AND health",

    "detox_myths": "detoxification diets AND scientific evidence",
    "ketogenic_diet": "ketogenic diet AND weight loss AND metabolic effects",
    "intermittent_fasting": "intermittent fasting AND time restricted eating",
    "popular_diets": "popular weight loss diets AND comparative effectiveness",

    "adolescent_nutrition": "adolescent nutrition AND dietary intake",
    "child_nutrition": "child nutrition AND dietary patterns AND growth",
    "older_adults": "nutrition in older adults AND healthy ageing",
}

PUBMED_FILTER = (
    "(review[pt] OR meta-analysis[pt] OR systematic review[pt] OR guideline[pt])"
    " AND humans[mh] AND hasabstract"
)

PER_TOPIC = 40

# Абстракт короче этого не даст ни одного полноценного чанка.
MIN_ABSTRACT_CHARS = 400


# ────────────────────────────────────────────────────────────
# Сеть
# ────────────────────────────────────────────────────────────


def _fetch(url: str, timeout: float = 120.0, max_retries: int = 4) -> bytes:
    """Скачать URL с повторами: сеть иногда отдаёт таймаут на ровном месте."""
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            # Экспоненциальная пауза: 1с, 2с, 4с, ...
            time.sleep(2**attempt)

    raise RuntimeError(f"Не удалось скачать {url}: {last_error}")


# ────────────────────────────────────────────────────────────
# 1. Справочник блюд (USDA FNDDS)
# ────────────────────────────────────────────────────────────


def download_usda(force: bool = False) -> Path:
    """Скачать и распаковать выгрузку FNDDS.

    Возвращает папку с CSV. Повторный запуск ничего не качает: архив весит
    ~4.4 МБ и не меняется между релизами.
    """
    missing = [name for name in FNDDS_REQUIRED if not (SURVEY_DIR / name).exists()]
    if not missing and not force:
        print(f"[usda] выгрузка уже на месте: {SURVEY_DIR}")
        return SURVEY_DIR

    SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = USDA_DIR / "survey.zip"

    if not archive_path.exists() or force:
        print(f"[usda] качаю {FNDDS_URL}")
        archive_path.write_bytes(_fetch(FNDDS_URL))
        print(f"[usda] скачано {archive_path.stat().st_size / 1024 / 1024:.1f} МБ")

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if not name.endswith(".csv"):
                continue
            target = SURVEY_DIR / name
            with archive.open(member) as source:
                target.write_bytes(source.read())
            print(f"[usda] распаковано {name} ({target.stat().st_size / 1024:.0f} КБ)")

    still_missing = [name for name in FNDDS_REQUIRED if not (SURVEY_DIR / name).exists()]
    if still_missing:
        raise RuntimeError(f"В выгрузке не хватает файлов: {still_missing}")

    return SURVEY_DIR


# ────────────────────────────────────────────────────────────
# 2. Корпус литературы (PubMed)
# ────────────────────────────────────────────────────────────


def _esearch(query: str, retmax: int) -> list[str]:
    """Найти PMID статей по запросу."""
    url = f"{EUTILS}/esearch.fcgi?" + urllib.parse.urlencode(
        {
            "db": "pubmed",
            "term": f"({query}) AND {PUBMED_FILTER}",
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
        }
    )
    payload = json.loads(_fetch(url, timeout=60.0))
    return payload["esearchresult"].get("idlist", [])


def _efetch(pmids: list[str]) -> list[dict]:
    """Забрать абстракты по списку PMID и разобрать XML."""
    if not pmids:
        return []

    url = f"{EUTILS}/efetch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    )
    root = ET.fromstring(_fetch(url, timeout=120.0))

    articles: list[dict] = []
    for node in root.findall(".//PubmedArticle"):
        # Абстракт бывает структурированным: BACKGROUND / METHODS / RESULTS.
        # Метки сохраняем — они помогают и чанкингу, и читаемости цитаты.
        parts: list[str] = []
        for block in node.findall(".//Abstract/AbstractText"):
            text = "".join(block.itertext()).strip()
            if not text:
                continue
            label = block.get("Label")
            parts.append(f"{label.title()}: {text}" if label else text)

        abstract = "\n\n".join(parts)
        if len(abstract) < MIN_ABSTRACT_CHARS:
            continue

        articles.append(
            {
                "pmid": node.findtext(".//PMID") or "",
                "title": (node.findtext(".//ArticleTitle") or "").strip(),
                "journal": (node.findtext(".//Journal/Title") or "").strip(),
                "year": node.findtext(".//JournalIssue/PubDate/Year") or "",
                "abstract": abstract,
            }
        )

    return articles


def download_pubmed(force: bool = False) -> list[dict]:
    """Собрать корпус абстрактов по темам из ``PUBMED_TOPICS``.

    Результат кладётся в ``data/raw/pubmed_abstracts.json`` одним файлом,
    чтобы сборка индекса не зависела от доступности NCBI.
    """
    if PUBMED_PATH.exists() and not force:
        articles = json.loads(PUBMED_PATH.read_text(encoding="utf-8"))
        print(f"[pubmed] корпус уже на месте: {len(articles)} статей")
        return articles

    PUBMED_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_pmid: dict[str, dict] = {}

    for topic, query in PUBMED_TOPICS.items():
        pmids = _esearch(query, PER_TOPIC)
        # NCBI просит не чаще трёх запросов в секунду без API-ключа.
        time.sleep(0.4)
        articles = _efetch(pmids)
        time.sleep(0.4)

        added = 0
        for article in articles:
            pmid = article["pmid"]
            if not pmid:
                continue
            if pmid in by_pmid:
                # Статья попала в несколько тем — помечаем все: потом видно,
                # какая тема покрыта своими статьями, а какая — заимствованными.
                by_pmid[pmid]["topics"].append(topic)
                continue
            article["topics"] = [topic]
            by_pmid[pmid] = article
            added += 1

        print(f"[pubmed] {topic}: найдено {len(pmids)}, добавлено {added}")

    corpus = sorted(by_pmid.values(), key=lambda article: article["pmid"])
    PUBMED_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pubmed] сохранено {len(corpus)} статей → {PUBMED_PATH}")
    return corpus


def load_pubmed() -> list[dict]:
    """Прочитать сохранённый корпус (скачав его, если файла ещё нет)."""
    if not PUBMED_PATH.exists():
        return download_pubmed()
    return json.loads(PUBMED_PATH.read_text(encoding="utf-8"))


def main() -> None:
    flags = set(sys.argv[1:])
    do_usda = not flags or "--usda" in flags
    do_pubmed = not flags or "--pubmed" in flags

    if do_usda:
        download_usda()
    if do_pubmed:
        download_pubmed()


if __name__ == "__main__":
    main()
