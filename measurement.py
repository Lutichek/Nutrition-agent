"""
Общая обвязка замеров: пути, колонки метрик, статистика.

Зачем модуль появился. Скрипты экспериментов начали импортировать друг
у друга: `experiments_chunking` брал `paired_bootstrap` и `METRIC_COLUMNS`
из `experiments`, то есть чтобы померить размер чанка, питон грузил модуль
про стратегии поиска со всеми его константами и путями. `experiments.py`
незаметно стал библиотекой, оставаясь при этом скриптом.

Побочные следствия, которые из-за этого накопились:

* `RESEARCH_DIR` был объявлен в девяти файлах;
* `paired_bootstrap` — в двух, одинаковый по смыслу и отдельный по коду;
* `METRIC_COLUMNS` — в двух, и **разного состава**: в `experiments_grade`
  не было `map`, поэтому одна и та же сводка в разных скриптах печатала
  разный набор колонок;
* перевод вопросов на английский писали в один и тот же `queries_en.json`
  две разные функции — `experiments.translate_questions`
  и `build_qrels.load_queries_en`.

Правило, по которому теперь разложено: **артефакт принадлежит модулю,
который его создаёт** (`qrels.csv` — за `build_qrels`, ответы агента —
за `run_eval`, датасет — за `make_eval_dataset`), а всё, чем пользуются
несколько скриптов, живёт здесь.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

# Выгрузки замеров живут в _research/dataset/ — в репозиторий не идут,
# пересобираются скриптами. В репозитории только журнал EVOLUTION.md.
RESEARCH_DIR = HERE / "_research" / "dataset"

# Английские переводы вопросов. Общий артефакт: и сборка пула, и сравнение
# стратегий поиска обязаны работать одними и теми же запросами, иначе
# разница между конфигурациями смешается с разницей переводов.
QUERIES_PATH = RESEARCH_DIR / "queries_en.json"

# Что именно нашла каждая конфигурация: {конфигурация: {id вопроса: [chunk_id]}}.
# Нужен, чтобы пересчитать метрики под другой разметкой, не платя заново
# за поиск: реранкинг — это вызов LLM на каждый вопрос.
RUNS_PATH = RESEARCH_DIR / "experiments_retrieval_runs.json"

# Порядок колонок в сводках. Один на все скрипты: раньше в одном из них
# не было map, и таблицы молча различались составом.
METRIC_COLUMNS = ["precision", "recall", "f1", "mrr", "map", "ndcg"]

# Сколько ресэмплов делаем для доверительного интервала разницы.
BOOTSTRAP_SAMPLES = 5000

# Сколько кандидатов набираем до слияния и реранкинга. Общее для всех
# конфигураций, чтобы сравнивались стратегии, а не размеры выборки.
CANDIDATE_K = 40


# ────────────────────────────────────────────────────────────
# Статистика
# ────────────────────────────────────────────────────────────


def paired_bootstrap(
    baseline: pd.Series,
    variant: pd.Series,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Доверительный интервал разницы двух конфигураций на одних вопросах.

    Зачем это нужно: на 80 вопросах разница в 0.013 recall — это ровно один
    вопрос. Без интервала такую разницу легко принять за улучшение и потащить
    в прод лишний приём, который на самом деле ничего не даёт.

    Бутстрэп парный (ресэмплим вопросы, а не конфигурации), потому что обе
    конфигурации прогнаны на одном и том же датасете — их ошибки коррелируют.

    ⚠️ Меряет только шум ВЫБОРКИ. Собственная нестабильность судьи сюда
    не входит, она замерена отдельно (`experiments/judge_noise.py`):
    порог различимости 0.010 для correctness, 0.004 для relevancy.
    Интервал уже этого порога ничего не доказывает.

    Вопросы, где хоть одна сторона не измерена, выбрасываются парой.
    Иначе один NaN отравляет всю разницу, и на выходе получается
    `nan [nan, nan]` — ровно это и случилось при сравнении промптов
    генерации, где faithfulness не посчиталась у 5 вопросов из 210.
    Само по себе это ещё полбеды, но автоматический вердикт сравнил
    NaN с нулём, получил `False` и напечатал «ПРОСЕЛА» — то есть
    отсутствие оценки превратилось в приговор.

    Returns:
        (средняя разница, нижняя граница 95% ДИ, верхняя граница).
    """
    pair = pd.concat([baseline, variant], axis=1).dropna()
    if pair.empty:
        return float("nan"), float("nan"), float("nan")

    dropped = len(baseline) - len(pair)
    if dropped:
        print(f"    (из сравнения выпало {dropped} вопросов без оценки)")

    difference = (pair.iloc[:, 1] - pair.iloc[:, 0]).to_numpy()
    rng = np.random.default_rng(seed)

    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    means = difference[indices].mean(axis=1)

    return (
        float(difference.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def report_significance(
    per_question: dict[str, pd.DataFrame],
    pairs: list[tuple[str, str]],
    metrics: tuple[str, ...] = ("recall", "mrr", "ndcg"),
) -> None:
    """Сравнить пары конфигураций и сказать, значима ли разница.

    Пары должны отличаться ровно одним приёмом — тогда разницу можно
    отнести к нему, а не к сумме изменений.
    """
    print("=" * 62)
    print("ЗНАЧИМОСТЬ РАЗНИЦЫ (парный бутстрэп, 95% ДИ)")
    print("=" * 62)

    for baseline_name, variant_name in pairs:
        if baseline_name not in per_question or variant_name not in per_question:
            continue

        print(f"\n{variant_name}")
        print(f"  против: {baseline_name}")

        baseline = per_question[baseline_name].set_index("id")
        variant = per_question[variant_name].set_index("id")
        common = baseline.index.intersection(variant.index)

        for metric in metrics:
            mean, low, high = paired_bootstrap(
                baseline.loc[common, metric], variant.loc[common, metric]
            )
            # Интервал, накрывающий ноль, означает «не отличили от шума».
            verdict = "значимо" if (low > 0 or high < 0) else "в пределах шума"
            print(f"    {metric:<8} {mean:+.3f}  [{low:+.3f}, {high:+.3f}]  {verdict}")


# ────────────────────────────────────────────────────────────
# Перевод запросов (общий кэш на все замеры)
# ────────────────────────────────────────────────────────────


def load_queries_en(questions: list[str], client: Any, model: str) -> list[str]:
    """Английские переводы вопросов тем же промптом, что и в агенте.

    Кэшируется на диск: перевод не зависит от конфигурации поиска, а платить
    за него на каждом прогоне незачем. Кэш общий намеренно — и сборка пула,
    и сравнение стратегий обязаны искать одинаковыми запросами.

    Раньше эту работу делали две функции в разных модулях, писавшие
    в один файл. Пока промпты совпадали, расхождения не было видно;
    разойдись они — пул собрался бы одними запросами, а замеры шли бы
    другими, и заметить это было бы нечем.
    """
    from agent import REWRITE_PROMPT

    if QUERIES_PATH.exists():
        cached = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
        if len(cached) == len(questions):
            print(f"Переводы взяты из кэша: {QUERIES_PATH.name}")
            return cached

    print(f"Перевожу {len(questions)} вопросов на английский...")

    def work(question: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
                temperature=0.0,
            )
            return (response.choices[0].message.content or question).strip().strip('"')
        except Exception:
            # Не смогли перевести — ищем как есть; это честнее, чем терять вопрос.
            return question

    with ThreadPoolExecutor(max_workers=6) as pool:
        translated = list(pool.map(work, questions))

    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUERIES_PATH.write_text(json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено → {QUERIES_PATH.name}")
    return translated
