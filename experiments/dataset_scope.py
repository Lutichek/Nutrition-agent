"""
Сколько в датасете вопросов, на которые агент не обязан отвечать.

Откуда вопрос. Разбор нулевых оценок correctness: из семи вопросов, где
агент получил 0.0, **пять вообще не дошли до поиска** — `classify` отправил
их в `off_topic`, и четыре из пяти были отправлены туда правильно:

    «Какой метод диагностики используется для выявления камней в жёлчном пузыре?»
    «Какие программы эффективнее для снижения стресса у подростков?»
    «Какие симптомы уменьшаются при йога-интервенциях?»
    «Что такое подростковый возраст и когда он начинается?»

Это не про питание. Агент честно ответил «помогаю только с питанием»
и получил ноль.

Причина системная: датасет **генерируется из корпуса**, а корпус набирался
поиском по PubMed и затянул статьи по смежным темам. Генератор вопросов
про границы продукта ничего не знает. Значит correctness штрафует агента
за то, что он эти границы соблюдает.

## Почему замер устроен именно так

Судья видит **только вопрос и описание продукта**. Ответа агента он
не видит вовсе — иначе это была бы подгонка: удобно объявить внеобластным
ровно то, на чём агент споткнулся.

По той же причине вопросы **не удаляются** из датасета. Результат
публикуется двумя числами: по всем 210 (сопоставимо с историей замеров)
и по вопросам внутри области (честное качество продукта).

Запуск::

    python -m experiments.dataset_scope
    python -m experiments.dataset_scope --metrics   # по сохранённой разметке
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm.auto import tqdm

from evaluation_answers import _judge
from make_eval_dataset import DATASET_PATH
from measurement import RESEARCH_DIR
from providers import get_polza_client

RESULTS_PATH = RESEARCH_DIR / "dataset_scope.csv"
METRICS_PATH = RESEARCH_DIR / "metrics.csv"

# Независимый судья: не та модель, что маршрутизирует запросы в агенте.
# Иначе замер проверял бы согласие классификатора с самим собой — ровно
# ту ошибку, которую поймали в experiments/qrels_bias.py.
JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"

BATCH = 10

SCOPE_PROMPT = """Ассистент по питанию умеет ровно три вещи:

1. считать норму калорий и БЖУ по параметрам человека;
2. собирать рацион из справочника продуктов;
3. отвечать на вопросы о питании, нутриентах, диетах, продуктах
   и их влиянии на здоровье — опираясь на научные статьи.

Всё остальное он обязан отклонять словами «помогаю только с питанием»:
диагностика и лечение болезней, психотерапия, физические упражнения,
общая физиология и возрастная периодизация, организация здравоохранения.

Ниже пронумерованные вопросы. Определи, на какие из них ассистент ОБЯЗАН
отвечать, то есть какие относятся к питанию.

Вопрос относится к питанию, если ответ на него — про еду, нутриенты,
режим питания или их влияние на организм. Не относится, если еда в нём
лишь фон или не упоминается вовсе.

Вопросы:
{questions}

Верни ТОЛЬКО JSON: {{"in_scope": [номера вопросов про питание]}}
Нумерация с 1. Если подходящих нет — пустой список."""


def judge_scope(dataset: pd.DataFrame, client) -> pd.DataFrame:
    """Разметить вопросы: внутри области продукта или снаружи."""
    records = dataset[["id", "question"]].to_dict("records")
    batches = [records[i : i + BATCH] for i in range(0, len(records), BATCH)]

    def work(batch: list[dict]) -> list[dict]:
        listing = "\n".join(f"{i + 1}. {r['question']}" for i, r in enumerate(batch))
        try:
            verdict = _judge(client, JUDGE_MODEL, SCOPE_PROMPT.format(questions=listing))
            keep = verdict.get("in_scope") or []
            indices = {int(n) - 1 for n in keep if isinstance(n, (int, float))}
        except Exception:
            indices = None  # вердикта нет — не выдаём молчание за «вне области»

        return [
            {
                "id": str(r["id"]),
                "question": r["question"],
                "in_scope": None if indices is None else int(i in indices),
            }
            for i, r in enumerate(batch)
        ]

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in tqdm(pool.map(work, batches), total=len(batches), desc="Разметка области"):
            rows.extend(batch)
    return pd.DataFrame(rows)


def main() -> None:
    if not DATASET_PATH.exists():
        print("Нет датасета. Выполните: python make_eval_dataset.py")
        return

    dataset = pd.read_excel(DATASET_PATH)
    print(f"Вопросов: {len(dataset)}")
    print(f"Судья: {JUDGE_MODEL} (ответы агента он НЕ видит)\n")

    if "--metrics" in sys.argv and RESULTS_PATH.exists():
        scope = pd.read_csv(RESULTS_PATH)
        print(f"Разметка из кэша: {RESULTS_PATH.name}\n")
    else:
        scope = judge_scope(dataset, get_polza_client())
        scope.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
        print(f"\nСохранено → {RESULTS_PATH.name}\n")

    missing = int(scope["in_scope"].isna().sum())
    if missing:
        print(f"⚠️ Судья не ответил на {missing} вопросов — они исключены\n")

    judged = scope.dropna(subset=["in_scope"])
    out_of_scope = judged[judged["in_scope"] == 0]

    print("=" * 72)
    print("ЗАСОРЁННОСТЬ ДАТАСЕТА")
    print("=" * 72)
    print(f"  вопросов размечено:     {len(judged)}")
    print(f"  вне области продукта:   {len(out_of_scope)} ({len(out_of_scope) / len(judged):.1%})")
    print()
    print("Примеры вне области:")
    for q in out_of_scope["question"].head(8):
        print(f"  • {str(q)[:100]}")

    if not METRICS_PATH.exists():
        print(f"\nНет {METRICS_PATH.name} — пересчитать метрики не по чему.")
        return

    metrics = pd.read_csv(METRICS_PATH)
    metrics["id"] = metrics["id"].astype(str)
    merged = metrics.merge(judged[["id", "in_scope"]], on="id", how="inner")
    inside = merged[merged["in_scope"] == 1]

    print()
    print("=" * 72)
    print("ЧТО ЭТО МЕНЯЕТ В ЦИФРАХ")
    print("=" * 72)
    print(f"{'Метрика':<16}{'все вопросы':>14}{'в области':>12}{'разница':>10}")
    print("-" * 72)
    for column in ("correctness", "faithfulness", "relevancy", "precision", "recall"):
        if column not in merged.columns:
            continue
        whole, part = merged[column].dropna(), inside[column].dropna()
        if whole.empty or part.empty:
            continue
        print(f"{column:<16}{whole.mean():>14.3f}{part.mean():>12.3f}"
              f"{part.mean() - whole.mean():>+10.3f}")

    print()
    print("Публиковать надо обе цифры: по всем вопросам — чтобы сравнивать")
    print("с прошлыми замерами, по вопросам в области — как качество продукта.")
    print("Вопросы из датасета не удаляются: выбрасывать неудобные — подгонка.")


if __name__ == "__main__":
    main()
