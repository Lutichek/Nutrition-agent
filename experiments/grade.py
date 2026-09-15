"""
Эксперимент с узлом ``grade``: размен между полнотой и честностью отказов.

Проблема, ради которой всё затевалось: мягкий критерий релевантности признаёт
полезным любой фрагмент «про питание». На вопрос «какое влияние оказывают
блюда с историей» агент находил абстракт про средиземноморскую диету и уверенно
отвечал. Из 25 заведомо бессмысленных вопросов он отказывался только на 20.

Очевидная правка — ужесточить критерий. Неочевидная её цена: тот же строгий
критерий может отбросить фрагменты, которые на нормальный вопрос отвечают.
Мерить одну сторону без другой бессмысленно: агент, который молчит всегда,
имеет идеальную честность отказов и нулевую полезность.

Поэтому скрипт гоняет обе конфигурации по ДВУМ датасетам сразу:

* 80 обычных вопросов — сколько раз агент дал ответ и не просел ли поиск;
* 25 бессмысленных    — сколько раз честно отказался.

Запуск::

    python -m experiments.grade
    python -m experiments.grade --n 40     # быстрее и дешевле
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from agent import RAGAgentAnswer, build_agent
from evaluation import evaluate
from measurement import METRIC_COLUMNS, RESEARCH_DIR, paired_bootstrap
from make_eval_dataset import DATASET_PATH, NEGATIVE_PATH
from providers import get_clients
from run_eval import looks_like_refusal

# Выгрузки замеров живут в _research/dataset/ — в репозиторий не идут,
# пересобираются скриптами. В репозитории только журнал EVOLUTION.md.
RESULTS_PATH = RESEARCH_DIR / "experiments_grade.csv"



def run_batch(agent, questions: list[tuple[str, str]], max_workers: int = 4) -> list[RAGAgentAnswer]:
    """Прогнать агента по списку (id, вопрос)."""

    def work(item: tuple[str, str]) -> RAGAgentAnswer:
        row_id, question = item
        return agent.run(question, dataset_row_id=row_id, remember=False)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(work, questions))


def evaluate_mode(name: str, strict: bool, positives: pd.DataFrame, negatives: pd.DataFrame) -> dict:
    """Померить одну конфигурацию ``grade`` на обоих датасетах."""
    print("=" * 62)
    print(name)
    print("=" * 62)

    client, model, embed_client = get_clients("polza")
    agent = build_agent(client, model=model, embed_client=embed_client, strict_grade=strict)

    started = time.perf_counter()

    # ── Обычные вопросы: не разучился ли агент отвечать ──────
    answers = run_batch(agent, list(zip(positives["id"].astype(str), positives["question"].astype(str), strict=True)))
    answered = [a for a in answers if a.retrieved_chunks and not looks_like_refusal(a.answer)]

    metrics = evaluate(
        answers=answers,
        dataset=positives,
        client=client,
        model=model,
        compute_faithfulness=False,
    )

    # ── Бессмысленные вопросы: отказывается ли ───────────────
    negative_answers = run_batch(
        agent, list(zip(negatives["id"].astype(str), negatives["question"].astype(str), strict=True))
    )
    refusal_by_id = {
        str(a.dataset_row_id): looks_like_refusal(a.answer) for a in negative_answers
    }
    negatives = negatives.copy()
    negatives["refused"] = negatives["id"].astype(str).map(refusal_by_id).fillna(False)

    # Трудная часть набора — правдоподобные псевдонаучные вопросы. Откровенную
    # эзотерику («порошок из облаков») отвергает любая конфигурация, и общая
    # цифра за счёт неё завышается.
    subtle = negatives[negatives["subtle"]] if "subtle" in negatives.columns else negatives

    row = {
        "config": name,
        "answer_rate": len(answered) / len(answers),
        "refusal_accuracy": float(negatives["refused"].mean()),
        "refusal_subtle": float(subtle["refused"].mean()),
        "elapsed_s": round(time.perf_counter() - started),
    }
    row.update({column: float(metrics[column].mean()) for column in METRIC_COLUMNS})

    print(
        f"  доля отвеченных: {row['answer_rate']:.1%}, "
        f"отказов: {row['refusal_accuracy']:.1%} "
        f"(на трудных: {row['refusal_subtle']:.1%})"
    )
    print("  " + "  ".join(f"{column}={row[column]:.3f}" for column in METRIC_COLUMNS))
    print()

    return row, metrics[["id", *METRIC_COLUMNS]].copy(), negatives[["id", "refused"]].copy()


def main() -> None:
    if not DATASET_PATH.exists() or not NEGATIVE_PATH.exists():
        print("Нет датасетов. Выполните: python make_eval_dataset.py "
              "и python make_eval_dataset.py --negative")
        return

    positives = pd.read_excel(DATASET_PATH)
    negatives = pd.read_excel(NEGATIVE_PATH)

    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
        positives = positives.head(n)

    print(f"Обычных вопросов: {len(positives)}, бессмысленных: {len(negatives)}\n")

    lenient_row, lenient_pos, lenient_neg = evaluate_mode(
        "Мягкий grade (текущий)", False, positives, negatives
    )
    strict_row, strict_pos, strict_neg = evaluate_mode("Строгий grade", True, positives, negatives)

    summary = pd.DataFrame([lenient_row, strict_row])
    summary.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    print("=" * 62)
    print("РАЗМЕН")
    print("=" * 62)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print()
    print("Значимость разницы (парный бутстрэп, 95% ДИ):")

    # Отказы: считаем по общему набору бессмысленных вопросов.
    common_neg = lenient_neg.set_index("id").index.intersection(strict_neg.set_index("id").index)
    mean, low, high = paired_bootstrap(
        lenient_neg.set_index("id").loc[common_neg, "refused"].astype(float),
        strict_neg.set_index("id").loc[common_neg, "refused"].astype(float),
    )
    verdict = "значимо" if (low > 0 or high < 0) else "в пределах шума"
    print(f"  отказы   {mean:+.3f}  [{low:+.3f}, {high:+.3f}]  {verdict}")

    # Полнота: считаем по обычным вопросам.
    common_pos = lenient_pos.set_index("id").index.intersection(strict_pos.set_index("id").index)
    for metric in ("recall", "precision", "ndcg"):
        mean, low, high = paired_bootstrap(
            lenient_pos.set_index("id").loc[common_pos, metric],
            strict_pos.set_index("id").loc[common_pos, metric],
        )
        verdict = "значимо" if (low > 0 or high < 0) else "в пределах шума"
        print(f"  {metric:<8} {mean:+.3f}  [{low:+.3f}, {high:+.3f}]  {verdict}")

    print()
    print(
        "Строгий вариант принимаем, только если прирост честности отказов\n"
        "значим, а потеря полноты — нет."
    )
    print(f"\nСохранено → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
