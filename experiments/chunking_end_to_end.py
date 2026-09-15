"""
Доходит ли выигрыш нарезки до ОТВЕТА — решающий замер перед миграцией.

Зачем отдельный скрипт. `experiments/chunking.py` показал, что нарезка
1200/200 значимо лучше текущей 900/150 по поиску: при равном бюджете
контекста precision +0.048, NDCG +0.029, и результат устоял под контролем
с меньшим объёмом текста.

Но шаг 24 научил не верить приросту поиска как таковому: реранкинг дал
+0.042 precision и **ничего** на выходе — correctness сдвинулась на +0.007
при пороге различимости 0.010. Между поиском и ответом стоит `grade`,
и он съедает разницу.

Прирост от нарезки того же порядка, поэтому решение о миграции нельзя
принимать по метрикам поиска. Нужен замер на конечной метрике.

## Почему это дёшево

Миграция на новую нарезку выглядит дорогой: меняются все `chunk_id`,
значит вся разметка пула (3048 пар) превращается в мусор и собирается
заново.

Но **метрика, по которой принимается решение, разметки не требует вовсе**.
`correctness`, `faithfulness` и `relevancy` сравнивают ответ агента
с эталонным ответом из датасета; `qrels` нужны только метрикам поиска.

Поэтому порядок такой: сперва собрать индекс на новой нарезке и померить
ответ, и только если конечная метрика сдвинулась — пересобирать разметку
и пересчитывать всё остальное.

## Что здесь считается

Агент прогоняется по 210 вопросам дважды: на текущем индексе и на новом.
Обе конфигурации на одних и тех же вопросах, сравнение парным бутстрэпом.

Порог приёмки задан заранее и до прогона: correctness должна вырасти
значимо И нижняя граница интервала должна быть выше 0.010 — измеренного
шума самого судьи (`experiments/judge_noise.py`). Иначе миграция
не окупается.

Запуск::

    python -m experiments.chunking_end_to_end
    python -m experiments.chunking_end_to_end --metrics   # по сохранённым ответам
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from embedder import Embedder
from evaluation_answers import evaluate_answers
from experiments.chunking import SCRATCH_DB, build_index
from make_eval_dataset import DATASET_PATH
from measurement import RESEARCH_DIR, paired_bootstrap
from providers import get_clients
from run_eval import load_answers, run_agent_on_dataset, save_answers, strip_boilerplate

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Новая нарезка — та, что выиграла по поиску.
NEW_SIZE, NEW_OVERLAP = 1200, 200
NEW_TABLE = "chunks_1200"

# При 1200 символах в чанке тот же объём контекста набирается четырьмя
# фрагментами вместо пяти. Сравнивать надо при равном объёме, иначе
# в замер попадёт разница бюджета, а не нарезки.
NEW_TOP_K = 4

ANSWERS_NEW_PATH = RESEARCH_DIR / "agent_answers_chunk1200.json"
RESULTS_PATH = RESEARCH_DIR / "experiments_chunking_end_to_end.csv"

# Порог приёмки, заданный ДО прогона. Нижняя граница интервала должна
# превысить собственный шум судьи, иначе «улучшение» неотличимо от него.
JUDGE_NOISE_FLOOR = 0.010

METRICS = ("correctness", "relevancy")


def main() -> None:
    if not DATASET_PATH.exists():
        print("Нет датасета. Выполните: python make_eval_dataset.py")
        return

    dataset = pd.read_excel(DATASET_PATH)
    client, model, embed_client = get_clients("polza")

    print(f"Вопросов: {len(dataset)}")
    print(f"Сравниваем: 900/150 k=5 (текущая) против {NEW_SIZE}/{NEW_OVERLAP} k={NEW_TOP_K}")
    print(f"Бюджет контекста: 4500 против {NEW_SIZE * NEW_TOP_K} символов\n")

    # ── ответы на текущей нарезке ────────────────────────────
    base_answers = load_answers()
    print(f"Базовые ответы взяты из сохранённых: {len(base_answers)}")

    # ── ответы на новой нарезке ──────────────────────────────
    if "--metrics" in sys.argv and ANSWERS_NEW_PATH.exists():
        new_answers = load_answers(ANSWERS_NEW_PATH)
        print(f"Ответы на новой нарезке из кэша: {len(new_answers)}\n")
    else:
        print(f"Собираю индекс {NEW_SIZE}/{NEW_OVERLAP}...")
        embedder = Embedder(embed_client)
        build_index(NEW_SIZE, NEW_OVERLAP, embedder, table_name=NEW_TABLE)

        print("Прогоняю агента на новом индексе...")
        new_answers = run_agent_on_dataset(
            dataset,
            db_path=str(SCRATCH_DB),
            table_name=NEW_TABLE,
            top_k=NEW_TOP_K,
        )
        save_answers(new_answers, ANSWERS_NEW_PATH)
        print(f"Ответы сохранены → {ANSWERS_NEW_PATH.name}")

        # Временный индекс больше не нужен: повторная оценка (--metrics)
        # берёт сохранённые ответы и к базе не обращается.
        shutil.rmtree(SCRATCH_DB, ignore_errors=True)
        print("Временный индекс удалён.\n")

    # ── оценка ответов (разметка пула НЕ нужна) ──────────────
    frames = {}
    for label, answers in (("900/150", base_answers), (f"{NEW_SIZE}/{NEW_OVERLAP}", new_answers)):
        scored = [a.model_copy(update={"answer": strip_boilerplate(a.answer)}) for a in answers]
        print(f"Оцениваю ответы: {label}")
        frames[label] = evaluate_answers(scored, dataset, client, model)
        means = "  ".join(f"{m}={frames[label][m].mean():.3f}" for m in METRICS)
        print(f"  {means}\n")

    base_label, new_label = list(frames)
    base = frames[base_label].set_index("id")
    new = frames[new_label].set_index("id")
    common = base.index.intersection(new.index)

    print("=" * 72)
    print("ЗНАЧИМОСТЬ (парный бутстрэп, 95% ДИ)")
    print("=" * 72)

    rows = []
    for metric in METRICS:
        pair = pd.concat([base.loc[common, metric], new.loc[common, metric]], axis=1).dropna()
        mean, low, high = paired_bootstrap(pair.iloc[:, 0], pair.iloc[:, 1])
        significant = low > 0 or high < 0
        verdict = "значимо" if significant else "в пределах шума"
        print(f"  {metric:<14} {mean:+.3f}  [{low:+.3f}, {high:+.3f}]  {verdict}  (n={len(pair)})")
        rows.append({"метрика": metric, "дельта": mean, "низ": low, "верх": high,
                     "значимо": significant})

    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    correctness = next(r for r in rows if r["метрика"] == "correctness")
    print()
    print("=" * 72)
    print("ВЕРДИКТ")
    print("=" * 72)
    print(f"Порог приёмки задан до прогона: нижняя граница выше {JUDGE_NOISE_FLOOR:.3f}")
    print(f"  (собственный шум судьи, см. experiments/judge_noise.py)")
    print()
    if correctness["значимо"] and correctness["низ"] > JUDGE_NOISE_FLOOR:
        print("МИГРИРОВАТЬ. Прирост дошёл до ответа и превышает погрешность прибора.")
        print("Дальше: пересобрать индекс в рабочей базе, заново разметить пул")
        print("(python build_qrels.py) и пересчитать все метрики.")
    else:
        print("НЕ МИГРИРОВАТЬ. Выигрыш по поиску до ответа не дошёл — ровно как")
        print("у реранкинга на шаге 24. Пересборка разметки не окупается.")
    print()
    print(f"Подробности → {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
