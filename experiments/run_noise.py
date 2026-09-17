"""
Отделить эффект конфигурации от шума прогона по контрольной группе.

Зачем. Сквозные прогоны делаются в разное время, а генерация стохастична:
та же продовая конфигурация в двух прогонах дала correctness 0.860 и 0.876.
Значит наблюдаемая дельта между конфигурациями — это

    эффект конфигурации + шум прогона + дрейф остального кода

и парный бутстрэп по вопросам их не разделяет: он ресэмплит вопросы,
а не прогоны. Порог различимости судьи (0.010) тоже не помогает — он
измерен на ФИКСИРОВАННЫХ ответах и про стохастичность генерации
не знает вовсе.

Приём. Контрольная группа ищется в самих данных: берём вопросы, где обе
конфигурации отдали в генерацию **один и тот же контекст**. Повлиять там
было не на что, значит вся разница — шум и дрейф. Остальные вопросы —
«лечение»: шум плюс эффект. Разность между группами и есть эффект,
не объяснимый шумом.

⚠️ Тонкость, на которой уже ошиблись: «один и тот же контекст» — это
состав **И ПОРЯДОК** чанков. Порядок фрагментов идёт в промпт генерации,
поэтому перестановка — это уже эффект проверяемой конфигурации. Считая
по одному составу, в контроль пускаешь переставленные и завышаешь оценку
шума: на выключении BM25 таких было 15 из 74, и оценка шума выходила
вдвое больше настоящей (+0.014 против +0.008).

⚠️ Чего этот приём НЕ умеет: чистить метрики ПОИСКА. Контрольная группа
определена через совпадение выдачи, поэтому дельта поиска в ней равна
нулю по построению. Если перед поиском стоит вызов модели (здесь — узел
``rewrite``), сквозные precision и recall наследуют шум прогона так же,
как correctness, и вычесть его нечем. Решать по поиску в таком случае —
только детерминированными замерами, где запрос берётся из кэша один
на все конфигурации (образец — ``experiments/bm25_weight.py``).

Замер бесплатный: ни одного вызова модели, всё считается по сохранённым
ответам и таблицам метрик.

Запуск::

    python -m experiments.run_noise                       # база против --hybrid
    python -m experiments.run_noise --rerank              # база против --rerank
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from measurement import BOOTSTRAP_SAMPLES, paired_bootstrap
from run_eval import (
    ANSWERS_HYBRID_PATH,
    ANSWERS_PATH,
    ANSWERS_RERANK_PATH,
    METRICS_HYBRID_PATH,
    METRICS_PATH,
    METRICS_RERANK_PATH,
    load_answers,
)

GENERATION_METRICS = ["faithfulness", "correctness", "relevancy"]


def contexts(answers) -> dict[str, tuple[int, ...]]:
    """Контекст каждого вопроса: чанки В ТОМ ПОРЯДКЕ, в каком ушли в промпт.

    Кортеж, а не множество — см. предупреждение в докстроке модуля.
    """
    return {
        str(answer.dataset_row_id): tuple(
            chunk.chunk_id for chunk in (answer.retrieved_chunks or [])
        )
        for answer in answers
    }


def _load(metrics_path) -> pd.DataFrame:
    frame = pd.read_csv(metrics_path)
    return frame.assign(id=frame["id"].astype(str)).set_index("id")


def between_groups(
    deltas: pd.Series,
    control: list[str],
    treated: list[str],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Непарный бутстрэп разности средних дельт двух групп.

    Группы независимы (вопросы разные), поэтому ресэмплим каждую отдельно.
    """
    rng = np.random.default_rng(seed)
    a = deltas.reindex(control).dropna().to_numpy()
    b = deltas.reindex(treated).dropna().to_numpy()
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan"), float("nan")

    draws = np.empty(samples)
    for i in range(samples):
        draws[i] = (
            rng.choice(b, size=len(b), replace=True).mean()
            - rng.choice(a, size=len(a), replace=True).mean()
        )
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(b.mean() - a.mean()), float(low), float(high)


def main() -> None:
    rerank = "--rerank" in sys.argv[1:]
    variant_answers = ANSWERS_RERANK_PATH if rerank else ANSWERS_HYBRID_PATH
    variant_metrics = METRICS_RERANK_PATH if rerank else METRICS_HYBRID_PATH
    variant_name = "реранкинг" if rerank else "гибрид (BM25+RRF)"

    for path in (ANSWERS_PATH, variant_answers, METRICS_PATH, variant_metrics):
        if not path.exists():
            print(f"Нет файла {path.name}. Сначала прогоните run_eval.py.")
            return

    base_ctx = contexts(load_answers(ANSWERS_PATH))
    var_ctx = contexts(load_answers(variant_answers))
    base = _load(METRICS_PATH)
    var = _load(variant_metrics)

    common = sorted(set(base.index) & set(var.index) & set(base_ctx) & set(var_ctx))
    control = [i for i in common if base_ctx[i] == var_ctx[i]]
    treated = [i for i in common if base_ctx[i] != var_ctx[i]]

    # Сколько вопросов различаются ТОЛЬКО порядком: если их много, значит
    # приём в основном переставляет, а не находит другое.
    reordered = [i for i in treated if set(base_ctx[i]) == set(var_ctx[i])]

    print("=" * 74)
    print(f"БАЗА против «{variant_name}»")
    print("=" * 74)
    print(f"вопросов: {len(common)}")
    print(f"  контекст совпал (состав и порядок): {len(control):>3}  — КОНТРОЛЬ")
    print(f"  контекст различается:               {len(treated):>3}")
    print(f"     из них только перестановкой:     {len(reordered):>3}")
    print()

    header = f"{'метрика':<14}{'контроль':>22}{'изменился':>22}{'разность':>26}"
    print(header)
    print("-" * len(header))

    for metric in GENERATION_METRICS:
        if metric not in base.columns or metric not in var.columns:
            continue
        deltas = (var[metric] - base[metric]).dropna()

        control_mean, *_ = paired_bootstrap(
            base.loc[base.index.intersection(control), metric].dropna(),
            var.loc[var.index.intersection(control), metric].dropna(),
        )
        treated_delta = deltas.reindex(treated).dropna()
        mean, low, high = between_groups(deltas, control, treated)

        verdict = "ЗНАЧИМО" if (low > 0 or high < 0) else "шум"
        n_control = len(deltas.reindex(control).dropna())
        print(
            f"{metric:<14}"
            f"{f'{control_mean:+.3f} (n={n_control})':>22}"
            f"{f'{treated_delta.mean():+.3f} (n={len(treated_delta)})':>22}"
            f"{f'{mean:+.3f} [{low:+.3f}, {high:+.3f}] {verdict}':>26}"
        )

    print()
    print("Как читать. «Контроль» — оценка шума прогона и дрейфа кода: эффекта")
    print("там нет по построению. «Разность» — то, что шумом не объясняется.")
    print("Значимой разности нет — значит влияние на ОТВЕТ не доказано, каким")
    print("бы убедительным ни выглядело сырое сравнение средних.")
    print()
    print("По метрикам ПОИСКА этот приём вывода не даёт — см. докстроку модуля.")


if __name__ == "__main__":
    main()
