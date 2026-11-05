"""
Не завышает ли correctness то, что эталон написала модель генератора.

Вопросы, эталонные ответы и ответы агента писала одна модель — gpt-4o-mini.
Судья другого семейства (llama, шаг 61) убрал завышение со стороны судьи,
но эталон остался «своим» для генератора: ответ может совпасть с ним
по фактам потому, что модель выбирает те же факты из того же фрагмента.

Замер выделяет ровно этот фактор. Всё общее:

* ответы агента — те же 210 (``agent_answers_llama.json``, текущий код);
* судья — тот же, llama-3.3-70b;
* эталон пишется по той же процедуре (``make_eval_dataset.REFERENCE_PROMPT``,
  тот же фрагмент, температура 0.7) — меняется только модель:
  gpt-4o-mini → google/gemini-2.5-flash (третье семейство).

В том же прогоне судья заново оценивает ответы и по прежним эталонам:
расхождение с шагом 61 — шум самого судьи llama.

Метрики поиска по двум разметкам (gpt-4o-mini и llama) — бесплатно, по тем же
выдачам, без вызовов модели.

    python -m experiments.reference_family
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_qrels import load_qrels  # noqa: E402
from evaluation import evaluate  # noqa: E402
from evaluation_answers import evaluate_answers  # noqa: E402
from make_eval_dataset import DATASET_PATH, REFERENCE_PROMPT, _complete  # noqa: E402
from measurement import RESEARCH_DIR, paired_bootstrap  # noqa: E402
from providers import get_clients  # noqa: E402
from run_eval import (  # noqa: E402
    ANSWERS_PATH,
    INDEPENDENT_QRELS_PATH,
    METRICS_PATH,
    _tagged,
    load_answers,
    strip_boilerplate,
)

REFERENCE_MODEL = "google/gemini-2.5-flash"
JUDGE = "meta-llama/llama-3.3-70b-instruct"
REFERENCES_PATH = RESEARCH_DIR / "references_gemini.csv"
RESULTS_PATH = RESEARCH_DIR / "experiments_reference_family.csv"


def gemini_references(dataset: pd.DataFrame, client) -> pd.Series:
    """Эталоны по той же процедуре, что make_eval_dataset, — другой моделью.
    Сохраняются: повторный запуск их не перегенерирует."""
    if REFERENCES_PATH.exists():
        saved = pd.read_csv(REFERENCES_PATH, dtype={"id": str})
        by_id = dict(zip(saved["id"], saved["reference_answer"], strict=True))
        return dataset["id"].astype(str).map(by_id)

    def work(row: dict) -> str:
        chunks = json.loads(row["relevant_text"])
        prompt = REFERENCE_PROMPT.format(question=row["question"], chunk=chunks[0] if chunks else "")
        return (_complete(client, REFERENCE_MODEL, prompt) or "").strip()

    with ThreadPoolExecutor(max_workers=6) as pool:
        references = list(pool.map(work, dataset.to_dict("records")))
    pd.DataFrame({"id": dataset["id"].astype(str), "reference_answer": references}).to_csv(
        REFERENCES_PATH, index=False, encoding="utf-8")
    return pd.Series(references, index=dataset.index)


def main() -> None:
    dataset = pd.read_excel(DATASET_PATH)
    dataset["id"] = dataset["id"].astype(str)
    client, _, _ = get_clients("polza")
    answers = load_answers(_tagged(ANSWERS_PATH, "llama"))
    scored = [a.model_copy(update={"answer": strip_boilerplate(a.answer)}) for a in answers]

    # ── эталоны ──────────────────────────────────────────────────────
    gemini = gemini_references(dataset, client)
    empty = int((gemini.fillna("").str.len() == 0).sum())
    print(f"Эталоны {REFERENCE_MODEL}: пустых {empty} из {len(gemini)}; "
          f"длина {gemini.fillna('').str.len().median():.0f} символов "
          f"(у gpt-4o-mini {dataset['reference_answer'].str.len().median():.0f})")

    # ── судья llama по обоим эталонам, на одних ответах ──────────────
    frames = {}
    for label, refs in (("эталон gpt-4o-mini", dataset["reference_answer"]),
                        (f"эталон {REFERENCE_MODEL}", gemini)):
        judged = evaluate_answers(scored, dataset.assign(reference_answer=refs), client, JUDGE,
                                  with_relevancy=False)
        frames[label] = judged.assign(id=judged["id"].astype(str)).set_index("id")["correctness"]

    mini, other = frames.values()
    pair = pd.concat([mini, other], axis=1).dropna()
    delta, low, high = paired_bootstrap(pair.iloc[:, 0], pair.iloc[:, 1])

    # Шум судьи: тот же судья, те же эталоны и ответы, что в шаге 61.
    step61 = pd.read_csv(_tagged(METRICS_PATH, "llama"), dtype={"id": str}).set_index("id")
    rejudge = pd.concat([step61["correctness"], mini], axis=1).dropna()
    noise_delta, noise_low, noise_high = paired_bootstrap(rejudge.iloc[:, 0], rejudge.iloc[:, 1])
    same = float((rejudge.iloc[:, 0] == rejudge.iloc[:, 1]).mean())

    print("\nCORRECTNESS, судья llama, одни и те же 210 ответов")
    for label, series in frames.items():
        counts = series.value_counts().reindex([0.0, 0.5, 1.0], fill_value=0).astype(int)
        print(f"  {label:<32} {series.mean():.3f}   (0: {counts[0.0]}, "
              f"0.5: {counts[0.5]}, 1: {counts[1.0]}; нет оценки {series.isna().sum()})")
    print(f"  разница (Gemini − gpt-4o-mini)   {delta:+.3f}  95% ДИ [{low:+.3f}, {high:+.3f}]")
    print(f"\nШум судьи llama (повтор шага 61): {noise_delta:+.3f} [{noise_low:+.3f}, "
          f"{noise_high:+.3f}], совпало оценок {same:.1%}")

    # ── поиск по двум разметкам, без вызовов модели ──────────────────
    print("\nПОИСК, те же выдачи, разметка разных моделей")
    rows = []
    for label, path in (("разметка gpt-4o-mini", None), ("разметка llama", INDEPENDENT_QRELS_PATH)):
        metrics = evaluate(answers=scored, dataset=dataset, compute_faithfulness=False,
                           qrels=load_qrels(path))
        row = {k: float(metrics[k].mean()) for k in ("precision", "recall", "ndcg", "mrr")}
        rows.append({"что": label, **row})
        print(f"  {label:<22} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))

    pd.DataFrame([
        {"что": "correctness, эталон gpt-4o-mini", "значение": mini.mean()},
        {"что": f"correctness, эталон {REFERENCE_MODEL}", "значение": other.mean()},
        {"что": "разница", "значение": delta, "низ": low, "верх": high},
        {"что": "шум судьи (повтор шага 61)", "значение": noise_delta,
         "низ": noise_low, "верх": noise_high},
        *[{"что": r["что"], **{k: r[k] for k in ("precision", "recall", "ndcg", "mrr")}}
          for r in rows],
    ]).to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nСохранено → {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
