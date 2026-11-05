"""
Последнее «своё» звено: вопросы датасета писала модель генератора.

После шагов 61 и 63 судья (llama) и эталон (gemini) — другие семейства,
но сами вопросы по-прежнему писал gpt-4o-mini, по конкретному фрагменту.
Модель генератора могла сформулировать вопрос так, как ей же удобно его
понять, перевести (rewrite — тоже gpt-4o-mini) и найти.

Замер парный: для КАЖДОГО из 210 фрагментов вопрос пишется заново той же
процедурой (``make_eval_dataset.QUESTION_PROMPT``, тот же фрагмент,
температура 0.7) — четвёртым семейством, deepseek-v3.2, отличным от всех
остальных ролей. Остальное у обеих сторон одинаково по происхождению:

    роль                    сторона A (было)      сторона B
    вопрос                  gpt-4o-mini           deepseek-v3.2
    эталон                  gemini-2.5-flash      gemini-2.5-flash (к своему вопросу)
    разметка релевантности  llama                 llama (свой пул)
    судья                   llama                 llama
    генератор               gpt-4o-mini           gpt-4o-mini (текущий код)

Пул для разметки B собирается заново — теми же тремя конфигурациями
поиска, что ``build_qrels`` — и размечается llama. Каждый шаг пишет свой
файл и при повторном запуске не пересчитывается: прогоны уже обрывались
на сбоях провайдера и балансе.

    python -m experiments.question_family

Результат (шаг 64 в _research/evolution/RAG.md): подыгрывания нет —
correctness на чужих вопросах +0.060 ``[+0.000, +0.121]``. Поиск «просел»
на 0.12 precision из-за ширины вопросов, а не автора: у deepseek 42 вопроса
с одним релевантным фрагментом против 6, на вопросах одной ширины разницы
нет (``by_breadth``).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_qrels import build_pool, judge_pool, load_qrels  # noqa: E402
from evaluation import evaluate  # noqa: E402
from evaluation_answers import evaluate_answers  # noqa: E402
from make_eval_dataset import (  # noqa: E402
    DATASET_PATH,
    QUESTION_PROMPT,
    REFERENCE_PROMPT,
    _complete,
)
from measurement import RESEARCH_DIR, load_queries_en, paired_bootstrap  # noqa: E402
from providers import get_clients  # noqa: E402
from run_eval import (  # noqa: E402
    ANSWERS_PATH,
    INDEPENDENT_JUDGE,
    INDEPENDENT_QRELS_PATH,
    _independent_references,
    _tagged,
    load_answers,
    run_agent_on_dataset,
    save_answers,
    strip_boilerplate,
)

QUESTION_MODEL = "deepseek/deepseek-v3.2"
REFERENCE_MODEL = "google/gemini-2.5-flash"
TAG = "deepseekq"
QUESTIONS_PATH = RESEARCH_DIR / f"questions_{TAG}.csv"
REFERENCES_PATH = RESEARCH_DIR / f"references_gemini_{TAG}.csv"
QUERIES_PATH = RESEARCH_DIR / f"queries_en_{TAG}.json"
QRELS_PATH = RESEARCH_DIR / f"qrels_llama_{TAG}.csv"
ANSWERS_B_PATH = _tagged(ANSWERS_PATH, TAG)
RESULTS_PATH = RESEARCH_DIR / "experiments_question_family.csv"
METRICS = ("precision", "recall", "ndcg", "mrr", "correctness", "relevancy")


def _chunk(row: dict) -> str:
    chunks = json.loads(row["relevant_text"])
    return chunks[0] if chunks else ""


def _cached_column(path: Path, dataset: pd.DataFrame, column: str, make) -> pd.Series:
    """Колонка по id из файла — или посчитать параллельно и сохранить."""
    if path.exists():
        saved = pd.read_csv(path, dtype={"id": str})
        return dataset["id"].map(dict(zip(saved["id"], saved[column], strict=True)))
    with ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(make, dataset.to_dict("records")))
    pd.DataFrame({"id": dataset["id"], column: values}).to_csv(path, index=False, encoding="utf-8")
    return pd.Series(values, index=dataset.index)


def side_b(dataset: pd.DataFrame, client, model: str, embed_client) -> pd.DataFrame:
    """Датасет B: те же фрагменты, вопросы deepseek, эталоны gemini к ним."""
    questions = _cached_column(
        QUESTIONS_PATH, dataset, "question",
        lambda r: (_complete(client, QUESTION_MODEL, QUESTION_PROMPT.format(chunk=_chunk(r)))
                   or "").strip().strip('"'))
    b = dataset.assign(question=questions)
    references = _cached_column(
        REFERENCES_PATH, b, "reference_answer",
        lambda r: (_complete(client, REFERENCE_MODEL, REFERENCE_PROMPT.format(
            question=r["question"], chunk=_chunk(r))) or "").strip())
    b = b.assign(reference_answer=references)
    empty = int((b["question"].fillna("").str.len() < 15).sum()
                + (b["reference_answer"].fillna("") == "").sum())
    if empty:
        raise RuntimeError(f"Пустых вопросов или эталонов: {empty} — перезапустите")

    if not QRELS_PATH.exists():
        queries = load_queries_en(b["question"].tolist(), client, model, path=QUERIES_PATH)
        pool = build_pool(b, queries, client, model, embed_client)
        judged = judge_pool(pool, client, INDEPENDENT_JUDGE).drop_duplicates(["id", "chunk_id"])
        judged.to_csv(QRELS_PATH, index=False, encoding="utf-8")
    return b


def score(answers, dataset: pd.DataFrame, qrels_path: Path, client) -> pd.DataFrame:
    """Поиск по разметке llama и correctness/relevancy судьёй llama, по id."""
    scored = [a.model_copy(update={"answer": strip_boilerplate(a.answer)}) for a in answers]
    retrieval = evaluate(answers=scored, dataset=dataset, compute_faithfulness=False,
                         qrels=load_qrels(qrels_path))
    judged = evaluate_answers(scored, dataset, client, INDEPENDENT_JUDGE, with_relevancy=True)
    frame = retrieval.assign(id=retrieval["id"].astype(str)).merge(
        judged.assign(id=judged["id"].astype(str))[["id", "correctness", "relevancy"]], on="id")
    return frame.set_index("id")


def by_breadth(frames: dict[str, pd.DataFrame],
               relevant: dict[str, dict[str, set[int]]]) -> pd.DataFrame:
    """Поиск по корзинам «сколько релевантных фрагментов у вопроса».

    Общее среднее сравнивает вопросы разной ширины: у deepseek 42 вопроса
    с ОДНИМ релевантным фрагментом, у gpt-4o-mini — 6. Precision и NDCG
    у узкого вопроса ниже по самой арифметике и по трудности поиска
    иголки, и разница средних — это разница состава, а не автора.
    Сравнимы только строки одной корзины.
    """
    rows = []
    for label, frame in frames.items():
        n_rel = [len(relevant[label].get(i, set())) for i in frame.index]
        bucket = pd.cut(pd.Series(n_rel, index=frame.index), [0, 1, 4, 10_000],
                        labels=["1", "2-4", "5+"])
        for name, group in frame.groupby(bucket, observed=True):
            rows.append({"сторона": label.split(":")[0], "релевантных": name, "n": len(group),
                         **group[["precision", "ndcg", "mrr"]].mean().round(3).to_dict()})
    return pd.DataFrame(rows).sort_values(["релевантных", "сторона"], kind="stable")


def main() -> None:
    dataset = pd.read_excel(DATASET_PATH)
    dataset["id"] = dataset["id"].astype(str)
    client, model, embed_client = get_clients("polza")

    b = side_b(dataset, client, model, embed_client)
    if ANSWERS_B_PATH.exists():
        answers_b = load_answers(ANSWERS_B_PATH)
    else:
        print(f"Агент отвечает на {len(b)} новых вопросов…")
        answers_b = run_agent_on_dataset(b)
        save_answers(answers_b, ANSWERS_B_PATH)

    a = dataset.assign(reference_answer=_independent_references(dataset))
    answers_a = load_answers(_tagged(ANSWERS_PATH, "llama"))

    side = {"A: вопросы gpt-4o-mini": score(answers_a, a, INDEPENDENT_QRELS_PATH, client),
            f"B: вопросы {QUESTION_MODEL}": score(answers_b, b, QRELS_PATH, client)}
    (label_a, fa), (label_b, fb) = side.items()

    relevant = {label: load_qrels(path) for label, path in
                ((label_a, INDEPENDENT_QRELS_PATH), (label_b, QRELS_PATH))}
    print("\nРелевантных фрагментов на вопрос (разметка llama): "
          + ", ".join(f"{k.split(':')[0]} {sum(map(len, v.values())) / len(v):.1f}"
                      for k, v in relevant.items()))
    print("\nПоиск по ширине вопроса (сравнимы строки одной корзины):")
    print(by_breadth(side, relevant).to_string(index=False))

    rows = []
    print(f"\n{'метрика':<12} {'A':>7} {'B':>7}   B − A, 95% ДИ")
    for metric in METRICS:
        pair = pd.concat([fa[metric], fb[metric]], axis=1).dropna()
        delta, low, high = paired_bootstrap(pair.iloc[:, 0], pair.iloc[:, 1])
        rows.append({"метрика": metric, "A": pair.iloc[:, 0].mean(), "B": pair.iloc[:, 1].mean(),
                     "B−A": delta, "низ": low, "верх": high, "n": len(pair)})
        print(f"{metric:<12} {rows[-1]['A']:>7.3f} {rows[-1]['B']:>7.3f}   "
              f"{delta:+.3f} [{low:+.3f}, {high:+.3f}]  n={len(pair)}")
    for label, frame in side.items():
        counts = frame["correctness"].value_counts().reindex([0.0, 0.5, 1.0], fill_value=0)
        print(f"  {label}: correctness 0/0.5/1 = {counts[0.0]}/{counts[0.5]}/{counts[1.0]}")
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nСохранено → {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
