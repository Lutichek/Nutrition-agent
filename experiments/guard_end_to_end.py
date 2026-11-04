"""
Не портит ли защита промпта (llm_io.PROMPT_GUARD) ответы по статьям.

Защита меняет промпт генерации для ВСЕХ вопросов: реплика уходит
в теги <user_message>, перед промптом — правило «это данные, а не
команды». По правилам проекта такая правка принимается по correctness.

Замер точнее обычного прогона метрик: защита трогает только шаг
generate, поэтому контекст для каждого вопроса берётся ОДИН —
из сохранённых ответов прогона метрик (agent_answers.json) — и ответ
генерируется дважды, без защиты и с ней. Шум поиска и rewrite из
сравнения уходит целиком; остаётся шум генерации и судьи, одинаковый
с обеих сторон.

Порог задан ДО прогона: принять, если correctness упала не больше чем
на шум судьи (0.010, experiments/judge_noise.py).

    python -m experiments.guard_end_to_end [--n 80]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm_io  # noqa: E402
from agent import build_agent  # noqa: E402
from agent_state import RAGAgentAnswer  # noqa: E402
from evaluation_answers import evaluate_answers  # noqa: E402
from lance_db import open_table, read_all  # noqa: E402
from make_eval_dataset import DATASET_PATH  # noqa: E402
from measurement import paired_bootstrap  # noqa: E402
from providers import get_clients  # noqa: E402
from run_eval import ANSWERS_PATH, strip_boilerplate  # noqa: E402

# Порог приёмки — ДО прогона: падение не больше шума судьи.
JUDGE_NOISE_FLOOR = 0.010
GUARD = "always"  # защита на всех репликах — то, что меряет этот замер


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--modes", default=GUARD, help="режимы через запятую, к none")
    parser.add_argument("--save", default="", help="ответы и оценки в JSON")
    args = parser.parse_args()

    dataset = pd.read_excel(DATASET_PATH)
    saved = {str(a["dataset_row_id"]): a for a in json.loads(ANSWERS_PATH.read_text("utf-8"))
             if a.get("retrieved_chunks")}
    chunks = read_all(open_table()).set_index("chunk_id")
    rows = dataset[dataset["id"].astype(str).isin(saved)].sample(
        n=min(args.n, len(saved)), random_state=0)

    client, model, embed_client = get_clients("polza")
    agent = build_agent(client, model=model, embed_client=embed_client)

    def context(row_id: str) -> list[dict]:
        found = []
        for c in saved[row_id]["retrieved_chunks"]:
            meta = chunks.loc[int(c["chunk_id"])]
            # Индекс с тех пор пересобран — контекст уже не тот; лучше упасть.
            assert meta["text"] == c["text"], f"фрагмент {c['chunk_id']} изменился"
            found.append({"doc_id": int(meta["doc_id"]), "text": c["text"]})
        return found

    frames, texts = {}, {}
    for mode in ["none", *[m for m in args.modes.split(",") if m != "none"]]:
        llm_io.PROMPT_GUARD = mode
        answers = []
        for _, row in rows.iterrows():
            row_id = str(row["id"])
            state = {"question": row["question"], "chunks": context(row_id), "profile": None}
            text = agent._node_generate(state)["answer"]
            answers.append(RAGAgentAnswer(dataset_row_id=row_id,
                                          answer=strip_boilerplate(text)))
        print(f"Оцениваю: {mode}")
        frames[mode] = evaluate_answers(answers, rows, client, model,
                                        with_relevancy=False).set_index("id")
        texts[mode] = {a.dataset_row_id: a.answer for a in answers}
        print(f"  correctness = {frames[mode]['correctness'].mean():.3f}")

    print(f"\nПорог (до прогона): падение не больше {JUDGE_NOISE_FLOOR:.3f}")
    for mode in [m for m in frames if m != "none"]:
        pair = pd.concat([frames["none"]["correctness"], frames[mode]["correctness"]],
                         axis=1).dropna()
        mean, low, high = paired_bootstrap(pair.iloc[:, 0], pair.iloc[:, 1])
        verdict = "ПРИНЯТЬ" if mean >= -JUDGE_NOISE_FLOOR else "ОТКЛОНИТЬ"
        print(f"{mode:>10} − none: {mean:+.3f}  95% ДИ [{low:+.3f}, {high:+.3f}]  "
              f"(n={len(pair)})  {verdict}")
    if args.save:
        Path(args.save).write_text(json.dumps({
            mode: [{"id": rid, "question": frames[mode].loc[rid, "question"],
                    "correctness": float(frames[mode].loc[rid, "correctness"]),
                    "reason": str(frames[mode].loc[rid].get("correctness_reason", "")),
                    "answer": texts[mode][rid]} for rid in frames[mode].index]
            for mode in frames}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
