"""
Parent-child (small-to-big): ищем по фрагменту, в генерацию — абстракт целиком.

Откуда вопрос. Поиск не узкое место: промахов в области 3 из 253 (шаг 65),
а два значимых прироста поиска до ответа не дошли (шаг 30). Если что-то
решается дальше поиска — в grade и генерации, — то контекстом: фрагмент
на 800 символов теряет начало и конец абстракта (цель, выборку, вывод).
Parent-child меняет именно то, что видит генерация, а индекс не трогает.

Конфигурации (все на текущем индексе, 900/150, k=5):

    base       — как в проде; прогоняется ДВАЖДЫ (base, base2): разница двух
                 прогонов одной конфигурации — шум генерации в тот же день
                 (шум прогона 0.016 мерили раньше, но между днями и с дрейфом)
    after      — grade видит фрагменты, как сейчас; оставленные им фрагменты
                 разворачиваются в абстракты (по одному на статью, в порядке
                 выдачи). Grade не дорожает.
    before     — разворот до grade: он судит по абстрактам. Дороже на grade,
                 зато может оставить статью, чей фрагмент сам по себе неясен.

Родитель — заголовок и очищенный абстракт ровно в том виде, в котором
он резался в индекс (``data_preparation.build_corpus``). Абстракты длиннее
``PARENT_MAX_CHARS`` (45 из 1604, самый длинный 32 тыс. символов) — окно
этой длины вокруг найденного фрагмента, иначе один абстракт раздувал бы
промпт впятеро.

Метрики — те же, что в ``run_eval``: correctness (судья по умолчанию —
сравнения в силе, абсолют не называть) и faithfulness (RAGAS по тому
контексту, что ушёл в генерацию). Служебные предложения вырезаются.
Отдельно — длина контекста генерации, токены и время: прирост, если он
есть, покупается контекстом вдвое длиннее.

Порог приёмки задан ДО прогона (``accept``): база — среднее двух прогонов
base по каждому вопросу; нижняя граница 95% интервала correctness выше
и шума судьи (0.010), и разницы двух прогонов base; faithfulness значимо
не просела.

Итог (шаг 68 в EVOLUTION.md): не принят ни один вариант — correctness
after −0.011, before −0.015 при шуме прогона −0.007. Абстракт короткий
(меньше трёх фрагментов), и родитель почти ничего не добавляет. Перемерить
при переходе на полные тексты статей.

    python -m experiments.parent_context              # прогоны + оценка
    python -m experiments.parent_context --metrics    # только оценка сохранённых

Ответы сохраняются по одному файлу на конфигурацию и при повторном запуске
не переснимаются: оборванный прогон продолжается с того, что не снято.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import NutritionAgent, build_agent  # noqa: E402
from agent_state import RAGAgentAnswer  # noqa: E402
from evaluation import evaluate  # noqa: E402
from evaluation_answers import evaluate_answers  # noqa: E402
from make_eval_dataset import DATASET_PATH  # noqa: E402
from measurement import RESEARCH_DIR, paired_bootstrap  # noqa: E402
from providers import get_clients  # noqa: E402
from run_eval import load_answers, save_answers, strip_boilerplate  # noqa: E402

TOP_K = 5
PARENT_MAX_CHARS = 4000
CONFIGS = ("base", "base2", "after", "before")
RESULTS_PATH = RESEARCH_DIR / "experiments_parent_context.csv"

# Порог приёмки — до прогона. Шум судьи на фиксированных ответах (шаг 25).
JUDGE_NOISE_FLOOR = 0.010
# Faithfulness «значимо не просела»: верхняя граница интервала не ниже нуля.


def answers_path(config: str) -> Path:
    return RESEARCH_DIR / f"agent_answers_parent_{config}.json"


def load_parents() -> dict[int, str]:
    """doc_id → «заголовок\\n\\nабстракт», как текст резался в индекс."""
    from data_preparation import build_corpus

    return {int(d["doc_id"]): f"{d['title']}\n\n{d['text']}" for d in build_corpus(verbose=False)}


def parent_text(chunk: dict, parents: dict[int, str]) -> str:
    """Абстракт статьи фрагмента; у длинного — окно вокруг фрагмента."""
    full = parents.get(int(chunk["doc_id"]))
    if full is None:
        return chunk["text"]
    if len(full) <= PARENT_MAX_CHARS:
        return full
    body = chunk["text"].split("\n\n", 1)[-1]
    start = full.find(body[:200])
    if start < 0:
        return chunk["text"]
    title = full.split("\n\n", 1)[0]
    centre = start + len(body) // 2
    lo = max(0, centre - PARENT_MAX_CHARS // 2)
    return f"{title}\n\n…{full[lo:lo + PARENT_MAX_CHARS]}…"


def expand(chunks: list[dict], parents: dict[int, str]) -> list[dict]:
    """По одному родителю на статью, в порядке первого появления."""
    seen: set[int] = set()
    result = []
    for chunk in chunks:
        doc = int(chunk["doc_id"])
        if doc in seen:
            continue
        seen.add(doc)
        result.append({**chunk, "text": parent_text(chunk, parents)})
    return result


class ParentAgent(NutritionAgent):
    """Агент, отдающий дальше поиска абстракты вместо фрагментов."""

    parents: dict[int, str] = {}
    mode: str = "after"

    def _node_retrieve(self, state):
        update = super()._node_retrieve(state)
        if self.mode == "before":
            update["chunks"] = expand(update.get("chunks") or [], self.parents)
        return update

    def _node_grade(self, state):
        update = super()._node_grade(state)
        # Разворачивается только то, что пойдёт в генерацию. При отказе
        # grade фрагменты остаются как есть: генерации не будет.
        if self.mode == "after" and update.get("has_relevant"):
            kept = update.get("chunks") or state.get("chunks") or []
            update["chunks"] = expand(kept, self.parents)
        return update


def make_agent(config: str, parents: dict[int, str]) -> NutritionAgent:
    """Продовый агент; для вариантов — подкласс на ТОМ ЖЕ поисковике и каталоге.

    Поисковик берётся из ``build_agent``, а не собирается заново: у
    ``Retriever`` свои умолчания, и разойдись они с продом — замер посчитал
    бы две правки разом.
    """
    client, model, embed_client = get_clients("polza")
    prod = build_agent(client, model=model, embed_client=embed_client, top_k=TOP_K)
    if config not in ("after", "before"):
        return prod
    agent = ParentAgent(client=client, model=model, retriever=prod.retriever,
                        catalog=prod.catalog, top_k=TOP_K,
                        structured_extract=prod.structured_extract)
    agent.parents, agent.mode = parents, config
    return agent


def run(config: str, dataset: pd.DataFrame, parents: dict[int, str]) -> list[RAGAgentAnswer]:
    agent = make_agent(config, parents)
    started = time.perf_counter()
    answers: list[RAGAgentAnswer] = []

    def work(row):
        return agent.run(str(row["question"]), dataset_row_id=str(row["id"]), remember=False)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(work, row) for _, row in dataset.iterrows()]
        for number, future in enumerate(as_completed(futures), start=1):
            answers.append(future.result())
            if number % 30 == 0:
                print(f"  {config}: {number}/{len(dataset)}", flush=True)
    print(f"  {config}: {time.perf_counter() - started:.0f} с", flush=True)
    return answers


def context_chars(answers: list[RAGAgentAnswer]) -> pd.Series:
    """Сколько символов контекста ушло в генерацию (у отказа — ничего)."""
    return pd.Series({str(a.dataset_row_id): sum(len(c.text) for c in (a.retrieved_chunks or []))
                      for a in answers})


def score(answers: list[RAGAgentAnswer], dataset: pd.DataFrame, client, model) -> pd.DataFrame:
    scored = [a.model_copy(update={"answer": strip_boilerplate(a.answer)}) for a in answers]
    faith = evaluate(scored, dataset, client=client, model=model, compute_faithfulness=True)
    corr = evaluate_answers(scored, dataset, client, model, with_relevancy=False)
    frame = faith[["id", "faithfulness"]].merge(corr[["id", "correctness"]], on="id")
    frame["id"] = frame["id"].astype(str)
    return frame.set_index("id")


def accept(delta_lo: float, aa: float, faith_hi: float) -> bool:
    """Порог приёмки, заданный до прогона (см. docstring модуля)."""
    return delta_lo > max(JUDGE_NOISE_FLOOR, abs(aa)) and faith_hi >= 0


def main() -> None:
    dataset = pd.read_excel(DATASET_PATH)
    dataset["id"] = dataset["id"].astype(str)
    parents = load_parents()
    print(f"Вопросов: {len(dataset)}, абстрактов: {len(parents)}, "
          f"длиннее {PARENT_MAX_CHARS}: {sum(len(t) > PARENT_MAX_CHARS for t in parents.values())}")

    answers = {}
    for config in CONFIGS:
        path = answers_path(config)
        if path.exists():
            answers[config] = load_answers(path)
            print(f"{config}: из файла ({len(answers[config])})")
        elif "--metrics" in sys.argv:
            print(f"{config}: ответов нет — запустите без --metrics")
            return
        else:
            print(f"{config}: прогон агента", flush=True)
            answers[config] = run(config, dataset, parents)
            save_answers(answers[config], path)

    client, model, _ = get_clients("polza")
    frames = {}
    for config in CONFIGS:
        print(f"Оценка: {config}", flush=True)
        frames[config] = score(answers[config], dataset, client, model)
        frames[config]["context"] = context_chars(answers[config])

    rows = [frame.assign(config=config).reset_index() for config, frame in frames.items()]
    pd.concat(rows).to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    print("\nСредние:")
    for config, frame in frames.items():
        print(f"  {config:<7} correctness {frame.correctness.mean():.3f}  "
              f"faithfulness {frame.faithfulness.mean():.3f}  "
              f"контекст {frame.context.mean():.0f} симв.")

    base = (frames["base"][["correctness", "faithfulness"]]
            + frames["base2"][["correctness", "faithfulness"]]) / 2
    aa, aa_lo, aa_hi = paired_bootstrap(frames["base"].correctness, frames["base2"].correctness)
    print(f"\nШум генерации (base2 − base): correctness {aa:+.3f} [{aa_lo:+.3f}, {aa_hi:+.3f}]")

    print("\nПротив среднего двух base, 95% ДИ:")
    for config in ("after", "before"):
        d, lo, hi = paired_bootstrap(base.correctness, frames[config].correctness)
        fd, flo, fhi = paired_bootstrap(base.faithfulness, frames[config].faithfulness)
        verdict = "ПРИНЯТЬ" if accept(lo, aa, fhi) else "не принимать"
        print(f"  {config:<7} correctness {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"faithfulness {fd:+.3f} [{flo:+.3f}, {fhi:+.3f}]  → {verdict}")
    print(f"\nПо вопросам → {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
