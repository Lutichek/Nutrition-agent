"""
Прогон метрик качества: поиск, генерация, отказы.

Скрипт разделён на этапы, чтобы не переплачивать за повторы: ответы агента
складываются на диск, и метрики можно пересчитать, не гоняя агента заново.

Запуск::

    python run_eval.py                 # прогнать агента и посчитать метрики
    python run_eval.py --metrics       # только пересчитать по сохранённым ответам
    python run_eval.py --retrieval     # только метрики поиска (быстро и дёшево)
    python run_eval.py --refusals      # проверка отказов на негативных вопросах
    python run_eval.py --single-gt     # по старой разметке (один эталон на вопрос)
    python run_eval.py --rerank        # то же, но с LLM-реранкингом в ретривере
    python run_eval.py --no-relevancy  # без насыщенной relevancy (дешевле)

Что меряется:

* **Retrieval** (Precision@K, Recall@K, F1, MRR, MAP, NDCG) — нашёл ли ретривер
  тот фрагмент, из которого сгенерирован вопрос.
* **Faithfulness** — не противоречит ли ответ найденному контексту.
* **Answer correctness** — совпадает ли ответ с эталонным ПО ФАКТАМ. Это не то же
  самое, что faithfulness: можно добросовестно пересказать найденный фрагмент,
  который к вопросу не относится, — faithfulness будет высокой, а ответ неверным.
* **Answer relevancy** — отвечает ли ответ на заданный вопрос, а не на соседний.
* **Refusal accuracy** — на вопросах без ответа в корпусе агент обязан
  отказаться, а не сочинить. Без этой метрики предыдущие поощряют болтливость:
  агент, который всегда что-то отвечает, выглядит лучше честного.

Как читать цифры. Если собрана разметка пула (``build_qrels.py``), релевантным
считается ЛЮБОЙ размеченный чанк, и метрики сопоставимы между собой напрямую.
Без неё эталон один на вопрос, и тогда precision физически не превысит 1/k
(при k=5 — 0.2), а F1 упирается в ≈0.33 — сравнивать конфигурации в этом
режиме можно только по recall, MRR и NDCG.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from agent import RAGAgentAnswer, build_agent
from build_qrels import QRELS_PATH, load_qrels
from evaluation import evaluate
from evaluation_answers import evaluate_answers
from evaluation_answers import evaluate_refusals as judge_refusals
from make_eval_dataset import DATASET_PATH, NEGATIVE_PATH
from measurement import RESEARCH_DIR
from providers import get_clients

HERE = Path(__file__).parent

# Выгрузки замеров живут в _research/dataset/ — в репозиторий не идут,
# пересобираются скриптами. В репозитории только журнал EVOLUTION.md.
DATASET_DIR = RESEARCH_DIR
ANSWERS_PATH = DATASET_DIR / "agent_answers.json"
# Отдельный файл для прогона с реранкингом: базовые ответы не затираются,
# иначе сравнить две конфигурации будет уже не с чем.
ANSWERS_RERANK_PATH = DATASET_DIR / "agent_answers_rerank.json"
METRICS_PATH = DATASET_DIR / "metrics.csv"
METRICS_RERANK_PATH = DATASET_DIR / "metrics_rerank.csv"
REFUSALS_PATH = DATASET_DIR / "refusals.csv"

# Формулировки, по которым видно, что агент отказался отвечать.
#
# Первые четыре — готовые тексты отказа из agent.py. Остальные добавлены после
# первого прогона: агент нередко отказывается «своими словами» внутри узла
# generate — «в приведённых фрагментах нет данных о...». По существу это тот же
# отказ, и не засчитывать его — значит занижать метрику. На первом прогоне
# из-за этого потерялось 2 отказа из 25.
REFUSAL_MARKERS = (
    # шаблонные отказы
    "нет информации",
    "не могу составить",
    "только с питанием",
    "переформулировать",
    # отказы «своими словами»
    "нет прямого упоминания",
    "нет конкретных данных",
    "нет данных",
    "не содержат информации",
    "нельзя однозначно",
    "не упоминается",
)


def run_agent_on_dataset(
    dataset: pd.DataFrame,
    max_workers: int = 4,
    use_rerank: bool = False,
    db_path: str = "lance_db/vectorstore",
    table_name: str = "chunks",
    top_k: int = 5,
) -> list[RAGAgentAnswer]:
    """Прогнать агента по всем вопросам датасета.

    ``remember=False`` обязателен: вопросы независимы, и профиль или история
    от одного не должны протекать в другой.

    ``use_rerank`` нужен, чтобы проверить реранкинг на полном графе, а не
    на голом ретривере: между поиском и ответом стоит узел ``grade``,
    и он сам отсеивает часть мусора. Прирост на ретривере может там
    и раствориться — так уже было при старом замере.
    """
    llm_client, model, embed_client = get_clients("polza")
    agent = build_agent(
        llm_client,
        model=model,
        embed_client=embed_client,
        use_rerank=use_rerank,
        # Индекс и k вынесены в параметры ради замера нарезки: чтобы сравнить
        # 900/150 с 1200/200 честно, второй конфигурации нужен свой индекс
        # и меньшее k — при вчетверо более длинных чанках тот же объём
        # контекста набирается четырьмя фрагментами, а не пятью.
        db_path=db_path,
        table_name=table_name,
        top_k=top_k,
    )

    def work(row: pd.Series) -> RAGAgentAnswer:
        return agent.run(str(row["question"]), dataset_row_id=str(row["id"]), remember=False)

    answers: list[RAGAgentAnswer] = []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, row) for _, row in dataset.iterrows()]
        for number, future in enumerate(as_completed(futures), start=1):
            answers.append(future.result())
            if number % 10 == 0:
                print(f"  обработано {number}/{len(dataset)}")

    print(f"Агент отработал за {time.perf_counter() - started:.0f} с")
    return answers


def save_answers(answers: list[RAGAgentAnswer], path: Path | None = None) -> None:
    path = path or ANSWERS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([answer.model_dump() for answer in answers], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_answers(path: Path | None = None) -> list[RAGAgentAnswer]:
    raw = json.loads((path or ANSWERS_PATH).read_text(encoding="utf-8"))
    return [RAGAgentAnswer(**item) for item in raw]


# Служебные предложения, которые агент обязан добавлять по промпту:
# напоминание про врача и перечень использованных PMID.
#
# Зачем их вырезать перед подсчётом faithfulness: эта метрика проверяет,
# подтверждается ли КАЖДОЕ утверждение ответа найденным контекстом. Фраза
# «это не заменяет консультацию врача» — требование техники безопасности,
# а не утверждение о предмете, и в абстрактах её, разумеется, нет. RAGAS
# честно считает её неподтверждённой, и метрика падает на ровном месте.
#
# Замер: на 23 ответах с самой низкой faithfulness удаление дисклеймера
# подняло её с 0.694 до 0.847 (выросла у 22 из 23). То есть мерили мы
# не галлюцинации агента, а собственное требование к нему.
BOILERPLATE = re.compile(
    r"[^.!?\n]*(врач\w*|диетолог\w*|специалист\w*)[^.!?\n]*[.!?]"
    r"|Использованные PMID[^\n]*",
    re.IGNORECASE,
)


def strip_boilerplate(text: str) -> str:
    """Убрать служебные предложения перед оценкой ответа.

    Обратите внимание: чистится только копия для метрик. Пользователю
    дисклеймер показывается — он там не для красоты.
    """
    return BOILERPLATE.sub("", text).strip()


def looks_like_refusal(answer: str) -> bool:
    """Отказался ли агент отвечать."""
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def evaluate_refusals(use_judge: bool = True) -> pd.DataFrame:
    """На вопросах без ответа в корпусе агент обязан отказаться.

    Args:
        use_judge: считать отказ LLM-судьёй, а не поиском ключевых фраз.
            Судья надёжнее: агент нередко отказывается своими словами
            («в приведённых фрагментах нет данных о...»), и список маркеров
            такие отказы пропускает. Маркеры оставлены как быстрый и бесплатный
            запасной вариант — и чтобы было с чем сравнить самого судью.
    """
    if not NEGATIVE_PATH.exists():
        print("Нет негативного датасета. Выполните: python make_eval_dataset.py --negative")
        return pd.DataFrame()

    dataset = pd.read_excel(NEGATIVE_PATH)
    llm_client, model, embed_client = get_clients("polza")
    agent = build_agent(llm_client, model=model, embed_client=embed_client)

    answers = []
    for number, row in enumerate(dataset.to_dict("records"), start=1):
        answers.append(agent.run(str(row["question"]), dataset_row_id=str(row["id"]), remember=False))
        if number % 20 == 0:
            print(f"  обработано {number}/{len(dataset)}")

    frame = pd.DataFrame(
        {
            "id": dataset["id"].astype(str),
            "question": dataset["question"],
            "subtle": dataset["subtle"] if "subtle" in dataset.columns else True,
        }
    )

    by_id = {str(a.dataset_row_id): a.answer for a in answers}
    frame["answer"] = frame["id"].map(by_id).fillna("")
    frame["refused_markers"] = frame["answer"].map(looks_like_refusal)

    if use_judge:
        judged = judge_refusals(answers, dataset.assign(id=dataset["id"].astype(str)),
                                llm_client, model)
        frame = frame.merge(
            judged[["id", "refused", "reason"]].rename(columns={"refused": "refused_judge"}),
            on="id",
            how="left",
        )
        # Где судья не ответил, берём оценку по маркерам. Работает это
        # только потому, что сбой судьи теперь даёт пусто, а не False.
        judge = frame["refused_judge"]
        frame["refused"] = judge.where(judge.notna(), frame["refused_markers"]).astype(bool)
    else:
        frame["refused"] = frame["refused_markers"]

    # Ответы сохраняем целиком: если промпт судьи придётся править (а его
    # пришлось), пересудить можно по этому файлу, не гоняя агента заново.
    frame.to_csv(REFUSALS_PATH, index=False, encoding="utf-8")

    subtle = frame[frame["subtle"]] if "subtle" in frame.columns else frame

    print()
    print(f"Refusal accuracy: {frame['refused'].mean():.1%} ({int(frame['refused'].sum())}/{len(frame)})")
    print(f"  на трудных вопросах: {subtle['refused'].mean():.1%} ({int(subtle['refused'].sum())}/{len(subtle)})")
    if use_judge:
        # Расхождение судьи с маркерами полезно видеть: если оно большое,
        # значит один из двух способов меряет не то, что нужно.
        disagreement = (frame["refused"] != frame["refused_markers"]).mean()
        print(f"  судья разошёлся с маркерами на: {disagreement:.1%} вопросов")
    print(f"Подробности → {REFUSALS_PATH}")
    return frame


# Метрики разделены на два яруса, и это не косметика.
#
# РЕШАЮЩИЕ — те, по которым принимаются решения о правках. Каждая отвечает
# на свой вопрос и не выводится из соседних.
#
# КОНТРОЛЬНЫЕ — те, что остаются как индикатор здоровья, но решать по ним
# нельзя. Причины разные и все измеренные:
#
#   f1        выводится из precision и recall арифметически, своего
#             содержания не несёт;
#   map       на бинарной разметке почти дублирует ndcg;
#   mrr       под пулинговой разметкой насыщен (0.93-0.97): первый
#             фрагмент релевантен почти всегда, различать конфигурации
#             он больше не может. При старой разметке «один эталон
#             на вопрос» он был осмысленным — ярус сменился вместе
#             с разметкой, а не с кодом;
#   relevancy насыщена (0.95-0.97) и одинакова у трёх независимых судей —
#             это предохранитель «агент не ушёл в сторону», а не метрика
#             решения.
DECISIVE = ("precision", "recall", "ndcg", "faithfulness", "correctness")
CONTROL = ("f1", "map", "mrr", "relevancy")


def report(metrics: pd.DataFrame) -> None:
    """Напечатать сводку метрик."""

    def block(title: str, names: tuple[str, ...]) -> None:
        columns = [c for c in names if c in metrics.columns]
        if not columns:
            return
        print(f"\n{title}")
        print("-" * 52)
        for column in columns:
            values = metrics[column].dropna()
            if values.empty:
                continue
            missing = int(metrics[column].isna().sum())
            note = f"  (нет оценки: {missing})" if missing else ""
            print(f"{column:<16}{values.mean():>10.3f}{values.median():>10.3f}"
                  f"{(values > 0).mean():>12.1%}{note}")

    print()
    print("=" * 52)
    print(f"{'Метрика':<16}{'Среднее':>10}{'Медиана':>10}{'Доля > 0':>12}")
    print("=" * 52)
    block("РЕШАЮЩИЕ — по ним принимаются решения", DECISIVE)
    block("КОНТРОЛЬНЫЕ — индикатор здоровья, решать по ним нельзя", CONTROL)
    print("=" * 52)
    print(f"Вопросов в прогоне: {len(metrics)}")

    # Потолок recall. Голая цифра recall@k под пулинговой разметкой занижена
    # арифметикой: релевантных чанков больше, чем мест в выдаче.
    if "recall_ceiling" in metrics.columns:
        ceiling = metrics["recall_ceiling"].dropna()
        if not ceiling.empty:
            common = metrics.loc[ceiling.index, "recall"]
            share = common.mean() / ceiling.mean() if ceiling.mean() else float("nan")
            print()
            print(f"Потолок recall@k при этой разметке: {ceiling.mean():.3f}")
            print(f"Взято от достижимого:               {share:.1%}")
            print("  (релевантных на вопрос больше, чем мест в выдаче —")
            print("   recall=1.0 недостижим даже идеальным поиском)")


def main() -> None:
    flags = set(sys.argv[1:])

    if "--refusals" in flags:
        evaluate_refusals()
        return

    if not DATASET_PATH.exists():
        print("Нет тестового датасета. Выполните: python make_eval_dataset.py")
        return

    dataset = pd.read_excel(DATASET_PATH)
    print(f"Датасет: {len(dataset)} вопросов, {dataset['document'].nunique()} статей")

    # Прогон с реранкингом пишется в свой файл и свою таблицу метрик,
    # чтобы базовый прогон остался на месте для сравнения.
    rerank = "--rerank" in flags
    answers_path = ANSWERS_RERANK_PATH if rerank else ANSWERS_PATH
    metrics_path = METRICS_RERANK_PATH if rerank else METRICS_PATH
    if rerank:
        print("Конфигурация: гибрид + LLM-реранкинг")

    if "--metrics" in flags:
        answers = load_answers(answers_path)
        print(f"Загружено сохранённых ответов: {len(answers)}")
    else:
        answers = run_agent_on_dataset(dataset, use_rerank=rerank)
        save_answers(answers, answers_path)
        print(f"Ответы сохранены → {answers_path}")

    llm_client, model, _ = get_clients("polza")
    retrieval_only = "--retrieval" in flags

    # Для метрик берём копию ответов без служебных предложений — см. BOILERPLATE.
    # На retrieval-метрики это не влияет, они считаются по чанкам.
    scored = [
        answer.model_copy(update={"answer": strip_boilerplate(answer.answer)})
        for answer in answers
    ]

    # Множественный эталон, если разметка пула собрана. Без неё метрики
    # считаются по одному эталонному чанку и занижают полноту — см. EVOLUTION.md.
    qrels = None
    if QRELS_PATH.exists() and "--single-gt" not in flags:
        qrels = load_qrels()
        print(f"Разметка пула: {len(qrels)} вопросов с множественным эталоном")
    else:
        print("Разметки пула нет — метрики по одному эталонному чанку")

    metrics = evaluate(
        answers=scored,
        dataset=dataset,
        client=llm_client,
        model=model,
        # Faithfulness требует отдельного вызова LLM на каждый ответ,
        # поэтому для быстрых прогонов её отключают.
        compute_faithfulness=not retrieval_only,
        qrels=qrels,
    )

    # Метрики самого ответа: верен ли он по фактам и отвечает ли на вопрос.
    # Retrieval-метрики на это не отвечают — можно найти нужный фрагмент
    # и всё равно ответить мимо.
    if not retrieval_only:
        if "reference_answer" not in dataset.columns:
            print(
                "\n⚠️ В датасете нет колонки reference_answer — correctness пропущен.\n"
                "   Выполните: python make_eval_dataset.py --references"
            )
        else:
            extra = evaluate_answers(
                scored, dataset, llm_client, model,
                with_relevancy="--no-relevancy" not in flags,
            )
            metrics = metrics.merge(
                extra[["id", "correctness", "relevancy", "correctness_reason"]],
                on="id",
                how="left",
            )

    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    report(metrics)
    print(f"Подробности → {metrics_path}")


if __name__ == "__main__":
    main()
