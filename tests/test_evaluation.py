"""
Тесты измерительного прибора.

Модули оценки до сих пор были без тестов — и это ровно то место, где ошибка
опаснее всего. Дефект в агенте видно в ответе; дефект в метрике не видно
никогда, а по её показаниям принимаются решения о продукте.

Оба проверяемых здесь случая уже случались:

* сбой провайдера превратил correctness в 0.498 вместо 0.869, и это
  выглядело как обвал качества, хотя агент не менялся ни на строку;
* пять вопросов, где поиск ничего не дал, занижали faithfulness
  с 0.941 до 0.919 — агент там честно отказался отвечать.

Общее правило, которое эти тесты стерегут: **отсутствие оценки — не ноль.**
Ноль означает «измерили, вышло плохо». Пропуск означает «не измерили».
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent import Chunk, RAGAgentAnswer
from evaluation import _ndcg_at_k, _recall_ceiling, _zero_row, evaluate
from evaluation_answers import evaluate_refusals


class FailingClient:
    """Клиент, который всегда падает — изображает недоступный провайдер."""

    def __init__(self) -> None:
        self.chat = self
        self.completions = self

    def create(self, **_kwargs):
        raise RuntimeError("503 Service Unavailable")


def make_dataset(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "id": ids,
        "question": [f"вопрос {i}" for i in ids],
        "relevant_text": ["эталонный фрагмент про белок" for _ in ids],
        "document": [1 for _ in ids],
    })


def make_answer(row_id: str, chunks: int = 1) -> RAGAgentAnswer:
    return RAGAgentAnswer(
        dataset_row_id=row_id,
        answer="Белка нужно 1.6 грамма на килограмм массы тела.",
        retrieved_chunks=[
            Chunk(text="protein intake 1.6 g per kg body mass", chunk_id=i, year=1, distance=0.1)
            for i in range(chunks)
        ] or None,
    )


class TestMissingContextIsNotZero:
    """Нет контекста — faithfulness не определена, а не равна нулю.

    Наблюдалось: пять вопросов из 210, где узел grade забраковал все
    найденные фрагменты. Агент честно отказался отвечать — проверять
    нечего, утверждений нет. Ноль читался как «выдумал».
    """

    def test_zero_row_leaves_faithfulness_unmeasured(self):
        row = _zero_row("1", "вопрос")
        assert np.isnan(row["faithfulness"])

    def test_retrieval_metrics_are_honest_zeros(self):
        """Метрики поиска при пустой выдаче — именно нули: искали и не нашли."""
        row = _zero_row("1", "вопрос")
        for metric in ("precision", "recall", "f1", "mrr", "map", "ndcg"):
            assert row[metric] == 0.0

    def test_answer_without_chunks_is_excluded_from_mean(self):
        dataset = make_dataset(["a", "b"])
        answers = [
            make_answer("a", chunks=1),
            RAGAgentAnswer(dataset_row_id="b", answer="В базе нет данных.", retrieved_chunks=None),
        ]
        frame = evaluate(answers, dataset, client=None, compute_faithfulness=False)

        assert len(frame) == 2
        assert frame.set_index("id").loc["b", "faithfulness"] != 0.0
        assert np.isnan(frame.set_index("id").loc["b", "faithfulness"])


class TestRecallCeiling:
    """Recall@k под пулинговой разметкой ограничен арифметикой, а не поиском.

    Наблюдалось: recall@5 = 0.436 публиковался голым и читался как «нашли
    меньше половины нужного». При 6.2 релевантных чанках на вопрос и k=5
    потолок — около 0.81, то есть больше половины разрыва создаёт разметка,
    а не качество поиска.
    """

    def test_ceiling_is_one_when_gold_fits_in_k(self):
        """Старая разметка: один эталон на вопрос — потолок был всегда 1.0."""
        assert _recall_ceiling(k=5, total_relevant=1) == 1.0

    def test_ceiling_drops_when_relevant_outnumber_slots(self):
        assert _recall_ceiling(k=5, total_relevant=10) == 0.5

    def test_ceiling_matches_the_observed_case(self):
        """6.2 релевантных на вопрос при k=5 — тот самый случай."""
        assert _recall_ceiling(k=5, total_relevant=6) == pytest.approx(0.833, abs=1e-3)

    def test_question_without_relevant_chunks_has_no_ceiling(self):
        """Ноль релевантных — потолок не определён, а не равен нулю."""
        assert np.isnan(_recall_ceiling(k=5, total_relevant=0))

    def test_ceiling_is_reported_per_question(self):
        """Потолок считается для каждого вопроса, а не берётся средним по датасету."""
        dataset = make_dataset(["a"])
        frame = evaluate(
            [make_answer("a", chunks=5)], dataset,
            client=None, compute_faithfulness=False,
            qrels={"a": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
        )
        assert frame.loc[0, "recall_ceiling"] == pytest.approx(0.5)

    def test_empty_retrieval_leaves_ceiling_unmeasured(self):
        """Выдачи не было — делить нечего, это NaN, а не ноль."""
        assert np.isnan(_zero_row("1", "вопрос")["recall_ceiling"])


class TestNdcgNormalisation:
    """Идеальная выдача обязана давать 1.0 при любом числе релевантных.

    Наблюдалось: при пулинговой разметке у вопроса бывает 20 релевантных
    чанков, а выдача — пять. Идеальный вектор строился из 20 единиц и DCG
    по нему считался по всем двадцати позициям, поэтому пять релевантных
    подряд получали 0.419. Заметили это только когда мелкая нарезка
    показала рост precision на 0.148 при падении NDCG на 0.215 —
    так быть не может, обе метрики считаются по одной выдаче.
    """

    @pytest.mark.parametrize("total_relevant", [1, 5, 6, 20, 100])
    def test_perfect_top_k_is_one(self, total_relevant):
        vector = [1] * min(5, total_relevant)
        assert _ndcg_at_k(vector, total_relevant) == pytest.approx(1.0)

    def test_empty_output_is_zero(self):
        assert _ndcg_at_k([0, 0, 0, 0, 0], 20) == 0.0

    def test_order_still_matters(self):
        """Нормировка починена, но ранжирование метрика различать не перестала."""
        top = _ndcg_at_k([1, 0, 0, 0, 0], 20)
        bottom = _ndcg_at_k([0, 0, 0, 0, 1], 20)
        assert top > bottom > 0

    def test_single_gold_case_is_unchanged(self):
        """Старая разметка считалась верно — правка не должна её тронуть."""
        assert _ndcg_at_k([1, 0, 0, 0, 0], 1) == pytest.approx(1.0)
        assert _ndcg_at_k([0, 1, 0, 0, 0], 1) == pytest.approx(1 / np.log2(3))


class TestJudgeFailureIsNotZero:
    """Сбой провайдера не должен выглядеть как плохое качество."""

    def test_faithfulness_failure_stays_unmeasured(self):
        dataset = make_dataset(["a"])
        frame = evaluate(
            [make_answer("a")], dataset,
            client=FailingClient(), model="whatever", compute_faithfulness=True,
        )
        assert np.isnan(frame.loc[0, "faithfulness"]), "сбой оценки стал нулём"

    def test_mean_ignores_missing_values(self):
        """NaN выпадает из среднего — так работает pandas, и на это расчёт."""
        values = pd.Series([1.0, 0.9, float("nan")])
        assert values.mean() == pytest.approx(0.95)


class TestRefusalJudgeFailure:
    """Сбой судьи отказов — отсутствие вердикта, а не провал агента.

    Раньше здесь стоял False, из-за чего запасной путь в run_eval
    (оценить по маркерам, если судья молчит) не срабатывал никогда:
    fillna не видит False.
    """

    def test_failure_leaves_verdict_empty(self):
        dataset = pd.DataFrame({"id": ["a"], "question": ["вопрос про несуществующее"]})
        answers = [RAGAgentAnswer(dataset_row_id="a", answer="В моей базе нет информации.")]

        frame = evaluate_refusals(answers, dataset, client=FailingClient(), model="whatever")

        assert frame.loc[0, "refused"] is None or pd.isna(frame.loc[0, "refused"])

    def test_empty_answer_is_a_real_failure_not_a_gap(self):
        """Агент не ответил вовсе — это False, а не пропуск: вердикт есть."""
        dataset = pd.DataFrame({"id": ["a"], "question": ["вопрос"]})
        frame = evaluate_refusals(
            [RAGAgentAnswer(dataset_row_id="a", answer="")],
            dataset, client=FailingClient(), model="whatever",
        )
        assert bool(frame.loc[0, "refused"]) is False
        assert not pd.isna(frame.loc[0, "refused"])

    def test_markers_fill_the_gap(self):
        """Проверка самого запасного пути: пусто от судьи → берём маркеры."""
        frame = pd.DataFrame({
            "refused_judge": [None, True, None],
            "refused_markers": [True, False, False],
        })
        judge = frame["refused_judge"]
        merged = judge.where(judge.notna(), frame["refused_markers"]).astype(bool)
        assert list(merged) == [True, True, False]
