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
from evaluation import _ndcg_at_k, _recall_ceiling, _unretrieved_row, evaluate
from evaluation_answers import evaluate_refusals
from prompts import OFF_TOPIC_ANSWER


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
    """Нет контекста — поиск не запускался: метрики не определены, а не нули.

    Наблюдалось: пять вопросов из 210 без контекста — все пять отказы
    маршрутизатора (grade фрагменты не стирает). Ноль faithfulness читался
    как «выдумал», а ноль поиска — как промах поиска, которого не было:
    у вопросов deepseek 13 верных отказов из 210 (шаги 65–66).
    """

    def test_unretrieved_row_leaves_faithfulness_unmeasured(self):
        row = _unretrieved_row("1", "вопрос")
        assert np.isnan(row["faithfulness"])

    def test_retrieval_metrics_are_unmeasured_not_zero(self):
        """Поиска не было — нет и оценки поиска."""
        row = _unretrieved_row("1", "вопрос")
        for metric in ("precision", "recall", "f1", "mrr", "map", "ndcg"):
            assert np.isnan(row[metric])

    def test_router_refusal_does_not_lower_retrieval_mean(self):
        """Верный отказ на вопрос вне области не тянет вниз precision поиска."""
        dataset = make_dataset(["a", "b"])
        answers = [
            make_answer("a", chunks=1),
            RAGAgentAnswer(dataset_row_id="b", answer=OFF_TOPIC_ANSWER, retrieved_chunks=None),
        ]
        frame = evaluate(answers, dataset, client=None, compute_faithfulness=False,
                         qrels={"a": {0}, "b": {7}})

        assert frame["precision"].mean() == 1.0
        assert int(frame["precision"].isna().sum()) == 1

    def test_reason_of_missing_search_is_printed(self, capsys):
        """Отказ маршрутизатора и пропавший ответ — разные причины, и обе видны."""
        dataset = make_dataset(["a", "b", "c"])
        answers = [
            make_answer("a", chunks=1),
            RAGAgentAnswer(dataset_row_id="b", answer=OFF_TOPIC_ANSWER, retrieved_chunks=None),
        ]
        evaluate(answers, dataset, client=None, compute_faithfulness=False)

        out = capsys.readouterr().out
        assert "не измерены у 2 из 3" in out
        assert "отказ маршрутизатора 1" in out
        assert "нет ответа 1" in out
        assert "проверьте прогон" in out

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
        assert np.isnan(_unretrieved_row("1", "вопрос")["recall_ceiling"])


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


class TestRagasImportsInTheLockedEnvironment:
    """ragas 0.4.3 импортирует ``langchain_community.chat_models.vertexai``,
    которого в langchain-community 0.4 уже нет. В зафиксированном окружении
    RAGAS не импортировался вовсе, и faithfulness не считалась ни в одном
    прогоне — NaN и предупреждение. Заглушка закрывает ровно этот модуль."""

    def test_missing_vertex_module_is_stubbed(self):
        import sys

        from evaluation import _ragas_compat

        _ragas_compat()
        stub = sys.modules.get("langchain_community.chat_models.vertexai")
        assert stub is not None and hasattr(stub, "ChatVertexAI")


class TestOtherJudgeNeverOverwritesBaseFiles:
    """Метрики другим судьёй — другая единица измерения: записанные в базовый
    metrics.csv, их сравнили бы с прежними как равные."""

    def test_tagged_run_writes_its_own_files(self):
        from run_eval import ANSWERS_PATH, REFUSALS_PATH, _tagged

        assert _tagged(ANSWERS_PATH, "llama").name == "agent_answers_llama.json"
        assert _tagged(REFUSALS_PATH, "llama").name == "refusals_llama.csv"
        assert _tagged(ANSWERS_PATH, None) == ANSWERS_PATH

    def test_judge_without_tag_is_refused(self, monkeypatch, capsys):
        import run_eval

        monkeypatch.setattr("sys.argv", ["run_eval.py", "--judge=meta-llama/x"])
        monkeypatch.setattr(run_eval, "run_agent_on_dataset",
                            lambda *a, **k: pytest.fail("агент не должен запускаться"))
        run_eval.main()
        assert "--tag" in capsys.readouterr().out


class TestQueryCacheBelongsToItsQuestions:
    """Кэш переводов сверял только КОЛИЧЕСТВО вопросов: 210 новых вопросов
    (замер авторов вопросов, шаг 64) получили бы переводы 210 старых, и пул
    молча собрался бы по чужим запросам."""

    def test_other_questions_are_translated_not_taken_from_cache(self, tmp_path):
        from conftest import FakeLLMClient

        from measurement import cached_queries, load_queries_en

        path = tmp_path / "queries.json"
        client = FakeLLMClient(default="english query")
        assert load_queries_en(["вопрос А"], client, "fake", path=path) == ["english query"]
        calls = len(client.calls)
        load_queries_en(["вопрос А"], client, "fake", path=path)
        assert len(client.calls) == calls, "те же вопросы — из кэша"
        load_queries_en(["вопрос Б"], client, "fake", path=path)
        assert len(client.calls) == calls + 1, "другие вопросы того же числа — заново"
        with pytest.raises(ValueError):
            cached_queries(["вопрос В"], path=path)


class TestIndependentModeSwitchesEveryLink:
    """--independent: ни судья, ни эталон, ни разметка — не модель генератора.
    Судья-генератор завышал correctness на 0.13, его эталон — ещё на 0.04
    (шаги 61, 63); режим, забывший одно звено, вернул бы завышение молча."""

    def test_judge_references_and_qrels_are_independent(self, monkeypatch, tmp_path):
        import run_eval

        dataset = pd.DataFrame({"id": ["a", "b"], "question": ["q1", "q2"],
                                "document": [1, 2], "reference_answer": ["mini", "mini"]})
        references = tmp_path / "references_gemini.csv"
        pd.DataFrame({"id": ["a", "b"], "reference_answer": ["gem-a", "gem-b"]}).to_csv(
            references, index=False)
        seen: dict = {}
        monkeypatch.setattr(run_eval, "INDEPENDENT_REFERENCES_PATH", references)
        monkeypatch.setattr(run_eval, "DATASET_PATH", tmp_path / "dataset.xlsx")
        (tmp_path / "dataset.xlsx").write_bytes(b"")
        monkeypatch.setattr(run_eval.pd, "read_excel", lambda _: dataset)
        monkeypatch.setattr(run_eval, "load_answers", lambda path: seen.setdefault("answers", path) and [])
        monkeypatch.setattr(run_eval, "QRELS_PATH", tmp_path / "qrels.csv")
        (tmp_path / "qrels.csv").write_text("x")
        monkeypatch.setattr(run_eval, "INDEPENDENT_QRELS_PATH", tmp_path / "qrels_llama.csv")
        (tmp_path / "qrels_llama.csv").write_text("x")
        monkeypatch.setattr(run_eval, "load_qrels", lambda path=None: seen.setdefault("qrels", path) and {})
        monkeypatch.setattr(run_eval, "get_clients", lambda _: (None, "openai/gpt-4o-mini", None))
        monkeypatch.setattr(run_eval, "evaluate", lambda **kw: pd.DataFrame({"id": ["a", "b"]}))

        def fake_answers(scored, data, client, model, **_):
            seen["judge"], seen["refs"] = model, list(data["reference_answer"])
            return pd.DataFrame({"id": ["a", "b"], "correctness": [1.0, 1.0],
                                 "relevancy": [1.0, 1.0], "correctness_reason": ["", ""]})

        monkeypatch.setattr(run_eval, "evaluate_answers", fake_answers)
        monkeypatch.setattr(run_eval, "report", lambda _m: None)
        monkeypatch.setattr(run_eval, "METRICS_PATH", tmp_path / "metrics.csv")
        monkeypatch.setattr("sys.argv", ["run_eval.py", "--independent", "--metrics"])
        run_eval.main()

        assert seen["judge"] == run_eval.INDEPENDENT_JUDGE
        assert seen["refs"] == ["gem-a", "gem-b"], "эталоны gemini, не модели генератора"
        assert seen["qrels"].name == "qrels_llama.csv"
        assert "independent" in seen["answers"].name
        assert (tmp_path / "metrics_independent.csv").exists(), "базовый metrics.csv не тронут"
