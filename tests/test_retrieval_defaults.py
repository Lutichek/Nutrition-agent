"""
Тесты умолчаний ретривера: BM25 и реранкинг отключены по результатам замеров.

Зачем тест на константу. Оба приёма — общепринятые, оба выглядят как
улучшение, и оба здесь измерены и отклонены. Включить их обратно —
одна правка `bool`, после которой всё работает: тесты зелёные, ответы
осмысленные, сервис поднимается. Просядут только метрики, и узнают
об этом в следующий полный прогон, если он вообще случится.

Это ровно тот класс дефектов, ради которого здесь пишутся тесты:
код работает, результат хуже, сигнала нет.

Цифры, по которым приняты решения:

* BM25 — при равном голосе в RRF выигрывает на 30% вопросов (+0.133 NDCG)
  и проигрывает на 42% (−0.198); веса без вреда не существует; сквозной
  прогон без него дал precision +0.031 [+0.001, +0.063] под обеими
  разметками. EVOLUTION.md, шаг 36.
* Реранкинг — precision@5 0.839 → 0.881, но correctness +0.007 при пороге
  различимости 0.010, ценой 23% времени ответа. EVOLUTION.md, шаги 24 и 28.

Тест не запрещает включать их: флаги на месте, отклонённые варианты
гоняются через `run_eval.py --hybrid` и `--rerank`. Он требует, чтобы
смена умолчания была осознанной — вместе с правкой обоснования и новым
прогоном.
"""

from __future__ import annotations

import inspect

import pytest

from agent import build_agent
from retriever import Retriever


def _default(func, name: str):
    return inspect.signature(func).parameters[name].default


class TestRetrieverDefaults:
    """Умолчания самого ретривера."""

    @pytest.mark.parametrize("flag", ["use_hybrid", "use_rerank"])
    def test_flag_is_off(self, flag: str) -> None:
        assert _default(Retriever.__init__, flag) is False, (
            f"{flag} включён по умолчанию — приём отклонён замером, "
            "см. EVOLUTION.md"
        )

    def test_bm25_index_not_built_when_hybrid_off(self) -> None:
        """Выключенный BM25 не должен строить индекс по всему корпусу.

        Проверяется не только скорость старта: пока индекс строится,
        «выключен» и «включён, но не используется» выглядят одинаково,
        и обратное включение однажды пройдёт незамеченным.
        """
        source = inspect.getsource(Retriever.__init__)
        assert "if use_hybrid else None" in source, (
            "BM25-индекс строится независимо от флага use_hybrid"
        )


class TestBuildAgentDefaults:
    """Сборка «под ключ» — то, чем пользуются api.py и demo.py."""

    @pytest.mark.parametrize("flag", ["use_hybrid", "use_rerank"])
    def test_flag_is_off(self, flag: str) -> None:
        assert _default(build_agent, flag) is False, (
            f"{flag} включён в build_agent — сервис поедет не в той "
            "конфигурации, в которой мерили"
        )

    @pytest.mark.parametrize("flag", ["use_hybrid", "use_rerank"])
    def test_matches_retriever(self, flag: str) -> None:
        """Умолчания обязаны совпадать.

        Разойдутся — и `Retriever(...)` напрямую (замеры в experiments/)
        будет мерить не то, что работает в сервисе.
        """
        assert _default(build_agent, flag) == _default(Retriever.__init__, flag)


class TestDecisionIsDocumented:
    """Решение должно быть объяснено там, где стоит флаг."""

    @pytest.mark.parametrize("flag", ["use_hybrid", "use_rerank"])
    def test_justification_next_to_flag(self, flag: str) -> None:
        """Числа рядом с флагом, а не только в журнале.

        Правило проекта: меняя число, обнови причину. Причина должна быть
        под рукой у того, кто трогает флаг.
        """
        source = inspect.getsource(build_agent)
        head = source[: source.index(f"{flag}: bool")]
        comment = head[head.rindex("\n\n") :] if "\n\n" in head else head
        assert "correctness" in comment or "precision" in comment, (
            f"рядом с {flag} нет цифр замера"
        )
