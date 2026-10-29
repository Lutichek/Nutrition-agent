"""
Сборка графа LangGraph из узлов агента.

Вынесено из agent.py (ревью R6). Примесь к NutritionAgent: методы узлов
лежат в nodes_plan.py и nodes_question.py, здесь — только как они
связаны и как каждый шаг пишется в trace.
"""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from agent_state import STAGE_LABELS, AgentState
from llm_io import _LLM_CALLS


class AgentGraph:
    """Граф агента: узлы, переходы и запись шагов в trace."""

    # ------------------------------------------------------------------
    # Сборка графа
    # ------------------------------------------------------------------
    @staticmethod
    def _traced(name: str, node):
        """Обернуть узел: записать в trace его время и ответы модели в нём.

        Ответ модели — главное для разбора тихих багов: «include=["protein"]»
        отсеивается дальше по коду, и по итоговому профилю его уже не видно.
        """
        def wrapper(state: AgentState) -> AgentState:
            emit = state.get("on_event")
            if emit is not None and name in STAGE_LABELS:
                emit({"type": "stage", "node": name, "label": STAGE_LABELS[name]})
            token = _LLM_CALLS.set([])
            started = time.perf_counter()
            try:
                update = node(state) or {}
            finally:
                calls = _LLM_CALLS.get() or []
                _LLM_CALLS.reset(token)
            step = {"node": name, "seconds": round(time.perf_counter() - started, 2)}
            if calls:
                step["llm"] = calls
            return {**update, "trace": [step]}

        return wrapper

    def _build_graph(self):
        builder = StateGraph(AgentState)
        nodes = {
            "classify": self._node_classify,
            "off_topic": self._node_off_topic,
            "red_flag": self._node_red_flag,

            "extract_profile": self._node_extract_profile,
            "match_eaten": self._node_match_eaten,
            "ask_profile": self._node_ask_profile,
            "make_plan": self._node_make_plan_in_graph,
            "explain_plan": self._node_explain_plan,

            "rewrite": self._node_rewrite,
            "retrieve": self._node_retrieve,
            "grade": self._node_grade,
            "refine": self._node_refine,
            "generate": self._node_generate,
            "cannot_answer": self._node_cannot_answer,
        }
        for name, node in nodes.items():
            builder.add_node(name, self._traced(name, node))

        builder.set_entry_point("classify")

        builder.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "off_topic": "off_topic",
                "red_flag": "red_flag",
                "extract_profile": "extract_profile",
                "rewrite": "rewrite",
            },
        )

        builder.add_conditional_edges(
            "extract_profile",
            self._route_after_profile,
            {"ask_profile": "ask_profile", "make_plan": "make_plan",
             "match_eaten": "match_eaten"},
        )
        builder.add_conditional_edges(
            "match_eaten",
            self._route_after_eaten,
            {"make_plan": "make_plan", END: END},
        )
        builder.add_conditional_edges(
            "make_plan",
            self._route_after_plan,
            {"explain_plan": "explain_plan", END: END},
        )

        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"generate": "generate", "refine": "refine", "cannot_answer": "cannot_answer"},
        )
        builder.add_edge("refine", "retrieve")

        for node in ("off_topic", "red_flag", "ask_profile", "explain_plan",
                     "generate", "cannot_answer"):
            builder.add_edge(node, END)

        return builder.compile()
