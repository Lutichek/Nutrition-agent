"""
Узлы ветки вопроса по литературе: переписать запрос, найти фрагменты,
отобрать, при неудаче переформулировать, ответить.

Вынесено из agent.py (ревью R6). Примесь к NutritionAgent: модель,
поиск и вспомогательные методы (_complete, _format_context и др.)
остаются в агенте и вызываются через self.
"""

from __future__ import annotations

from pydantic import ValidationError

from agent_state import AgentState
from llm_io import parse_json_response
from profile_rules import REQUIRED_PROFILE_FIELDS
from prompts import (
    ANSWER_PROMPTS,
    CANNOT_ANSWER_TEMPLATE,
    FRAGMENTS_HEADER,
    GRADE_PROMPT,
    GRADE_PROMPT_STRICT,
    PERSONAL_NORMS_BLOCK,
    REFINE_PROMPT,
)
from targets import Profile, compute_targets, explain


class QuestionNodes:
    """Узлы ветки «вопрос по статьям»."""

    # ------------------------------------------------------------------
    # Узлы: ветка вопроса по литературе
    # ------------------------------------------------------------------
    def _node_rewrite(self, state: AgentState) -> AgentState:
        """Перевести вопрос на английский и превратить в поисковый запрос."""
        query = self._ask(state, "rewrite", self._rewrite_prompt(state))
        query = query.strip().strip('"')
        return {"search_query": query or state["question"], "refine_count": 0}

    def _node_retrieve(self, state: AgentState) -> AgentState:
        chunks = self.retriever.retrieve(state["search_query"], k=self.top_k)
        return {"chunks": chunks}

    def _node_grade(self, state: AgentState) -> AgentState:
        """Проверить, отвечает ли найденное на вопрос."""
        chunks = state.get("chunks") or []
        if not chunks:
            return {"has_relevant": False}

        prompt = GRADE_PROMPT_STRICT if self.strict_grade else GRADE_PROMPT
        raw = self._complete(
            prompt.format(
                question=state["question"],
                context=self._format_context(chunks),
            ),
            json_mode=True,
        )
        parsed = parse_json_response(raw)

        has_relevant = bool(parsed.get("has_relevant"))

        # Оставляем только те фрагменты, которые модель назвала полезными:
        # мусор в контексте портит и ответ, и faithfulness.
        keep = parsed.get("keep") or []
        if has_relevant and keep:
            indices = [int(number) - 1 for number in keep if isinstance(number, (int, float))]
            filtered = [chunks[i] for i in indices if 0 <= i < len(chunks)]
            if filtered:
                return {"has_relevant": True, "chunks": filtered}

        return {"has_relevant": has_relevant}

    def _route_after_grade(self, state: AgentState) -> str:
        if state.get("has_relevant"):
            return "generate"
        if state.get("refine_count", 0) < self.max_refines:
            return "refine"
        return "cannot_answer"

    def _node_refine(self, state: AgentState) -> AgentState:
        query = self._complete(
            REFINE_PROMPT.format(
                question=state["question"],
                search_query=state.get("search_query", ""),
            )
        )
        return {
            "search_query": query.strip().strip('"'),
            "refine_count": state.get("refine_count", 0) + 1,
        }

    def _node_generate(self, state: AgentState) -> AgentState:
        prompt = ANSWER_PROMPTS[self.answer_style].format(
            question=state["question"],
            context=self._format_context(state.get("chunks") or []),
        )
        personal = self._personal_norms(state.get("profile"),
                                        (state.get("adaptive") or {}).get("tdee"))
        if personal:
            prompt = prompt.replace(FRAGMENTS_HEADER, personal + FRAGMENTS_HEADER, 1)
        answer = self._complete(prompt, temperature=0.2, on_text=self._text_sink(state))
        return {"answer": answer.strip()}

    def _personal_norms(self, raw_profile: dict | None, tdee: float | None = None) -> str:
        """Расчёт нормы человека для ветки вопросов — или пусто.

        Наблюдалось: после плана «А почему так мало углеводов?» уходило
        в вопросы, и человек получал рассказ о низкоуглеводных диетах
        вообще, а не о своей норме — хотя она посчитана и объяснима.
        Маршрутизатор тут не трогается: на вопросах оценки профиля нет,
        блок не вставляется, и промпт совпадает с прежним до байта.
        """
        # Неполный профиль — не считаем: у Profile есть умолчания (активность),
        # и норма по ним была бы домыслом, а не расчётом.
        if not raw_profile or any(f not in raw_profile for f in REQUIRED_PROFILE_FIELDS):
            return ""
        try:
            profile = Profile(**raw_profile)
        except ValidationError:
            return ""
        return PERSONAL_NORMS_BLOCK.format(
            profile=self._describe_profile(profile),
            # Та же норма, что в плане: с фактическим расходом, если он есть.
            targets_explained=explain(profile, compute_targets(profile, tdee_override=tdee)),
        )

    def _node_cannot_answer(self, state: AgentState) -> AgentState:
        return {"answer": CANNOT_ANSWER_TEMPLATE}
