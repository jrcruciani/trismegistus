"""Stage 3b (Delegation): System-1 subagent router for `delegate_task` (`delegation.backend: trismegistus`).

Unlike `llm_request` middleware inside a live session (which shares a single wire mode),
each `delegate_task` subagent spawns its own `AIAgent` instance. That means Trismegistus can
freely dispatch System-1 children to `claude-haiku-4.5` / `gpt-5-mini` (`chat_completions`)
and System-2 children to `gpt-5.4` / `claude-sonnet-5` (`codex_responses` or `chat_completions`)
in the very same parallel batch.
"""

from __future__ import annotations

from typing import Any

from .backends import build_decision_backend
from .compat import BaseDelegationProvider, DelegationRouteError, redact_text
from .contracts import DecisionQuestion
from .copilot_lanes import normalize_copilot_model

SUBAGENT_QUESTIONS: dict[str, DecisionQuestion] = {
    "tier": DecisionQuestion.choice(
        "Which subagent capability tier fits this delegated task?",
        {
            "system1": "Focused file search, summarization, mechanical extraction, formatting, or routine check",
            "system2": "Complex multi-file implementation, root-cause debugging, architecture, or deep synthesis",
        },
    ),
}


class TrismegistusDelegationProvider(BaseDelegationProvider):
    """Routes `delegate_task` children between System-1 and System-2 Copilot models."""

    display_name = "Trismegistus (System 1 / System 2)"
    summary = "Routes subagents between fast System-1 and deep System-2 models via Microsoft-Decision-1.1."

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    @property
    def name(self) -> str:
        return "trismegistus"

    def task_parameters(self, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {
            "cognitive_tier": {
                "type": "string",
                "enum": ["auto", "system1", "system2"],
                "description": (
                    "Cognitive routing tier: 'auto' (default, classified by Microsoft-Decision-1.1), "
                    "'system1' (fast/cheap Copilot model), or 'system2' (deep-reasoning Copilot model)."
                ),
            }
        }

    def description(self, config: dict[str, Any]) -> str:
        del config
        return (
            "Optional `cognitive_tier` per task ('auto', 'system1', 'system2'): Trismegistus "
            "automatically routes routine subagents to System 1 and complex tasks to System 2."
        )

    def route_task(self, task: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        own = self.settings(config)
        requested = str(task.get("cognitive_tier") or "auto").strip().lower()
        if requested not in ("auto", "system1", "system2"):
            raise DelegationRouteError(
                f"Unsupported cognitive_tier {requested!r}; expected 'auto', 'system1', or 'system2'."
            )

        provider = str(
            own.get("provider")
            or self._ctx.get_config("delegation_provider", default="copilot")
            or "copilot"
        ).strip()
        s1_model = str(
            own.get("system1_model")
            or self._ctx.get_config("delegation_system1_model", default="claude-haiku-4.5")
            or "claude-haiku-4.5"
        ).strip()
        s2_model = str(
            own.get("system2_model")
            or self._ctx.get_config("delegation_system2_model", default="gpt-5.4")
            or "gpt-5.4"
        ).strip()

        if provider in ("copilot", "github-copilot", "github"):
            s1_model = normalize_copilot_model(s1_model)
            s2_model = normalize_copilot_model(s2_model)

        effective_tier = requested
        if requested == "auto":
            goal_text = redact_text(str(task.get("goal") or "")[:1600])
            ctx_text = redact_text(str(task.get("context") or "")[:800])
            conf_threshold = float(self._ctx.get_config("confidence_threshold", default=0.65) or 0.65)
            backend = build_decision_backend(self._ctx)
            batch = backend.decide(
                {"goal": goal_text, "context": ctx_text},
                SUBAGENT_QUESTIONS,
            )
            ans = batch.answers.get("tier")
            if batch.ok and ans and ans.choice == "system1" and ans.top_probability >= conf_threshold:
                effective_tier = "system1"
            else:
                effective_tier = "system2"

        if effective_tier == "system1":
            route: dict[str, Any] = {
                "provider": provider,
                "model": s1_model,
                "label": "trismegistus:system1",
            }
            if s1_model.startswith("gpt-5"):
                route["reasoning_effort"] = "low"
            return route

        route = {
            "provider": provider,
            "model": s2_model,
            "label": "trismegistus:system2",
        }
        if s2_model.startswith(("gpt-5", "o3", "o4", "claude-sonnet-5", "claude-opus-5", "gemini-4")):
            route["reasoning_effort"] = "high"
        return route
