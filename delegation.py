"""Stage 3b (Delegation): Cognitive specialty router for `delegate_task` (`delegation.backend: trismegistus`).

Because each `delegate_task` subagent spawns its own `AIAgent` instance, Trismegistus routes
children by **cognitive specialty** across the entire GitHub Copilot catalog for maximum speed
and domain precision — without single-wire session constraints:
  - `fast_scout`   -> fast parallel search & extraction (`gemini-3-flash-preview`, low latency)
  - `deep_code`    -> algorithms, root-cause debugging, concurrency, tests (`gpt-5.4`, `xhigh`)
  - `architecture` -> multi-file design, code review, synthesis (`claude-sonnet-5`, `high`)
  - `long_context` -> massive log/trace/documentation cross-referencing (`gemini-3.1-pro-preview`, `high`)
"""

from __future__ import annotations

from typing import Any

from .backends import build_decision_backend
from .compat import BaseDelegationProvider, DelegationRouteError, redact_text
from .contracts import DecisionQuestion
from .copilot_lanes import (
    clamp_effort,
    is_copilot_provider,
    normalize_copilot_model,
    supported_reasoning_efforts,
)

VALID_COGNITIVE_TIERS: tuple[str, ...] = (
    "auto",
    "fast_scout",
    "deep_code",
    "architecture",
    "long_context",
    "system1",
    "system2",
)

SUBAGENT_SPECIALTY_QUESTIONS: dict[str, DecisionQuestion] = {
    "specialty": DecisionQuestion.choice(
        "Which cognitive specialist profile will execute this subagent task with highest precision and speed?",
        {
            "fast_scout": (
                "Fast file/web search, grep/glob discovery, straightforward data extraction, "
                "or mechanical inventory with minimal ambiguity"
            ),
            "deep_code": (
                "Algorithmic implementation, root-cause debugging, concurrency/race conditions, "
                "test failure diagnosis, or formal verification"
            ),
            "architecture": (
                "Multi-module software design, critical security/code review, API contract synthesis, "
                "or nuanced technical writing"
            ),
            "long_context": (
                "Cross-referencing large logs, multi-document research corpora, transcripts, "
                "or wide codebase audits"
            ),
        },
    ),
}


class TrismegistusDelegationProvider(BaseDelegationProvider):
    """Routes `delegate_task` children to specialized Copilot models via Microsoft-Decision-1.1."""

    display_name = "Trismegistus (Precision Specialty Router)"
    summary = "Routes subagents across Copilot specialty models (scout / deep_code / architecture / long_context) via Microsoft-Decision-1.1."

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
                "enum": list(VALID_COGNITIVE_TIERS),
                "description": (
                    "Cognitive specialty route: 'auto' (default, classified by Microsoft-Decision-1.1), "
                    "'fast_scout' (rapid search/extraction), 'deep_code' (debugging/algorithms at xhigh effort), "
                    "'architecture' (design/code review at high effort), 'long_context' (large corpus analysis), "
                    "or explicit 'system1' / 'system2'."
                ),
            }
        }

    def description(self, config: dict[str, Any]) -> str:
        del config
        return (
            "Optional `cognitive_tier` per task ('auto', 'fast_scout', 'deep_code', 'architecture', "
            "'long_context', 'system1', 'system2'): Trismegistus matches each subagent to the optimal Copilot model."
        )

    def _resolve_specialty_target(self, specialty: str, own: dict[str, Any], provider: str) -> tuple[str, str]:
        """Return `(model_id, desired_effort)` for a resolved specialty."""
        mapping: dict[str, tuple[str, str, str, str]] = {
            "fast_scout": ("scout_model", "delegation_scout_model", "gemini-3-flash-preview", "low"),
            "system1": ("system1_model", "delegation_system1_model", "gemini-3-flash-preview", "low"),
            "deep_code": ("deep_code_model", "delegation_deep_code_model", "gpt-5.4", "xhigh"),
            "system2": ("system2_model", "delegation_system2_model", "gpt-5.4", "xhigh"),
            "architecture": ("architecture_model", "delegation_architecture_model", "claude-sonnet-5", "high"),
            "long_context": ("long_context_model", "delegation_long_context_model", "gemini-3.1-pro-preview", "high"),
        }
        own_key, cfg_key, fallback_model, desired_effort = mapping.get(
            specialty,
            ("deep_code_model", "delegation_deep_code_model", "gpt-5.4", "xhigh"),
        )
        is_copilot = is_copilot_provider(provider)
        configured = str(
            own.get(own_key)
            or self._ctx.get_config(cfg_key, default="")
            or ""
        ).strip()
        if is_copilot:
            raw_model = normalize_copilot_model(configured or fallback_model)
            return raw_model, desired_effort
        return configured, desired_effort

    def route_task(self, task: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        own = self.settings(config)
        requested = str(task.get("cognitive_tier") or "auto").strip().lower()
        if requested not in VALID_COGNITIVE_TIERS:
            raise DelegationRouteError(
                f"Unsupported cognitive_tier {requested!r}; expected one of {', '.join(VALID_COGNITIVE_TIERS)}."
            )

        provider = str(
            own.get("provider")
            or self._ctx.get_config("delegation_provider", default="copilot")
            or "copilot"
        ).strip()

        specialty = requested
        if requested == "auto":
            goal_text = redact_text(str(task.get("goal") or "")[:1600])
            ctx_text = redact_text(str(task.get("context") or "")[:800])
            conf_threshold = float(self._ctx.get_config("confidence_threshold", default=0.82) or 0.82)
            backend = build_decision_backend(self._ctx)
            batch = backend.decide(
                {"goal": goal_text, "context": ctx_text},
                SUBAGENT_SPECIALTY_QUESTIONS,
            )
            ans = batch.answers.get("specialty") or batch.answers.get("tier")
            if batch.ok and ans and ans.choice:
                picked = ans.choice
                if picked == "system1":
                    picked = "fast_scout"
                elif picked == "system2":
                    picked = "deep_code"
                # Asymmetric precision guard on subagents: only use fast_scout when confidence >= threshold
                if picked == "fast_scout" and ans.top_probability < conf_threshold:
                    specialty = "deep_code"
                elif picked in ("fast_scout", "deep_code", "architecture", "long_context"):
                    specialty = picked
                else:
                    specialty = "deep_code"
            else:
                specialty = "deep_code"

        model_id, desired_effort = self._resolve_specialty_target(specialty, own, provider)
        if not is_copilot_provider(provider):
            if not model_id:
                return {"label": f"trismegistus:{specialty}"}
            return {
                "provider": provider,
                "model": model_id,
                "label": f"trismegistus:{specialty}",
            }

        route: dict[str, Any] = {
            "provider": provider,
            "model": model_id,
            "label": f"trismegistus:{specialty}",
        }
        supported = supported_reasoning_efforts(model_id, provider=provider)
        clamped = clamp_effort(desired_effort, supported)
        if clamped:
            route["reasoning_effort"] = clamped
        return route
