"""Trismegistus — Three-stage System-1 / System-2 cognitive router for Hermes Agent.

Stages:
  1. Triage (`pre_llm_call`, `pre_tool_call`, `pre_verify`):
     Evaluates turn complexity, effort, and safety via Microsoft-Decision-1.1 on Foundry
     (`/providers/microsoft/v1/systemone`) or a local decision model.
  2. Modulation (`llm_request` middleware):
     Adjusts `reasoning_effort` / `reasoning.effort` and wire-compatible GitHub Copilot models
     while preserving KV prompt caching and signed reasoning replay carriers.
  3. Call & Delegation (`llm_execution` middleware, `post_api_request`, `DelegationProvider`):
     Executes with fail-open entitlement recovery, records turn-deduplicated telemetry to a
     profile-scoped SQLite ledger, and routes `delegate_task` subagents across System-1 / System-2 Copilot models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .backends import build_decision_backend
from .compat import redact_text
from .contracts import DecisionQuestion
from .delegation import TrismegistusDelegationProvider
from .ledger import handle_slash_command, record_turn_completion
from .modulation import TurnSessionRegistry, execute_with_failopen, modulate_llm_request
from .triage import (
    MUTATING_TOOLS,
    READ_ONLY_SCOUT_TOOLS,
    evaluate_tool_risk,
    evaluate_verify_gate,
    run_turn_triage,
)

__version__ = "0.2.2"

DECIDE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "trismegistus_decide",
    "description": (
        "Run a fast calibrated System-1 decision query (Microsoft-Decision-1.1 / local decision head) "
        "over structured state using typed questions: 'noul' (yes/no probability 0..1), "
        "'choice' (option probabilities), or 'score' (expected ordinal level)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "description": "Text or JSON object describing the situation to evaluate.",
                "anyOf": [{"type": "string"}, {"type": "object"}],
            },
            "question_type": {
                "type": "string",
                "enum": ["noul", "choice", "score"],
                "description": "Primitive decision type: 'noul' (boolean probability), 'choice' (pick option), or 'score' (ordered scale).",
            },
            "instructions": {
                "type": "string",
                "description": "Clear, neutral decision question to evaluate against state.",
            },
            "options": {
                "type": "object",
                "description": "Required for 'choice' (map of option_name -> description) and optional for 'noul' ({'true': ..., 'false': ...}).",
            },
            "levels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required for 'score': ordered level labels from lowest (0) to highest (N-1).",
            },
        },
        "required": ["state", "question_type", "instructions"],
    },
}

ESCALATE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "trismegistus_escalate",
    "description": (
        "Explicitly escalate the current session turn from System-1 to System-2 deep deliberative "
        "reasoning (high effort / frontier model) when encountering subtle bugs, architectural "
        "trade-offs, or unexpected ambiguity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Concise explanation of why deep System-2 deliberation is needed.",
            }
        },
        "required": ["reason"],
    },
}


def _handle_decide_tool(ctx: Any, params: dict[str, Any]) -> str:
    q_type = str(params.get("question_type") or "").strip()
    instructions = str(params.get("instructions") or "").strip()
    raw_state = params.get("state")
    if not instructions:
        return json.dumps({"ok": False, "error": "instructions must not be empty"})

    safe_state: Any
    if isinstance(raw_state, str):
        safe_state = redact_text(raw_state[:8000])
    else:
        safe_state = json.loads(redact_text(json.dumps(raw_state, ensure_ascii=False)[:8000]))

    try:
        if q_type == "noul":
            opts = params.get("options") if isinstance(params.get("options"), dict) else {}
            q = DecisionQuestion.noul(
                instructions,
                when_true=opts.get("true"),
                when_false=opts.get("false"),
            )
        elif q_type == "choice":
            opts = params.get("options")
            if not isinstance(opts, dict) or len(opts) < 2:
                return json.dumps({"ok": False, "error": "'choice' requires an 'options' object with >=2 entries"})
            q = DecisionQuestion.choice(instructions, {str(k): str(v) for k, v in opts.items()})
        elif q_type == "score":
            levels = params.get("levels")
            if not isinstance(levels, list) or len(levels) < 2:
                return json.dumps({"ok": False, "error": "'score' requires a 'levels' array with >=2 strings"})
            q = DecisionQuestion.score(instructions, [str(x) for x in levels])
        else:
            return json.dumps({"ok": False, "error": f"Unsupported question_type: {q_type!r}"})
    except ValueError as exc:
        return json.dumps({"ok": False, "error": str(exc)})

    backend = build_decision_backend(ctx)
    batch = backend.decide(safe_state, {"decision": q})
    if not batch.ok or "decision" not in batch.answers:
        return json.dumps(
            {
                "ok": False,
                "backend": batch.backend,
                "model": batch.model,
                "latency_ms": round(batch.latency_ms, 1),
                "error": redact_text(str(batch.error or "No answer returned")[:240]),
            }
        )

    return json.dumps(
        {
            "ok": True,
            "backend": batch.backend,
            "model": batch.model,
            "latency_ms": round(batch.latency_ms, 1),
            "answer": batch.answers["decision"].to_dict(),
        }
    )


def register(ctx: Any) -> None:
    """Hermes plugin entry point wiring Triage -> Modulation -> Call."""
    registry = TurnSessionRegistry()

    # Auxiliary model slot for zero-config fallback (`auxiliary.trismegistus_decider`)
    ctx.register_auxiliary_task(
        "trismegistus_decider",
        display_name="Trismegistus System-1 Decider",
        description="Fast model used when backend='hermes_aux' for System-1 / System-2 triage.",
        defaults={"timeout": 10},
    )

    # Delegation backend (`delegation.backend: trismegistus`)
    ctx.register_delegation_provider(TrismegistusDelegationProvider(ctx))

    # Bundled skill (`skill_view("trismegistus:trismegistus")`)
    skill_md = Path(__file__).parent / "skills" / "trismegistus" / "SKILL.md"
    if skill_md.is_file():
        ctx.register_skill("trismegistus", skill_md)

    # Tools
    ctx.register_tool(
        name="trismegistus_decide",
        toolset="trismegistus",
        schema=DECIDE_TOOL_SCHEMA,
        handler=lambda params, **kw: _handle_decide_tool(ctx, params or {}),
    )

    def _handle_escalate(params: dict[str, Any], session_id: str = "", **kwargs: Any) -> str:
        del kwargs
        reason = str((params or {}).get("reason") or "agent requested System-2 escalation")[:300]
        if session_id:
            ctx.state.set(f"force_s2:{session_id}", True)
        verdict = registry.get(session_id)
        if verdict is not None:
            verdict.effective_tier = "system2"
            verdict.effective_effort = "xhigh"
            verdict.sticky_override = True
        return json.dumps(
            {
                "ok": True,
                "escalated_to": "system2",
                "effective_effort": "xhigh",
                "reason": reason,
            }
        )

    ctx.register_tool(
        name="trismegistus_escalate",
        toolset="trismegistus",
        schema=ESCALATE_TOOL_SCHEMA,
        handler=_handle_escalate,
    )

    # Stage 1: Triage hooks
    def on_pre_llm_call(
        session_id: str = "",
        user_message: Any = "",
        conversation_history: list[Any] | None = None,
        model: str = "",
        platform: str = "",
        **kwargs: Any,
    ) -> dict[str, str] | None:
        del kwargs
        verdict = run_turn_triage(
            ctx,
            session_id=session_id,
            user_message=user_message,
            conversation_history=conversation_history,
            model=model,
            platform=platform,
        )
        registry.put(session_id, verdict)

        mode = str(ctx.get_config("mode", default="shadow") or "shadow").strip()
        if mode != "shadow" and verdict.effective_tier == "system2" and verdict.needs_plan_prob >= 0.85:
            return {
                "context": (
                    "[Trismegistus System-2 Note: High structural complexity detected "
                    f"(p={verdict.needs_plan_prob:.2f}). Verify assumptions before mutating files.]"
                )
            }
        return None

    def on_pre_tool_call(
        tool_name: str = "",
        args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        del kwargs
        if not bool(ctx.get_config("tool_gate_enabled", default=False)):
            return None
        if str(ctx.get_config("mode", default="shadow")) == "shadow":
            return None
        if tool_name not in MUTATING_TOOLS:
            return None
        threshold = float(ctx.get_config("tool_risk_threshold", default=0.82) or 0.82)
        risk_p, latency_ms, _err = evaluate_tool_risk(ctx, tool_name=tool_name, args=args or {})
        if risk_p >= threshold:
            return {
                "action": "approve",
                "message": (
                    f"Trismegistus System-1 Gate flagged {tool_name} as high-impact "
                    f"(p={risk_p:.2f}, {latency_ms:.0f}ms)."
                ),
                "rule_key": f"trismegistus:{tool_name}",
            }
        return None

    def on_post_tool_call(
        tool_name: str = "",
        result: str = "",
        status: str = "",
        session_id: str = "",
        **kwargs: Any,
    ) -> None:
        del kwargs
        if not session_id:
            return
        err_key = f"err_streak:{session_id}"
        read_key = f"read_streak:{session_id}"
        phase_key = f"last_phase:{session_id}"

        head = str(result or "")[:500]
        is_error = status == "error" or '"error":' in head or '"success": false' in head.lower()
        current_err = int(ctx.state.get(err_key, default=0) or 0)
        if is_error:
            ctx.state.set(err_key, current_err + 1)
            ctx.state.set(read_key, 0)
            ctx.state.set(phase_key, "verify_or_act")
            return

        ctx.state.set(err_key, 0)
        if tool_name in READ_ONLY_SCOUT_TOOLS:
            cur_reads = int(ctx.state.get(read_key, default=0) or 0)
            ctx.state.set(read_key, cur_reads + 1)
            ctx.state.set(phase_key, "scout")
        else:
            ctx.state.set(read_key, 0)
            ctx.state.set(phase_key, "verify_or_act")

    def on_pre_verify(
        coding: bool = False,
        attempt: int = 0,
        final_response: str = "",
        changed_paths: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, str] | None:
        del kwargs
        if attempt > 0 or not coding or not changed_paths:
            return None
        if not bool(ctx.get_config("verify_gate_enabled", default=True)):
            return None
        if str(ctx.get_config("mode", default="shadow")) == "shadow":
            return None
        verified_p, _ms, _err = evaluate_verify_gate(
            ctx,
            changed_paths=changed_paths,
            final_response=final_response,
        )
        if verified_p < 0.35:
            return {
                "action": "continue",
                "message": (
                    "Trismegistus Verify Gate: Code files were modified, but your response lacks "
                    "concrete verification output. Run the relevant tests or syntax check before finishing."
                ),
            }
        return None

    # Stage 2: Modulation middleware
    def on_llm_request(
        request: dict[str, Any],
        session_id: str = "",
        provider: str = "",
        model: str = "",
        api_mode: str = "",
        api_call_count: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        del kwargs
        return modulate_llm_request(
            ctx,
            registry,
            request=request,
            session_id=session_id,
            provider=provider,
            model=model,
            api_mode=api_mode,
            api_call_count=api_call_count,
        )

    # Stage 3: Call execution middleware + ledger observer
    def on_llm_execution(
        request: dict[str, Any],
        original_request: dict[str, Any] | None = None,
        next_call: Any = None,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        return execute_with_failopen(
            request=request,
            original_request=original_request,
            next_call=next_call,
        )

    def on_post_api_request(
        session_id: str = "",
        api_duration: float = 0.0,
        usage: Any = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        verdict = registry.get(session_id)
        if verdict is None:
            return
        mode = str(ctx.get_config("mode", default="shadow") or "shadow")
        in_tok = 0
        out_tok = 0
        if isinstance(usage, dict):
            in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        elif usage is not None:
            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        try:
            record_turn_completion(
                verdict=verdict,
                mode=mode,
                provider_duration_ms=float(api_duration or 0.0) * 1000.0,
                provider_input_tokens=in_tok,
                provider_output_tokens=out_tok,
            )
        except Exception:
            pass

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_verify", on_pre_verify)
    ctx.register_hook("post_api_request", on_post_api_request)

    ctx.register_middleware("llm_request", on_llm_request)
    ctx.register_middleware("llm_execution", on_llm_execution)

    ctx.register_command(
        "trismegistus",
        handler=lambda raw_args, session_id="", **kw: handle_slash_command(
            ctx, raw_args, session_id=session_id
        ),
        description="Inspect or switch Trismegistus System-1 / System-2 cognitive routing",
        args_hint="[status|stats|last|mode <shadow|effort_only|tier_and_effort>|escalate]",
    )
