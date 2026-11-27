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
from .ledger import handle_slash_command, record_turn_completion, record_turn_outcome
from .modulation import TurnSessionRegistry, execute_with_failopen, modulate_llm_request
from .triage import (
    MUTATING_TOOLS,
    READ_ONLY_SCOUT_TOOLS,
    _STACKTRACE_RE,
    evaluate_tool_risk,
    evaluate_verify_gate,
    run_turn_triage,
)

__version__ = "0.3.2"

_CONSTANT_SYSTEM2_PLAN_NOTE = (
    "[Trismegistus System-2 Note: High structural complexity detected. "
    "Verify assumptions and invariants before mutating files.]"
)


def _extract_usage_metrics(usage: Any) -> tuple[int, int, int, int, int]:
    """Extract `(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)`.

    Supports OpenAI Responses (`input_tokens_details`), OpenAI Chat (`prompt_tokens_details`,
    `completion_tokens_details`), Anthropic (`cache_read_input_tokens`, `cache_creation_input_tokens`),
    and Gemini usage payloads.
    """
    if usage is None:
        return 0, 0, 0, 0, 0

    def _get(obj: Any, *keys: str) -> Any:
        for k in keys:
            if isinstance(obj, dict):
                v = obj.get(k)
            else:
                v = getattr(obj, k, None)
            if v is not None:
                return v
        return None

    in_tok = int(_get(usage, "input_tokens", "prompt_tokens") or 0)
    out_tok = int(_get(usage, "output_tokens", "completion_tokens") or 0)

    in_details = _get(usage, "input_tokens_details", "prompt_tokens_details")
    out_details = _get(usage, "output_tokens_details", "completion_tokens_details")

    cache_read = int(
        _get(
            usage,
            "cache_read_tokens",
            "cache_read_input_tokens",
            "cached_input_tokens",
            "cached_tokens",
            "prompt_cache_hit_tokens",
        )
        or (_get(in_details, "cached_tokens", "cache_read_tokens") if in_details is not None else 0)
        or 0
    )
    cache_write = int(
        _get(usage, "cache_write_tokens", "cache_creation_input_tokens", "cache_creation_tokens")
        or (_get(in_details, "cache_creation_tokens", "cache_write_tokens") if in_details is not None else 0)
        or 0
    )
    reasoning_tok = int(
        _get(usage, "reasoning_tokens", "thoughts_token_count")
        or (_get(out_details, "reasoning_tokens") if out_details is not None else 0)
        or 0
    )
    return in_tok, out_tok, cache_read, cache_write, reasoning_tok


def _is_tool_result_error(tool_name: str, result: Any, status: str) -> bool:
    """Detect real tool failures without false positives on file contents containing `'error':`."""
    if str(status or "").strip().lower() == "error":
        return True
    # Read-only scout tools return raw source files / JSON documents that legitimately contain
    # substrings like `"error":` — never treat file contents as a tool execution failure!
    if tool_name in READ_ONLY_SCOUT_TOOLS:
        return False
    raw = str(result or "").strip()
    if not raw:
        return False
    if raw.startswith("{") and len(raw) <= 65536:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                if obj.get("success") is False or obj.get("ok") is False:
                    return True
                err_val = obj.get("error")
                if err_val not in (None, False, "", 0):
                    return True
                exit_code = obj.get("exit_code")
                if isinstance(exit_code, int) and exit_code != 0:
                    return True
                return False
        except Exception:
            pass
    head = raw[:300].lower()
    return '"success": false' in head or '"ok": false' in head

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
            if verdict.initial_effective_tier == "system1" and verdict.turn_outcome == "clean":
                verdict.turn_outcome = "false_s1_escalated"
                try:
                    record_turn_outcome(verdict.turn_id, verdict.turn_outcome)
                except Exception:
                    pass
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
            # Constant string (no variable float) so prompt prefix caches stay warm across turns.
            return {"context": _CONSTANT_SYSTEM2_PLAN_NOTE}
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
        chars_key = f"last_read_chars:{session_id}"
        complex_key = f"last_read_complex:{session_id}"

        is_error = _is_tool_result_error(tool_name, result, status)
        current_err = int(ctx.state.get(err_key, default=0) or 0)
        if is_error:
            new_err = current_err + 1
            ctx.state.set(err_key, new_err)
            ctx.state.set(read_key, 0)
            ctx.state.set(phase_key, "verify_or_act")
            ctx.state.set(chars_key, 0)
            ctx.state.set(complex_key, False)
            err_cap = max(1, int(ctx.get_config("escalate_after_errors", default=2) or 2))
            if new_err >= err_cap:
                verdict = registry.get(session_id)
                if verdict is not None and verdict.initial_effective_tier == "system1" and verdict.turn_outcome == "clean":
                    verdict.turn_outcome = "false_s1_err_streak"
                    try:
                        record_turn_outcome(verdict.turn_id, verdict.turn_outcome)
                    except Exception:
                        pass
            return

        ctx.state.set(err_key, 0)
        if tool_name in READ_ONLY_SCOUT_TOOLS:
            cur_reads = int(ctx.state.get(read_key, default=0) or 0)
            result_str = str(result or "")
            ctx.state.set(read_key, cur_reads + 1)
            ctx.state.set(phase_key, "scout")
            ctx.state.set(chars_key, len(result_str))
            ctx.state.set(complex_key, bool(_STACKTRACE_RE.search(result_str[:4000])))
        else:
            ctx.state.set(read_key, 0)
            ctx.state.set(phase_key, "verify_or_act")
            ctx.state.set(chars_key, 0)
            ctx.state.set(complex_key, False)

    def on_pre_verify(
        coding: bool = False,
        attempt: int = 0,
        final_response: str = "",
        changed_paths: list[str] | None = None,
        session_id: str = "",
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
            if session_id:
                verdict = registry.get(session_id)
                if verdict is not None and verdict.initial_effective_tier == "system1" and verdict.turn_outcome == "clean":
                    verdict.turn_outcome = "false_s1_verify_failed"
                    try:
                        record_turn_outcome(verdict.turn_id, verdict.turn_outcome)
                    except Exception:
                        pass
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
        in_tok, out_tok, cache_read, cache_write, reasoning_tok = _extract_usage_metrics(usage)
        try:
            record_turn_completion(
                verdict=verdict,
                mode=mode,
                provider_duration_ms=float(api_duration or 0.0) * 1000.0,
                provider_input_tokens=in_tok,
                provider_output_tokens=out_tok,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                reasoning_tokens=reasoning_tok,
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
        args_hint="[status|stats|last|outcomes|calibrate|mode <shadow|effort_only|tier_and_effort>|escalate]",
    )
