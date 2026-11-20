"""Stage 2 (Modulation) & Stage 3 (Execution): wire-safe request shaping and fail-open execution."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .contracts import TriageVerdict
from .copilot_lanes import (
    apply_effort_to_request,
    can_safely_swap_model,
    default_copilot_peer,
    infer_copilot_api_mode,
    normalize_copilot_model,
)

logger = logging.getLogger("plugins.trismegistus")


class TurnSessionRegistry:
    """Thread-safe in-memory store of active `TriageVerdict` per session/turn."""

    def __init__(self, max_sessions: int = 256) -> None:
        self._lock = threading.Lock()
        self._verdicts: dict[str, TriageVerdict] = {}
        self._max_sessions = max_sessions

    def put(self, session_id: str, verdict: TriageVerdict) -> None:
        key = session_id or "_default"
        with self._lock:
            if key not in self._verdicts and len(self._verdicts) >= self._max_sessions:
                oldest = next(iter(self._verdicts))
                self._verdicts.pop(oldest, None)
            self._verdicts[key] = verdict

    def get(self, session_id: str) -> TriageVerdict | None:
        key = session_id or "_default"
        with self._lock:
            return self._verdicts.get(key)


def modulate_llm_request(
    ctx: Any,
    registry: TurnSessionRegistry,
    *,
    request: dict[str, Any],
    session_id: str = "",
    provider: str = "",
    model: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
) -> dict[str, Any] | None:
    """Stage 2 (`llm_request` middleware): apply effort & wire-safe Copilot model selection."""
    verdict = registry.get(session_id)
    if verdict is None:
        return None

    mode = str(ctx.get_config("mode", default="shadow") or "shadow").strip()
    sticky_s2 = bool(ctx.get_config("sticky_system2_within_turn", default=True))
    err_cap = max(1, int(ctx.get_config("escalate_after_errors", default=2) or 2))

    # Mid-turn metacognitive escalation: check if tool errors or explicit escalate tool fired
    err_streak = int(ctx.state.get(f"err_streak:{session_id}", default=0) or 0) if session_id else 0
    forced_s2 = bool(ctx.state.get(f"force_s2:{session_id}", default=False)) if session_id else False

    if err_streak >= err_cap or forced_s2:
        verdict.effective_tier = "system2"
        verdict.effective_effort = "high"
        verdict.error_streak_override = err_streak >= err_cap
        verdict.sticky_override = verdict.sticky_override or forced_s2
        if session_id and forced_s2:
            ctx.state.set(f"force_s2:{session_id}", False)
    elif sticky_s2 and api_call_count > 1 and verdict.effective_tier == "system2":
        # Never downgrade mid-turn once System 2 KV cache is warm
        verdict.sticky_override = True

    active_provider = str(provider or "").strip().lower()
    current_model = str(request.get("model") or model or "").strip()
    effective_wire = str(api_mode or "").strip() or (
        infer_copilot_api_mode(current_model)
        if active_provider in ("copilot", "github-copilot", "github")
        else "chat_completions"
    )

    # Compute target model candidate (for shadow telemetry and tier_and_effort mode)
    cfg_s1 = str(ctx.get_config("system1_model", default="") or "").strip()
    cfg_s2 = str(ctx.get_config("system2_model", default="") or "").strip()
    if verdict.effective_tier == "system1":
        candidate_model = cfg_s1 or (
            default_copilot_peer(current_model, "system1", effective_wire)
            if active_provider in ("copilot", "github-copilot", "github")
            else current_model
        )
    else:
        candidate_model = cfg_s2 or (
            default_copilot_peer(current_model, "system2", effective_wire)
            if active_provider in ("copilot", "github-copilot", "github")
            else current_model
        )

    if active_provider in ("copilot", "github-copilot", "github"):
        candidate_model = normalize_copilot_model(candidate_model)

    if mode == "shadow":
        verdict.applied_model = current_model
        verdict.applied_effort = f"shadow({verdict.effective_effort})"
        return {
            "request": dict(request),
            "source": "trismegistus",
            "reason": f"shadow:{verdict.summary_tag()} would_model={candidate_model}",
        }

    chosen_model = current_model
    guard_note = ""
    has_model_cap = True
    if hasattr(ctx, "has_capability"):
        try:
            probe_cap = ctx.has_capability("llm.model_override")
            if isinstance(probe_cap, bool):
                has_model_cap = probe_cap
        except Exception:
            has_model_cap = True

    if mode == "tier_and_effort" and candidate_model and candidate_model != current_model:
        if not has_model_cap:
            verdict.wire_guarded = True
            guard_note = "missing_capability(llm.model_override)"
        else:
            safe, why = can_safely_swap_model(
                provider=active_provider,
                api_mode=effective_wire,
                current_model=current_model,
                candidate_model=candidate_model,
                request=request,
            )
            if safe:
                chosen_model = candidate_model
            else:
                verdict.wire_guarded = True
                guard_note = why

    updated = dict(request)
    if chosen_model and chosen_model != current_model:
        updated["model"] = chosen_model

    updated, applied_effort = apply_effort_to_request(
        updated,
        effort=verdict.effective_effort,
        effective_model=chosen_model or current_model,
        provider=active_provider,
        api_mode=effective_wire,
    )

    verdict.applied_model = chosen_model
    verdict.applied_effort = applied_effort or "native"

    reason = f"{mode}:{verdict.summary_tag()} model={chosen_model}"
    if guard_note:
        reason = f"{reason} guard={guard_note}"

    return {
        "request": updated,
        "source": "trismegistus",
        "reason": reason,
    }


def execute_with_failopen(
    *,
    request: dict[str, Any],
    original_request: dict[str, Any] | None,
    next_call: Any,
) -> Any:
    """Stage 3 (`llm_execution` middleware): execute call and fail-open to original model if rejected."""
    orig_model = (original_request or {}).get("model")
    req_model = request.get("model")
    swapped_model = bool(orig_model and req_model and orig_model != req_model)

    started = time.perf_counter()
    try:
        return next_call(request)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        msg = str(exc).lower()
        # If our swapped model isn't entitled on this Copilot seat (400/403/404 model_not_supported),
        # retry transparently once with the session's original request kwargs!
        entitlement_err = status in (400, 403, 404, 422) or any(
            tok in msg for tok in ("model_not_supported", "model_not_found", "unsupported model", "does not exist")
        )
        if swapped_model and original_request and entitlement_err:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.warning(
                "Trismegistus model %r rejected after %.0fms (%s); failing open to session model %r",
                req_model,
                elapsed_ms,
                exc,
                orig_model,
            )
            return next_call(dict(original_request))
        raise
