"""Stage 2 (Modulation) & Stage 3 (Execution): wire-safe request shaping, dynamic step pacing, and fail-open execution."""

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
    preserves_cache_on_effort_change,
)
from .providers_adapter import get_provider_adapter

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


def _pace_effort_for_step(
    base_effort: str,
    *,
    api_call_count: int,
    last_phase: str,
    read_streak: int,
    max_paced_reads: int = 2,
    last_read_chars: int = 0,
    last_read_complex: bool = False,
    cache_safe: bool = True,
) -> tuple[str, bool]:
    """Precision-first Dynamic Intra-Turn Step Pacing:

    - Call 1 (initial planning): full `base_effort` (`high` / `xhigh`).
    - Early intermediate calls (`read_streak` in `1..max_paced_reads`) immediately following small,
      clean read-only scout tools on cache-resilient families (e.g. OpenAI Responses / GPT-5+):
      pace down (`xhigh` -> `medium`, `high`/`medium` -> `low`) to cut 5-12s of redundant hidden
      thinking per scout call while keeping the exact same model and warm KV cache.
    - Synthesis / dense-read protection: once `read_streak > max_paced_reads`, or the last read
      returned a large/complex block (`>= 6000` chars or embedded error/stacktrace), restore full
      `base_effort` BEFORE the synthesis/patch step so root-cause diagnosis is never degraded.
    - Prompt-cache protection: when `cache_safe=False` (e.g. Claude, where effort changes alter the
      rendered prefix and can invalidate cache breakpoints), keep `base_effort` stable for lower TTFT.
    """
    if not cache_safe:
        return base_effort, False
    if api_call_count <= 1 or last_phase != "scout" or read_streak < 1:
        return base_effort, False
    if read_streak > max(1, int(max_paced_reads)):
        return base_effort, False
    if last_read_complex or last_read_chars >= 6000:
        return base_effort, False
    if base_effort in ("xhigh", "max"):
        return "medium", True
    if base_effort in ("high", "medium"):
        return "low", True
    return base_effort, False


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
    """Stage 2 (`llm_request` middleware): apply effort, step pacing, & wire-safe Copilot model selection."""
    verdict = registry.get(session_id)
    if verdict is None:
        return None

    mode = str(ctx.get_config("mode", default="shadow") or "shadow").strip()
    sticky_s2 = bool(ctx.get_config("sticky_system2_within_turn", default=True))
    step_pacing = bool(ctx.get_config("step_pacing_enabled", default=True))
    max_paced_reads = max(1, int(ctx.get_config("step_pacing_max_reads", default=2) or 2))
    protect_claude_cache = bool(ctx.get_config("step_pacing_protect_claude_cache", default=True))
    err_cap = max(1, int(ctx.get_config("escalate_after_errors", default=2) or 2))

    err_streak = int(ctx.state.get(f"err_streak:{session_id}", default=0) or 0) if session_id else 0
    forced_s2 = bool(ctx.state.get(f"force_s2:{session_id}", default=False)) if session_id else False
    last_phase = str(ctx.state.get(f"last_phase:{session_id}", default="turn_start") or "turn_start") if session_id else "turn_start"
    read_streak = int(ctx.state.get(f"read_streak:{session_id}", default=0) or 0) if session_id else 0
    last_read_chars = int(ctx.state.get(f"last_read_chars:{session_id}", default=0) or 0) if session_id else 0
    last_read_complex = bool(ctx.state.get(f"last_read_complex:{session_id}", default=False)) if session_id else False

    verdict.step_paced = False

    if err_streak >= err_cap or forced_s2:
        verdict.effective_tier = "system2"
        verdict.effective_effort = "xhigh"
        verdict.error_streak_override = err_streak >= err_cap
        verdict.sticky_override = verdict.sticky_override or forced_s2
        if session_id and forced_s2:
            ctx.state.set(f"force_s2:{session_id}", False)
    elif sticky_s2 and api_call_count > 1 and verdict.effective_tier == "system2":
        verdict.sticky_override = True

    active_provider = str(provider or "").strip().lower()
    current_model = str(request.get("model") or model or "").strip()
    adapter = get_provider_adapter(active_provider, api_mode)
    if api_mode:
        effective_wire = str(api_mode).strip()
    else:
        effective_wire = adapter.infer_api_mode(current_model, request)

    cache_safe = (
        True
        if not protect_claude_cache
        else preserves_cache_on_effort_change(
            current_model,
            provider=active_provider,
            api_mode=effective_wire,
            request=request,
        )
    )

    step_effort = verdict.effective_effort
    if step_pacing and err_streak == 0 and not forced_s2:
        step_effort, paced = _pace_effort_for_step(
            verdict.effective_effort,
            api_call_count=api_call_count,
            last_phase=last_phase,
            read_streak=read_streak,
            max_paced_reads=max_paced_reads,
            last_read_chars=last_read_chars,
            last_read_complex=last_read_complex,
            cache_safe=cache_safe,
        )
        verdict.step_paced = paced

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
        verdict.applied_effort = f"shadow({step_effort})"
        return {
            "request": dict(request),
            "source": "trismegistus",
            "reason": f"shadow:{verdict.summary_tag()} step_effort={step_effort} would_model={candidate_model}",
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
        effort=step_effort,
        effective_model=chosen_model or current_model,
        provider=active_provider,
        api_mode=effective_wire,
    )

    verdict.applied_model = chosen_model
    verdict.applied_effort = applied_effort or "native"

    reason = f"{mode}:{verdict.summary_tag()} applied_effort={verdict.applied_effort} model={chosen_model}"
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
