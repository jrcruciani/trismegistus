"""Stage 1 (Triage): precision-calibrated System-1 turn classification, trap gate, and verify gate."""

from __future__ import annotations

import json
import re
from typing import Any

from .backends import build_decision_backend
from .calibration import compute_closed_loop_adjustment, load_calibration_profile
from .compat import redact_text
from .contracts import EFFORT_LADDER, CognitiveTier, DecisionQuestion, TriageVerdict
from .ledger import recent_closed_loop_window

TURN_QUESTIONS: dict[str, DecisionQuestion] = {
    "tier": DecisionQuestion.choice(
        "Which cognitive regime should handle this user request?",
        {
            "system1": (
                "Direct factual answer, greeting, simple formatting, translation, "
                "single-command lookup, or straightforward mechanical follow-up"
            ),
            "system2": (
                "Multi-step software engineering, root-cause debugging, architecture design, "
                "reverse engineering, security analysis, ambiguous trade-offs, or multi-file edits"
            ),
        },
    ),
    "effort": DecisionQuestion.score(
        "How much deliberative reasoning depth does this request require?",
        [
            "Minimal: trivial acknowledgement, pure formatting, or simple lookup",
            "Low: routine single-step task or clear localized edit",
            "Medium: multi-step reasoning, code analysis, or structured synthesis",
            "High: complex debugging, multi-file refactor, or architectural design",
            "XHigh: subtle race condition, cryptographic/security invariant, formal root-cause proof, or deep reverse engineering",
        ],
    ),
    "trap_risk": DecisionQuestion.noul(
        "Could this request hide subtle side effects, concurrency/state hazards, security/auth invariants, "
        "edge-case pitfalls, or implicit constraints from prior context despite looking short?",
        when_true="Contains hidden coupling, subtle edge cases, security/state risk, or implicit context dependencies",
        when_false="Self-contained, transparent request with low risk of subtle misinterpretation",
    ),
    "needs_plan": DecisionQuestion.noul(
        "Should the agent formulate an explicit step-by-step verification plan before acting?",
        when_true="Complex multi-step change where acting without a plan risks breaking invariants",
        when_false="Direct question or bounded task that needs no upfront planning scaffold",
    ),
}

MUTATING_TOOLS: frozenset[str] = frozenset({"terminal", "write_file", "patch", "browser_exec"})
READ_ONLY_SCOUT_TOOLS: frozenset[str] = frozenset(
    {"read_file", "search_files", "skills_list", "skill_view", "session_search", "web_search", "web_extract"}
)


_STACKTRACE_RE = re.compile(
    r"(Traceback \(most recent call last\):|\bat\s+[A-Za-z0-9_$.]+\([^)]+:\d+\)|panic:|segmentation fault|SIGSEGV|goroutine \d+)",
    re.IGNORECASE,
)
_DIFF_HUNK_RE = re.compile(r"(^diff --git |^@@ -\d+,\d+ \+\d+,\d+ @@)", re.MULTILINE)
_FILE_PATH_RE = re.compile(r"(?:^|[\s`'\"(])([\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|c|cc|cpp|h|java|rb|sql|yaml|yml|toml|json|sh))\b")


def _head_tail_excerpt(raw: str, max_chars: int) -> str:
    """Preserve both the opening context (head 60%) and the closing question/request (tail 40%)."""
    budget = max(200, int(max_chars))
    if len(raw) <= budget:
        return raw
    marker = "\n...[truncated middle]...\n"
    usable = max(120, budget - len(marker))
    head_len = int(usable * 0.60)
    tail_len = usable - head_len
    return raw[:head_len] + marker + raw[-tail_len:]


def _extract_user_text(user_message: Any, max_chars: int) -> str:
    text, _meta = _extract_user_state(user_message, max_chars)
    return text


def _extract_user_state(user_message: Any, max_chars: int) -> tuple[str, dict[str, Any]]:
    """Extract redacted head+tail text plus multimodal & structural signals."""
    image_refs: list[dict[str, Any]] = []
    attachment_types: list[str] = []

    if isinstance(user_message, str):
        raw = user_message
    elif isinstance(user_message, list):
        parts: list[str] = []
        for item in user_message:
            if isinstance(item, dict):
                itype = str(item.get("type") or "").lower()
                if itype in ("text", "input_text"):
                    parts.append(str(item.get("text") or ""))
                elif itype in ("image_url", "image", "input_image"):
                    attachment_types.append("image")
                    url_val = ""
                    img_obj = item.get("image_url")
                    if isinstance(img_obj, dict):
                        url_val = str(img_obj.get("url") or "")
                    elif isinstance(img_obj, str):
                        url_val = img_obj
                    elif isinstance(item.get("url"), str):
                        url_val = str(item["url"])
                    # Keep safe bounded HTTPS URLs or small data URLs for multimodal Decision-1.1
                    if url_val.startswith("https://") and len(url_val) <= 2048:
                        image_refs.append({"type": "image_url", "image_url": {"url": redact_text(url_val)}})
                    elif url_val.startswith("data:image/") and len(url_val) <= 250_000:
                        image_refs.append({"type": "image_url", "image_url": {"url": url_val}})
                elif itype in ("file", "input_file", "audio", "input_audio", "video"):
                    attachment_types.append(itype)
            elif isinstance(item, str):
                parts.append(item)
        raw = "\n".join(parts)
    else:
        raw = str(user_message or "")

    redacted = redact_text(_head_tail_excerpt(raw, max_chars))
    has_stacktrace = bool(_STACKTRACE_RE.search(raw))
    diff_hunks = len(_DIFF_HUNK_RE.findall(raw))
    has_code_blocks = "```" in raw
    mentioned_files = sorted(set(_FILE_PATH_RE.findall(raw)))[:20]

    meta: dict[str, Any] = {
        "char_length": len(raw),
        "has_images": "image" in attachment_types,
        "image_count": attachment_types.count("image"),
        "attachment_types": sorted(set(attachment_types)),
        "has_code_blocks": has_code_blocks,
        "has_stacktrace": has_stacktrace,
        "has_diff": diff_hunks >= 2,
        "diff_hunk_count": diff_hunks,
        "mentioned_files_count": len(mentioned_files),
        "image_refs": image_refs[:4],
    }
    return redacted, meta


def _evaluate_s2_fast_path(
    raw_text: str,
    meta: dict[str, Any],
    *,
    backend_supports_multimodal: bool,
) -> tuple[bool, str, str]:
    """Sub-millisecond precision-first fast-path strictly upward to System 2.

    NEVER fast-paths downward to System 1 (short prompts can hide subtle traps).
    Saves 150-500ms of triage network latency on unmistakably deep engineering inputs.
    Returns `(triggered, reason, effort)`.
    """
    if meta.get("has_diff") and int(meta.get("diff_hunk_count", 0)) >= 3:
        return True, "multi_hunk_diff", "high"
    if meta.get("has_stacktrace") and (meta.get("has_code_blocks") or int(meta.get("char_length", 0)) >= 600):
        return True, "stacktrace_debug", "xhigh"
    if meta.get("has_images") and not backend_supports_multimodal and len(raw_text.strip()) < 160:
        return True, "multimodal_blind_guard", "high"
    if (
        int(meta.get("char_length", 0)) >= 4500
        and meta.get("has_code_blocks")
        and int(meta.get("mentioned_files_count", 0)) >= 3
    ):
        return True, "large_multifile_code", "high"
    return False, "", ""


def _tail_digest(conversation_history: list[Any] | None, max_chars: int = 800) -> list[dict[str, Any]]:
    if not isinstance(conversation_history, list) or not conversation_history:
        return []
    digest: list[dict[str, Any]] = []
    budget = max_chars
    for msg in reversed(conversation_history[-6:]):
        if not isinstance(msg, dict) or budget <= 0:
            break
        role = str(msg.get("role") or "")
        if role == "system":
            continue
        content = msg.get("content")
        snippet = content if isinstance(content, str) else ""
        snippet = redact_text(snippet[: min(240, budget)])
        tool_count = len(msg.get("tool_calls") or ()) if isinstance(msg.get("tool_calls"), list) else 0
        budget -= len(snippet)
        digest.append({"role": role, "preview": snippet, "tool_calls": tool_count})
    digest.reverse()
    return digest


def score_to_effort(score: float, effective_tier: CognitiveTier, *, trap_risk_prob: float = 0.0) -> str:
    """Map a probability-weighted 0..4 score onto Hermes's 5-rung precision effort ladder."""
    if score >= 3.35 or (score >= 2.80 and trap_risk_prob >= 0.55):
        effort = "xhigh"
    else:
        clamped_idx = max(0, min(len(EFFORT_LADDER) - 1, round(score)))
        effort = EFFORT_LADDER[clamped_idx]

    if effective_tier == "system2" and effort in ("minimal", "low"):
        return "high" if trap_risk_prob >= 0.60 else "medium"
    if effective_tier == "system1" and effort in ("high", "xhigh"):
        return "low"
    return effort


def run_turn_triage(
    ctx: Any,
    *,
    session_id: str,
    user_message: Any,
    conversation_history: list[Any] | None = None,
    model: str = "",
    platform: str = "",
    opener_factory: Any = None,
) -> TriageVerdict:
    """Run Stage 1 (Triage) in a single multi-question System-1 pass with precision fast-paths."""
    mode = str(ctx.get_config("mode", default="shadow") or "shadow").strip()
    backend_kind = str(ctx.get_config("backend", default="foundry_systemone") or "foundry_systemone").strip()
    max_chars = int(ctx.get_config("max_excerpt_chars", default=4000) or 4000)
    conf_threshold = float(ctx.get_config("confidence_threshold", default=0.82) or 0.82)
    trap_threshold = float(ctx.get_config("trap_risk_threshold", default=0.50) or 0.50)
    err_cap = max(1, int(ctx.get_config("escalate_after_errors", default=2) or 2))
    fast_path_enabled = bool(ctx.get_config("fast_path_enabled", default=True))
    use_calibrated = bool(ctx.get_config("use_calibrated_thresholds", default=True))
    closed_loop_enabled = bool(ctx.get_config("closed_loop_enabled", default=True))
    multimodal_enabled = bool(ctx.get_config("multimodal_triage_enabled", default=True))

    calibration_source = "config"
    if use_calibrated:
        cal_profile = load_calibration_profile(backend_kind)
        if cal_profile is not None:
            conf_threshold = cal_profile.confidence_threshold
            trap_threshold = cal_profile.trap_risk_threshold
            calibration_source = f"conformal:{cal_profile.backend}"

    if closed_loop_enabled:
        try:
            recent_s1, recent_false_s1 = recent_closed_loop_window(backend_kind, limit=50)
            target_alpha = float(ctx.get_config("closed_loop_target_alpha", default=0.03) or 0.03)
            adj_conf, adj_trap, tightened = compute_closed_loop_adjustment(
                base_conf_threshold=conf_threshold,
                base_trap_threshold=trap_threshold,
                recent_s1_turns=recent_s1,
                recent_false_s1_turns=recent_false_s1,
                target_alpha=target_alpha,
            )
            if tightened:
                conf_threshold = adj_conf
                trap_threshold = adj_trap
                calibration_source = f"{calibration_source}+closed_loop"
        except Exception:
            pass

    err_streak = int(ctx.state.get(f"err_streak:{session_id}", default=0) or 0) if session_id else 0
    forced_s2 = bool(ctx.state.get(f"force_s2:{session_id}", default=False)) if session_id else False

    # Reset per-turn step-pacing counters on fresh user turn entry
    if session_id:
        ctx.state.set(f"read_streak:{session_id}", 0)
        ctx.state.set(f"last_phase:{session_id}", "turn_start")
        ctx.state.set(f"last_read_chars:{session_id}", 0)
        ctx.state.set(f"last_read_complex:{session_id}", False)

    user_excerpt, meta = _extract_user_state(user_message, max_chars)
    is_multimodal = bool(meta.get("has_images") or meta.get("attachment_types"))
    error_override = err_streak >= err_cap

    # Speed optimization: in active modes, skip the network triage RTT entirely when System-2 xhigh
    # is already forced by explicit escalation or error streak!
    if mode != "shadow" and (forced_s2 or error_override):
        if session_id and forced_s2:
            ctx.state.set(f"force_s2:{session_id}", False)
        reason = "forced_s2" if forced_s2 else "err_streak"
        return TriageVerdict(
            session_id=session_id,
            raw_tier="system2",
            effective_tier="system2",
            initial_effective_tier="system2",
            effort_score=3.8,
            effective_effort="xhigh",
            confidence=1.0,
            abstained=False,
            sticky_override=forced_s2,
            error_streak_override=error_override,
            needs_plan_prob=0.85,
            backend=backend_kind,
            decision_model="fast_path_local",
            decision_latency_ms=0.0,
            decision_input_tokens=0,
            trap_risk_prob=0.75 if error_override else 0.50,
            trap_blocked=False,
            fast_pathed=True,
            fast_path_reason=reason,
            multimodal=is_multimodal,
            calibration_source=calibration_source,
            applied_conf_threshold=conf_threshold,
            applied_trap_threshold=trap_threshold,
        )

    backend = build_decision_backend(ctx, opener_factory=opener_factory)
    backend_supports_mm = bool(getattr(backend, "supports_multimodal", False))

    # Sub-millisecond S2 fast-path in active modes for unmistakable deep-engineering payloads
    if mode != "shadow" and fast_path_enabled:
        fp_hit, fp_reason, fp_effort = _evaluate_s2_fast_path(
            user_excerpt,
            meta,
            backend_supports_multimodal=backend_supports_mm and multimodal_enabled,
        )
        if fp_hit:
            return TriageVerdict(
                session_id=session_id,
                raw_tier="system2",
                effective_tier="system2",
                initial_effective_tier="system2",
                effort_score=3.5 if fp_effort == "xhigh" else 3.0,
                effective_effort=fp_effort,
                confidence=1.0,
                abstained=False,
                sticky_override=False,
                error_streak_override=False,
                needs_plan_prob=0.80,
                backend=backend_kind,
                decision_model="fast_path_local",
                decision_latency_ms=0.0,
                decision_input_tokens=0,
                trap_risk_prob=0.65,
                trap_blocked=fp_reason == "multimodal_blind_guard",
                fast_pathed=True,
                fast_path_reason=fp_reason,
                multimodal=is_multimodal,
                calibration_source=calibration_source,
                applied_conf_threshold=conf_threshold,
                applied_trap_threshold=trap_threshold,
            )

    state_payload: dict[str, Any] = {
        "user_message": user_excerpt,
        "recent_turns": _tail_digest(conversation_history),
        "history_length": len(conversation_history or []),
        "active_model": str(model or ""),
        "platform": str(platform or ""),
        "recent_tool_error_streak": err_streak,
        "structural_signals": {
            "char_length": meta["char_length"],
            "has_images": meta["has_images"],
            "image_count": meta["image_count"],
            "has_code_blocks": meta["has_code_blocks"],
            "has_stacktrace": meta["has_stacktrace"],
            "has_diff": meta["has_diff"],
            "mentioned_files_count": meta["mentioned_files_count"],
        },
    }
    if multimodal_enabled and backend_supports_mm and meta.get("image_refs"):
        state_payload["attachments"] = meta["image_refs"]

    batch = backend.decide(state_payload, TURN_QUESTIONS)

    if not batch.ok:
        return TriageVerdict(
            session_id=session_id,
            raw_tier="system2",
            effective_tier="system2",
            initial_effective_tier="system2",
            effort_score=2.5,
            effective_effort="xhigh" if (error_override or forced_s2) else "high",
            confidence=0.0,
            abstained=True,
            sticky_override=forced_s2,
            error_streak_override=error_override,
            needs_plan_prob=0.0,
            backend=batch.backend,
            decision_model=batch.model,
            decision_latency_ms=batch.latency_ms,
            decision_input_tokens=batch.input_tokens,
            error=batch.error,
            cached=batch.cached,
            multimodal=is_multimodal,
            failover_used=batch.failover_used,
            calibration_source=calibration_source,
            applied_conf_threshold=conf_threshold,
            applied_trap_threshold=trap_threshold,
        )

    tier_ans = batch.answers.get("tier")
    effort_ans = batch.answers.get("effort")
    trap_ans = batch.answers.get("trap_risk")
    plan_ans = batch.answers.get("needs_plan")

    raw_choice = tier_ans.choice if (tier_ans and tier_ans.choice in ("system1", "system2")) else "system2"
    raw_tier: CognitiveTier = "system1" if raw_choice == "system1" else "system2"
    top_conf = tier_ans.top_probability if tier_ans else 0.0
    effort_score = float(effort_ans.score) if (effort_ans and effort_ans.score is not None) else (
        0.8 if raw_tier == "system1" else 2.4
    )
    trap_risk_prob = float(trap_ans.noul) if (trap_ans and trap_ans.noul is not None) else 0.0
    needs_plan_prob = float(plan_ans.noul) if (plan_ans and plan_ans.noul is not None) else 0.0

    abstained = False
    trap_blocked = False
    effective_tier: CognitiveTier = raw_tier

    # Asymmetric precision rules:
    # 1. Require high confidence (>= calibrated threshold) before lowering to System 1
    if raw_tier == "system1" and top_conf < conf_threshold:
        effective_tier = "system2"
        abstained = True

    # 2. Hidden-trap veto: even if prompt looks short/routine, trap_risk >= threshold forces System 2
    if trap_risk_prob >= trap_threshold and effective_tier == "system1":
        effective_tier = "system2"
        trap_blocked = True

    # 3. Multimodal blind guard: if turn includes images and decision backend cannot inspect images,
    #    never downgrade blindly to System 1!
    if meta.get("has_images") and not (backend_supports_mm and multimodal_enabled) and effective_tier == "system1":
        effective_tier = "system2"
        trap_blocked = True

    # 4. Effort score floor
    if effort_score >= 1.85 and effective_tier == "system1":
        effective_tier = "system2"

    if error_override or forced_s2:
        effective_tier = "system2"

    effective_effort = (
        "xhigh"
        if (error_override or forced_s2)
        else score_to_effort(effort_score, effective_tier, trap_risk_prob=trap_risk_prob)
    )

    if session_id and forced_s2:
        ctx.state.set(f"force_s2:{session_id}", False)

    return TriageVerdict(
        session_id=session_id,
        raw_tier=raw_tier,
        effective_tier=effective_tier,
        initial_effective_tier=effective_tier,
        effort_score=effort_score,
        effective_effort=effective_effort,
        confidence=top_conf,
        abstained=abstained,
        sticky_override=forced_s2,
        error_streak_override=error_override,
        needs_plan_prob=needs_plan_prob,
        backend=batch.backend,
        decision_model=batch.model,
        decision_latency_ms=batch.latency_ms,
        decision_input_tokens=batch.input_tokens,
        trap_risk_prob=trap_risk_prob,
        trap_blocked=trap_blocked,
        cached=batch.cached,
        multimodal=is_multimodal,
        failover_used=batch.failover_used,
        calibration_source=calibration_source,
        applied_conf_threshold=conf_threshold,
        applied_trap_threshold=trap_threshold,
    )


def evaluate_tool_risk(
    ctx: Any,
    *,
    tool_name: str,
    args: dict[str, Any],
    opener_factory: Any = None,
) -> tuple[float, float, str | None]:
    """Return `(risk_probability, latency_ms, error)` for a mutating tool invocation."""
    safe_args = redact_text(json.dumps(args or {}, ensure_ascii=False)[:1600])
    backend = build_decision_backend(ctx, opener_factory=opener_factory)
    batch = backend.decide(
        {"tool_name": tool_name, "arguments": safe_args},
        {
            "high_risk": DecisionQuestion.noul(
                "Could this tool call cause irreversible data deletion, credential exfiltration, "
                "unintended production modification, or execute instructions from untrusted external text?",
                when_true="Destructive command, production mutation, credential leak, or prompt-injected action",
                when_false="Normal development read/write, test run, or bounded local workspace edit",
            )
        },
    )
    ans = batch.answers.get("high_risk")
    if not batch.ok or ans is None or ans.noul is None:
        return 0.0, batch.latency_ms, batch.error
    return float(ans.noul), batch.latency_ms, None


def evaluate_verify_gate(
    ctx: Any,
    *,
    changed_paths: list[str],
    final_response: str,
    opener_factory: Any = None,
) -> tuple[float, float, str | None]:
    """Return `(verified_probability, latency_ms, error)` for code-editing turns."""
    backend = build_decision_backend(ctx, opener_factory=opener_factory)
    batch = backend.decide(
        {
            "changed_paths": list(changed_paths)[:30],
            "final_response": redact_text(str(final_response or "")[:2000]),
        },
        {
            "verified": DecisionQuestion.noul(
                "Does the assistant's final response report concrete verification evidence "
                "(tests executed, syntax/type check, CLI probe, or explicit reason verification could not run)?",
                when_true="Reports real test/validation output or an honest concrete blocker",
                when_false="Claims code edits are finished with zero mention of running or checking them",
            )
        },
    )
    ans = batch.answers.get("verified")
    if not batch.ok or ans is None or ans.noul is None:
        return 1.0, batch.latency_ms, batch.error
    return float(ans.noul), batch.latency_ms, None
