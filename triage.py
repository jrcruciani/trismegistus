"""Stage 1 (Triage): precision-calibrated System-1 turn classification, trap gate, and verify gate."""

from __future__ import annotations

import json
from typing import Any

from .backends import build_decision_backend
from .compat import redact_text
from .contracts import EFFORT_LADDER, CognitiveTier, DecisionQuestion, TriageVerdict

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


def _extract_user_text(user_message: Any, max_chars: int) -> str:
    if isinstance(user_message, str):
        raw = user_message
    elif isinstance(user_message, list):
        parts: list[str] = []
        for item in user_message:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        raw = "\n".join(parts)
    else:
        raw = str(user_message or "")
    return redact_text(raw[: max(200, int(max_chars))])


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
    """Run Stage 1 (Triage) in a single multi-question System-1 pass."""
    max_chars = int(ctx.get_config("max_excerpt_chars", default=2400) or 2400)
    conf_threshold = float(ctx.get_config("confidence_threshold", default=0.82) or 0.82)
    trap_threshold = float(ctx.get_config("trap_risk_threshold", default=0.50) or 0.50)
    err_cap = max(1, int(ctx.get_config("escalate_after_errors", default=2) or 2))

    err_streak = int(ctx.state.get(f"err_streak:{session_id}", default=0) or 0) if session_id else 0
    forced_s2 = bool(ctx.state.get(f"force_s2:{session_id}", default=False)) if session_id else False

    # Reset per-turn step-pacing counters on fresh user turn entry
    if session_id:
        ctx.state.set(f"read_streak:{session_id}", 0)
        ctx.state.set(f"last_phase:{session_id}", "turn_start")

    state_payload = {
        "user_message": _extract_user_text(user_message, max_chars),
        "recent_turns": _tail_digest(conversation_history),
        "history_length": len(conversation_history or []),
        "active_model": str(model or ""),
        "platform": str(platform or ""),
        "recent_tool_error_streak": err_streak,
    }

    backend = build_decision_backend(ctx, opener_factory=opener_factory)
    batch = backend.decide(state_payload, TURN_QUESTIONS)

    if not batch.ok:
        return TriageVerdict(
            session_id=session_id,
            raw_tier="system2",
            effective_tier="system2",
            effort_score=2.5,
            effective_effort="xhigh" if (err_streak >= err_cap or forced_s2) else "high",
            confidence=0.0,
            abstained=True,
            sticky_override=forced_s2,
            error_streak_override=err_streak >= err_cap,
            needs_plan_prob=0.0,
            backend=batch.backend,
            decision_model=batch.model,
            decision_latency_ms=batch.latency_ms,
            decision_input_tokens=batch.input_tokens,
            error=batch.error,
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
    # 1. Require high confidence (>= 0.82 default) before lowering to System 1
    if raw_tier == "system1" and top_conf < conf_threshold:
        effective_tier = "system2"
        abstained = True

    # 2. Hidden-trap veto: even if prompt looks short/routine, trap_risk >= 0.50 default forces System 2
    if trap_risk_prob >= trap_threshold and effective_tier == "system1":
        effective_tier = "system2"
        trap_blocked = True

    # 3. Effort score floor
    if effort_score >= 1.85 and effective_tier == "system1":
        effective_tier = "system2"

    error_override = err_streak >= err_cap
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
