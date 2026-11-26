"""Profile-scoped SQLite audit ledger and `/trismegistus` slash command diagnostics."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from .compat import open_plugin_db, read_secret, redact_text
from .contracts import TriageVerdict

FOUNDRY_PRICE_PER_1M_INPUT_USD = 0.042

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trismegistus_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT DEFAULT '',
    step_index INTEGER DEFAULT 1,
    mode TEXT NOT NULL,
    backend TEXT NOT NULL,
    decision_model TEXT NOT NULL,
    raw_tier TEXT NOT NULL,
    effective_tier TEXT NOT NULL,
    effort_score REAL NOT NULL,
    effective_effort TEXT NOT NULL,
    confidence REAL NOT NULL,
    abstained INTEGER NOT NULL,
    sticky_override INTEGER NOT NULL,
    error_streak_override INTEGER NOT NULL,
    wire_guarded INTEGER NOT NULL,
    trap_risk_prob REAL DEFAULT 0,
    trap_blocked INTEGER DEFAULT 0,
    step_paced INTEGER DEFAULT 0,
    decision_latency_ms REAL NOT NULL,
    decision_input_tokens INTEGER NOT NULL,
    applied_model TEXT NOT NULL,
    applied_effort TEXT NOT NULL,
    provider_duration_ms REAL DEFAULT 0,
    provider_input_tokens INTEGER DEFAULT 0,
    provider_output_tokens INTEGER DEFAULT 0,
    error TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_trismegistus_ts ON trismegistus_decisions(ts DESC);
"""

_MIGRATIONS = (
    "ALTER TABLE trismegistus_decisions ADD COLUMN trap_risk_prob REAL DEFAULT 0",
    "ALTER TABLE trismegistus_decisions ADD COLUMN trap_blocked INTEGER DEFAULT 0",
    "ALTER TABLE trismegistus_decisions ADD COLUMN step_paced INTEGER DEFAULT 0",
    "ALTER TABLE trismegistus_decisions ADD COLUMN turn_id TEXT DEFAULT ''",
    "ALTER TABLE trismegistus_decisions ADD COLUMN step_index INTEGER DEFAULT 1",
)

_db_lock = threading.Lock()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass


def classify_error_category(error: str | None) -> str:
    """Bucket a triage error string into a canonical audit category without leaking details."""
    raw = str(error or "").strip()
    if not raw:
        return "ok"
    low = raw.lower()
    if raw.startswith("HTTP 429") or "ratelimit" in low or "rate limit" in low:
        return "http_429"
    if raw.startswith("HTTP 401") or "unauthorized" in low or "token expired" in low:
        return "http_401"
    if "missing trismegistus_" in low or "missing systemone endpoint" in low:
        return "missing_secret"
    if raw.startswith("HTTP 404") or "deploymentnotfound" in low:
        return "http_404"
    if "remote end closed" in low or "urlerror" in low or "timed out" in low or "timeout" in low:
        return "network_or_timeout"
    return "other"


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = rank - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def record_turn_completion(
    *,
    verdict: TriageVerdict,
    mode: str,
    provider_duration_ms: float = 0.0,
    provider_input_tokens: int = 0,
    provider_output_tokens: int = 0,
) -> None:
    """Persist one completed API step while attributing triage tokens/errors strictly once per turn."""
    verdict.recorded_steps += 1
    step_idx = verdict.recorded_steps
    is_first_step = step_idx == 1

    triage_ms = verdict.decision_latency_ms if is_first_step else 0.0
    triage_tokens = verdict.decision_input_tokens if is_first_step else 0
    step_error = redact_text(str(verdict.error or "")[:240]) if is_first_step else ""

    with _db_lock:
        conn = open_plugin_db("trismegistus", "ledger.db")
        try:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO trismegistus_decisions (
                    ts, session_id, turn_id, step_index, mode, backend, decision_model,
                    raw_tier, effective_tier, effort_score, effective_effort,
                    confidence, abstained, sticky_override, error_streak_override,
                    wire_guarded, trap_risk_prob, trap_blocked, step_paced,
                    decision_latency_ms, decision_input_tokens,
                    applied_model, applied_effort, provider_duration_ms,
                    provider_input_tokens, provider_output_tokens, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    verdict.session_id or "",
                    verdict.turn_id or "",
                    int(step_idx),
                    mode,
                    verdict.backend,
                    verdict.decision_model,
                    verdict.raw_tier,
                    verdict.effective_tier,
                    verdict.effort_score,
                    verdict.effective_effort,
                    verdict.confidence,
                    int(verdict.abstained),
                    int(verdict.sticky_override),
                    int(verdict.error_streak_override),
                    int(verdict.wire_guarded),
                    float(verdict.trap_risk_prob),
                    int(verdict.trap_blocked),
                    int(verdict.step_paced),
                    float(triage_ms),
                    int(triage_tokens),
                    verdict.applied_model,
                    verdict.applied_effort,
                    float(provider_duration_ms),
                    int(provider_input_tokens),
                    int(provider_output_tokens),
                    step_error,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _deduplicate_turns_from_rows(rows: list[tuple[Any, ...]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collapse multi-step tool-loop rows (both legacy pre-v0.2.2 and `step_index`-tagged) into unique turns."""
    unique_turns: list[dict[str, Any]] = []
    seen_turn_ids: dict[str, int] = {}
    last_legacy_by_session: dict[str, tuple[tuple[Any, ...], float, int]] = {}
    step_flags = {"total_steps": len(rows), "step_paced": 0, "wire_guarded": 0}
    step_error_counts: Counter[str] = Counter()

    for r in rows:
        (
            rid,
            ts,
            sid,
            turn_id,
            step_idx,
            mode,
            backend,
            dmodel,
            rtier,
            etier,
            escore,
            eeffort,
            conf,
            abst,
            trap_p,
            trap_b,
            paced,
            wguard,
            lat_ms,
            in_tok,
            amodel,
            aeffort,
            err,
        ) = r
        step_flags["step_paced"] += int(bool(paced))
        step_flags["wire_guarded"] += int(bool(wguard))
        step_error_counts[classify_error_category(err)] += 1

        tid = str(turn_id or "").strip()
        s_idx = int(step_idx or 1)
        if tid:
            if tid in seen_turn_ids:
                unique_turns[seen_turn_ids[tid]]["steps"] += 1
                continue
            if s_idx > 1:
                continue
            seen_turn_ids[tid] = len(unique_turns)
            unique_turns.append(
                {
                    "id": rid,
                    "ts": float(ts or 0.0),
                    "session_id": str(sid or ""),
                    "turn_id": tid,
                    "steps": 1,
                    "mode": str(mode or ""),
                    "backend": str(backend or ""),
                    "decision_model": str(dmodel or ""),
                    "raw_tier": str(rtier or ""),
                    "effective_tier": str(etier or ""),
                    "effort_score": float(escore or 0.0),
                    "effective_effort": str(eeffort or ""),
                    "confidence": float(conf or 0.0),
                    "abstained": bool(abst),
                    "trap_risk_prob": float(trap_p or 0.0),
                    "trap_blocked": bool(trap_b),
                    "decision_latency_ms": float(lat_ms or 0.0),
                    "decision_input_tokens": int(in_tok or 0),
                    "applied_model": str(amodel or ""),
                    "applied_effort": str(aeffort or ""),
                    "error": str(err or ""),
                    "error_category": classify_error_category(err),
                }
            )
            continue

        # Legacy pre-v0.2.2 row without `turn_id`: within a multi-step tool loop, `pre_llm_call`
        # stamped one TriageVerdict whose `(lat_ms, in_tok, conf, trap_p, escore, rtier, err)` was
        # written verbatim on every `post_api_request` step of that session.
        sig = (
            round(float(lat_ms or 0.0), 5),
            int(in_tok or 0),
            round(float(conf or 0.0), 5),
            round(float(trap_p or 0.0), 5),
            round(float(escore or 0.0), 5),
            str(rtier or ""),
            str(err or ""),
        )
        sess_key = str(sid or "_default")
        prev = last_legacy_by_session.get(sess_key)
        if prev is not None and prev[0] == sig and (float(ts or 0.0) - prev[1]) < 1800.0:
            t_pos = prev[2]
            unique_turns[t_pos]["steps"] += 1
            last_legacy_by_session[sess_key] = (sig, float(ts or 0.0), t_pos)
            continue

        t_pos = len(unique_turns)
        last_legacy_by_session[sess_key] = (sig, float(ts or 0.0), t_pos)
        unique_turns.append(
            {
                "id": rid,
                "ts": float(ts or 0.0),
                "session_id": sess_key,
                "turn_id": f"legacy-{rid}",
                "steps": 1,
                "mode": str(mode or ""),
                "backend": str(backend or ""),
                "decision_model": str(dmodel or ""),
                "raw_tier": str(rtier or ""),
                "effective_tier": str(etier or ""),
                "effort_score": float(escore or 0.0),
                "effective_effort": str(eeffort or ""),
                "confidence": float(conf or 0.0),
                "abstained": bool(abst),
                "trap_risk_prob": float(trap_p or 0.0),
                "trap_blocked": bool(trap_b),
                "decision_latency_ms": float(lat_ms or 0.0),
                "decision_input_tokens": int(in_tok or 0),
                "applied_model": str(amodel or ""),
                "applied_effort": str(aeffort or ""),
                "error": str(err or ""),
                "error_category": classify_error_category(err),
            }
        )

    step_flags["step_ok"] = step_error_counts.get("ok", 0)
    step_flags["step_http_429"] = step_error_counts.get("http_429", 0)
    step_flags["step_http_401"] = step_error_counts.get("http_401", 0)
    step_flags["step_missing_secret"] = step_error_counts.get("missing_secret", 0)
    step_flags["step_other_error"] = (
        len(rows)
        - step_flags["step_ok"]
        - step_flags["step_http_429"]
        - step_flags["step_http_401"]
        - step_flags["step_missing_secret"]
    )
    return unique_turns, step_flags


def audit_ledger_db(db_path: str | Path | None = None, *, limit_recent: int = 5) -> dict[str, Any]:
    """Read and deduplicate a Trismegistus SQLite ledger (read-only when an explicit path is given)."""
    if db_path is not None:
        resolved = Path(db_path)
        conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        readonly = True
    else:
        conn = open_plugin_db("trismegistus", "ledger.db")
        readonly = False

    try:
        if not readonly:
            _ensure_schema(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trismegistus_decisions)").fetchall()}
        if not cols:
            return {
                "total": 0,
                "total_steps": 0,
                "ok_turns": 0,
                "error_turns": 0,
                "error_rate": 0.0,
                "errors_by_category": {},
                "step_errors_by_category": {},
                "system1": 0,
                "system2": 0,
                "abstained": 0,
                "trap_blocked": 0,
                "step_paced": 0,
                "wire_guarded": 0,
                "avg_triage_ms": 0.0,
                "p50_triage_ms": 0.0,
                "p95_triage_ms": 0.0,
                "avg_confidence": 0.0,
                "decision_input_tokens": 0,
                "estimated_cost_usd": 0.0,
                "recent": [],
            }
        turn_expr = "turn_id" if "turn_id" in cols else "''"
        step_expr = "step_index" if "step_index" in cols else "1"
        trap_p_expr = "trap_risk_prob" if "trap_risk_prob" in cols else "0.0"
        trap_b_expr = "trap_blocked" if "trap_blocked" in cols else "0"
        paced_expr = "step_paced" if "step_paced" in cols else "0"

        cur = conn.execute(
            f"""
            SELECT id, ts, session_id, {turn_expr}, {step_expr}, mode, backend, decision_model,
                   raw_tier, effective_tier, effort_score, effective_effort, confidence,
                   abstained, {trap_p_expr}, {trap_b_expr}, {paced_expr}, wire_guarded,
                   decision_latency_ms, decision_input_tokens, applied_model, applied_effort, error
            FROM trismegistus_decisions
            ORDER BY id ASC
            """
        )
        all_rows = cur.fetchall()
    finally:
        conn.close()

    turns, step_flags = _deduplicate_turns_from_rows(all_rows)
    total_turns = len(turns)
    turn_err_counts = Counter(t["error_category"] for t in turns)
    ok_turns = [t for t in turns if t["error_category"] == "ok"]
    ok_lats = sorted(t["decision_latency_ms"] for t in ok_turns if t["decision_latency_ms"] > 0)
    all_lats = sorted(t["decision_latency_ms"] for t in turns if t["decision_latency_ms"] >= 0)

    s1_count = sum(1 for t in turns if t["effective_tier"] == "system1")
    s2_count = sum(1 for t in turns if t["effective_tier"] == "system2")
    abst_count = sum(1 for t in turns if t["abstained"])
    trap_count = sum(1 for t in turns if t["trap_blocked"])
    in_tokens = sum(t["decision_input_tokens"] for t in turns)
    avg_ms = (sum(ok_lats) / len(ok_lats)) if ok_lats else ((sum(all_lats) / len(all_lats)) if all_lats else 0.0)
    avg_conf = (sum(t["confidence"] for t in ok_turns) / len(ok_turns)) if ok_turns else 0.0

    recent_turns = list(reversed(turns[-max(1, int(limit_recent)) :])) if turns else []
    recent = [
        {
            "mode": t["mode"],
            "tier": t["effective_tier"],
            "effort": t["applied_effort"],
            "confidence": t["confidence"],
            "trap_risk": t["trap_risk_prob"],
            "triage_ms": t["decision_latency_ms"],
            "model": t["applied_model"],
            "abstained": t["abstained"],
            "trap_blocked": t["trap_blocked"],
            "step_paced": False,
            "wire_guarded": False,
            "steps": t["steps"],
            "error": redact_text(t["error"][:120]) if t["error"] else "",
        }
        for t in recent_turns
    ]

    err_turns_count = total_turns - len(ok_turns)
    return {
        "total": total_turns,
        "total_steps": step_flags["total_steps"],
        "ok_turns": len(ok_turns),
        "error_turns": err_turns_count,
        "error_rate": round(err_turns_count / total_turns, 4) if total_turns else 0.0,
        "errors_by_category": dict(turn_err_counts),
        "step_errors_by_category": {
            "ok": step_flags["step_ok"],
            "http_429": step_flags["step_http_429"],
            "http_401": step_flags["step_http_401"],
            "missing_secret": step_flags["step_missing_secret"],
            "other": step_flags["step_other_error"],
        },
        "system1": s1_count,
        "system2": s2_count,
        "ok_system1": sum(1 for t in ok_turns if t["effective_tier"] == "system1"),
        "ok_system2": sum(1 for t in ok_turns if t["effective_tier"] == "system2"),
        "abstained": abst_count,
        "ok_abstained": sum(1 for t in ok_turns if t["abstained"]),
        "trap_blocked": trap_count,
        "step_paced": step_flags["step_paced"],
        "wire_guarded": step_flags["wire_guarded"],
        "avg_triage_ms": round(avg_ms, 2),
        "p50_triage_ms": round(_percentile(ok_lats or all_lats, 50.0), 2),
        "p95_triage_ms": round(_percentile(ok_lats or all_lats, 95.0), 2),
        "avg_confidence": round(avg_conf, 4),
        "decision_input_tokens": in_tokens,
        "estimated_cost_usd": round((in_tokens / 1_000_000.0) * FOUNDRY_PRICE_PER_1M_INPUT_USD, 6),
        "recent": recent,
    }


def read_summary_stats(limit_recent: int = 5) -> dict[str, Any]:
    with _db_lock:
        return audit_ledger_db(None, limit_recent=limit_recent)


def handle_slash_command(ctx: Any, raw_args: str, *, session_id: str = "") -> str:
    """Handler for `/trismegistus [status|stats|last|mode <shadow|effort_only|tier_and_effort>|escalate]`."""
    tokens = (raw_args or "").strip().split()
    sub = tokens[0].lower() if tokens else "status"

    if sub == "mode":
        if len(tokens) < 2 or tokens[1] not in ("shadow", "effort_only", "tier_and_effort"):
            return "Usage: /trismegistus mode <shadow|effort_only|tier_and_effort>"
        new_mode = tokens[1]
        ctx.set_config("mode", new_mode)
        return f"Trismegistus operating mode updated to: {new_mode}"

    if sub == "escalate":
        if not session_id:
            return "No active session_id bound to escalate."
        ctx.state.set(f"force_s2:{session_id}", True)
        return "Next turn in this session armed for System-2 (xhigh effort)."

    stats = read_summary_stats(limit_recent=5)
    mode = ctx.get_config("mode", default="shadow")
    backend = ctx.get_config("backend", default="foundry_systemone")
    fallback = ctx.get_config("fallback_backend", default="none")
    endpoint = ctx.get_config("foundry_endpoint", default="") or ctx.get_config("local_endpoint", default="")
    has_foundry_key = bool(read_secret("TRISMEGISTUS_FOUNDRY_API_KEY"))

    lines = [
        "Trismegistus v0.2.2 — Precision & Performance Cognitive Router",
        f"  Mode:     {mode} (step_pacing={'on' if ctx.get_config('step_pacing_enabled', True) else 'off'}, verify_gate={'on' if ctx.get_config('verify_gate_enabled', True) else 'off'})",
        f"  Backend:  {backend} ({endpoint or 'not set'}) | fallback={fallback} | keep_alive={'on' if ctx.get_config('keep_alive_enabled', True) else 'off'}",
        f"  Foundry API Key configured: {'yes' if has_foundry_key else 'no'}",
        (
            f"  Unique turns: {stats['total']} across {stats['total_steps']} API steps "
            f"(S1: {stats['system1']}, S2: {stats['system2']}, Abstained->S2: {stats['abstained']}, "
            f"Trap-guarded: {stats['trap_blocked']}, Step-paced: {stats['step_paced']}, Wire-guarded: {stats['wire_guarded']})"
        ),
    ]
    if stats["total"] > 0:
        s1_pct = (stats["system1"] / stats["total"]) * 100.0
        lines.append(
            f"  System-1 share: {s1_pct:.1f}% | Avg confidence (OK): {stats['avg_confidence']:.2f} | "
            f"Triage latency p50/p95: {stats['p50_triage_ms']:.0f}/{stats['p95_triage_ms']:.0f} ms | "
            f"Error rate: {stats['error_rate'] * 100.0:.1f}% | Est. Foundry S1 spend: ${stats['estimated_cost_usd']:.5f}"
        )

    if sub in ("last", "stats") and stats["recent"]:
        lines.append("  Recent turns:")
        for item in stats["recent"]:
            flags = []
            if item["abstained"]:
                flags.append("abstained")
            if item["trap_blocked"]:
                flags.append("trap_guard")
            if item.get("steps", 1) > 1:
                flags.append(f"steps={item['steps']}")
            if item["error"]:
                flags.append("err")
            flag_str = f" [{'|'.join(flags)}]" if flags else ""
            lines.append(
                f"    - [{item['mode']}] {item['tier']} effort={item['effort']} "
                f"conf={item['confidence']:.2f} trap={item['trap_risk']:.2f} "
                f"({item['triage_ms']:.0f}ms) -> {item['model'] or '(active)'}{flag_str}"
            )
    elif sub == "help":
        lines.append("  Commands: /trismegistus [status | stats | last | mode <shadow|effort_only|tier_and_effort> | escalate]")

    return "\n".join(lines)
