"""Profile-scoped SQLite audit ledger and `/trismegistus` slash command diagnostics."""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any

from .compat import open_plugin_db, read_secret
from .contracts import TriageVerdict

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trismegistus_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
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
)

_db_lock = threading.Lock()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass


def record_turn_completion(
    *,
    verdict: TriageVerdict,
    mode: str,
    provider_duration_ms: float = 0.0,
    provider_input_tokens: int = 0,
    provider_output_tokens: int = 0,
) -> None:
    """Persist one completed request's Triage + Modulation + Call metrics."""
    with _db_lock:
        conn = open_plugin_db("trismegistus", "ledger.db")
        try:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO trismegistus_decisions (
                    ts, session_id, mode, backend, decision_model,
                    raw_tier, effective_tier, effort_score, effective_effort,
                    confidence, abstained, sticky_override, error_streak_override,
                    wire_guarded, trap_risk_prob, trap_blocked, step_paced,
                    decision_latency_ms, decision_input_tokens,
                    applied_model, applied_effort, provider_duration_ms,
                    provider_input_tokens, provider_output_tokens, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    verdict.session_id or "",
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
                    verdict.decision_latency_ms,
                    verdict.decision_input_tokens,
                    verdict.applied_model,
                    verdict.applied_effort,
                    float(provider_duration_ms),
                    int(provider_input_tokens),
                    int(provider_output_tokens),
                    str(verdict.error or "")[:240],
                ),
            )
            conn.commit()
        finally:
            conn.close()


def read_summary_stats(limit_recent: int = 5) -> dict[str, Any]:
    with _db_lock:
        conn = open_plugin_db("trismegistus", "ledger.db")
        try:
            _ensure_schema(conn)
            cur = conn.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN effective_tier = 'system1' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN effective_tier = 'system2' THEN 1 ELSE 0 END),
                    SUM(abstained),
                    SUM(trap_blocked),
                    SUM(step_paced),
                    SUM(wire_guarded),
                    AVG(decision_latency_ms),
                    AVG(confidence),
                    SUM(decision_input_tokens)
                FROM trismegistus_decisions
                """
            )
            row = cur.fetchone() or (0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0)
            recent_cur = conn.execute(
                """
                SELECT mode, effective_tier, applied_effort, confidence, trap_risk_prob,
                       decision_latency_ms, applied_model, abstained, trap_blocked,
                       step_paced, wire_guarded, error
                FROM trismegistus_decisions
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit_recent),),
            )
            recent = [
                {
                    "mode": r[0],
                    "tier": r[1],
                    "effort": r[2],
                    "confidence": r[3],
                    "trap_risk": r[4],
                    "triage_ms": r[5],
                    "model": r[6],
                    "abstained": bool(r[7]),
                    "trap_blocked": bool(r[8]),
                    "step_paced": bool(r[9]),
                    "wire_guarded": bool(r[10]),
                    "error": r[11],
                }
                for r in recent_cur.fetchall()
            ]
        finally:
            conn.close()

    total = int(row[0] or 0)
    return {
        "total": total,
        "system1": int(row[1] or 0),
        "system2": int(row[2] or 0),
        "abstained": int(row[3] or 0),
        "trap_blocked": int(row[4] or 0),
        "step_paced": int(row[5] or 0),
        "wire_guarded": int(row[6] or 0),
        "avg_triage_ms": float(row[7] or 0.0),
        "avg_confidence": float(row[8] or 0.0),
        "decision_input_tokens": int(row[9] or 0),
        "recent": recent,
    }


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
    endpoint = ctx.get_config("foundry_endpoint", default="") or ctx.get_config("local_endpoint", default="")
    has_foundry_key = bool(read_secret("TRISMEGISTUS_FOUNDRY_API_KEY"))

    lines = [
        "Trismegistus v0.2 — Precision & Performance Cognitive Router",
        f"  Mode:     {mode} (step_pacing={'on' if ctx.get_config('step_pacing_enabled', True) else 'off'}, verify_gate={'on' if ctx.get_config('verify_gate_enabled', True) else 'off'})",
        f"  Backend:  {backend} ({endpoint or 'not set'}) | keep_alive={'on' if ctx.get_config('keep_alive_enabled', True) else 'off'}",
        f"  Foundry API Key configured: {'yes' if has_foundry_key else 'no'}",
        (
            f"  Recorded calls: {stats['total']} (S1: {stats['system1']}, S2: {stats['system2']}, "
            f"Abstained->S2: {stats['abstained']}, Trap-guarded: {stats['trap_blocked']}, "
            f"Step-paced: {stats['step_paced']}, Wire-guarded: {stats['wire_guarded']})"
        ),
    ]
    if stats["total"] > 0:
        s1_pct = (stats["system1"] / stats["total"]) * 100.0
        est_cost = (stats["decision_input_tokens"] / 1_000_000.0) * 0.042
        lines.append(
            f"  System-1 share: {s1_pct:.1f}% | Avg confidence: {stats['avg_confidence']:.2f} | "
            f"Avg triage latency: {stats['avg_triage_ms']:.0f} ms | Est. Foundry S1 spend: ${est_cost:.5f}"
        )

    if sub in ("last", "stats") and stats["recent"]:
        lines.append("  Recent decisions:")
        for item in stats["recent"]:
            flags = []
            if item["abstained"]:
                flags.append("abstained")
            if item["trap_blocked"]:
                flags.append("trap_guard")
            if item["step_paced"]:
                flags.append("step_paced")
            if item["wire_guarded"]:
                flags.append("wire_guard")
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
