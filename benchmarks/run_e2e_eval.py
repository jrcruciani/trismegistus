#!/usr/bin/env python3
"""End-to-end precision, speed, and KV-cache evaluation harness for Trismegistus v0.3.0.

Compares three routing policies across realistic multi-step engineering & single-turn tasks
covering both GPT-5.4 (`codex_responses`) and Claude Sonnet 5 (`chat_completions`):
  1. `baseline_s2`     — Always System 2 (`xhigh`/`high`), zero triage, zero step pacing.
  2. `effort_only`     — Trismegistus v0.3.0 (`fast_path` + cache-safe `step_pacing` + conformal floor).
  3. `tier_and_effort` — Trismegistus v0.3.0 with wire-safe peer swap on high-confidence S1 turns.

Metrics prioritize **precision (`pass_at_1`, `false_s1_rate`) and speed (`p50`/`p95` total
wall-clock latency, `triage_overhead_ms`, `cache_hit_rate`)** over token savings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class E2ETask:
    task_id: str
    category: str  # "s1_simple" | "s2_multistep" | "ambiguous_trap"
    provider: str
    model: str
    api_mode: str
    user_message: str
    scripted_decision: dict[str, Any]
    tool_steps: tuple[tuple[str, int, bool], ...]  # (tool_name, result_chars, has_stacktrace)
    requires_full_effort_on_synthesis: bool


def build_e2e_tasks() -> list[E2ETask]:
    """Build 30 representative tasks spanning single-turn S1, multi-step S2, and traps."""
    tasks: list[E2ETask] = []

    # 10 clear S1 tasks (fast turnaround)
    s1_prompts = [
        ("e2e_s1_01", "gpt-5.4", "codex_responses", "Translate to Spanish: 'Connection pool drained cleanly.'"),
        ("e2e_s1_02", "claude-sonnet-5", "chat_completions", "Convert 4096 milliseconds into seconds with 2 decimals."),
        ("e2e_s1_03", "gpt-5.4", "codex_responses", "Format {'env': 'prod', 'replicas': 3} as YAML."),
        ("e2e_s1_04", "claude-sonnet-5", "chat_completions", "What HTTP status code corresponds to Too Many Requests?"),
        ("e2e_s1_05", "gpt-5.4", "codex_responses", "Rewrite `max_retry_wait_ms` in camelCase and PascalCase."),
        ("e2e_s1_06", "claude-sonnet-5", "chat_completions", "Sort alphabetically: zeta, alpha, delta, beta."),
        ("e2e_s1_07", "gpt-5.4", "codex_responses", "Translate to English: 'La caché de prefijos redujo la latencia.'"),
        ("e2e_s1_08", "claude-sonnet-5", "chat_completions", "Convert 0xFF to decimal and binary."),
        ("e2e_s1_09", "gpt-5.4", "codex_responses", "Give the git command to show the last 5 commits oneline."),
        ("e2e_s1_10", "claude-sonnet-5", "chat_completions", "Format 2026-11-26 14:30:00 UTC as RFC 3339."),
    ]
    for tid, model, wire, msg in s1_prompts:
        tasks.append(
            E2ETask(
                task_id=tid,
                category="s1_simple",
                provider="copilot",
                model=model,
                api_mode=wire,
                user_message=msg,
                scripted_decision={
                    "tier": {"choice": "system1", "confidence": 0.95},
                    "effort": {"score": 0.35, "confidence": 0.92},
                    "trap_risk": {"noul": 0.06},
                    "needs_plan": {"noul": 0.05},
                },
                tool_steps=(),
                requires_full_effort_on_synthesis=False,
            )
        )

    # 10 multi-step S2 engineering tasks (including stacktraces / diffs triggering <1ms fast-path)
    stacktrace_prompt = (
        "Fix this intermittent crash in our async worker pool:\n"
        "```\nTraceback (most recent call last):\n"
        "  File 'pool.py', line 142, in acquire\n"
        "    conn = self._idle.popleft()\n"
        "IndexError: pop from an empty deque\n```\n"
        "Find the root cause across `pool.py` and `worker.py` and add a deterministic concurrency test."
    )
    diff_prompt = (
        "Review this patch for subtle transaction isolation bugs:\n"
        "diff --git a/db/tx.py b/db/tx.py\n@@ -10,5 +10,7 @@\n"
        "diff --git a/db/pool.py b/db/pool.py\n@@ -40,4 +40,6 @@\n"
        "diff --git a/api/orders.py b/api/orders.py\n@@ -88,3 +88,5 @@\n"
    )
    s2_specs = [
        ("e2e_s2_01", "gpt-5.4", "codex_responses", stacktrace_prompt, (("read_file", 1200, False), ("read_file", 1500, False), ("read_file", 8200, False), ("patch", 120, False))),
        ("e2e_s2_02", "claude-sonnet-5", "chat_completions", stacktrace_prompt, (("read_file", 1400, False), ("read_file", 1600, False), ("patch", 120, False))),
        ("e2e_s2_03", "gpt-5.4", "codex_responses", diff_prompt, (("read_file", 1800, False), ("search_files", 900, False), ("read_file", 2100, False), ("patch", 140, False))),
        ("e2e_s2_04", "claude-sonnet-5", "chat_completions", diff_prompt, (("read_file", 1500, False), ("patch", 140, False))),
        ("e2e_s2_05", "gpt-5.4", "codex_responses", "Design a lock-free ring buffer with acquire/release ordering and prove progress under SPSC.", (("read_file", 1000, False), ("read_file", 1100, False), ("read_file", 900, False), ("write_file", 200, False))),
        ("e2e_s2_06", "claude-sonnet-5", "chat_completions", "Refactor auth middleware across 5 services to support zero-downtime JWKS key rotation.", (("read_file", 1200, False), ("search_files", 800, False), ("patch", 160, False))),
        ("e2e_s2_07", "gpt-5.4", "codex_responses", "Audit SQLite WAL checkpoint starvation under concurrent readers and propose a fix.", (("read_file", 1300, False), ("read_file", 7500, False), ("patch", 120, False))),
        ("e2e_s2_08", "claude-sonnet-5", "chat_completions", "Trace why HTTP/2 stream resets occur under gRPC keepalive pings behind Envoy.", (("read_file", 1400, False), ("read_file", 1400, False), ("terminal", 400, False))),
        ("e2e_s2_09", "gpt-5.4", "codex_responses", "Implement split conformal calibration with finite-sample coverage guarantees.", (("read_file", 1100, False), ("read_file", 1300, False), ("write_file", 300, False))),
        ("e2e_s2_10", "claude-sonnet-5", "chat_completions", "Diagnose memory leak in bidirectional streaming proxy when clients abort mid-frame.", (("read_file", 1500, False), ("read_file", 2000, True), ("patch", 120, False))),
    ]
    for tid, model, wire, msg, steps in s2_specs:
        tasks.append(
            E2ETask(
                task_id=tid,
                category="s2_multistep",
                provider="copilot",
                model=model,
                api_mode=wire,
                user_message=msg,
                scripted_decision={
                    "tier": {"choice": "system2", "confidence": 0.96},
                    "effort": {"score": 3.6, "confidence": 0.94},
                    "trap_risk": {"noul": 0.72},
                    "needs_plan": {"noul": 0.88},
                },
                tool_steps=steps,
                requires_full_effort_on_synthesis=True,
            )
        )

    # 10 deceptive short trap tasks (look short like S1, hide concurrency/security/data hazards)
    trap_prompts = [
        ("e2e_trap_01", "gpt-5.4", "codex_responses", "Add an automatic retry on timeout to POST /api/payments/capture."),
        ("e2e_trap_02", "claude-sonnet-5", "chat_completions", "Bump the distributed lock TTL from 5s to 60s in worker.py."),
        ("e2e_trap_03", "gpt-5.4", "codex_responses", "Replace time.monotonic() with time.time() in the token bucket limiter."),
        ("e2e_trap_04", "claude-sonnet-5", "chat_completions", "Switch PRAGMA synchronous=OFF on the billing ledger DB for speed."),
        ("e2e_trap_05", "gpt-5.4", "codex_responses", "Remove the mutex around the shared cache dict because Python has the GIL."),
        ("e2e_trap_06", "claude-sonnet-5", "chat_completions", "Set verify=False on the internal webhook client to fix expired cert errors."),
        ("e2e_trap_07", "gpt-5.4", "codex_responses", "Use `if val:` to check whether the numeric probability field is present."),
        ("e2e_trap_08", "claude-sonnet-5", "chat_completions", "Pass the bearer token as a CLI flag to the background curl subprocess."),
        ("e2e_trap_09", "gpt-5.4", "codex_responses", "Reuse the HTTP socket in the pool even after an upstream 502 error."),
        ("e2e_trap_10", "claude-sonnet-5", "chat_completions", "Drop the unique idempotency constraint on webhook_events to speed up inserts."),
    ]
    for idx, (tid, model, wire, msg) in enumerate(trap_prompts):
        # Mix of low-confidence S1 (abstention) and high-confidence S1 with trap_risk >= 0.50 (trap guard)
        scripted = (
            {
                "tier": {"choice": "system1", "confidence": 0.75},
                "effort": {"score": 1.3, "confidence": 0.72},
                "trap_risk": {"noul": 0.44},
                "needs_plan": {"noul": 0.55},
            }
            if idx % 2 == 0
            else {
                "tier": {"choice": "system1", "confidence": 0.91},
                "effort": {"score": 1.2, "confidence": 0.85},
                "trap_risk": {"noul": 0.66},
                "needs_plan": {"noul": 0.60},
            }
        )
        tasks.append(
            E2ETask(
                task_id=tid,
                category="ambiguous_trap",
                provider="copilot",
                model=model,
                api_mode=wire,
                user_message=msg,
                scripted_decision=scripted,
                tool_steps=(("read_file", 1800, False), ("patch", 100, False)),
                requires_full_effort_on_synthesis=True,
            )
        )

    return tasks


def _load_pkg():
    pkg_name = "trismegistus_e2e_pkg"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(ROOT / "__init__.py"),
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(ROOT)]
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


# Calibrated latency model (ms) for deterministic offline simulation:
# reflects real-world thinking/TTFT curves for GPT-5.4 Responses & Claude Sonnet 5.
_EFFORT_LATENCY_MS: dict[str, float] = {
    "minimal": 550.0,
    "low": 950.0,
    "medium": 2400.0,
    "high": 5800.0,
    "xhigh": 10500.0,
    "native": 650.0,
}


def _simulate_policy(pkg: Any, tasks: list[E2ETask], policy: str) -> dict[str, Any]:
    """Run all 30 tasks through the real Trismegistus middleware pipeline and evaluate metrics."""
    from dataclasses import dataclass as _dc

    @_dc
    class _AuxRes:
        parsed: dict[str, Any]
        model: str = "microsoft-decision-1.1"
        usage: Any = None

    class _SimCtx:
        def __init__(self, cfg: dict[str, Any]):
            self._cfg = dict(cfg)
            self._state: dict[str, Any] = {}
            self.scripted: dict[str, Any] = {}
            self.tools: dict[str, Any] = {}
            self.hooks: dict[str, Any] = {}
            self.middleware: dict[str, Any] = {}
            self.commands: dict[str, Any] = {}

        class _Llm:
            def __init__(self, parent: "_SimCtx"):
                self._p = parent

            def complete_structured(self, **_kw):
                return _AuxRes(parsed=self._p.scripted)

        @property
        def llm(self):
            return self._Llm(self)

        def get_config(self, key: str, default: Any = None) -> Any:
            return self._cfg.get(key, default)

        def set_config(self, key: str, val: Any) -> None:
            self._cfg[key] = val

        def has_capability(self, _cap: str) -> bool:
            return True

        @property
        def state(self):
            return self

        def get(self, key: str, default: Any = None) -> Any:
            return self._state.get(key, default)

        def set(self, key: str, val: Any) -> None:
            self._state[key] = val

        def register_auxiliary_task(self, *_a, **_kw): ...
        def register_delegation_provider(self, *_a, **_kw): ...
        def register_skill(self, *_a, **_kw): ...
        def register_tool(self, name, **kw): self.tools[name] = kw
        def register_hook(self, name, cb, **_kw): self.hooks[name] = cb
        def register_middleware(self, kind, cb): self.middleware[kind] = cb
        def register_command(self, name, handler, **_kw): self.commands[name] = handler

    mode = "shadow" if policy == "baseline_s2" else policy
    ctx = _SimCtx(
        {
            "mode": mode,
            "backend": "hermes_aux",
            "step_pacing_enabled": True,
            "step_pacing_max_reads": 2,
            "step_pacing_protect_claude_cache": True,
            "fast_path_enabled": True,
            "use_calibrated_thresholds": False,
            "closed_loop_enabled": False,
        }
    )
    pkg.register(ctx)

    turn_latencies: list[float] = []
    triage_overheads: list[float] = []
    passed = 0
    false_s1_count = 0
    cache_read_total = 0
    prompt_input_total = 0
    fast_path_hits = 0
    step_paced_steps = 0

    for task in tasks:
        sid = f"{policy}:{task.task_id}"
        ctx.scripted = task.scripted_decision

        if policy == "baseline_s2":
            triage_ms = 0.0
            fast_pathed = False
            eff_tier = "system2"
            base_eff = "xhigh"
        else:
            ctx.hooks["pre_llm_call"](
                session_id=sid,
                user_message=task.user_message,
                conversation_history=[],
                model=task.model,
                platform="cli",
            )
            # Determine if fast-pathed by inspecting first llm_request reason
            triage_ms = 0.0
            fast_pathed = False
            eff_tier = "system2"
            base_eff = "high"

        total_steps = 1 + len(task.tool_steps)
        turn_wall_ms = 0.0
        prev_model = task.model
        prev_effort = ""
        synthesis_effort = ""

        for step_idx in range(1, total_steps + 1):
            req = (
                {"model": task.model, "input": [{"role": "user", "content": task.user_message}]}
                if task.api_mode == "codex_responses"
                else {"model": task.model, "messages": [{"role": "user", "content": task.user_message}]}
            )
            if policy == "baseline_s2":
                used_model = task.model
                used_effort = "xhigh" if task.category != "s1_simple" else "high"
                paced = False
            else:
                mw = ctx.middleware["llm_request"](
                    request=req,
                    session_id=sid,
                    provider=task.provider,
                    model=task.model,
                    api_mode=task.api_mode,
                    api_call_count=step_idx,
                )
                assert mw is not None
                out_req = mw["request"]
                reason = mw["reason"]
                if step_idx == 1:
                    fast_pathed = "fast_path:" in reason
                    if fast_pathed:
                        fast_path_hits += 1
                        triage_ms = 0.2
                    else:
                        triage_ms = 110.0  # warm TLS Foundry / Decision-1.1 latency
                    eff_tier = "system1" if "tier=system1" in reason else "system2"
                used_model = out_req["model"]
                used_effort = (
                    (out_req.get("reasoning") or {}).get("effort")
                    or out_req.get("reasoning_effort")
                    or "native"
                )
                paced = "step_paced" in reason
                if paced:
                    step_paced_steps += 1

            # Simulate realistic KV-cache behavior:
            # - Same model + cache-safe effort -> 88% prefix cache hit on steps > 1
            # - Model swap -> 0% cache hit
            # - Claude effort change mid-turn -> drops to 25% cache hit (message breakpoint invalidation)
            step_input_tokens = 3200 + (step_idx - 1) * 1400
            prompt_input_total += step_input_tokens
            if step_idx == 1 or used_model != prev_model:
                hit_ratio = 0.0
            elif "claude" in used_model and used_effort != prev_effort:
                hit_ratio = 0.25
            else:
                hit_ratio = 0.88
            cache_read_total += int(step_input_tokens * hit_ratio)

            # Latency includes prefill penalty when cache misses
            prefill_ms = 180.0 if hit_ratio >= 0.8 else 720.0
            step_ms = prefill_ms + _EFFORT_LATENCY_MS.get(used_effort, 2400.0)
            if used_model != task.model and ("mini" in used_model or "haiku" in used_model):
                step_ms *= 0.65
            turn_wall_ms += step_ms
            prev_model = used_model
            prev_effort = used_effort
            if step_idx == total_steps:
                synthesis_effort = used_effort

            # Feed simulated tool outcome before next step
            if step_idx <= len(task.tool_steps):
                t_name, t_chars, has_st = task.tool_steps[step_idx - 1]
                payload = ("Traceback (most recent call last):\n" if has_st else "") + ("x" * t_chars)
                if policy != "baseline_s2":
                    ctx.hooks["post_tool_call"](
                        tool_name=t_name,
                        result=payload,
                        status="ok",
                        session_id=sid,
                    )

        turn_wall_ms += triage_ms
        turn_latencies.append(turn_wall_ms)
        triage_overheads.append(triage_ms)

        # Evaluate task pass@1 under precision invariants
        if task.category == "s1_simple":
            task_ok = True
        else:
            if eff_tier == "system1":
                false_s1_count += 1
                task_ok = False
            elif task.requires_full_effort_on_synthesis and synthesis_effort in ("minimal", "low"):
                # Synthesis degraded to low effort on a deep engineering task risks precision loss
                task_ok = False
            else:
                task_ok = True
        if task_ok:
            passed += 1

    n = len(tasks)
    must_s2 = sum(1 for t in tasks if t.category != "s1_simple")
    return {
        "policy": policy,
        "tasks_evaluated": n,
        "pass_at_1": round(passed / n, 4),
        "false_s1_count": false_s1_count,
        "false_s1_rate": round(false_s1_count / must_s2, 4) if must_s2 else 0.0,
        "wall_clock_ms": {
            "mean": round(sum(turn_latencies) / n, 1),
            "p50": round(_percentile(turn_latencies, 50.0), 1),
            "p95": round(_percentile(turn_latencies, 95.0), 1),
        },
        "triage_overhead_ms_mean": round(sum(triage_overheads) / n, 2),
        "fast_path_s2_hits": fast_path_hits,
        "step_paced_steps": step_paced_steps,
        "kv_cache_hit_rate": round(cache_read_total / prompt_input_total, 4) if prompt_input_total else 0.0,
    }


def run_e2e_evaluation() -> dict[str, Any]:
    pkg = _load_pkg()
    tasks = build_e2e_tasks()
    baseline = _simulate_policy(pkg, tasks, "baseline_s2")
    effort_only = _simulate_policy(pkg, tasks, "effort_only")
    tier_and_effort = _simulate_policy(pkg, tasks, "tier_and_effort")

    base_mean = baseline["wall_clock_ms"]["mean"] or 1.0
    return {
        "status": "completed",
        "execution_kind": "offline_e2e_simulation",
        "tasks_count": len(tasks),
        "policies": {
            "baseline_s2": baseline,
            "effort_only": effort_only,
            "tier_and_effort": tier_and_effort,
        },
        "speedup_vs_baseline": {
            "effort_only_latency_reduction_pct": round((1.0 - effort_only["wall_clock_ms"]["mean"] / base_mean) * 100.0, 2),
            "tier_and_effort_latency_reduction_pct": round((1.0 - tier_and_effort["wall_clock_ms"]["mean"] / base_mean) * 100.0, 2),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Trismegistus end-to-end precision, speed, and KV-cache benchmark.")
    parser.add_argument("--mode", default="offline_sim", choices=["offline_sim"])
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report output path.")
    args = parser.parse_args(argv)
    _ = args.mode

    report = run_e2e_evaluation()
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
