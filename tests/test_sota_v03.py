"""Tests for Trismegistus v0.3.0 SOTA precision & speed upgrades (Priorities 1, 2, and 3)."""

from __future__ import annotations

import email.message
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import urllib.error

ROOT = Path(__file__).resolve().parent.parent


class _FakeHttpResp:
    def __init__(self, payload: dict, status: int = 200):
        self._raw = json.dumps(payload).encode("utf-8")
        self.status = status
        self.headers = {"Connection": "keep-alive"}

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_step_pacing_protects_synthesis_dense_reads_and_claude_cache(pkg):
    """Step pacing must restore full effort before synthesis (read_streak > max), on dense reads, and on Claude."""
    from test_pipeline import _StubPluginContext

    ctx = _StubPluginContext(
        config={
            "mode": "effort_only",
            "backend": "hermes_aux",
            "step_pacing_enabled": True,
            "step_pacing_max_reads": 2,
            "step_pacing_protect_claude_cache": True,
        },
        scripted_decisions=[
            {
                "tier": {"choice": "system2", "confidence": 0.97},
                "effort": {"score": 3.7, "confidence": 0.95},
                "trap_risk": {"noul": 0.70},
                "needs_plan": {"noul": 0.90},
            },
            {
                "tier": {"choice": "system2", "confidence": 0.97},
                "effort": {"score": 3.7, "confidence": 0.95},
                "trap_risk": {"noul": 0.70},
                "needs_plan": {"noul": 0.90},
            },
        ],
    )
    pkg.register(ctx)

    # 1. GPT-5.4 Responses: check constant cache-safe note (no variable float!)
    note = ctx.hooks["pre_llm_call"](
        session_id="ses_synth",
        user_message="Diagnose subtle deadlock across pool.py and worker.py",
        conversation_history=[],
        model="gpt-5.4",
    )
    assert note is not None
    assert "p=0." not in note["context"]

    # Read 1 (short, clean) -> paced to medium
    ctx.hooks["post_tool_call"](tool_name="read_file", result="1|class Pool: pass", status="ok", session_id="ses_synth")
    c2 = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_synth",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=2,
    )
    assert c2["request"]["reasoning"]["effort"] == "medium"

    # Read 2 (large >= 6000 chars) -> restores FULL xhigh BEFORE synthesis/patch!
    ctx.hooks["post_tool_call"](tool_name="read_file", result="x" * 6500, status="ok", session_id="ses_synth")
    c3 = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_synth",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=3,
    )
    assert c3["request"]["reasoning"]["effort"] == "xhigh"
    assert "step_paced" not in c3["reason"]

    # Read 3 (small again, but read_streak == 3 > max_paced_reads=2) -> STILL keeps full xhigh!
    ctx.hooks["post_tool_call"](tool_name="read_file", result="1|short", status="ok", session_id="ses_synth")
    c4 = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_synth",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=4,
    )
    assert c4["request"]["reasoning"]["effort"] == "xhigh"
    assert "step_paced" not in c4["reason"]

    # 2. Claude Sonnet 5: prompt-cache protection keeps effort steady mid-turn (no breakpoint invalidation)
    ctx.hooks["pre_llm_call"](
        session_id="ses_claude",
        user_message="Refactor multi-module auth boundaries safely",
        conversation_history=[],
        model="claude-sonnet-5",
    )
    ctx.hooks["post_tool_call"](tool_name="read_file", result="1|def auth(): pass", status="ok", session_id="ses_claude")
    c_claude = ctx.middleware["llm_request"](
        request={"model": "claude-sonnet-5", "messages": []},
        session_id="ses_claude",
        provider="copilot",
        model="claude-sonnet-5",
        api_mode="chat_completions",
        api_call_count=2,
    )
    assert c_claude["request"]["reasoning_effort"] == "xhigh"
    assert "step_paced" not in c_claude["reason"]


def test_read_only_file_containing_error_text_never_triggers_err_streak(pkg):
    """Reading source code or JSON containing literal '\"error\":' must NEVER increment err_streak."""
    from test_pipeline import _StubPluginContext

    ctx = _StubPluginContext(
        config={"mode": "effort_only", "backend": "hermes_aux", "escalate_after_errors": 2},
        scripted_decisions=[
            {
                "tier": {"choice": "system1", "confidence": 0.95},
                "effort": {"score": 0.5, "confidence": 0.91},
                "trap_risk": {"noul": 0.05},
                "needs_plan": {"noul": 0.05},
            }
        ],
    )
    pkg.register(ctx)
    ctx.hooks["pre_llm_call"](
        session_id="ses_ro",
        user_message="Inspect schema fields in ledger.json",
        conversation_history=[],
        model="gpt-5.4",
    )
    # Two clean read_file calls whose file content literally includes `"error":` and `"success": false`
    ctx.hooks["post_tool_call"](
        tool_name="read_file",
        result='{"error": "example_field", "success": false}',
        status="ok",
        session_id="ses_ro",
    )
    ctx.hooks["post_tool_call"](
        tool_name="search_files",
        result='backends.py:148: "error": "SystemOne response"',
        status="ok",
        session_id="ses_ro",
    )
    assert ctx.state.get("err_streak:ses_ro") == 0
    assert ctx.state.get("read_streak:ses_ro") == 2


def test_fast_path_skips_network_for_forced_s2_and_stacktraces_and_preserves_tail(pkg):
    """Active modes bypass triage RTT on forced_s2 and deep stacktraces; head+tail keeps closing question."""
    from test_pipeline import _StubPluginContext

    ctx = _StubPluginContext(
        config={"mode": "effort_only", "backend": "hermes_aux", "fast_path_enabled": True},
        scripted_decisions=[],
    )
    pkg.register(ctx)

    # 1. Forced S2 bypasses LLM/Foundry triage in 0ms
    ctx.state.set("force_s2:ses_fp1", True)
    v1 = pkg.triage.run_turn_triage(ctx, session_id="ses_fp1", user_message="Check lock ordering", model="gpt-5.4")
    assert v1.fast_pathed is True
    assert v1.fast_path_reason == "forced_s2"
    assert v1.effective_tier == "system2"
    assert v1.effective_effort == "xhigh"
    assert v1.decision_latency_ms == 0.0
    assert len(ctx.llm.calls) == 0

    # 2. Stacktrace + code block triggers <1ms S2 fast-path
    st_msg = (
        "Crash in worker:\n```python\nTraceback (most recent call last):\n"
        "  File 'worker.py', line 91, in run\n    raise RuntimeError('boom')\n```\nFix root cause."
    )
    v2 = pkg.triage.run_turn_triage(ctx, session_id="ses_fp2", user_message=st_msg, model="gpt-5.4")
    assert v2.fast_pathed is True
    assert v2.fast_path_reason == "stacktrace_debug"
    assert v2.effective_effort == "xhigh"
    assert len(ctx.llm.calls) == 0

    # 3. Head + Tail preserves both start and closing question of long logs
    long_msg = "OPENING_HEADER " + ("A" * 5000) + " CLOSING_QUESTION_ABOUT_RACE_CONDITION"
    excerpt, meta = pkg.triage._extract_user_state(long_msg, 800)
    assert "OPENING_HEADER" in excerpt
    assert "CLOSING_QUESTION_ABOUT_RACE_CONDITION" in excerpt
    assert "[truncated middle]" in excerpt
    assert meta["char_length"] == len(long_msg)


def test_multimodal_triage_forwarding_and_text_only_blind_guard(pkg):
    """Multimodal turns forward image refs to Decision-1.1 and veto blind S1 on text-only backends."""
    from test_pipeline import _StubPluginContext

    mm_msg = [
        {"type": "text", "text": "Mira esto"},
        {"type": "image_url", "image_url": {"url": "https://example.org/screenshot.png"}},
    ]

    # Text-only backend (`hermes_aux`) with fast_path disabled -> multimodal blind guard vetoes S1!
    ctx_txt = _StubPluginContext(
        config={
            "mode": "effort_only",
            "backend": "hermes_aux",
            "fast_path_enabled": False,
            "multimodal_triage_enabled": True,
        },
        scripted_decisions=[
            {
                "tier": {"choice": "system1", "confidence": 0.96},
                "effort": {"score": 0.4, "confidence": 0.92},
                "trap_risk": {"noul": 0.10},
                "needs_plan": {"noul": 0.10},
            }
        ],
    )
    v_txt = pkg.triage.run_turn_triage(ctx_txt, session_id="ses_mm1", user_message=mm_msg, model="gpt-5.4")
    assert v_txt.multimodal is True
    assert v_txt.effective_tier == "system2"
    assert v_txt.trap_blocked is True

    # Multimodal Decision-1.1 backend receives `attachments` and `structural_signals.has_images=True`
    seen_payloads: list[dict] = []

    class _CapOpener:
        def open(self, req, timeout=None):
            del timeout
            seen_payloads.append(json.loads(req.data.decode("utf-8")))
            return _FakeHttpResp(
                {
                    "model": "microsoft-decision-1.1",
                    "answers": {
                        "tier": {"type": "choice", "choice": "system2", "confidence": 0.94, "probabilities": {"system1": 0.06, "system2": 0.94}},
                        "effort": {"type": "score", "score": 2.8, "confidence": 0.90},
                        "trap_risk": {"type": "noul", "noul": 0.55},
                        "needs_plan": {"type": "noul", "noul": 0.70},
                    },
                    "usage": {"input_tokens": 220, "output_tokens": 0},
                }
            )

    import unittest.mock

    ctx_mm = _StubPluginContext(
        config={
            "mode": "effort_only",
            "backend": "foundry_systemone",
            "foundry_endpoint": "https://demo.services.ai.azure.com",
            "fast_path_enabled": False,
            "multimodal_triage_enabled": True,
        }
    )
    with unittest.mock.patch.object(pkg.backends, "read_secret", return_value="placeholder-key"):
        v_mm = pkg.triage.run_turn_triage(
            ctx_mm,
            session_id="ses_mm2",
            user_message=mm_msg,
            model="gpt-5.4",
            opener_factory=lambda _h: _CapOpener(),
        )
    assert v_mm.multimodal is True
    assert len(seen_payloads) == 1
    assert seen_payloads[0]["state"]["structural_signals"]["has_images"] is True
    assert seen_payloads[0]["state"]["attachments"][0]["image_url"]["url"] == "https://example.org/screenshot.png"


def test_same_model_secondary_failover_and_parallel_local_logits(pkg):
    """Foundry 429 fails over immediately to OpenRouter/Vercel Decision-1.1; LocalLogits runs in parallel."""
    from test_pipeline import _StubPluginContext
    import unittest.mock

    assert pkg.backends.normalize_secondary_systemone_url("openrouter") == "https://openrouter.ai/api/v1/decisions"
    assert pkg.backends.normalize_secondary_systemone_url("vercel") == "https://ai-gateway.vercel.sh/v1/decisions"

    seen_urls: list[str] = []

    class _FailoverOpener:
        def open(self, req, timeout=None):
            del timeout
            seen_urls.append(req.full_url)
            if "azure.com" in req.full_url:
                hdrs = email.message.Message()
                hdrs["retry-after-ms"] = "9000"
                raise urllib.error.HTTPError(
                    url=req.full_url,
                    code=429,
                    msg="RateLimitReached",
                    hdrs=hdrs,
                    fp=io.BytesIO(b'{"error":"RateLimitReached"}'),
                )
            return _FakeHttpResp(
                {
                    "model": "microsoft/microsoft-decision-1.1",
                    "answers": {
                        "tier": {"type": "choice", "choice": "system1", "confidence": 0.95, "probabilities": {"system1": 0.95, "system2": 0.05}},
                        "effort": {"type": "score", "score": 0.3, "confidence": 0.91},
                        "trap_risk": {"type": "noul", "noul": 0.05},
                        "needs_plan": {"type": "noul", "noul": 0.04},
                    },
                    "usage": {"input_tokens": 110, "output_tokens": 0},
                }
            )

    ctx = _StubPluginContext(
        config={
            "mode": "shadow",
            "backend": "foundry_systemone",
            "foundry_endpoint": "https://primary.services.ai.azure.com",
            "secondary_systemone_endpoint": "openrouter",
            "max_retry_wait_seconds": 0.2,
            "triage_cache_ttl_seconds": 0.0,
        }
    )
    with unittest.mock.patch.object(pkg.backends, "read_secret", return_value="placeholder-key"):
        verdict = pkg.triage.run_turn_triage(
            ctx,
            session_id="ses_fo",
            user_message="Translate hello to French",
            model="gpt-5.4",
            opener_factory=lambda _h: _FailoverOpener(),
        )
    assert verdict.error is None
    assert verdict.failover_used is True
    assert verdict.backend == "secondary_systemone"
    assert verdict.effective_tier == "system1"
    assert any("openrouter.ai" in u for u in seen_urls)

    # LocalLogitsBackend executes 4 questions concurrently under a single global deadline
    DQ = pkg.contracts.DecisionQuestion

    class _SlowParallelOpener:
        def open(self, req, timeout=None):
            del req, timeout
            time.sleep(0.06)
            return _FakeHttpResp(
                {
                    "choices": [
                        {
                            "message": {"content": "A"},
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "A",
                                        "logprob": -0.05,
                                        "top_logprobs": [
                                            {"token": "A", "logprob": -0.05},
                                            {"token": "B", "logprob": -3.0},
                                            {"token": "Y", "logprob": -0.1},
                                            {"token": "N", "logprob": -2.5},
                                        ],
                                    }
                                ]
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 40},
                }
            )

    local = pkg.backends.LocalLogitsBackend(
        url="http://127.0.0.1:8080/v1",
        model="gemma-4-26b-a4b",
        timeout=1.0,
        opener_factory=lambda _h: _SlowParallelOpener(),
    )
    t0 = time.perf_counter()
    batch = local.decide({"msg": "test"}, pkg.triage.TURN_QUESTIONS)
    elapsed = time.perf_counter() - t0
    assert batch.ok
    assert len(batch.answers) == 4
    # 4 x 60ms in series would take >= 240ms; parallel runs in ~60-140ms
    assert elapsed < 0.22


def test_conformal_calibration_closed_loop_hardening_and_cache_telemetry(pkg):
    """Split conformal calibration, closed-loop false-S1 hardening, and KV-cache metrics work end to end."""
    from test_pipeline import _StubPluginContext

    # 1. Split conformal calibration selects safe thresholds bounded by target_alpha
    samples = []
    for i in range(60):
        samples.append(
            {
                "expected_effective_tier": "system1",
                "raw_tier": "system1",
                "confidence": 0.92 if i < 54 else 0.78,
                "trap_risk_prob": 0.12,
                "effort_score": 0.5,
                "error_category": "ok",
            }
        )
    for i in range(100):
        samples.append(
            {
                "expected_effective_tier": "system2",
                "raw_tier": "system1" if i < 20 else "system2",
                "confidence": 0.86 if i < 20 else 0.94,
                "trap_risk_prob": 0.54 if i < 20 else 0.75,
                "effort_score": 1.3 if i < 20 else 3.2,
                "error_category": "ok",
            }
        )

    profile = pkg.calibration.calibrate_thresholds(samples, backend="hermes_aux", target_alpha=0.02)
    assert profile.empirical_false_fast_route_rate <= 0.02
    assert profile.conformal_risk_bound <= 0.02
    assert profile.achieved_s1_recall >= 0.85

    saved = pkg.calibration.save_calibration_profile(profile)
    loaded = pkg.calibration.load_calibration_profile("hermes_aux")
    assert loaded is not None and saved.is_file()
    assert loaded.confidence_threshold == profile.confidence_threshold
    assert loaded.trap_risk_threshold == profile.trap_risk_threshold

    # 2. Closed-loop hardening tightens thresholds when recent false-S1 rate > target_alpha
    c_new, t_new, tightened = pkg.calibration.compute_closed_loop_adjustment(
        base_conf_threshold=0.82,
        base_trap_threshold=0.50,
        recent_s1_turns=10,
        recent_false_s1_turns=2,  # 20% > 3%
        target_alpha=0.03,
    )
    assert tightened is True
    assert c_new > 0.82
    assert t_new < 0.50

    # 3. Real pipeline records cache_read_tokens, reasoning_tokens, step_paced in `recent`, and false_s1_escalated!
    ctx = _StubPluginContext(
        config={
            "mode": "effort_only",
            "backend": "hermes_aux",
            "use_calibrated_thresholds": True,
            "closed_loop_enabled": True,
        },
        scripted_decisions=[
            {
                "tier": {"choice": "system1", "confidence": 0.95},
                "effort": {"score": 0.4, "confidence": 0.92},
                "trap_risk": {"noul": 0.08},
                "needs_plan": {"noul": 0.05},
            }
        ],
    )
    pkg.register(ctx)
    ctx.hooks["pre_llm_call"](session_id="ses_cl", user_message="Quick lookup", model="gpt-5.4")
    ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_cl",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=1,
    )
    ctx.hooks["post_api_request"](
        session_id="ses_cl",
        api_duration=0.45,
        usage={
            "input_tokens": 4000,
            "output_tokens": 120,
            "input_tokens_details": {"cached_tokens": 3200},
            "output_tokens_details": {"reasoning_tokens": 64},
        },
    )
    # Agent discovers hidden bug mid-turn and calls trismegistus_escalate -> recorded as false_s1_escalated!
    ctx.tools["trismegistus_escalate"]["handler"]({"reason": "Hidden concurrency bug"}, session_id="ses_cl")

    stats = pkg.ledger.read_summary_stats(limit_recent=5)
    assert stats["cache_read_tokens"] == 3200
    assert stats["reasoning_tokens"] == 64
    assert abs(stats["cache_hit_rate"] - 0.8) < 1e-4
    assert stats["false_s1_signals"] == 1
    assert stats["recent"][0]["turn_outcome"] == "false_s1_escalated"

    cal_cmd = ctx.commands["trismegistus"]("calibrate", session_id="ses_cl")
    assert "Conformal Calibration Profile (hermes_aux)" in cal_cmd
    out_cmd = ctx.commands["trismegistus"]("outcomes", session_id="ses_cl")
    assert "false_s1_escalated" in out_cmd


def test_provider_adapters_anthropic_adaptive_and_structured_delegation_routes(pkg):
    """Anthropic adapter respects adaptive thinking & never injects Copilot fields; delegation supports multi-provider routes."""
    from test_pipeline import _StubPluginContext

    # 1. Anthropic adaptive thinking modulates output_config.effort safely
    adapt_req = {
        "model": "claude-sonnet-5",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    out_req, eff = pkg.copilot_lanes.apply_effort_to_request(
        adapt_req,
        effort="xhigh",
        effective_model="claude-sonnet-5",
        provider="anthropic",
        api_mode="anthropic_messages",
    )
    assert eff == "high"
    assert out_req["output_config"]["effort"] == "high"
    assert "reasoning_effort" not in out_req

    # 2. Multi-provider structured delegation routes
    ctx = _StubPluginContext(
        config={"backend": "hermes_aux", "delegation_provider": "openrouter"},
        scripted_decisions=[{"specialty": {"choice": "deep_code", "confidence": 0.96}}],
    )
    pkg.register(ctx)
    dp = ctx.delegation_provider
    cfg = {
        "backends": {
            "trismegistus": {
                "routes": {
                    "deep_code": {
                        "provider": "openai",
                        "model": "gpt-5.4",
                        "api_mode": "codex_responses",
                        "reasoning_effort": "xhigh",
                    }
                }
            }
        }
    }
    routed = dp.route_task({"goal": "Prove ring buffer ordering"}, cfg)
    assert routed == {
        "provider": "openai",
        "model": "gpt-5.4",
        "api_mode": "codex_responses",
        "reasoning_effort": "xhigh",
        "label": "trismegistus:deep_code",
    }


def test_e2e_eval_harness_runs_and_preserves_full_precision_with_speedup(pkg):
    del pkg
    e2e_path = ROOT / "benchmarks" / "run_e2e_eval.py"
    spec = importlib.util.spec_from_file_location("trismegistus_e2e_runner", str(e2e_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    report = mod.run_e2e_evaluation()
    assert report["status"] == "completed"
    assert report["tasks_count"] == 30
    assert report["policies"]["effort_only"]["pass_at_1"] == 1.0
    assert report["policies"]["effort_only"]["false_s1_rate"] == 0.0
    assert report["policies"]["effort_only"]["fast_path_s2_hits"] >= 4
    assert report["speedup_vs_baseline"]["effort_only_latency_reduction_pct"] > 10.0


def test_hermes_top_level_cache_read_tokens_and_closed_loop_monotonicity_regressions(pkg):
    """Regression tests for Hermes top-level `cache_read_tokens` and strict-threshold monotonicity."""
    from dataclasses import dataclass as _dc
    from test_pipeline import _StubPluginContext

    # 1. Hermes delivers `cache_read_tokens` at top-level of `usage` (both dict and object forms)
    assert pkg._extract_usage_metrics(
        {"input_tokens": 4000, "output_tokens": 120, "cache_read_tokens": 3200, "cache_write_tokens": 200}
    ) == (4000, 120, 3200, 200, 0)

    @_dc
    class _UsageObj:
        input_tokens: int = 4000
        output_tokens: int = 120
        cache_read_tokens: int = 3200
        cache_write_tokens: int = 0
        reasoning_tokens: int = 48

    assert pkg._extract_usage_metrics(_UsageObj()) == (4000, 120, 3200, 0, 48)

    # Full hook path records Hermes's top-level `cache_read_tokens` into ledger.db
    ctx = _StubPluginContext(
        config={"mode": "shadow", "backend": "hermes_aux", "use_calibrated_thresholds": False, "closed_loop_enabled": False},
        scripted_decisions=[
            {
                "tier": {"choice": "system1", "confidence": 0.94},
                "effort": {"score": 0.4, "confidence": 0.90},
                "trap_risk": {"noul": 0.05},
                "needs_plan": {"noul": 0.05},
            }
        ],
    )
    pkg.register(ctx)
    ctx.hooks["pre_llm_call"](session_id="ses_hcache", user_message="Translate hi to French", model="gpt-5.4")
    ctx.hooks["post_api_request"](
        session_id="ses_hcache",
        api_duration=0.30,
        usage={"input_tokens": 4000, "output_tokens": 40, "cache_read_tokens": 3200},
    )
    stats = pkg.ledger.read_summary_stats(limit_recent=5)
    assert stats["cache_read_tokens"] == 3200
    assert abs(stats["cache_hit_rate"] - 0.8) < 1e-4

    # 2. Closed-loop adjustment MUST NEVER loosen already-strict thresholds (e.g. conf=0.98, trap=0.25)
    c_both, t_both, tightened_both = pkg.calibration.compute_closed_loop_adjustment(
        base_conf_threshold=0.98,
        base_trap_threshold=0.25,
        recent_s1_turns=10,
        recent_false_s1_turns=3,
        target_alpha=0.03,
    )
    assert c_both == 0.98
    assert t_both == 0.25
    assert tightened_both is False

    # Mixed cases: tighten only the dimension that still has headroom, preserve the stricter one
    c_mix1, t_mix1, tightened_mix1 = pkg.calibration.compute_closed_loop_adjustment(
        base_conf_threshold=0.98,
        base_trap_threshold=0.50,
        recent_s1_turns=10,
        recent_false_s1_turns=2,
        target_alpha=0.03,
    )
    assert c_mix1 == 0.98
    assert t_mix1 < 0.50 and t_mix1 >= 0.30
    assert tightened_mix1 is True

    c_mix2, t_mix2, tightened_mix2 = pkg.calibration.compute_closed_loop_adjustment(
        base_conf_threshold=0.82,
        base_trap_threshold=0.25,
        recent_s1_turns=10,
        recent_false_s1_turns=2,
        target_alpha=0.03,
    )
    assert c_mix2 > 0.82 and c_mix2 <= 0.96
    assert t_mix2 == 0.25
    assert tightened_mix2 is True
