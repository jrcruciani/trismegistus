"""End-to-end tests for Trismegistus v0.2: Precision Trap Gate, Dynamic Step Pacing, Specialty Delegation & Ledger."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass
class _FakeAuxResult:
    parsed: dict[str, Any] | None
    model: str = "microsoft-decision-1.1"
    usage: Any = None


class _FakeLlmSurface:
    def __init__(self, scripted: list[dict[str, Any]]):
        self.scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        nxt = self.scripted.pop(0) if self.scripted else {}
        return _FakeAuxResult(parsed=nxt)


class _StubPluginContext:
    def __init__(self, config: dict[str, Any] | None = None, scripted_decisions: list[dict[str, Any]] | None = None):
        self._config = dict(config or {})
        self._state: dict[str, Any] = {}
        self.llm = _FakeLlmSurface(scripted_decisions or [])
        self.tools: dict[str, Any] = {}
        self.hooks: dict[str, Any] = {}
        self.middleware: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}
        self.delegation_provider: Any = None
        self.aux_tasks: dict[str, Any] = {}
        self.skills: dict[str, Any] = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        self._config[key] = value

    def has_capability(self, cap: str) -> bool:
        return True

    @property
    def state(self):
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def register_auxiliary_task(self, name: str, **kwargs: Any) -> None:
        self.aux_tasks[name] = kwargs

    def register_delegation_provider(self, provider: Any) -> None:
        self.delegation_provider = provider

    def register_skill(self, name: str, path: Any) -> None:
        self.skills[name] = path

    def register_tool(self, name: str, toolset: str, schema: dict, handler: Any, **kwargs: Any) -> None:
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler}

    def register_hook(self, name: str, callback: Any, **kwargs: Any) -> None:
        self.hooks[name] = callback

    def register_middleware(self, kind: str, callback: Any) -> None:
        self.middleware[kind] = callback

    def register_command(self, name: str, handler: Any, **kwargs: Any) -> None:
        self.commands[name] = handler


def test_full_three_stage_system1_downgrade_and_ledger(pkg):
    ctx = _StubPluginContext(
        config={"mode": "tier_and_effort", "backend": "hermes_aux", "confidence_threshold": 0.82},
        scripted_decisions=[
            {
                "tier": {"choice": "system1", "confidence": 0.94},
                "effort": {"score": 0.2, "confidence": 0.91},
                "trap_risk": {"noul": 0.04},
                "needs_plan": {"noul": 0.05},
            }
        ],
    )
    pkg.register(ctx)

    inject = ctx.hooks["pre_llm_call"](
        session_id="ses_1",
        user_message="Translate 'good morning' to Spanish",
        conversation_history=[],
        model="claude-sonnet-4.6",
        platform="cli",
    )
    assert inject is None

    mw_out = ctx.middleware["llm_request"](
        request={"model": "claude-sonnet-4.6", "reasoning_effort": "medium", "messages": [{"role": "user", "content": "hi"}]},
        session_id="ses_1",
        provider="copilot",
        model="claude-sonnet-4.6",
        api_mode="chat_completions",
        api_call_count=1,
    )
    assert mw_out is not None
    assert mw_out["request"]["model"] == "claude-haiku-4.5"
    assert "reasoning_effort" not in mw_out["request"]

    resp = ctx.middleware["llm_execution"](
        request=mw_out["request"],
        original_request={"model": "claude-sonnet-4.6"},
        next_call=lambda req: {"ok": True, "used_model": req["model"]},
    )
    assert resp["used_model"] == "claude-haiku-4.5"

    ctx.hooks["post_api_request"](
        session_id="ses_1",
        api_duration=0.35,
        usage={"input_tokens": 420, "output_tokens": 12},
    )

    stats_text = ctx.commands["trismegistus"]("stats", session_id="ses_1")
    assert "S1: 1" in stats_text
    assert "claude-haiku-4.5" in stats_text


def test_hidden_trap_gate_vetoes_false_system1_downgrade(pkg):
    """Even when tier confidence is 0.91 for system1, trap_risk >= 0.30 blocks low effort!"""
    ctx = _StubPluginContext(
        config={
            "mode": "effort_only",
            "backend": "hermes_aux",
            "confidence_threshold": 0.82,
            "trap_risk_threshold": 0.30,
        },
        scripted_decisions=[
            {
                # Looks like a 1-line edit ("Change timeout from 5 to 30"), so tier=system1 (0.91),
                # but trap_risk spots hidden state/concurrency coupling (0.64 >= 0.30)!
                "tier": {"choice": "system1", "confidence": 0.91},
                "effort": {"score": 1.1, "confidence": 0.85},
                "trap_risk": {"noul": 0.64},
                "needs_plan": {"noul": 0.35},
            }
        ],
    )
    pkg.register(ctx)
    ctx.hooks["pre_llm_call"](
        session_id="ses_trap",
        user_message="Bump lock timeout from 5s to 30s in worker.py",
        conversation_history=[],
        model="gpt-5.4",
        platform="cli",
    )
    mw_out = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_trap",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=1,
    )
    assert mw_out["request"]["model"] == "gpt-5.4"
    # Trap risk >= 0.60 escalates floor directly to `high`!
    assert mw_out["request"]["reasoning"]["effort"] == "high"
    assert "trap_guard->s2" in mw_out["reason"]


def test_default_trap_risk_threshold_is_calibrated_to_050(pkg):
    """Default trap_risk_threshold (0.50) allows routine 0.42 turns in System 1 while vetoing 0.52."""
    ctx = _StubPluginContext(
        config={"mode": "effort_only", "backend": "hermes_aux"},
        scripted_decisions=[
            {
                "tier": {"choice": "system1", "confidence": 0.96},
                "effort": {"score": 0.6, "confidence": 0.92},
                "trap_risk": {"noul": 0.42},
                "needs_plan": {"noul": 0.08},
            },
            {
                "tier": {"choice": "system1", "confidence": 0.96},
                "effort": {"score": 0.6, "confidence": 0.92},
                "trap_risk": {"noul": 0.52},
                "needs_plan": {"noul": 0.08},
            },
        ],
    )
    pkg.register(ctx)

    ctx.hooks["pre_llm_call"](
        session_id="ses_t042",
        user_message="Show weather in Vienna",
        conversation_history=[],
        model="gpt-5.4",
        platform="cli",
    )
    out_ok = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_t042",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=1,
    )
    assert out_ok["request"]["reasoning"]["effort"] == "low"
    assert "trap_guard->s2" not in out_ok["reason"]

    ctx.hooks["pre_llm_call"](
        session_id="ses_t052",
        user_message="Restart service if config drifted",
        conversation_history=[],
        model="gpt-5.4",
        platform="cli",
    )
    out_veto = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_t052",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=1,
    )
    assert out_veto["request"]["reasoning"]["effort"] == "medium"
    assert "trap_guard->s2" in out_veto["reason"]


def test_dynamic_intra_turn_step_pacing_preserves_model_and_restores_xhigh(pkg):
    """Step 1 plans at `xhigh`; read_file steps pace to `medium` for speed; patch restores `xhigh`!"""
    ctx = _StubPluginContext(
        config={"mode": "effort_only", "backend": "hermes_aux", "step_pacing_enabled": True},
        scripted_decisions=[
            {
                "tier": {"choice": "system2", "confidence": 0.98},
                "effort": {"score": 3.8, "confidence": 0.95},
                "trap_risk": {"noul": 0.72},
                "needs_plan": {"noul": 0.88},
            }
        ],
    )
    pkg.register(ctx)
    ctx.hooks["pre_llm_call"](
        session_id="ses_pace",
        user_message="Find and fix the deadlock in connection pool checkout",
        conversation_history=[],
        model="gpt-5.4",
        platform="cli",
    )

    # Call 1 (initial planning): full `xhigh`!
    c1 = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_pace",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=1,
    )
    assert c1["request"]["reasoning"]["effort"] == "xhigh"
    assert "step_paced" not in c1["reason"]

    # Agent calls read_file cleanly -> scout phase
    ctx.hooks["post_tool_call"](tool_name="read_file", result="1|class Pool: ...", status="ok", session_id="ses_pace")

    # Call 2 (after read_file): paced down to `medium` on same `gpt-5.4` (100% warm KV cache!)
    c2 = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_pace",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=2,
    )
    assert c2["request"]["model"] == "gpt-5.4"
    assert c2["request"]["reasoning"]["effort"] == "medium"
    assert "step_paced" in c2["reason"]

    # Agent edits code via `patch` -> verify_or_act phase
    ctx.hooks["post_tool_call"](tool_name="patch", result='{"success": true}', status="ok", session_id="ses_pace")

    # Call 3 (after mutation): immediately restores full `xhigh` precision!
    c3 = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_pace",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=3,
    )
    assert c3["request"]["reasoning"]["effort"] == "xhigh"
    assert "step_paced" not in c3["reason"]


def test_calibrated_abstention_keeps_system2_when_below_precision_floor(pkg):
    ctx = _StubPluginContext(
        config={"mode": "tier_and_effort", "backend": "hermes_aux", "confidence_threshold": 0.82},
        scripted_decisions=[
            {
                # Confidence 0.74 < 0.82 precision floor -> must abstain upward to System 2!
                "tier": {"choice": "system1", "confidence": 0.74},
                "effort": {"score": 1.2, "confidence": 0.70},
                "trap_risk": {"noul": 0.15},
                "needs_plan": {"noul": 0.25},
            }
        ],
    )
    pkg.register(ctx)
    ctx.hooks["pre_llm_call"](
        session_id="ses_abstain",
        user_message="Update the auth config",
        conversation_history=[],
        model="gpt-5.4",
        platform="cli",
    )
    mw_out = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_abstain",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=1,
    )
    assert mw_out["request"]["model"] == "gpt-5.4"
    assert "abstained->s2" in mw_out["reason"]


def test_tool_error_streak_promotes_to_system2_xhigh(pkg):
    ctx = _StubPluginContext(
        config={"mode": "tier_and_effort", "backend": "hermes_aux", "escalate_after_errors": 2},
        scripted_decisions=[
            {
                "tier": {"choice": "system1", "confidence": 0.95},
                "effort": {"score": 0.3, "confidence": 0.90},
                "trap_risk": {"noul": 0.05},
                "needs_plan": {"noul": 0.05},
            }
        ],
    )
    pkg.register(ctx)
    ctx.hooks["pre_llm_call"](
        session_id="ses_err",
        user_message="Run quick script",
        conversation_history=[],
        model="gpt-5.4",
        platform="cli",
    )
    first = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_err",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=1,
    )
    assert first["request"]["model"] == "gpt-5.4-mini"

    ctx.hooks["post_tool_call"](tool_name="terminal", result='{"error": "exit 1"}', status="error", session_id="ses_err")
    ctx.hooks["post_tool_call"](tool_name="terminal", result='{"error": "exit 2"}', status="error", session_id="ses_err")

    second = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_err",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=2,
    )
    assert second["request"]["model"] == "gpt-5.4"
    assert second["request"]["reasoning"]["effort"] == "xhigh"
    assert "err_streak->s2" in second["reason"]


def test_llm_execution_fails_open_on_unentitled_copilot_model(pkg):
    ctx = _StubPluginContext(config={"mode": "tier_and_effort", "backend": "hermes_aux"})
    pkg.register(ctx)

    calls = []

    def fake_next(req):
        calls.append(req["model"])
        if req["model"] == "claude-haiku-4.5":
            raise RuntimeError("HTTP 400: model_not_supported for this organization")
        return {"ok": True, "model": req["model"]}

    out = ctx.middleware["llm_execution"](
        request={"model": "claude-haiku-4.5", "messages": []},
        original_request={"model": "claude-sonnet-4.6", "messages": []},
        next_call=fake_next,
    )
    assert calls == ["claude-haiku-4.5", "claude-sonnet-4.6"]
    assert out["model"] == "claude-sonnet-4.6"


def test_specialty_delegation_provider_routes_four_cognitive_lanes(pkg):
    ctx = _StubPluginContext(
        config={
            "backend": "hermes_aux",
            "delegation_provider": "copilot",
            "confidence_threshold": 0.82,
        },
        scripted_decisions=[
            {"specialty": {"choice": "fast_scout", "confidence": 0.93}},
            {"specialty": {"choice": "deep_code", "confidence": 0.96}},
            {"specialty": {"choice": "architecture", "confidence": 0.91}},
            {"specialty": {"choice": "long_context", "confidence": 0.89}},
            # Uncertain fast_scout (0.68 < 0.82) -> asymmetric guard upgrades to deep_code!
            {"specialty": {"choice": "fast_scout", "confidence": 0.68}},
        ],
    )
    pkg.register(ctx)
    dp = ctx.delegation_provider
    assert dp is not None and dp.name == "trismegistus"

    r_scout = dp.route_task({"goal": "List all YAML files under config/"}, {})
    assert r_scout == {
        "provider": "copilot",
        "model": "gemini-3-flash-preview",
        "label": "trismegistus:fast_scout",
    }

    r_code = dp.route_task({"goal": "Prove lock-free ring buffer memory ordering and fix ABA bug"}, {})
    assert r_code == {
        "provider": "copilot",
        "model": "gpt-5.4",
        "label": "trismegistus:deep_code",
        "reasoning_effort": "xhigh",
    }

    r_arch = dp.route_task({"goal": "Review plugin boundary contracts across 6 modules"}, {})
    assert r_arch == {
        "provider": "copilot",
        "model": "claude-sonnet-5",
        "label": "trismegistus:architecture",
        "reasoning_effort": "high",
    }

    r_long = dp.route_task({"goal": "Cross-reference 40k lines of crash logs against docs"}, {})
    assert r_long == {
        "provider": "copilot",
        "model": "gemini-3.1-pro-preview",
        "label": "trismegistus:long_context",
        "reasoning_effort": "high",
    }

    r_guarded = dp.route_task({"goal": "Check subtle auth middleware"}, {})
    assert r_guarded["model"] == "gpt-5.4"
    assert r_guarded["reasoning_effort"] == "xhigh"


def test_non_copilot_delegation_inherits_host_route_instead_of_copilot_ids(pkg):
    ctx = _StubPluginContext(
        config={
            "backend": "hermes_aux",
            "delegation_provider": "openrouter",
            "confidence_threshold": 0.82,
        },
        scripted_decisions=[
            {"specialty": {"choice": "architecture", "confidence": 0.94}},
            {"specialty": {"choice": "fast_scout", "confidence": 0.95}},
        ],
    )
    pkg.register(ctx)
    dp = ctx.delegation_provider

    # 1. Without explicit OpenRouter model IDs, never inject Copilot's `claude-sonnet-5`:
    #    return label-only route so Hermes core inherits the parent's configured route cleanly.
    r_inherit = dp.route_task({"goal": "Design adapter boundaries"}, {})
    assert "model" not in r_inherit
    assert "reasoning_effort" not in r_inherit
    assert r_inherit["label"] == "trismegistus:architecture"

    # 2. When user explicitly configures a specialty model on that provider, route to it without Copilot xhigh.
    ctx.set_config("delegation_scout_model", "google/gemini-2.5-flash")
    r_explicit = dp.route_task({"goal": "Find YAML manifests"}, {})
    assert r_explicit == {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
        "label": "trismegistus:fast_scout",
    }


def test_tool_risk_gate_default_verify_gate_and_tools(pkg):
    # Note: verify_gate_enabled is NOT passed here — testing that its new v0.2 default is True!
    ctx = _StubPluginContext(
        config={
            "mode": "effort_only",
            "backend": "hermes_aux",
            "tool_gate_enabled": True,
            "tool_risk_threshold": 0.80,
        },
        scripted_decisions=[
            {"high_risk": {"noul": 0.92}},
            {"verified": {"noul": 0.12}},
            {"decision": {"choice": "system2", "confidence": 0.97}},
        ],
    )
    pkg.register(ctx)

    gate = ctx.hooks["pre_tool_call"](
        tool_name="terminal",
        args={"command": "rm -rf /var/lib/prod-data"},
    )
    assert gate is not None
    assert gate["action"] == "approve"

    verify_nudge = ctx.hooks["pre_verify"](
        coding=True,
        attempt=0,
        final_response="Done! Edited router.py.",
        changed_paths=["router.py"],
    )
    assert verify_nudge is not None
    assert verify_nudge["action"] == "continue"

    decide_raw = ctx.tools["trismegistus_decide"]["handler"](
        {
            "state": "Database migration touches auth schema",
            "question_type": "choice",
            "instructions": "Which review depth?",
            "options": {"system1": "fast", "system2": "deep"},
        }
    )
    decide_out = json.loads(decide_raw)
    assert decide_out["ok"] is True
    assert decide_out["answer"]["choice"] == "system2"

    esc_raw = ctx.tools["trismegistus_escalate"]["handler"](
        {"reason": "Found subtle race condition"}, session_id="ses_tools"
    )
    esc_out = json.loads(esc_raw)
    assert esc_out["escalated_to"] == "system2"
    assert esc_out["effective_effort"] == "xhigh"
