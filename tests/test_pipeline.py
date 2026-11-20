"""End-to-end tests for Trismegistus's 3-stage pipeline: Triage -> Modulation -> Call."""

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
        config={"mode": "tier_and_effort", "backend": "hermes_aux", "confidence_threshold": 0.65},
        scripted_decisions=[
            {
                "tier": {"choice": "system1", "confidence": 0.94},
                "effort": {"score": 0.2, "confidence": 0.91},
                "needs_plan": {"noul": 0.05},
            }
        ],
    )
    pkg.register(ctx)

    # Stage 1: Triage
    inject = ctx.hooks["pre_llm_call"](
        session_id="ses_1",
        user_message="Translate 'good morning' to Spanish",
        conversation_history=[],
        model="claude-sonnet-4.6",
        platform="cli",
    )
    assert inject is None

    # Stage 2: Modulation (Claude on Copilot -> claude-haiku-4.5, stale reasoning_effort stripped)
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

    # Stage 3: Call + Ledger recording
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
    assert "System 1: 1" in stats_text
    assert "claude-haiku-4.5" in stats_text


def test_calibrated_abstention_keeps_system2_when_uncertain(pkg):
    ctx = _StubPluginContext(
        config={"mode": "tier_and_effort", "backend": "hermes_aux", "confidence_threshold": 0.70},
        scripted_decisions=[
            {
                # Model leans system1, but confidence 0.56 < 0.70 -> must abstain to System 2!
                "tier": {"choice": "system1", "confidence": 0.56},
                "effort": {"score": 1.4, "confidence": 0.55},
                "needs_plan": {"noul": 0.40},
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


def test_tool_error_streak_and_explicit_escalate_promote_to_system2(pkg):
    ctx = _StubPluginContext(
        config={"mode": "tier_and_effort", "backend": "hermes_aux", "escalate_after_errors": 2},
        scripted_decisions=[
            {
                "tier": {"choice": "system1", "confidence": 0.92},
                "effort": {"score": 0.3, "confidence": 0.90},
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
    # Iteration 1 uses gpt-5.4-mini (System 1)
    first = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_err",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=1,
    )
    assert first["request"]["model"] == "gpt-5.4-mini"

    # Two consecutive tool errors happen mid-turn
    ctx.hooks["post_tool_call"](result='{"error": "exit 1"}', status="error", session_id="ses_err")
    ctx.hooks["post_tool_call"](result='{"error": "exit 2"}', status="error", session_id="ses_err")

    # Iteration 2 automatically escalates back to System 2 (gpt-5.4, high effort)!
    second = ctx.middleware["llm_request"](
        request={"model": "gpt-5.4", "input": []},
        session_id="ses_err",
        provider="copilot",
        model="gpt-5.4",
        api_mode="codex_responses",
        api_call_count=2,
    )
    assert second["request"]["model"] == "gpt-5.4"
    assert second["request"]["reasoning"]["effort"] == "high"
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


def test_delegation_provider_routes_across_copilot_models(pkg):
    ctx = _StubPluginContext(
        config={
            "backend": "hermes_aux",
            "delegation_provider": "copilot",
            "delegation_system1_model": "claude-haiku-4.5",
            "delegation_system2_model": "gpt-5.4",
        },
        scripted_decisions=[
            {"tier": {"choice": "system1", "confidence": 0.91}},
            {"tier": {"choice": "system2", "confidence": 0.96}},
        ],
    )
    pkg.register(ctx)
    dp = ctx.delegation_provider
    assert dp is not None and dp.name == "trismegistus"

    r1 = dp.route_task({"goal": "Count lines in docs/"}, {})
    assert r1 == {"provider": "copilot", "model": "claude-haiku-4.5", "label": "trismegistus:system1"}

    r2 = dp.route_task({"goal": "Design lock-free ring buffer and prove memory ordering"}, {})
    assert r2["provider"] == "copilot"
    assert r2["model"] == "gpt-5.4"
    assert r2["reasoning_effort"] == "high"


def test_tool_risk_gate_and_verify_gate_and_tools(pkg):
    ctx = _StubPluginContext(
        config={
            "mode": "effort_only",
            "backend": "hermes_aux",
            "tool_gate_enabled": True,
            "tool_risk_threshold": 0.80,
            "verify_gate_enabled": True,
        },
        scripted_decisions=[
            {"high_risk": {"noul": 0.92}},
            {"verified": {"noul": 0.12}},
            {"decision": {"choice": "system2", "confidence": 0.97}},
        ],
    )
    pkg.register(ctx)

    # 1. pre_tool_call escalates risky mutating tool to human approval
    gate = ctx.hooks["pre_tool_call"](
        tool_name="terminal",
        args={"command": "rm -rf /var/lib/prod-data"},
    )
    assert gate is not None
    assert gate["action"] == "approve"

    # 2. pre_verify nudges agent when code changed without verification evidence
    verify_nudge = ctx.hooks["pre_verify"](
        coding=True,
        attempt=0,
        final_response="Done! Edited router.py.",
        changed_paths=["router.py"],
    )
    assert verify_nudge is not None
    assert verify_nudge["action"] == "continue"

    # 3. Direct trismegistus_decide tool invocation
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

    # 4. Direct trismegistus_escalate tool invocation
    esc_raw = ctx.tools["trismegistus_escalate"]["handler"](
        {"reason": "Found subtle race condition"}, session_id="ses_tools"
    )
    assert json.loads(esc_raw)["escalated_to"] == "system2"

