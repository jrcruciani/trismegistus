"""Tests for GitHub Copilot wire-mode safety, effort clamping, and replay-carrier guards."""

from __future__ import annotations


def test_copilot_api_mode_classification(pkg):
    infer = pkg.copilot_lanes.infer_copilot_api_mode
    # Responses wire on Copilot
    assert infer("gpt-5.4") == "codex_responses"
    assert infer("gpt-5.4-mini") == "codex_responses"
    assert infer("gpt-5.3-codex") == "codex_responses"
    assert infer("mai-code-2") == "codex_responses"
    assert infer("grok-4") == "codex_responses"
    # Chat Completions wire on Copilot
    assert infer("gpt-5-mini") == "chat_completions"
    assert infer("gpt-4.1") == "chat_completions"
    assert infer("claude-haiku-4.5") == "chat_completions"
    assert infer("claude-sonnet-4.6") == "chat_completions"
    assert infer("claude-sonnet-5") == "chat_completions"
    assert infer("gemini-3-flash-preview") == "chat_completions"
    assert infer("gemini-4-carbon") == "chat_completions"


def test_wire_mismatch_blocks_unsafe_copilot_swap(pkg):
    can_swap = pkg.copilot_lanes.can_safely_swap_model
    # Session started on Claude (chat_completions) cannot swap in-place to gpt-5.4 (codex_responses)
    ok, why = can_swap(
        provider="copilot",
        api_mode="chat_completions",
        current_model="claude-sonnet-4.6",
        candidate_model="gpt-5.4",
        request={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert not ok
    assert "wire_mismatch" in why

    # Same wire (`claude-sonnet-4.6` -> `claude-haiku-4.5`) is safe
    ok2, _ = can_swap(
        provider="copilot",
        api_mode="chat_completions",
        current_model="claude-sonnet-4.6",
        candidate_model="claude-haiku-4.5",
        request={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert ok2

    # Same wire on codex_responses (`gpt-5.4` -> `gpt-5.4-mini`) is safe
    ok3, _ = can_swap(
        provider="copilot",
        api_mode="codex_responses",
        current_model="gpt-5.4",
        candidate_model="gpt-5.4-mini",
        request={"input": [{"role": "user", "content": "hi"}]},
    )
    assert ok3


def test_gemini_thought_signature_carrier_blocks_mid_session_swap(pkg):
    can_swap = pkg.copilot_lanes.can_safely_swap_model
    req_with_sig = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{}",
                            "thought_signature": "opaque-sig-bytes",
                        },
                    }
                ],
            }
        ]
    }
    ok, why = can_swap(
        provider="copilot",
        api_mode="chat_completions",
        current_model="gemini-4-carbon",
        candidate_model="gemini-3-flash-preview",
        request=req_with_sig,
    )
    assert not ok
    assert "thought_signature" in why


def test_effort_encoding_and_stripping_per_model_and_wire(pkg):
    apply_eff = pkg.copilot_lanes.apply_effort_to_request

    # 1. Downgrading to claude-haiku-4.5 (non-reasoning) MUST strip stale reasoning_effort
    updated, eff = apply_eff(
        {"model": "claude-haiku-4.5", "reasoning_effort": "high", "messages": []},
        effort="low",
        effective_model="claude-haiku-4.5",
        provider="copilot",
        api_mode="chat_completions",
    )
    assert "reasoning_effort" not in updated
    assert eff is None

    # 2. Gemini 4 Carbon clamps 'minimal' -> 'low' on chat_completions
    updated_gem, eff_gem = apply_eff(
        {"model": "gemini-4-carbon", "messages": []},
        effort="minimal",
        effective_model="gemini-4-carbon",
        provider="copilot",
        api_mode="chat_completions",
    )
    assert updated_gem["reasoning_effort"] == "low"
    assert eff_gem == "low"

    # 3. GPT-5.4 on codex_responses writes nested `reasoning: {effort, summary: 'auto'}`
    updated_codex, eff_codex = apply_eff(
        {"model": "gpt-5.4", "input": []},
        effort="high",
        effective_model="gpt-5.4",
        provider="copilot",
        api_mode="codex_responses",
    )
    assert updated_codex["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "reasoning_effort" not in updated_codex
    assert eff_codex == "high"
