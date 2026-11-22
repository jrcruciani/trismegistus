"""Tests for GitHub Copilot wire-mode safety, xhigh/max effort unlocking, and replay-carrier guards."""

from __future__ import annotations


def test_copilot_api_mode_classification(pkg):
    infer = pkg.copilot_lanes.infer_copilot_api_mode
    # Responses wire on Copilot
    assert infer("gpt-5.4") == "codex_responses"
    assert infer("gpt-5.4-mini") == "codex_responses"
    assert infer("gpt-5.3-codex") == "codex_responses"
    assert infer("gpt-6-astra") == "codex_responses"
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
    ok, why = can_swap(
        provider="copilot",
        api_mode="chat_completions",
        current_model="claude-sonnet-4.6",
        candidate_model="gpt-5.4",
        request={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert not ok
    assert "wire_mismatch" in why

    ok2, _ = can_swap(
        provider="copilot",
        api_mode="chat_completions",
        current_model="claude-sonnet-4.6",
        candidate_model="claude-haiku-4.5",
        request={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert ok2

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


def test_non_copilot_signed_replay_carrier_blocks_mid_session_swap(pkg):
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
    ok_gemini, why_gemini = can_swap(
        provider="openrouter",
        api_mode="chat_completions",
        current_model="google/gemini-2.5-pro",
        candidate_model="google/gemini-2.5-flash",
        request=req_with_sig,
    )
    assert not ok_gemini
    assert "thought_signature" in why_gemini

    req_with_encrypted = {
        "input": [{"type": "reasoning", "encrypted_content": "enc_abc123"}]
    }
    ok_cross, why_cross = can_swap(
        provider="openrouter",
        api_mode="codex_responses",
        current_model="openai/gpt-5.4",
        candidate_model="anthropic/claude-sonnet-4.6",
        request=req_with_encrypted,
    )
    assert not ok_cross
    assert "signed_carrier_guard" in why_cross


def test_effort_encoding_xhigh_unlocking_and_responses_minimal_clamp(pkg):
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

    # 2. Gemini 4 Carbon clamps 'xhigh' -> 'high' and 'minimal' -> 'low' on chat_completions
    updated_gem_hi, eff_gem_hi = apply_eff(
        {"model": "gemini-4-carbon", "messages": []},
        effort="xhigh",
        effective_model="gemini-4-carbon",
        provider="copilot",
        api_mode="chat_completions",
    )
    assert updated_gem_hi["reasoning_effort"] == "high"
    assert eff_gem_hi == "high"

    updated_gem_lo, eff_gem_lo = apply_eff(
        {"model": "gemini-4-carbon", "messages": []},
        effort="minimal",
        effective_model="gemini-4-carbon",
        provider="copilot",
        api_mode="chat_completions",
    )
    assert updated_gem_lo["reasoning_effort"] == "low"
    assert eff_gem_lo == "low"

    # 3. GPT-5.4 on codex_responses unlocks `xhigh`, AND clamps `minimal` -> `low` (never `none`!)
    updated_codex_xh, eff_codex_xh = apply_eff(
        {"model": "gpt-5.4", "input": []},
        effort="xhigh",
        effective_model="gpt-5.4",
        provider="copilot",
        api_mode="codex_responses",
    )
    assert updated_codex_xh["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    assert "reasoning_effort" not in updated_codex_xh
    assert eff_codex_xh == "xhigh"

    updated_codex_min, eff_codex_min = apply_eff(
        {"model": "gpt-5.4", "input": []},
        effort="minimal",
        effective_model="gpt-5.4",
        provider="copilot",
        api_mode="codex_responses",
    )
    assert updated_codex_min["reasoning"]["effort"] == "low"
    assert eff_codex_min == "low"

    # 4. Claude Sonnet 5 on chat_completions unlocks `xhigh` natively
    updated_c5, eff_c5 = apply_eff(
        {"model": "claude-sonnet-5", "messages": []},
        effort="xhigh",
        effective_model="claude-sonnet-5",
        provider="copilot",
        api_mode="chat_completions",
    )
    assert updated_c5["reasoning_effort"] == "xhigh"
    assert eff_c5 == "xhigh"


def test_non_copilot_providers_passthrough_without_injecting_or_stripping_fields(pkg):
    apply_eff = pkg.copilot_lanes.apply_effort_to_request

    # 1. Direct Anthropic provider must NEVER receive Copilot's `reasoning_effort` field
    anth_req = {
        "model": "claude-sonnet-5",
        "thinking": {"type": "enabled", "budget_tokens": 8192},
        "messages": [{"role": "user", "content": "hi"}],
    }
    out_anth, eff_anth = apply_eff(
        anth_req,
        effort="xhigh",
        effective_model="claude-sonnet-5",
        provider="anthropic",
        api_mode="chat_completions",
    )
    assert out_anth == anth_req
    assert "reasoning_effort" not in out_anth
    assert eff_anth is None

    # 2. Custom Azure/OpenAI Responses deployment must NOT have its existing `reasoning` stripped
    az_req = {
        "model": "my-enterprise-deployment",
        "reasoning": {"effort": "high", "summary": "concise"},
        "input": [],
    }
    out_az, eff_az = apply_eff(
        az_req,
        effort="low",
        effective_model="my-enterprise-deployment",
        provider="azure",
        api_mode="codex_responses",
    )
    assert out_az["reasoning"] == {"effort": "low", "summary": "concise"}
    assert eff_az == "low"

    # 3. Unknown OpenRouter model without existing reasoning_effort stays untouched
    or_req = {"model": "meta-llama/llama-4-maverick", "messages": []}
    out_or, eff_or = apply_eff(
        or_req,
        effort="high",
        effective_model="meta-llama/llama-4-maverick",
        provider="openrouter",
        api_mode="chat_completions",
    )
    assert out_or == or_req
    assert eff_or is None
