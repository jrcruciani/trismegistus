"""GitHub Copilot wire-mode, reasoning-effort clamping, and System-1/System-2 lane pairs.

Aligned with Hermes's `agent/reasoning_effort.py` and `github-copilot` provider profile:
- `codex_responses` (`/responses`): GPT-5+ except `gpt-5-mini`, `gpt-6*`, `mai-code-*`, `grok-*`.
  `minimal` is rejected on `/responses` (clamps to `low`), while `xhigh` (and `max` on GPT-5.6 /
  GPT-6 Astra / Sol) IS supported! Reasoning lives at `request["reasoning"] = {"effort": ..., "summary": "auto"}`.
- `chat_completions` (`/chat/completions`): `gpt-5-mini`, `gpt-4.1`, `gpt-4o*`, `claude-*`, `gemini-*`.
  `claude-sonnet-5` / `claude-opus-5` support `("low", "medium", "high", "xhigh", "max")`.
  `gemini-3.8+` / `gemini-4*` support `("low", "medium", "high")` (`minimal` 400s).
  Non-reasoning models (`claude-haiku-4.5`, `gpt-4.1`, `gpt-4o-mini`) reject `reasoning_effort` (400).
- Signed carrier guard: Gemini 3/4 thinking turns attach `thought_signature` on tool calls,
  and Copilot Responses/Chat attaches `reasoning_opaque`.
"""

from __future__ import annotations

from typing import Any

from .providers_adapter import (
    COPILOT_ID_ALIASES,
    EFFORT_ORDER,
    EFFORTS_CHAT_GPT5,
    EFFORTS_CLAUDE5,
    EFFORTS_CODEX_GPT56_PLUS,
    EFFORTS_CODEX_LEGACY,
    EFFORTS_GEMINI_THINKING,
    EFFORTS_GPT6_NO_DISABLE,
    EFFORTS_GROK46_PLUS,
    EFFORTS_O_SERIES,
    clamp_effort,
    get_provider_adapter,
    has_signed_replay_carriers,
    infer_copilot_api_mode,
    is_copilot_provider,
    model_family,
    normalize_copilot_model,
)


def supported_reasoning_efforts(
    model_id: str | None,
    *,
    provider: str = "",
    api_mode: str = "",
) -> tuple[str, ...]:
    """Return wire-valid `reasoning_effort` levels for `(model_id, provider, api_mode)`."""
    if not provider or is_copilot_provider(provider):
        return get_provider_adapter("copilot", api_mode).supported_efforts(model_id, api_mode=api_mode)
    return get_provider_adapter(provider, api_mode).supported_efforts(model_id, api_mode=api_mode)


def default_copilot_peer(active_model: str, target_tier: str, api_mode: str) -> str:
    """Pick a wire-compatible Copilot peer model when explicit overrides are left empty."""
    return get_provider_adapter("copilot", api_mode).default_peer(active_model, target_tier, api_mode)


def can_safely_swap_model(
    *,
    provider: str,
    api_mode: str,
    current_model: str,
    candidate_model: str,
    request: dict[str, Any],
) -> tuple[bool, str]:
    """Verify wire-mode and replay-carrier safety before changing `request['model']`."""
    adapter = get_provider_adapter(provider, api_mode)
    return adapter.validate_swap(
        api_mode=api_mode,
        current_model=current_model,
        candidate_model=candidate_model,
        request=request,
    )


def apply_effort_to_request(
    request: dict[str, Any],
    *,
    effort: str,
    effective_model: str,
    provider: str,
    api_mode: str,
) -> tuple[dict[str, Any], str | None]:
    """Apply or strip reasoning effort through the resolved `ProviderAdapter`."""
    adapter = get_provider_adapter(provider, api_mode)
    return adapter.apply_effort(
        request,
        effort=effort,
        effective_model=effective_model,
        api_mode=api_mode,
    )


def preserves_cache_on_effort_change(
    model_id: str | None,
    *,
    provider: str = "",
    api_mode: str = "",
    request: dict[str, Any] | None = None,
) -> bool:
    """Whether mid-turn effort pacing is cache-safe for `(provider, model_id, api_mode)`."""
    adapter = get_provider_adapter(provider, api_mode)
    return adapter.preserves_cache_on_effort_change(model_id, api_mode=api_mode, request=request)
