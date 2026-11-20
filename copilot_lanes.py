"""GitHub Copilot wire-mode, reasoning-effort clamping, and System-1/System-2 lane pairs.

Derived from Hermes's `github-copilot` provider profile:
- `codex_responses` (`/responses`): GPT-5+ except `gpt-5-mini`, `mai-code-*`, `grok-*`.
  Reasoning payload lives at `request["reasoning"] = {"effort": ..., "summary": "auto"}`.
- `chat_completions` (`/chat/completions`): `gpt-5-mini`, `gpt-4.1`, `gpt-4o*`,
  `claude-*`, `gemini-*`. Reasoning lives at top-level `request["reasoning_effort"]` ONLY
  for models that advertise reasoning support; non-reasoning models reject the field (400).
- Signed carrier guard: Gemini 3/4 thinking turns attach `thought_signature` on tool calls,
  and Copilot Responses sessions may attach `reasoning_opaque`. Switching model families mid-
  session when signed carriers are present in history will 400, so Trismegistus stays on the
  active model and modulates effort only.
"""

from __future__ import annotations

import re
from typing import Any

_DUAL_RESPONSES_PREFIXES = ("mai-code-", "grok-")

EFFORT_ORDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

EFFORTS_GPT5: tuple[str, ...] = ("minimal", "low", "medium", "high")
EFFORTS_O_SERIES: tuple[str, ...] = ("low", "medium", "high")
EFFORTS_GEMINI_THINKING: tuple[str, ...] = ("low", "medium", "high")
EFFORTS_CLAUDE5: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# Copilot model ID normalization (mirrors github-copilot MODEL_ALIASES for hyphen vs dot Claude IDs)
COPILOT_ID_ALIASES: dict[str, str] = {
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-sonnet-4-0": "claude-sonnet-4",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "anthropic/claude-opus-4.6": "claude-opus-4.6",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5": "claude-haiku-4.5",
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5.4": "gpt-5.4",
    "openai/gpt-5.4-mini": "gpt-5.4-mini",
}


def normalize_copilot_model(model_id: str | None) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return ""
    if raw in COPILOT_ID_ALIASES:
        return COPILOT_ID_ALIASES[raw]
    lower = raw.lower()
    if lower in COPILOT_ID_ALIASES:
        return COPILOT_ID_ALIASES[lower]
    return raw


def infer_copilot_api_mode(model_id: str | None) -> str:
    """Return the wire protocol (`codex_responses` vs `chat_completions`) Copilot uses for `model_id`."""
    norm = normalize_copilot_model(model_id).lower()
    if not norm:
        return "chat_completions"
    if norm.startswith(_DUAL_RESPONSES_PREFIXES):
        return "codex_responses"
    match = re.match(r"^gpt-(\d+)", norm)
    if match and int(match.group(1)) >= 5 and not norm.startswith("gpt-5-mini"):
        return "codex_responses"
    return "chat_completions"


def model_family(model_id: str | None) -> str:
    norm = normalize_copilot_model(model_id).lower()
    if "claude" in norm:
        return "claude"
    if "gemini" in norm:
        return "gemini"
    if norm.startswith(("gpt-", "o1", "o3", "o4", "mai-code")):
        return "openai"
    if norm.startswith("grok"):
        return "grok"
    return "other"


def supported_reasoning_efforts(model_id: str | None, *, provider: str = "") -> tuple[str, ...]:
    """Return supported `reasoning_effort` levels for a model (empty tuple if unsupported)."""
    norm = normalize_copilot_model(model_id).lower()
    if not norm:
        return ()
    if norm.startswith(("o1", "o3", "o4", "openai/o1", "openai/o3", "openai/o4")):
        return EFFORTS_O_SERIES
    if norm.startswith("gpt-5"):
        return EFFORTS_GPT5
    if norm.startswith(("gemini-4", "gemini-3.8", "gemini-3.1-pro", "gemini-3-pro", "gemini-2.5-pro")):
        return EFFORTS_GEMINI_THINKING
    if norm.startswith(("claude-opus-5", "claude-sonnet-5", "claude-haiku-5")):
        return EFFORTS_CLAUDE5
    # Non-Copilot generic reasoning models (e.g. DeepSeek-R1, QwQ, OpenRouter o-series)
    if provider.lower() not in ("copilot", "github-copilot", "github"):
        if any(tag in norm for tag in ("reasoner", "thinking", "-r1", "qwq")):
            return EFFORTS_O_SERIES
    return ()


def clamp_effort(desired: str, supported: tuple[str, ...]) -> str | None:
    """Clamp `desired` to `supported`, preferring the nearest weaker supported level."""
    if not supported:
        return None
    target = str(desired or "medium").strip().lower()
    if target in supported:
        return target
    if target in EFFORT_ORDER:
        idx = EFFORT_ORDER.index(target)
        # Search downward first (nearest weaker level), then upward
        for i in range(idx - 1, -1, -1):
            if EFFORT_ORDER[i] in supported:
                return EFFORT_ORDER[i]
        for i in range(idx + 1, len(EFFORT_ORDER)):
            if EFFORT_ORDER[i] in supported:
                return EFFORT_ORDER[i]
    return "medium" if "medium" in supported else supported[0]


def has_signed_replay_carriers(request: dict[str, Any]) -> bool:
    """Detect provider/family-bound cryptographic replay carriers in the outgoing request.

    Gemini 3/4 thinking attaches `thought_signature` / `extra_content` to tool calls, and
    Copilot Responses/Chat attaches `reasoning_opaque`. Switching model families mid-session
    once these exist in history triggers HTTP 400/401 from upstream.
    """
    messages = request.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("reasoning_opaque"):
                return True
            for tc in msg.get("tool_calls") or ():
                if not isinstance(tc, dict):
                    continue
                if tc.get("extra_content"):
                    return True
                fn = tc.get("function")
                if isinstance(fn, dict) and fn.get("thought_signature"):
                    return True
    items = request.get("input")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and (
                item.get("encrypted_content") or item.get("type") == "reasoning"
            ):
                return True
    return False


def default_copilot_peer(active_model: str, target_tier: str, api_mode: str) -> str:
    """Pick a wire-compatible Copilot peer model when explicit overrides are left empty."""
    norm = normalize_copilot_model(active_model)
    low = norm.lower()
    fam = model_family(norm)

    if api_mode == "codex_responses":
        if target_tier == "system1":
            return "gpt-5.4-mini"
        # System 2: keep active high-end Responses model, or upgrade gpt-5.4-mini -> gpt-5.4
        return "gpt-5.4" if low in ("gpt-5.4-mini",) else (norm or "gpt-5.4")

    # chat_completions lane
    if fam == "claude":
        if target_tier == "system1":
            return "claude-haiku-4.5"
        return "claude-sonnet-4.6" if "haiku" in low else norm
    if fam == "gemini":
        if target_tier == "system1":
            return "gemini-3-flash-preview"
        return "gemini-3.1-pro-preview" if "flash" in low else norm
    if fam == "openai":
        if target_tier == "system1":
            return "gpt-5-mini"
        # Stay on chat_completions wire! gpt-5.4 would require codex_responses, so upgrade to claude-sonnet-4.6
        return "claude-sonnet-4.6" if low in ("gpt-5-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini") else norm

    return norm


def can_safely_swap_model(
    *,
    provider: str,
    api_mode: str,
    current_model: str,
    candidate_model: str,
    request: dict[str, Any],
) -> tuple[bool, str]:
    """Verify wire-mode and replay-carrier safety before changing `request['model']`."""
    cur = normalize_copilot_model(current_model) if provider == "copilot" else str(current_model or "").strip()
    cand = normalize_copilot_model(candidate_model) if provider == "copilot" else str(candidate_model or "").strip()
    if not cand or cand == cur:
        return True, "same_model"

    if provider in ("copilot", "github-copilot", "github"):
        cand_wire = infer_copilot_api_mode(cand)
        effective_wire = api_mode or infer_copilot_api_mode(cur)
        if cand_wire != effective_wire:
            return (
                False,
                f"wire_mismatch({cand} requires {cand_wire}, session is {effective_wire})",
            )
        if has_signed_replay_carriers(request) and model_family(cand) != model_family(cur):
            return (
                False,
                f"signed_carrier_guard(cannot cross {model_family(cur)}->{model_family(cand)} mid-session)",
            )
        # Gemini thinking models require thought_signature continuity even across Flash/Pro variants
        if model_family(cur) == "gemini" and has_signed_replay_carriers(request) and cand != cur:
            return (
                False,
                "gemini_thought_signature_guard(keep active Gemini model)",
            )

    return True, "ok"


def apply_effort_to_request(
    request: dict[str, Any],
    *,
    effort: str,
    effective_model: str,
    provider: str,
    api_mode: str,
) -> tuple[dict[str, Any], str | None]:
    """Apply or strip reasoning effort according to `api_mode` and model support."""
    updated = dict(request)
    supported = supported_reasoning_efforts(effective_model, provider=provider)
    clamped = clamp_effort(effort, supported)

    if api_mode == "codex_responses":
        if clamped:
            existing_reasoning = updated.get("reasoning")
            base_reasoning = dict(existing_reasoning) if isinstance(existing_reasoning, dict) else {}
            base_reasoning["effort"] = clamped
            base_reasoning.setdefault("summary", "auto")
            updated["reasoning"] = base_reasoning
            updated.pop("reasoning_effort", None)
            return updated, clamped
        updated.pop("reasoning", None)
        updated.pop("reasoning_effort", None)
        return updated, None

    if api_mode == "chat_completions":
        if clamped:
            updated["reasoning_effort"] = clamped
            return updated, clamped
        # On Copilot, our reasoning-effort table is authoritative: models outside it
        # (e.g. claude-haiku-4.5, gpt-4.1, gpt-4o-mini) reject `reasoning_effort` with HTTP 400.
        if (
            provider.lower() not in ("copilot", "github-copilot", "github")
            and "reasoning_effort" in updated
            and effective_model == request.get("model")
        ):
            fallback_clamped = clamp_effort(effort, EFFORTS_O_SERIES)
            updated["reasoning_effort"] = fallback_clamped
            return updated, fallback_clamped
        updated.pop("reasoning_effort", None)
        return updated, None

    return updated, None
