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

import re
from typing import Any

_DUAL_RESPONSES_PREFIXES = ("mai-code-", "grok-")

EFFORT_ORDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

EFFORTS_CHAT_GPT5: tuple[str, ...] = ("minimal", "low", "medium", "high")
EFFORTS_CODEX_LEGACY: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh")
EFFORTS_CODEX_GPT56_PLUS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh", "max")
EFFORTS_GPT6_NO_DISABLE: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
EFFORTS_O_SERIES: tuple[str, ...] = ("low", "medium", "high")
EFFORTS_GEMINI_THINKING: tuple[str, ...] = ("low", "medium", "high")
EFFORTS_CLAUDE5: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
EFFORTS_GROK46_PLUS: tuple[str, ...] = ("low", "medium", "high", "xhigh")

COPILOT_ID_ALIASES: dict[str, str] = {
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-sonnet-4-0": "claude-sonnet-4",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "anthropic/claude-opus-4.6": "claude-opus-4.6",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
    "anthropic/claude-opus-5": "claude-opus-5",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5": "claude-haiku-4.5",
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5.4": "gpt-5.4",
    "openai/gpt-5.4-mini": "gpt-5.4-mini",
}


def is_copilot_provider(provider: str | None) -> bool:
    return str(provider or "").strip().lower() in ("copilot", "github-copilot", "github")


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
    bare = norm.split("/", 1)[-1] if "/" in norm else norm
    if "claude" in bare:
        return "claude"
    if "gemini" in bare:
        return "gemini"
    if bare.startswith(("gpt-", "o1", "o3", "o4", "mai-code")):
        return "openai"
    if bare.startswith("grok"):
        return "grok"
    return "other"


def supported_reasoning_efforts(
    model_id: str | None,
    *,
    provider: str = "",
    api_mode: str = "",
) -> tuple[str, ...]:
    """Return wire-valid `reasoning_effort` levels for `(model_id, provider, api_mode)`."""
    norm = normalize_copilot_model(model_id).lower()
    if not norm:
        return ()

    effective_mode = api_mode or infer_copilot_api_mode(norm)

    if norm.startswith(("o1", "o3", "o4", "openai/o1", "openai/o3", "openai/o4")):
        return EFFORTS_O_SERIES
    if norm.startswith(("gpt-6-astra", "gpt-6.1-sol", "gpt-6.2-astra", "gpt-6.2-sol")):
        return EFFORTS_GPT6_NO_DISABLE
    if norm.startswith("gpt-6") or norm.startswith("gpt-5.6"):
        return EFFORTS_CODEX_GPT56_PLUS if effective_mode == "codex_responses" else EFFORTS_CHAT_GPT5
    if norm.startswith(("gpt-5", "mai-code-")):
        if effective_mode == "codex_responses":
            return EFFORTS_CODEX_LEGACY
        return EFFORTS_CHAT_GPT5
    if norm.startswith("grok-"):
        return EFFORTS_GROK46_PLUS
    if norm.startswith(("gemini-4", "gemini-3.8", "gemini-3.1-pro", "gemini-3-pro", "gemini-2.5-pro")):
        return EFFORTS_GEMINI_THINKING
    if norm.startswith(("claude-opus-5", "claude-sonnet-5", "claude-haiku-5")):
        return EFFORTS_CLAUDE5

    if provider.lower() not in ("copilot", "github-copilot", "github"):
        if any(tag in norm for tag in ("reasoner", "thinking", "-r1", "qwq")):
            return EFFORTS_O_SERIES
    return ()


def clamp_effort(desired: str, supported: tuple[str, ...]) -> str | None:
    """Clamp `desired` onto `supported`, never collapsing to `none` unless explicitly requested."""
    if not supported:
        return None
    target = str(desired or "medium").strip().lower()
    if target in supported:
        return target
    candidates = [lvl for lvl in supported if lvl != "none"] or list(supported)
    if target in EFFORT_ORDER:
        idx = EFFORT_ORDER.index(target)
        below = [lvl for lvl in candidates if EFFORT_ORDER.index(lvl) < idx]
        if below:
            return max(below, key=EFFORT_ORDER.index)
        return min(candidates, key=EFFORT_ORDER.index)
    return "medium" if "medium" in candidates else candidates[0]


def has_signed_replay_carriers(request: dict[str, Any]) -> bool:
    """Detect provider/family-bound cryptographic replay carriers in the outgoing request."""
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
        return "gpt-5.4" if low in ("gpt-5.4-mini",) else (norm or "gpt-5.4")

    if fam == "claude":
        if target_tier == "system1":
            return "claude-haiku-4.5"
        return "claude-sonnet-5" if "haiku" in low else norm
    if fam == "gemini":
        if target_tier == "system1":
            return "gemini-3-flash-preview"
        return "gemini-3.1-pro-preview" if "flash" in low else norm
    if fam == "openai":
        if target_tier == "system1":
            return "gpt-5-mini"
        return "claude-sonnet-5" if low in ("gpt-5-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini") else norm

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
    is_copilot = provider.lower() in ("copilot", "github-copilot", "github")
    cur = normalize_copilot_model(current_model) if is_copilot else str(current_model or "").strip()
    cand = normalize_copilot_model(candidate_model) if is_copilot else str(candidate_model or "").strip()
    if not cand or cand == cur:
        return True, "same_model"

    if is_copilot:
        cand_wire = infer_copilot_api_mode(cand)
        effective_wire = api_mode or infer_copilot_api_mode(cur)
        if cand_wire != effective_wire:
            return (
                False,
                f"wire_mismatch({cand} requires {cand_wire}, session is {effective_wire})",
            )

    if has_signed_replay_carriers(request):
        if model_family(cand) != model_family(cur):
            return (
                False,
                f"signed_carrier_guard(cannot cross {model_family(cur)}->{model_family(cand)} mid-session)",
            )
        if model_family(cur) == "gemini" and cand != cur:
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
    """Apply or strip reasoning effort on Copilot; use safe passthrough on other providers."""
    updated = dict(request)

    if not is_copilot_provider(provider):
        prov = str(provider or "").strip().lower()
        if prov == "anthropic" or api_mode == "anthropic_messages":
            return updated, None
        if api_mode == "codex_responses":
            existing_reasoning = updated.get("reasoning")
            if isinstance(existing_reasoning, dict) and "effort" in existing_reasoning:
                clamped_non_cp = clamp_effort(effort, EFFORTS_O_SERIES)
                if clamped_non_cp:
                    base_reasoning = dict(existing_reasoning)
                    base_reasoning["effort"] = clamped_non_cp
                    updated["reasoning"] = base_reasoning
                    return updated, clamped_non_cp
            return updated, None
        if (
            api_mode == "chat_completions"
            and "reasoning_effort" in updated
            and effective_model == request.get("model")
        ):
            clamped_non_cp = clamp_effort(effort, EFFORTS_O_SERIES)
            if clamped_non_cp:
                updated["reasoning_effort"] = clamped_non_cp
                return updated, clamped_non_cp
        return updated, None

    supported = supported_reasoning_efforts(effective_model, provider=provider, api_mode=api_mode)
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
        updated.pop("reasoning_effort", None)
        return updated, None

    return updated, None
