"""Provider-aware execution adapters with capability discovery and KV-cache safety.

Separates provider protocol logic from `modulation.py` and `delegation.py`:
- `CopilotAdapter`: first-class GitHub Copilot adapter (`codex_responses` vs `chat_completions`,
  `xhigh`/`max` effort unlocking, Claude/GPT/Gemini lane pairs, cache-safe effort pacing).
- `OpenAIAdapter`: direct OpenAI & Azure OpenAI Responses / Chat Completions reasoning modulation.
- `AnthropicAdapter`: direct Anthropic Messages API — never injects Copilot `reasoning_effort`,
  protects prompt-cache breakpoints when fixed `thinking.budget_tokens` is active, and only
  modulates `output_config.effort` when the caller already opted into adaptive thinking.
- `GeminiAdapter`: direct Gemini / Vertex adapter with strict `thought_signature` carrier guards.
- `PassthroughAdapter`: safe zero-mutation default for OpenRouter, vLLM, Ollama, and custom gateways
  while still enforcing universal signed replay-carrier guards (`thought_signature`,
  `reasoning_opaque`, `encrypted_content`).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .compat import hermes_clamp_effort

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
    # Try Hermes public helper as fallback for exotic level names
    bridged = hermes_clamp_effort(target, tuple(candidates))
    if bridged:
        return bridged
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
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and (
                        block.get("signature")
                        or block.get("type") in ("thinking", "redacted_thinking")
                    ):
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


def discover_runtime_effort_support(
    model_id: str,
    *,
    provider: str = "",
    api_mode: str = "",
) -> tuple[str, ...] | None:
    """Capability discovery hook checking public Hermes Gemini/effort helpers when available.

    Returns a tuple of supported effort levels if definitively discovered from the host runtime,
    or `None` to fall back to the adapter's static capability table.
    """
    norm = str(model_id or "").strip().lower()
    if not norm:
        return None
    if "gemini" in norm:
        try:
            from agent.reasoning_effort import bare_gemini_model_id

            bare = bare_gemini_model_id(norm)
            if bare and ("2.5" in bare or "3" in bare or "4" in bare) and "flash-lite" not in bare:
                return EFFORTS_GEMINI_THINKING
        except Exception:
            pass
    del provider, api_mode
    return None


@dataclass(frozen=True)
class ProviderCapabilities:
    """Declared execution properties for a provider adapter."""

    name: str
    preserves_cache_on_effort_step_pacing: bool
    supports_native_model_swap: bool


class ProviderAdapter:
    """Base execution adapter interface for wire-safe, cache-aware request modulation."""

    name: str = "passthrough"

    def matches(self, provider: str, api_mode: str = "") -> bool:
        del provider, api_mode
        return False

    def normalize_model(self, model_id: str | None) -> str:
        return str(model_id or "").strip()

    def infer_api_mode(self, model_id: str | None, request: dict[str, Any] | None = None) -> str:
        del model_id
        if isinstance(request, dict) and "input" in request and "messages" not in request:
            return "codex_responses"
        return "chat_completions"

    def supported_efforts(self, model_id: str | None, api_mode: str = "") -> tuple[str, ...]:
        norm = self.normalize_model(model_id).lower()
        if not norm:
            return ()
        discovered = discover_runtime_effort_support(norm, provider=self.name, api_mode=api_mode)
        if discovered:
            return discovered
        if any(tag in norm for tag in ("reasoner", "thinking", "-r1", "qwq")):
            return EFFORTS_O_SERIES
        return ()

    def preserves_cache_on_effort_change(self, model_id: str | None, api_mode: str = "", request: dict[str, Any] | None = None) -> bool:
        """Whether mid-turn effort pacing is safe for prompt KV-cache hits on this model."""
        del api_mode, request
        fam = model_family(model_id)
        # Anthropic renders thinking/effort config into the prompt prefix; changing effort
        # mid-turn can invalidate message cache breakpoints and hurt TTFT.
        if fam == "claude":
            return False
        return True

    def default_peer(self, active_model: str, target_tier: str, api_mode: str) -> str:
        del target_tier, api_mode
        return active_model

    def validate_swap(
        self,
        *,
        api_mode: str,
        current_model: str,
        candidate_model: str,
        request: dict[str, Any],
    ) -> tuple[bool, str]:
        del api_mode
        cur = self.normalize_model(current_model)
        cand = self.normalize_model(candidate_model)
        if not cand or cand == cur:
            return True, "same_model"
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

    def apply_effort(
        self,
        request: dict[str, Any],
        *,
        effort: str,
        effective_model: str,
        api_mode: str,
    ) -> tuple[dict[str, Any], str | None]:
        """Safe passthrough: only adjust an already-present reasoning field when the model is unchanged."""
        updated = dict(request)
        if api_mode == "codex_responses":
            existing_reasoning = updated.get("reasoning")
            if isinstance(existing_reasoning, dict) and "effort" in existing_reasoning:
                clamped = clamp_effort(effort, EFFORTS_O_SERIES)
                if clamped:
                    base_reasoning = dict(existing_reasoning)
                    base_reasoning["effort"] = clamped
                    updated["reasoning"] = base_reasoning
                    return updated, clamped
            return updated, None
        if (
            api_mode == "chat_completions"
            and "reasoning_effort" in updated
            and effective_model == request.get("model")
        ):
            clamped = clamp_effort(effort, EFFORTS_O_SERIES)
            if clamped:
                updated["reasoning_effort"] = clamped
                return updated, clamped
        return updated, None


class CopilotAdapter(ProviderAdapter):
    """First-class GitHub Copilot execution adapter."""

    name = "copilot"

    def matches(self, provider: str, api_mode: str = "") -> bool:
        del api_mode
        return is_copilot_provider(provider)

    def normalize_model(self, model_id: str | None) -> str:
        return normalize_copilot_model(model_id)

    def infer_api_mode(self, model_id: str | None, request: dict[str, Any] | None = None) -> str:
        del request
        return infer_copilot_api_mode(model_id)

    def supported_efforts(self, model_id: str | None, api_mode: str = "") -> tuple[str, ...]:
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

        discovered = discover_runtime_effort_support(norm, provider="copilot", api_mode=effective_mode)
        if discovered:
            return discovered
        return ()

    def default_peer(self, active_model: str, target_tier: str, api_mode: str) -> str:
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

    def validate_swap(
        self,
        *,
        api_mode: str,
        current_model: str,
        candidate_model: str,
        request: dict[str, Any],
    ) -> tuple[bool, str]:
        cur = normalize_copilot_model(current_model)
        cand = normalize_copilot_model(candidate_model)
        if not cand or cand == cur:
            return True, "same_model"
        cand_wire = infer_copilot_api_mode(cand)
        effective_wire = api_mode or infer_copilot_api_mode(cur)
        if cand_wire != effective_wire:
            return (
                False,
                f"wire_mismatch({cand} requires {cand_wire}, session is {effective_wire})",
            )
        return super().validate_swap(
            api_mode=effective_wire,
            current_model=cur,
            candidate_model=cand,
            request=request,
        )

    def apply_effort(
        self,
        request: dict[str, Any],
        *,
        effort: str,
        effective_model: str,
        api_mode: str,
    ) -> tuple[dict[str, Any], str | None]:
        updated = dict(request)
        supported = self.supported_efforts(effective_model, api_mode=api_mode)
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


class AnthropicAdapter(ProviderAdapter):
    """Direct Anthropic Messages API adapter — cache-safe & never injects Copilot fields."""

    name = "anthropic"

    def matches(self, provider: str, api_mode: str = "") -> bool:
        return str(provider or "").strip().lower() == "anthropic" or api_mode == "anthropic_messages"

    def infer_api_mode(self, model_id: str | None, request: dict[str, Any] | None = None) -> str:
        del model_id, request
        return "anthropic_messages"

    def preserves_cache_on_effort_change(
        self,
        model_id: str | None,
        api_mode: str = "",
        request: dict[str, Any] | None = None,
    ) -> bool:
        del model_id, api_mode, request
        # Anthropic renders thinking/effort config into the prompt prefix — keep stable mid-turn.
        return False

    def apply_effort(
        self,
        request: dict[str, Any],
        *,
        effort: str,
        effective_model: str,
        api_mode: str,
    ) -> tuple[dict[str, Any], str | None]:
        del effective_model, api_mode
        updated = dict(request)
        # Never inject Copilot `reasoning_effort` or break fixed `thinking.budget_tokens` caches.
        # Only modulate `output_config.effort` when caller explicitly uses adaptive thinking!
        thinking = updated.get("thinking")
        output_cfg = updated.get("output_config")
        if (
            isinstance(thinking, dict)
            and thinking.get("type") == "adaptive"
            and isinstance(output_cfg, dict)
            and "effort" in output_cfg
        ):
            clamped = clamp_effort(effort, ("low", "medium", "high", "max"))
            if clamped:
                new_cfg = dict(output_cfg)
                new_cfg["effort"] = clamped
                updated["output_config"] = new_cfg
                return updated, clamped
        return updated, None


class OpenAIAdapter(ProviderAdapter):
    """Direct OpenAI / Codex / Azure OpenAI adapter."""

    name = "openai"

    def matches(self, provider: str, api_mode: str = "") -> bool:
        del api_mode
        return str(provider or "").strip().lower() in ("openai", "openai-codex", "codex", "azure", "azure-openai")

    def supported_efforts(self, model_id: str | None, api_mode: str = "") -> tuple[str, ...]:
        norm = str(model_id or "").strip().lower()
        if norm.startswith(("o1", "o3", "o4")):
            return EFFORTS_O_SERIES
        if norm.startswith(("gpt-5", "gpt-6")):
            return EFFORTS_CODEX_LEGACY if api_mode == "codex_responses" else EFFORTS_CHAT_GPT5
        return super().supported_efforts(model_id, api_mode=api_mode)


class GeminiAdapter(ProviderAdapter):
    """Direct Google Gemini / Vertex adapter with strict thought_signature safety."""

    name = "gemini"

    def matches(self, provider: str, api_mode: str = "") -> bool:
        del api_mode
        return str(provider or "").strip().lower() in ("gemini", "google", "google-gemini", "vertex")

    def supported_efforts(self, model_id: str | None, api_mode: str = "") -> tuple[str, ...]:
        norm = str(model_id or "").strip().lower()
        discovered = discover_runtime_effort_support(norm, provider="gemini", api_mode=api_mode)
        if discovered:
            return discovered
        if "gemini-2.5" in norm or "gemini-3" in norm or "gemini-4" in norm:
            return EFFORTS_GEMINI_THINKING
        return ()


_ADAPTERS: tuple[ProviderAdapter, ...] = (
    CopilotAdapter(),
    AnthropicAdapter(),
    OpenAIAdapter(),
    GeminiAdapter(),
)
_PASSTHROUGH = ProviderAdapter()


def get_provider_adapter(provider: str | None, api_mode: str = "") -> ProviderAdapter:
    """Resolve the execution adapter for `(provider, api_mode)`."""
    prov = str(provider or "").strip().lower()
    mode = str(api_mode or "").strip().lower()
    for adapter in _ADAPTERS:
        if adapter.matches(prov, mode):
            return adapter
    return _PASSTHROUGH
