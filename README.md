# Trismegistus

**Three-stage System-1 / System-2 cognitive router for [Hermes Agent](https://github.com/NousResearch/hermes-agent)** (`Triage → Modulation → Call`), built to pair **Microsoft-Decision-1.1 on Microsoft Foundry** (or local System-1 decision models) with deliberative **System-2 models on GitHub Copilot**.

---

## Why "Trismegistus"?

Every turn passes through **three coordinated stages** inside Hermes's public plugin surface — without patching core internals or breaking prompt KV caching:

```text
1. TRIAGE (System 1)          2. MODULATION (Wire-Safe)       3. CALL & DELEGATION
──────────────────────────    ────────────────────────────    ──────────────────────────────
pre_llm_call / pre_tool_call  llm_request middleware          llm_execution + post_api_request
POST .../v1/systemone         • Effort modulation:            • Fail-open entitlement retry
  • tier (choice: s1 vs s2)     minimal / low / medium / high   if a swapped model 400/403s
  • effort (score: 0..3)      • Copilot wire-mode guard:      • SQLite WAL telemetry ledger
  • needs_plan (noul: 0..1)     chat_completions vs             (/trismegistus stats | last)
• Calibrated abstention:        codex_responses               • DelegationProvider: routes
  max(p) < threshold -> S2    • Signed carrier guard:           parallel delegate_task children
• Tool error streak -> S2       thought_signature / opaque      across ANY Copilot model
```

---

## Key Capabilities

- **Native Microsoft-Decision-1.1 (`foundry_systemone`)**: Calls `<foundry-endpoint>/providers/microsoft/v1/systemone` in a **single multi-question pass** (`choice`, `score`, `noul`), including explicit `Content-Length` framing and `retry-after-ms` 429 handling.
- **Pluggable Local Decision Backends**:
  - `local_systemone`: Compatible with local `/v1/systemone` servers such as `open-jev`.
  - `local_logits`: Calls any OpenAI-compatible local server (`llama-server`, `vLLM`, `Ollama`) with `max_tokens=1, logprobs=True, top_logprobs=10` and computes softmax probabilities directly from token log-probabilities.
  - `hermes_aux`: Zero-config structured fallback using Hermes's `auxiliary.trismegistus_decider` model slot.
- **GitHub Copilot Wire-Mode & Replay-Carrier Safety**:
  - GitHub Copilot serves some models on `/chat/completions` (`claude-*`, `gemini-*`, `gpt-5-mini`, `gpt-4.1`, `gpt-4o*`) and others on `/responses` (`gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `mai-code-*`, `grok-*`).
  - Trismegistus enforces wire-protocol parity on in-session swaps (`claude-haiku-4.5` ↔ `claude-sonnet-4.6`/`claude-sonnet-5`; `gpt-5.4-mini` ↔ `gpt-5.4`), strips `reasoning_effort` when downgrading onto non-reasoning models (avoiding Copilot HTTP 400s), encodes nested `reasoning: {effort, summary: "auto"}` on `codex_responses`, and keeps the active model when cryptographic replay carriers (`thought_signature` / `reasoning_opaque`) are present in session history.
- **Calibrated Abstention & Asymmetric Hysteresis**:
  - When `Microsoft-Decision-1.1` reports low confidence (`top_probability < confidence_threshold`), Trismegistus abstains from downgrading and stays on System 2.
  - Consecutive tool errors (`post_tool_call`) or calling `trismegistus_escalate` immediately promote the rest of the turn to System 2 (`high` effort), while `sticky_system2_within_turn` prevents mid-turn downgrades that would evict warm KV cache.
- **Unconstrained Subagent Routing (`delegation.backend: trismegistus`)**:
  - Because each `delegate_task` child constructs its own `AIAgent`, Trismegistus can dispatch routine subagents to `claude-haiku-4.5` / `gpt-5-mini` and complex architectural subagents to `gpt-5.4` / `claude-sonnet-5` in the same parallel call.

---

## Installation & Quickstart

```bash
# Install from GitHub into your active Hermes profile
hermes plugins install jrcruciani/trismegistus --enable

# Configure your Microsoft Foundry endpoint & API key (saved to .env, never config.yaml)
hermes plugins config trismegistus
```

Or configure directly via CLI:

```bash
# 1. Set your Foundry resource endpoint and deployment name
hermes config set plugins.entries.trismegistus.settings.backend foundry_systemone
hermes config set plugins.entries.trismegistus.settings.foundry_endpoint "https://<your-resource>.services.ai.azure.com"
hermes config set plugins.entries.trismegistus.settings.foundry_deployment "microsoft-decision-1.1"

# 2. Add TRISMEGISTUS_FOUNDRY_API_KEY in ~/.hermes/.env (or via `hermes plugins config trismegistus`)

# 3. Start in shadow mode (default), inspect verdicts, then switch to effort_only or tier_and_effort
# Inside a Hermes session:
#   /trismegistus status
#   /trismegistus stats
#   /trismegistus mode effort_only
#   /trismegistus mode tier_and_effort
```

Optional — enable System-1 / System-2 routing for `delegate_task` subagents:

```bash
hermes config set delegation.backend trismegistus
```

---

## Operating Modes

| Mode | What it mutates | KV Prompt Cache | Best for |
|---|---|---|---|
| `shadow` *(default)* | Nothing — records System-1 verdicts & latencies only | 100% untouched | Calibrating thresholds on your real workload |
| `effort_only` | Modulates `reasoning_effort` / `reasoning.effort` on the active model | **100% warm** (same model, same system prompt) | Models with elastic thinking (`gpt-5.4`, `claude-sonnet-5`, `gemini-4-carbon`) |
| `tier_and_effort` | Switches between wire-compatible System-1 / System-2 Copilot models + effort | Warm within sticky turns; cold only at tier transitions | Maximizing speed & premium-request savings on Copilot |

---

## Slash Commands & Tools

- `/trismegistus status` — Active mode, backend, Foundry key status, and aggregate System-1 / System-2 split.
- `/trismegistus stats` / `/trismegistus last` — Recent decisions with calibrated confidence, triage latency (ms), and applied model/effort.
- `/trismegistus mode <shadow|effort_only|tier_and_effort>` — Switch operating mode live.
- `/trismegistus escalate` — Force the next turn in the current session onto System 2 (`high` effort).
- Tool `trismegistus_decide` — Query `Microsoft-Decision-1.1` (or the configured local backend) directly with `noul`, `choice`, or `score` questions.
- Tool `trismegistus_escalate` — Allows a System-1 model to self-escalate to System 2 mid-turn when it hits deep ambiguity.

---

## Privacy & Security Disclosure (Hermes Plugin Catalog Rule 13)

- **Network calls**: When `backend` is `foundry_systemone` (and `TRISMEGISTUS_FOUNDRY_API_KEY` + `foundry_endpoint` are configured), Trismegistus sends a force-redacted excerpt of the current user turn (`max_excerpt_chars`, default 2,400 chars) and a short tail digest (up to 800 chars) to `<foundry_endpoint>/providers/microsoft/v1/systemone` on every turn (`shadow` mode included), and sends redacted tool arguments / changed file paths when `tool_gate_enabled` or `verify_gate_enabled` are explicitly enabled. With `local_systemone` or `local_logits`, requests go only to your configured `local_endpoint`. With `hermes_aux`, calls route through Hermes's host-owned `ctx.llm` on `auxiliary.trismegistus_decider`.
- **Secret handling**: Secrets (`TRISMEGISTUS_FOUNDRY_API_KEY`, `TRISMEGISTUS_LOCAL_API_KEY`) are read exclusively via `agent.secret_scope.get_secret`, outbound text is scrubbed with `agent.redact.redact_sensitive_text(..., force=True)`, and HTTP calls use `hermes_cli.urllib_security.open_credentialed_url` so credentials are never forwarded across cross-origin redirects.
- **Local storage**: Writes non-message telemetry metrics (timestamps, tier, effort, confidence, latencies, model IDs — never prompt text) to `<HERMES_HOME>/plugin-data/trismegistus/ledger.db`.
- **No core overrides**: Uses only documented Hermes public plugin APIs (`register_hook`, `register_middleware`, `register_tool`, `register_command`, `register_auxiliary_task`, `register_delegation_provider`, `register_skill`); compatible with `plugins.isolation: host`.

---

## Development & Validation

```bash
pytest -v
hermes plugins validate .
hermes plugins doctor . --ci
```

## License

MIT
