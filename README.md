# Trismegistus (`v0.2.0`)

**Precision & performance System-1 / System-2 cognitive router for [Hermes Agent](https://github.com/NousResearch/hermes-agent)** (`Triage → Modulation → Call`), built to pair **Microsoft-Decision-1.1 on Microsoft Foundry** (`0` output tokens, warm TLS Keep-Alive) with deliberative **frontier models on GitHub Copilot** — without sacrificing accuracy or invalidating prompt KV cache.

---

## Architecture: Precision + Speed Across Three Stages

```text
1. TRIAGE (System 1, ~60-120ms)    2. MODULATION (KV-Warm & Safe)      3. CALL & SPECIALTY DELEGATION
───────────────────────────────    ────────────────────────────────    ──────────────────────────────────
pre_llm_call / pre_tool_call       llm_request middleware              llm_execution + post_api_request
Warm TLS 1.3 Keep-Alive POST       • 5-rung effort ladder:             • Fail-open entitlement retry
.../providers/microsoft/v1/          minimal/low/medium/high/xhigh       if a swapped model 400/403s
    systemone (0 output tokens)    • Dynamic Intra-Turn Step Pacing:   • Pre-Verify QA Gate (default ON):
  • tier (choice: s1 vs s2)          - Plan/Edit/Synth -> high/xhigh     blocks unverified code claims
  • effort (score: 0..4 -> xhigh)    - Read-only scouts -> low/medium  • Specialty DelegationProvider:
  • trap_risk (noul: 0..1)           - Same model = 100% warm KV!        - fast_scout -> gemini-3-flash
  • needs_plan (noul: 0..1)        • Copilot wire & carrier guards:      - deep_code  -> gpt-5.4 (xhigh)
• Asymmetric precision floor:        chat_completions vs responses,      - architect  -> claude-sonnet-5
  conf < 0.82 OR trap >= 0.30        thought_signature / opaque          - long_ctx   -> gemini-3.1-pro
  -> Veto S1, lock System 2
```

---

## What Makes v0.2 Both Faster and More Accurate

1. **Zero-Generation Triage + Persistent TLS Keep-Alive (`compat.py` / `backends.py`)**:
   - `Microsoft-Decision-1.1` evaluates **four orthogonal questions in a single forward pass** (`tier`, `effort`, `trap_risk`, `needs_plan`) with `0` generated tokens (`$0.042 / 1M` input tokens on Foundry).
   - Thread-safe `KeepAliveHttpPool` reuses warm TLS 1.3 sockets across turns (saving ~80–150 ms per turn), automatically recovers from idle upstream disconnects, and caps triage timeout at `1.2s` (failing open immediately to System 2).
2. **Asymmetric Precision Floor & Hidden-Trap Gate (`triage.py`)**:
   - Lowering effort is only allowed when **both** `confidence >= 0.82` **and** `trap_risk < 0.30`.
   - Deceptively short prompts that hide concurrency hazards, state coupling, auth/security invariants, or subtle context dependencies trip `trap_risk` and stay on System 2 (`high`/`xhigh`).
3. **5-Rung Effort Ladder Unlocking `xhigh` (`copilot_lanes.py`)**:
   - Unlocks `xhigh` reasoning on `gpt-5.4` (`/responses`), `claude-sonnet-5` (`/chat/completions`), and `gpt-6*`, while clamping `minimal -> low` on `/responses` so Copilot never returns HTTP 400.
4. **Dynamic Intra-Turn Step Pacing (`modulation.py`)**:
   - In multi-step agent loops (`effort_only` mode), Step 1 plans at full `high`/`xhigh` effort; intermediate read-only scout iterations (`read_file`, `search_files`) pace down to `medium`/`low` on the **exact same model** (preserving **100% warm prompt KV cache** while cutting 5–12s of redundant hidden thinking per file read); and any `patch`, `write_file`, `terminal` call, or tool error **immediately restores full `high`/`xhigh` precision**.
5. **4-Way Cognitive Specialty Routing for `delegate_task` (`delegation.py`)**:
   - Because each `delegate_task` child runs in its own isolated `AIAgent`, Trismegistus matches subagents to the best Copilot model for each cognitive profile:
     - `fast_scout` → `gemini-3-flash-preview` (`low` effort)
     - `deep_code` → `gpt-5.4` (`xhigh` effort)
     - `architecture` → `claude-sonnet-5` (`high` effort)
     - `long_context` → `gemini-3.1-pro-preview` (`high` effort)
6. **Default-On Code Verification Gate (`pre_verify`)**:
   - When code files were modified in a turn (in `effort_only` or `tier_and_effort` mode), `Microsoft-Decision-1.1` checks whether the final response reports real verification evidence (`pytest`, CLI run, syntax check) and nudges the agent once before stopping if missing.

---

## Installation & Quickstart

```bash
# Install from GitHub into your active Hermes profile
hermes plugins install jrcruciani/trismegistus --enable

# Configure your Microsoft Foundry endpoint & API key (saved to .env, never config.yaml)
hermes plugins config trismegistus
```

Recommended configuration for **maximum precision + speed** (`effort_only` + specialty subagent routing):

```bash
hermes config set plugins.entries.trismegistus.settings.backend foundry_systemone
hermes config set plugins.entries.trismegistus.settings.foundry_endpoint "https://<your-resource>.services.ai.azure.com"
hermes config set plugins.entries.trismegistus.settings.foundry_deployment "microsoft-decision-1.1"
hermes config set plugins.entries.trismegistus.settings.mode effort_only
hermes config set delegation.backend trismegistus
```

---

## Operating Modes

| Mode | What it mutates | KV Prompt Cache | Best for |
|---|---|---|---|
| `shadow` *(default)* | Nothing — records System-1 verdicts, trap scores & latencies | 100% untouched | Auditing calibration on your workload |
| `effort_only` *(recommended)* | Modulates `minimal..xhigh` effort + intra-turn step pacing on the active model | **100% warm** across every step | **Highest precision + fastest multi-step loops** (`gpt-5.4`, `claude-sonnet-5`, `gemini-4-carbon`) |
| `tier_and_effort` | Swaps wire-compatible Copilot models (`claude-haiku-4.5` ↔ `claude-sonnet-5`, `gpt-5.4-mini` ↔ `gpt-5.4`) + effort | Warm within sticky System-2 turns | Extra Copilot premium-request savings on trivial turns |

---

## Slash Commands & Tools

- `/trismegistus status` — Active mode, step-pacing & keep-alive status, Foundry key check, and System-1 / System-2 / Trap-guarded / Step-paced counts + estimated Foundry spend.
- `/trismegistus stats` / `/trismegistus last` — Recent decisions with confidence, `trap_risk`, triage latency (ms), and applied model/effort.
- `/trismegistus mode <shadow|effort_only|tier_and_effort>` — Switch operating mode live.
- `/trismegistus escalate` — Arm the next turn for System 2 (`xhigh` effort).
- Tool `trismegistus_decide` — Query `Microsoft-Decision-1.1` (or local backend) directly with `noul`, `choice`, or `score` questions.
- Tool `trismegistus_escalate` — Self-escalate mid-turn to System 2 (`xhigh` effort).

---

## Privacy & Security Disclosure (Hermes Plugin Catalog Rule 13)

- **Network calls**: When `backend` is `foundry_systemone` (and `TRISMEGISTUS_FOUNDRY_API_KEY` + `foundry_endpoint` are configured), Trismegistus sends a force-redacted excerpt of the current user turn (`max_excerpt_chars`, default 2,400 chars) and a short tail digest (up to 800 chars) over a verified TLS 1.3 connection (`KeepAliveHttpPool`, no cross-origin redirect forwarding) to `<foundry_endpoint>/providers/microsoft/v1/systemone` on every turn (`shadow` mode included), plus redacted tool arguments / changed file paths when `tool_gate_enabled` or `verify_gate_enabled` run. With `local_systemone` or `local_logits`, requests go only to your configured `local_endpoint`. With `hermes_aux`, calls route through Hermes's host-owned `ctx.llm` on `auxiliary.trismegistus_decider`.
- **Secret handling**: Secrets (`TRISMEGISTUS_FOUNDRY_API_KEY`, `TRISMEGISTUS_LOCAL_API_KEY`) are read exclusively via `agent.secret_scope.get_secret`, and all outbound text is scrubbed with `agent.redact.redact_sensitive_text(..., force=True)`.
- **Local storage**: Writes non-message telemetry metrics (timestamps, tier, effort, confidence, `trap_risk`, latencies, model IDs — never prompt text) to `<HERMES_HOME>/plugin-data/trismegistus/ledger.db`.
- **No core overrides**: Uses only documented Hermes public plugin APIs; compatible with `plugins.isolation: host`.

---

## Development & Validation

```bash
pytest -v
hermes plugins validate .
hermes plugins doctor . --ci
```

## License

MIT
