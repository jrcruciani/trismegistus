# Trismegistus (`v0.2.2`)

**Precision & performance System-1 / System-2 cognitive router for [Hermes Agent](https://github.com/NousResearch/hermes-agent)** (`Triage → Modulation → Call`).

> **Reference design:** Trismegistus was designed and tuned for **GitHub Copilot as the execution provider** and **Microsoft Foundry / `Microsoft-Decision-1.1` as the decision backend**. That pairing is the tested fast path: model tiers, reasoning effort, delegation presets, and wire-level safeguards encode Copilot behavior, while non-Copilot execution providers fall back to **safe passthrough** (never injecting Copilot model IDs or incompatible reasoning fields, and still enforcing signed replay-carrier guards).
>
> For provider-neutral planning, safe rollout guidance, and the current limitations of OpenAI-compatible, Anthropic, Gemini, Azure, and local endpoints, see [Recommendations for Other Providers](docs/other-providers.md).

The reference configuration pairs **Microsoft-Decision-1.1 on Microsoft Foundry** (`0` output tokens, warm TLS Keep-Alive) with deliberative **frontier models on GitHub Copilot**, without sacrificing accuracy or invalidating prompt KV cache.

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
  conf < 0.82 OR trap >= 0.50        thought_signature / opaque          - long_ctx   -> gemini-3.1-pro
  -> Veto S1, lock System 2
```

---

## Key Capabilities & Resilience (`v0.2.2`)

1. **Zero-Generation Triage + Persistent TLS Keep-Alive (`compat.py` / `backends.py`)**:
   - `Microsoft-Decision-1.1` evaluates **four orthogonal questions in a single forward pass** (`tier`, `effort`, `trap_risk`, `needs_plan`) with `0` generated tokens (`$0.042 / 1M` input tokens on Foundry).
   - Thread-safe `KeepAliveHttpPool` reuses warm TLS 1.3 sockets across turns, discards upstream error sockets (`4xx`/`5xx`) so Azure APIM disconnects never poison later requests, and retries transient disconnects cleanly.
2. **401 / 429 / Secret-Rotation Resilience (`compat.py` / `backends.py`)**:
   - **Live `.env` & Secret-Scope Refresh**: Single-profile sessions read rotated credentials via `hermes_cli.config.get_env_value_prefer_dotenv` (while multiplexed gateways strictly honor `agent.secret_scope.get_secret` + `refresh_installed_secret_scope`), auto-detect Entra JWTs vs static Foundry API keys (`foundry_auth_header: auto`), pre-check JWT `exp` locally before making wasted 401 network calls, and retry once when an external rotator updates `.env`.
   - **Rate-Limit Backoff, Cooldown Gate & Triage Cache**: Honors standard `Retry-After` and Foundry `retry-after-ms` / `x-ms-retry-after-ms` headers up to `max_retry_wait_seconds` (`2.0s`) with jitter, enforces per-origin cooldowns on sustained 429s, and caches error-free verdicts across identical repeated prompts (`triage_cache_ttl_seconds: 180s`) so recurring heartbeats spend `0` Foundry tokens.
   - **Optional Supported Fallback (`fallback_backend`)**: Defaults to `none` (clean fail-open to System 2 without extra LLM calls); can be set to `hermes_aux` (`ctx.llm`) or `local_systemone` / `local_logits`.
3. **Asymmetric Precision Floor & Hidden-Trap Gate (`triage.py`)**:
   - Lowering effort is only allowed when **both** `confidence >= 0.82` **and** `trap_risk < 0.50`.
   - Deceptively short prompts that hide concurrency hazards, state coupling, auth/security invariants, or subtle context dependencies trip `trap_risk` and stay on System 2 (`high`/`xhigh`).
4. **Turn-Deduplicated SQLite Telemetry & Held-Out 300-Turn Benchmark (`ledger.py` / `benchmarks/`)**:
   - Multi-step tool loops attribute `decision_latency_ms`, `decision_input_tokens`, and triage errors strictly once per turn (`turn_id` + `step_index`), and `audit_ledger_db` deduplicates legacy pre-`v0.2.2` rows automatically.
   - Ships `benchmarks/heldout_300.jsonl` (100 `s1`, 100 `s2`, 100 `ambiguous_trap` independent turns) and `benchmarks/run_heldout.py` enforcing `mode: shadow` with Wilson 95% confidence intervals, `false_fast_route_rate`, `false_escalation_rate`, abstention/trap-guard rates, `p50`/`p95` latency, error breakdown, and Foundry spend.
5. **Hermes 0.23.x Manifest Compatibility (`plugin.yaml`)**:
   - Uses the documented unversioned/v1 `plugin.yaml` envelope so `provides_middleware` passes `hermes plugins validate` capability checks while avoiding `plugins_manifest.py` unknown-field warnings on `manifest_version >= 2`, keeping all `config_schema` and `capabilities` declarations active.

---

## Installation & Safe Shadow Rollout

```bash
# Install from GitHub into your active Hermes profile
hermes plugins install jrcruciani/trismegistus --enable

# Configure your Microsoft Foundry endpoint & API key (saved to .env, never config.yaml)
hermes plugins config trismegistus
```

Start in **`shadow` mode** (the default) and validate calibration with the 300-turn held-out benchmark before enabling active modulation:

```bash
hermes config set plugins.entries.trismegistus.settings.backend foundry_systemone
hermes config set plugins.entries.trismegistus.settings.foundry_endpoint "https://<your-resource>.services.ai.azure.com"
hermes config set plugins.entries.trismegistus.settings.foundry_deployment "microsoft-decision-1.1"
hermes config set plugins.entries.trismegistus.settings.mode shadow

# Run the 300-turn held-out shadow benchmark (~$0.007 total on Foundry)
python3 benchmarks/run_heldout.py --backend foundry_systemone --pacing-ms 220 --output benchmarks/results/foundry_heldout_300.json
```

---

## Operating Modes

| Mode | What it mutates | KV Prompt Cache | Best for |
|---|---|---|---|
| `shadow` *(default)* | Nothing — records System-1 verdicts, trap scores & latencies | 100% untouched | Auditing calibration & error rates before rollout |
| `effort_only` | Modulates `minimal..xhigh` effort + intra-turn step pacing on the active model | **100% warm** across every step | **Highest precision + fastest multi-step loops** once Foundry error rate < 1% |
| `tier_and_effort` | Swaps wire-compatible Copilot models (`claude-haiku-4.5` ↔ `claude-sonnet-5`, `gpt-5.4-mini` ↔ `gpt-5.4`) + effort | Warm within sticky System-2 turns | Extra Copilot premium-request savings on trivial turns |

---

## Slash Commands & Tools

- `/trismegistus status` — Active mode, backend/fallback status, Foundry key check, turn-deduplicated S1 / S2 / Trap-guarded counts, p50/p95 latency, error rate, and estimated Foundry spend.
- `/trismegistus stats` / `/trismegistus last` — Recent unique turns with step counts, confidence, `trap_risk`, triage latency (ms), and applied model/effort.
- `/trismegistus mode <shadow|effort_only|tier_and_effort>` — Switch operating mode live.
- `/trismegistus escalate` — Arm the next turn for System 2 (`xhigh` effort).
- Tool `trismegistus_decide` — Query `Microsoft-Decision-1.1` (or local backend) directly with `noul`, `choice`, or `score` questions.
- Tool `trismegistus_escalate` — Self-escalate mid-turn to System 2 (`xhigh` effort).

---

## Privacy & Security Disclosure (Hermes Plugin Catalog Rule 13)

- **Network calls**: When `backend` is `foundry_systemone` (and `TRISMEGISTUS_FOUNDRY_API_KEY` + `foundry_endpoint` are configured), Trismegistus sends a force-redacted excerpt of the current user turn (`max_excerpt_chars`, default 2,400 chars) and a short tail digest (up to 800 chars) over a verified TLS 1.3 connection (`KeepAliveHttpPool`, no cross-origin redirect forwarding) to `<foundry_endpoint>/providers/microsoft/v1/systemone` once per user turn (`shadow` mode included), plus redacted tool arguments / changed file paths when `tool_gate_enabled` or `verify_gate_enabled` run in active modes. With `local_systemone` or `local_logits`, requests go only to your configured `local_endpoint`. With `hermes_aux`, calls route through Hermes's host-owned `ctx.llm` on `auxiliary.trismegistus_decider`.
- **Secret handling**: Secrets (`TRISMEGISTUS_FOUNDRY_API_KEY`, `TRISMEGISTUS_LOCAL_API_KEY`) are read exclusively via Hermes's documented secret APIs (`agent.secret_scope` / `hermes_cli.config.get_env_value_prefer_dotenv`), never from raw credential files, and all outbound text is scrubbed with `agent.redact.redact_sensitive_text(..., force=True)`.
- **Local storage**: Writes non-message telemetry metrics (timestamps, `turn_id`, `step_index`, tier, effort, confidence, `trap_risk`, latencies, model IDs — never prompt text) to `<HERMES_HOME>/plugin-data/trismegistus/ledger.db`.
- **No core overrides**: Uses only documented Hermes public plugin APIs; compatible with `plugins.isolation: host`.

---

## Development & Validation

```bash
python3 tests/run_tests.py
hermes plugins validate .
hermes plugins doctor . --ci
python3 benchmarks/run_heldout.py --backend offline_wire_harness
```

## License

MIT
