# Trismegistus (`v0.3.1`)

**Precision & speed System-1 / System-2 cognitive router for [Hermes Agent](https://github.com/NousResearch/hermes-agent)** (`Triage → Modulation → Call`).

> **Design objective:** Precision and wall-clock speed first — never sacrificing correctness or breaking prompt KV cache merely to save tokens.
>
> **Reference design:** Trismegistus was designed and tuned for **GitHub Copilot as the execution provider** and **Microsoft Foundry / `Microsoft-Decision-1.1` as the decision backend** (with optional same-model failover to OpenRouter / Vercel `Microsoft-Decision-1.1`). First-class provider adapters (`providers_adapter.py`) cover Copilot, OpenAI/Azure, Anthropic (cache-safe adaptive thinking), and Gemini, while unknown providers default to **safe passthrough** with universal signed replay-carrier guards.
>
> For provider-neutral planning, safe rollout guidance, and multi-provider delegation routes, see [Recommendations for Other Providers](docs/other-providers.md).

The reference configuration pairs **Microsoft-Decision-1.1 on Microsoft Foundry** (`0` output tokens, warm TLS Keep-Alive, multimodal + 32K context) with deliberative **frontier models on GitHub Copilot**, backed by split-conformal threshold calibration, closed-loop false-S1 hardening, and measured KV-cache telemetry.

---

## Architecture: Precision + Speed Across Three Stages

```text
1. TRIAGE (System 1, <1ms fast-path / ~110ms warm)   2. MODULATION (KV-Aware & Safe)      3. CALL, DELEGATION & CLOSED-LOOP
──────────────────────────────────────────────────   ──────────────────────────────────   ───────────────────────────────────
pre_llm_call / pre_tool_call                         llm_request middleware               llm_execution + post_api_request
• Sub-ms S2 Fast-Path (active modes):                • ProviderAdapter dispatch:          • Fail-open entitlement retry
  forced_s2 / err_streak / multi-hunk diffs /          Copilot, OpenAI, Anthropic,          if a swapped model 400/403s
  stacktraces -> 0ms RTT directly to S2                Gemini, Passthrough                • Pre-Verify QA Gate (default ON)
• Head+Tail (60/40, 4k chars) + multimodal signals   • Precision-first Step Pacing:       • Closed-Loop Telemetry (ledger.db):
• Warm TLS 1.3 Keep-Alive + same-model failover        - Paces early small reads (<=2)      cache_read_tokens, cache_hit_rate,
  Foundry -> OpenRouter / Vercel (Decision-1.1)        - Restores xhigh BEFORE synthesis    reasoning_tokens, false_s1_* signals
• Split-conformal thresholds + closed-loop guard       - Keeps Claude effort steady to    • Multi-provider Specialty Delegation:
  per backend (never loosens precision floor)            preserve cache breakpoints         fast_scout / deep_code / arch / long_ctx
```

---

## Key Capabilities & Resilience (`v0.3.0`)

1. **Zero-Generation Multimodal Triage + Same-Model Failover (`backends.py` / `triage.py`)**:
   - `Microsoft-Decision-1.1` evaluates **four orthogonal questions in a single forward pass** (`tier`, `effort`, `trap_risk`, `needs_plan`) with `0` generated tokens (`$0.042 / 1M` input tokens).
   - Preserves **Head + Tail** (60% start + 40% end, default `4000` chars) plus structural signals (`has_images`, `has_stacktrace`, `has_diff`, `has_code_blocks`), forwards bounded image attachments to multimodal Decision-1.1 backends, and vetoes blind S1 downgrades on text-only backends.
   - Supports **same-model failover** (`secondary_systemone_endpoint`: `'openrouter'`, `'vercel'`, or secondary Foundry URL) on `429`/`401`/timeout without losing probability calibration.
   - `LocalLogitsBackend` evaluates all questions **concurrently** (`ThreadPoolExecutor`) under a single global wall-clock deadline.
2. **Sub-Millisecond S2 Fast-Path & Cache-Safe Step Pacing (`triage.py` / `modulation.py`)**:
   - In active modes, skips triage RTT (`0ms`) when `forced_s2`, `err_streak`, multi-hunk diffs, or stacktrace debug payloads already mandate System 2 (never fast-paths downward to System 1).
   - **Synthesis-safe Step Pacing**: paces only early small read-only scout steps (`read_streak <= step_pacing_max_reads`, default `2`, `< 6000` chars, no stacktrace) and restores full `high`/`xhigh` effort **before** synthesis/editing; protects Claude prompt-cache breakpoints (`step_pacing_protect_claude_cache: true`); uses constant System-2 context notes with no variable floats.
   - Robust tool-error detection in `post_tool_call` never misclassifies file contents containing `"error":` from `read_file`/`search_files` as tool failures.
3. **Split-Conformal Calibration & Closed-Loop Precision Hardening (`calibration.py` / `ledger.py`)**:
   - Per-backend split-conformal calibration (`python3 benchmarks/run_heldout.py --calibrate --target-alpha 0.02 --apply-calibration`) computes finite-sample risk-controlled `(confidence_threshold, trap_risk_threshold)` profiles stored in `<HERMES_HOME>/plugin-data/trismegistus/calibration.json`.
   - Closed-loop runtime monitoring tracks implicit false-S1 signals (`false_s1_err_streak`, `false_s1_escalated`, `false_s1_verify_failed`) and conservatively tightens thresholds under distribution shift without ever loosening below the precision floor.
4. **KV-Cache Telemetry, Provider Adapters & End-to-End Evaluation (`providers_adapter.py` / `benchmarks/`)**:
   - Records real `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`, `cache_hit_rate`, and per-turn `step_paced`/`wire_guarded`/`turn_outcome` flags in `ledger.db`.
   - Ships de-biased `benchmarks/heldout_300.jsonl` (300 independent turns: `100 s1`, `100 s2`, `100 ambiguous_trap`, 0 self-referential prompts, 65 multi-turn histories) and `benchmarks/run_e2e_eval.py` comparing `baseline_s2`, `effort_only`, and `tier_and_effort` on `pass@1`, wall-clock latency, and KV-cache hit rate.
5. **Hermes 0.23.x Manifest Compatibility (`plugin.yaml`)**:
   - Uses the documented unversioned/v1 `plugin.yaml` envelope with zero private-import warnings and full `config_schema` / `capabilities` validation.

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

- `/trismegistus status` — Active mode, backend/failover status, turn-deduplicated S1 / S2 / Fast-path / Trap-guarded counts, p50/p95 triage latency, KV-cache hit rate, reasoning tokens, error rate, and false-S1 signals.
- `/trismegistus stats` / `/trismegistus last` — Recent unique turns with step counts, `step_paced`/`wire_guard`/`fast_path` flags, outcome tags, confidence, `trap_risk`, triage latency (ms), and applied model/effort.
- `/trismegistus outcomes` — Closed-loop false-S1 signal breakdown (`false_s1_err_streak`, `false_s1_escalated`, `false_s1_verify_failed`).
- `/trismegistus calibrate` — Inspect the active backend's saved split-conformal calibration profile.
- `/trismegistus mode <shadow|effort_only|tier_and_effort>` — Switch operating mode live.
- `/trismegistus escalate` — Arm the next turn for System 2 (`xhigh` effort).
- Tool `trismegistus_decide` — Query `Microsoft-Decision-1.1` (or local backend) directly with `noul`, `choice`, or `score` questions.
- Tool `trismegistus_escalate` — Self-escalate mid-turn to System 2 (`xhigh` effort) and record a closed-loop signal if the turn started in System 1.

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
python3 benchmarks/run_heldout.py --backend offline_wire_harness --calibrate
python3 benchmarks/run_e2e_eval.py --mode offline_sim
```

## License

MIT
