---
name: trismegistus
description: "Use when escalating to System-2 reasoning, querying System-1 decisions, or routing subagents."
version: "0.3.1"
---

# Trismegistus v0.3.1 — Precision & Speed Cognitive Routing

Trismegistus pairs `Microsoft-Decision-1.1` on Microsoft Foundry (`0` output tokens, warm TLS Keep-Alive, multimodal + 32K context, optional same-model OpenRouter/Vercel failover) with deliberative frontier models across three stages: **Triage → Modulation → Call** (defaulting to non-mutating `shadow` mode until validated).

## How Dynamic Effort, Fast-Path & Step Pacing Work

- **Sub-Millisecond S2 Fast-Path**: Unmistakable deep-engineering payloads (multi-hunk diffs, stacktraces with code, forced escalations, error streaks) route directly to System 2 (`high`/`xhigh`) in `0 ms` without paying network RTT. Never fast-paths down to System 1.
- **Asymmetric Precision Floor & Conformal Calibration**: Lowering effort requires calibrated confidence AND low `trap_risk` (auto-tuned per backend via split-conformal calibration and conservatively hardened by closed-loop false-S1 signals). Multimodal turns on text-only decision backends never downgrade blindly to System 1.
- **5-Rung Effort Ladder (`minimal`..`xhigh`)**: Complex engineering tasks unlock `xhigh` reasoning on `gpt-5.4` (`/responses`) and `claude-sonnet-5` (`/chat/completions`).
- **Synthesis-Safe & Cache-Safe Step Pacing**: Paces only early small read-only scout steps (`read_streak <= 2`, `< 6000` chars, no stacktrace) on cache-resilient families (e.g., GPT-5.4 Responses), automatically restoring full `high`/`xhigh` effort **before** synthesis/editing and keeping Claude effort steady to preserve prompt-cache breakpoints.

## When to Use `trismegistus_escalate`

Call `trismegistus_escalate(reason=...)` immediately when you encounter:

- Subtle race conditions, memory-ordering bugs, or cross-module state coupling.
- Security, authentication, cryptographic, or infrastructure invariants.
- Two consecutive failed hypotheses or unexpected test failures.

Escalating locks the remainder of the turn to System 2 at `xhigh` effort.

## When to Use `trismegistus_decide`

Call `trismegistus_decide` when you need calibrated probabilities (`0..1`) rather than free-form prose:

- **`noul`**: Binary risk/invariant gates (`p(true)`).
- **`choice`**: Closed-set selection with calibrated uncertainty (`top_probability`).
- **`score`**: Ordered rubric scoring (`0 .. K-1`).

## Specialty Subagent Routing (`delegate_task`)

When `delegation.backend: trismegistus` is active, `delegate_task` accepts `cognitive_tier` to match each subagent to the fastest/sharpest Copilot specialist:

- `"auto"` (default): `Microsoft-Decision-1.1` classifies each subagent's `goal` + `context` into one of the four lanes below (upgrading uncertain scouts to `deep_code`).
- `"fast_scout"`: `gemini-3-flash-preview` — parallel search, file discovery, mechanical extraction.
- `"deep_code"`: `gpt-5.4` (`xhigh` effort) — algorithms, root-cause debugging, concurrency, test fixing.
- `"architecture"`: `claude-sonnet-5` (`high` effort) — multi-module design, security/code review, synthesis.
- `"long_context"`: `gemini-3.1-pro-preview` (`high` effort) — large log, trace, or corpus cross-referencing.
