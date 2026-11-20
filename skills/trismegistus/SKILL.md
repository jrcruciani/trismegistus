---
name: trismegistus
description: "Use when escalating to System-2 reasoning, querying System-1 decisions, or routing subagents."
version: "0.2.0"
---

# Trismegistus v0.2 — Precision & Performance Cognitive Routing

Trismegistus pairs `Microsoft-Decision-1.1` on Microsoft Foundry (`0` output tokens, warm TLS Keep-Alive) with deliberative frontier models on GitHub Copilot across three stages: **Triage → Modulation → Call**.

## How Dynamic Effort & Step Pacing Work

- **Asymmetric Precision Floor (`confidence >= 0.82` & `trap_risk < 0.30`)**: Short prompts that hide concurrency, state, security, or cross-file traps are automatically vetoed up to System 2 (`high`/`xhigh` effort).
- **5-Rung Effort Ladder (`minimal`..`xhigh`)**: Complex engineering tasks unlock `xhigh` reasoning on `gpt-5.4` (`/responses`) and `claude-sonnet-5` (`/chat/completions`).
- **Dynamic Intra-Turn Step Pacing**: During multi-step tool loops, Trismegistus paces intermediate read-only scout steps (`read_file`, `search_files`) down to `medium`/`low` effort on the **same model** (100% warm KV cache, saving 5–12s per read), and immediately restores full `high`/`xhigh` effort when you edit files (`patch`, `write_file`), run `terminal`, hit an error, or synthesize the final answer.

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
