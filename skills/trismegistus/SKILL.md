---
name: trismegistus
description: "Use when escalating to System-2 reasoning, querying System-1 decisions, or routing subagents."
version: "0.1.0"
---

# Trismegistus — System-1 / System-2 Cognitive Routing

Trismegistus pairs a fast calibrated System-1 decision model (`Microsoft-Decision-1.1` on Microsoft Foundry or a local decision head) with deliberative System-2 models on GitHub Copilot across three stages: **Triage → Modulation → Call**.

## When to Use `trismegistus_escalate`

If the current turn was routed to a fast System-1 model or low reasoning effort, call `trismegistus_escalate(reason=...)` immediately when you encounter:

- Ambiguous root causes across multiple modules or race conditions.
- Security-sensitive logic, authentication, or infrastructure invariants.
- Two consecutive failed hypotheses or unexpected test failures.

Escalating upgrades subsequent iterations in the turn to System-2 (`high` effort / System-2 Copilot model) while keeping System-2 sticky for the rest of the turn.

## When to Use `trismegistus_decide`

Call `trismegistus_decide` when you need calibrated probabilities (`0..1`) rather than free-form prose:

- **`noul`**: Binary gates (e.g., checking whether a diff touches public API contracts).
- **`choice`**: Picking among closed alternatives with calibrated uncertainty (`top_probability`).
- **`score`**: Ranking items along an ordered rubric (`0 .. K-1`).

## Subagent Routing (`delegate_task`)

When `delegation.backend: trismegistus` is active, `delegate_task` accepts `cognitive_tier`:

- `"auto"` (default): `Microsoft-Decision-1.1` classifies each subagent's `goal`.
- `"system1"`: Forces the fast Copilot model (e.g., `claude-haiku-4.5` or `gpt-5-mini`) for Parallel searches, log extraction, or mechanical checks.
- `"system2"`: Forces the deep Copilot model (e.g., `gpt-5.4` or `claude-sonnet-5`) for complex synthesis or debugging.
