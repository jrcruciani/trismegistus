# Recommendations for Other Providers

Trismegistus was designed and tuned around this reference stack:

- **Decision backend:** Microsoft Foundry with `Microsoft-Decision-1.1`.
- **Execution provider:** GitHub Copilot.

The architecture intentionally separates triage, modulation, and execution. Active model-tier presets, `xhigh` reasoning-effort tables, and 4-lane specialty defaults in `copilot_lanes.py`, `modulation.py`, and `delegation.py` target **GitHub Copilot**. On non-Copilot execution providers, `v0.2.1+` defaults to **safe passthrough**: it enforces signed replay-carrier guards (`thought_signature`, `reasoning_opaque`, `encrypted_content`), never injects Copilot model IDs or unsupported reasoning fields, and inherits the host's configured `delegate_task` route unless explicit provider-matching models are configured.

This document describes how to use other providers safely today and what should be implemented before treating them as first-class.

## Safe starting point

For any provider other than the Copilot/Foundry reference stack:

1. Run with `mode: shadow` first.
2. Keep the currently active execution model unchanged.
3. Do not enable `tier_and_effort` until the model swap has a provider-specific safety adapter and has been tested end to end.
4. Calibrate `confidence_threshold` and `trap_risk_threshold` independently for the selected decision backend.
5. Treat missing, partial, or malformed decisions as conservative abstentions to System 2.
6. Verify with real requests; unit tests do not prove model availability, API compatibility, entitlement, or cost behavior.

`shadow` is the portable mode: it observes and records triage while leaving the execution request unchanged.

## Supported decision-backend shapes

The current plugin can choose among these decision paths:

| Backend | Use case | Current caveat |
|---|---|---|
| `foundry_systemone` | Microsoft-Decision-1.1 on Foundry | The validated reference backend; supports API-key or bearer header modes. |
| `hermes_aux` | A Hermes auxiliary model | Portable through Hermes, but confidence is model-reported rather than necessarily calibrated. |
| `local_systemone` | A local `/v1/systemone` service | Requires the same typed multi-question response contract. |
| `local_logits` | A local OpenAI-compatible service with one-token logprobs | Provider-specific; `logprobs` and response shapes vary, and normalized top-token probabilities are not automatically calibrated. |

For provider portability, use `hermes_aux` when the host already has a suitable auxiliary provider. Use `local_systemone` for a service that explicitly implements the decision contract. Use `local_logits` only after checking batch behavior, `logprobs`, option-label mapping, HTTPS/auth policy, and deadline semantics.

Confidence from these backends is not interchangeable. A value from Foundry, self-reported confidence from another LLM, and a softmax over local logprobs should be recorded with different calibration semantics.

## Execution providers

### OpenAI and OpenAI-compatible endpoints

Potentially usable in `shadow`. Before enabling effort modulation or model swaps:

- Confirm whether the endpoint uses Chat Completions or Responses.
- Confirm the exact encoding of reasoning: `reasoning_effort`, `reasoning.effort`, or a provider-specific extension.
- Confirm which effort values each model accepts.
- Preserve encrypted reasoning items and opaque carrier fields when the provider requires replay.
- Confirm that changing `model` on an in-flight conversation is valid.

Do not assume OpenAI-compatible means reasoning-compatible. Several services accept the same endpoint shape but reject specific reasoning fields or models.

### Anthropic

Use `shadow` until there is an Anthropic adapter. Direct model execution and delegation require explicit handling for:

- Anthropic message format and tool-use blocks;
- extended-thinking configuration and budget semantics;
- signed or opaque thinking blocks that must survive a tool loop;
- provider-specific model IDs and fallback policies.

Copilot aliases for Claude models must not be assumed to be valid direct Anthropic IDs.

### Gemini

Gemini requires special care around thought signatures and other replay data. The existing Copilot guards do not prove compatibility with the direct Gemini API or another Gemini gateway. Test the entire tool loop, not only a standalone completion.

Use shadow observation first, and avoid changing the model after signed thinking or tool-call carriers have appeared in the request.

### Microsoft Azure OpenAI / Azure AI Foundry

An Azure-hosted execution model is separate from a Foundry decision deployment. Configure them as independent surfaces:

- Decision endpoint, deployment, protocol, audience, and authentication for System 1.
- Execution provider, deployment/model ID, API mode, and credentials for the main model.

Do not reuse Copilot model names as Azure deployment names. Azure credentials and Entra audiences must be resolved independently.

For Entra, prefer a token provider with explicit scope, expiry, refresh, and retry behavior. A bearer token copied into `.env` has a short lifetime; an external refresh job is host-specific and should not be presented as a universal plugin contract.

### Local and self-hosted models

Local endpoints can reduce external data flow but need explicit capability checks:

- Does the endpoint preserve the selected message/tool schema?
- Does it support `logprobs` and `top_logprobs` for all required questions?
- Are probabilities calibrated for the question format and model?
- Can it satisfy the configured triage deadline?
- Does it require authentication even on loopback?

Unknown capabilities should fail closed or fall back to the active model in System 2. Do not silently route to another backend because the local backend is unavailable.

## Recommended delegation design

The current `delegation.py` contains Copilot-specific defaults for `fast_scout`, `deep_code`, `architecture`, and `long_context`. With another provider, configure complete routes instead of selecting only a provider name.

A future provider-neutral route should be able to specify:

```yaml
routes:
  fast_scout:
    provider: <provider>
    model: <model-id>
    api_mode: <chat_completions|responses|provider-native>
  deep_code:
    provider: <provider>
    model: <model-id>
    reasoning_effort: <provider-supported-value>
```

If no route exists for a specialty, delegation should inherit the host's configured route rather than substitute a Copilot model. Credentials, base URLs, API modes, and fallback providers belong to the resolved route and must not cross provider boundaries.

## Roadmap for first-class provider support

Before promoting a provider beyond experimental:

1. **Add an execution adapter** with `validate_swap`, reasoning encoding, supported effort levels, and model-ID normalization.
2. **Default unknown providers to passthrough**: never mutate model or reasoning fields.
3. **Require fail-closed capability checks** for model and provider changes.
4. **Represent authentication separately from protocol**, including static credentials, bearer tokens, and renewable token providers.
5. **Enforce one total decision deadline**, including authentication, retries, and all questions.
6. **Record confidence semantics**: calibrated, self-reported, logprob-derived, or unknown.
7. **Preserve the requested and effective routes** in telemetry, including fallback reasons.
8. **Exercise real wire-level integrations** for a simple turn, a tool loop, a model swap, a failed entitlement, and a timeout.
9. **Document a support matrix** as tested, passthrough-only, or unsupported rather than implying all OpenAI-compatible providers are equivalent.

## Practical recommendation

For a new provider, start with:

```yaml
mode: shadow
backend: hermes_aux
```

Use the provider through Hermes's auxiliary-model slot, keep the active main model unchanged, and collect decisions before enabling any behavior-changing mode. If the provider is added as a full execution adapter later, retain `shadow` as the migration and regression-testing path.
