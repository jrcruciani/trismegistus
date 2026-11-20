"""System-1 decision backends for Trismegistus:

1. `FoundrySystemOneBackend` (`foundry_systemone` / `local_systemone`):
   Speaks the native System One HTTP protocol (`POST .../providers/microsoft/v1/systemone` on
   Microsoft Foundry for `Microsoft-Decision-1.1`, or `.../v1/systemone` for `open-jev` / Jev).
   Sends explicit `Content-Length` (required by Foundry gateway) and handles `retry-after-ms`.
2. `LocalLogitsBackend` (`local_logits`):
   Calls any local OpenAI-compatible inference engine (`llama-server`, `vLLM`, `Ollama`) with
   `max_tokens=1, logprobs=True, top_logprobs=10, temperature=0.0` to extract calibrated
   single-token softmax probabilities without prose generation.
3. `HermesAuxBackend` (`hermes_aux`):
   Uses Hermes's host-owned `ctx.llm.complete_structured` on the plugin's registered
   `trismegistus_decider` auxiliary task slot.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request

from .compat import open_safe_url, parse_retry_after, read_secret, user_agent
from .contracts import DecisionAnswer, DecisionBatchResult, DecisionQuestion


def normalize_foundry_systemone_url(endpoint: str) -> str:
    """Normalize a Microsoft Foundry resource URL to `/providers/microsoft/v1/systemone`."""
    raw = str(endpoint or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.endswith("/v1/systemone"):
        return raw
    if raw.endswith("/providers/microsoft"):
        return f"{raw}/v1/systemone"
    # Strip accidental `/openai` or `/models` suffixes pasted from other Foundry blades
    for suffix in ("/openai/v1", "/openai", "/models"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)].rstrip("/")
            break
    return f"{raw}/providers/microsoft/v1/systemone"


def normalize_local_systemone_url(endpoint: str) -> str:
    raw = str(endpoint or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.endswith("/systemone"):
        return raw
    if raw.endswith("/v1"):
        return f"{raw}/systemone"
    return f"{raw}/v1/systemone"


def normalize_chat_completions_url(endpoint: str) -> str:
    raw = str(endpoint or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.endswith("/chat/completions"):
        return raw
    if raw.endswith("/v1"):
        return f"{raw}/chat/completions"
    return f"{raw}/v1/chat/completions"


def parse_systemone_payload(
    payload: Any,
    questions: Mapping[str, DecisionQuestion],
    *,
    backend_name: str,
    fallback_model: str,
    latency_ms: float,
) -> DecisionBatchResult:
    """Parse a `/v1/systemone` JSON response into calibrated `DecisionAnswer` objects."""
    if not isinstance(payload, dict):
        return DecisionBatchResult(
            backend=backend_name,
            model=fallback_model,
            answers={},
            latency_ms=latency_ms,
            error="SystemOne response was not a JSON object",
        )
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, dict):
        return DecisionBatchResult(
            backend=backend_name,
            model=str(payload.get("model") or fallback_model),
            answers={},
            latency_ms=latency_ms,
            error="SystemOne response missing 'answers' object",
        )

    parsed_answers: dict[str, DecisionAnswer] = {}
    for q_name, q_spec in questions.items():
        ans = raw_answers.get(q_name)
        if not isinstance(ans, dict):
            continue
        q_type = str(ans.get("type") or q_spec.type)
        probs_raw = ans.get("probabilities")
        probs: dict[str, float] = {}
        if isinstance(probs_raw, dict):
            for k, v in probs_raw.items():
                try:
                    probs[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue

        conf_raw = ans.get("confidence")
        conf = float(conf_raw) if isinstance(conf_raw, (int, float)) else None

        if q_type == "noul":
            noul_val = ans.get("noul")
            if isinstance(noul_val, (int, float)):
                p_true = max(0.0, min(1.0, float(noul_val)))
            elif "true" in probs:
                p_true = max(0.0, min(1.0, probs["true"]))
            else:
                continue
            if not probs:
                probs = {"true": p_true, "false": 1.0 - p_true}
            parsed_answers[q_name] = DecisionAnswer(
                type="noul",
                noul=p_true,
                confidence=conf if conf is not None else max(p_true, 1.0 - p_true),
                probabilities=probs,
            )
        elif q_type == "choice":
            choice_val = ans.get("choice")
            if not isinstance(choice_val, str) and probs:
                choice_val = max(probs.items(), key=lambda kv: kv[1])[0]
            if not isinstance(choice_val, str):
                continue
            parsed_answers[q_name] = DecisionAnswer(
                type="choice",
                choice=choice_val,
                confidence=conf,
                probabilities=probs,
            )
        elif q_type == "score":
            score_val = ans.get("score")
            if isinstance(score_val, (int, float)):
                s_float = float(score_val)
            elif probs:
                s_float = sum(int(k) * v for k, v in probs.items() if str(k).isdigit())
            else:
                continue
            parsed_answers[q_name] = DecisionAnswer(
                type="score",
                score=s_float,
                confidence=conf,
                probabilities=probs,
            )

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

    if not parsed_answers:
        return DecisionBatchResult(
            backend=backend_name,
            model=str(payload.get("model") or fallback_model),
            answers={},
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            error="No valid answers parsed from SystemOne response",
        )

    return DecisionBatchResult(
        backend=backend_name,
        model=str(payload.get("model") or fallback_model),
        answers=parsed_answers,
        latency_ms=latency_ms,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


class SystemOneHttpBackend:
    """Client for Microsoft Foundry (`Microsoft-Decision-1.1`) and local `/v1/systemone` servers."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str = "",
        auth_mode: str = "api-key",
        backend_name: str = "foundry_systemone",
        timeout: float = 4.0,
        max_attempts: int = 2,
        opener_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.auth_mode = auth_mode
        self.backend_name = backend_name
        self.timeout = max(0.5, float(timeout))
        self.max_attempts = max(1, int(max_attempts))
        self.opener_factory = opener_factory

    def decide(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
    ) -> DecisionBatchResult:
        started = time.perf_counter()
        if not self.url:
            return DecisionBatchResult(
                backend=self.backend_name,
                model=self.model,
                answers={},
                latency_ms=0.0,
                error="Missing SystemOne endpoint URL",
            )
        if self.backend_name == "foundry_systemone" and not self.api_key:
            return DecisionBatchResult(
                backend=self.backend_name,
                model=self.model,
                answers={},
                latency_ms=0.0,
                error="Missing TRISMEGISTUS_FOUNDRY_API_KEY secret",
            )

        wire_questions = {name: q.to_wire() for name, q in questions.items()}
        body_bytes = json.dumps(
            {"model": self.model, "state": state, "questions": wire_questions},
            ensure_ascii=False,
        ).encode("utf-8")

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(len(body_bytes)),
            "User-Agent": user_agent(),
        }
        if self.api_key:
            if self.auth_mode == "bearer":
                headers["Authorization"] = f"Bearer {self.api_key}"
            else:
                headers["api-key"] = self.api_key

        last_error = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            req = urllib.request.Request(
                self.url,
                data=body_bytes,
                headers=headers,
                method="POST",
            )
            try:
                with open_safe_url(
                    req, timeout=self.timeout, opener_factory=self.opener_factory
                ) as resp:
                    raw = resp.read()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                payload = json.loads(raw.decode("utf-8"))
                return parse_systemone_payload(
                    payload,
                    questions,
                    backend_name=self.backend_name,
                    fallback_model=self.model,
                    latency_ms=elapsed_ms,
                )
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                if exc.code == 429 and attempt < self.max_attempts:
                    wait_s = parse_retry_after(exc.headers) or 0.25
                    if wait_s <= max(0.5, self.timeout * 0.5):
                        time.sleep(wait_s)
                        continue
                last_error = f"HTTP {exc.code}: {err_body or exc.reason}"
                break
            except Exception as exc:
                last_error = str(exc)
                break

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DecisionBatchResult(
            backend=self.backend_name,
            model=self.model,
            answers={},
            latency_ms=elapsed_ms,
            error=last_error,
        )


def _softmax_from_logprobs(top_logprobs: list[dict[str, Any]], label_map: dict[str, str]) -> dict[str, float]:
    """Compute normalized probabilities over single-token labels (`A`, `B`, ... or `Y`, `N`)."""
    best_logprob_per_key: dict[str, float] = {}
    for entry in top_logprobs:
        tok = str(entry.get("token") or "").strip().upper()
        lp = entry.get("logprob")
        if not tok or not isinstance(lp, (int, float)):
            continue
        matched = label_map.get(tok)
        if matched and (matched not in best_logprob_per_key or float(lp) > best_logprob_per_key[matched]):
            best_logprob_per_key[matched] = float(lp)

    if not best_logprob_per_key:
        return {}
    max_lp = max(best_logprob_per_key.values())
    exps = {k: math.exp(v - max_lp) for k, v in best_logprob_per_key.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


class LocalLogitsBackend:
    """Calibrated single-token logprobs client for local `llama-server`, `vLLM`, or `Ollama`."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str = "",
        timeout: float = 4.0,
        opener_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.url = normalize_chat_completions_url(url)
        self.model = model
        self.api_key = api_key
        self.timeout = max(0.5, float(timeout))
        self.opener_factory = opener_factory

    def _ask_one(self, state_json: str, q: DecisionQuestion) -> tuple[DecisionAnswer | None, int, str | None]:
        label_map: dict[str, str] = {}
        prompt_lines = [
            "You are a calibrated System-1 decision head. Reply with ONLY the single letter token of the best answer.",
            f"State:\n{state_json}",
            f"Question: {q.instructions}",
            "Options:",
        ]
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if q.type == "noul":
            label_map = {"Y": "true", "N": "false", "YES": "true", "NO": "false", "T": "true", "F": "false"}
            true_desc = q.criteria.get("true", "Yes / True") if isinstance(q.criteria, dict) else "Yes / True"
            false_desc = q.criteria.get("false", "No / False") if isinstance(q.criteria, dict) else "No / False"
            prompt_lines.append(f"Y: {true_desc}")
            prompt_lines.append(f"N: {false_desc}")
            prompt_lines.append("Answer (Y or N):")
        elif q.type == "choice" and isinstance(q.criteria, dict):
            for idx, (opt_key, opt_desc) in enumerate(q.criteria.items()):
                letter = letters[idx]
                label_map[letter] = opt_key
                prompt_lines.append(f"{letter}: [{opt_key}] {opt_desc}")
            prompt_lines.append(f"Answer ({'/'.join(label_map.keys())}):")
        elif q.type == "score" and isinstance(q.criteria, list):
            for idx, level_desc in enumerate(q.criteria):
                letter = letters[idx]
                label_map[letter] = str(idx)
                prompt_lines.append(f"{letter}: [level {idx}] {level_desc}")
            prompt_lines.append(f"Answer ({'/'.join(label_map.keys())}):")
        else:
            return None, 0, f"Unsupported question shape for {q.type}"

        body_bytes = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "\n".join(prompt_lines)}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 10,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(len(body_bytes)),
            "User-Agent": user_agent(),
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(self.url, data=body_bytes, headers=headers, method="POST")
        try:
            with open_safe_url(req, timeout=self.timeout, opener_factory=self.opener_factory) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return None, 0, str(exc)

        choices = payload.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return None, 0, "Empty choices from local_logits server"
        c0 = choices[0]
        raw_text = str(((c0.get("message") or {}).get("content")) or "").strip().upper()
        lp_content = ((c0.get("logprobs") or {}).get("content")) or []
        top_lps = lp_content[0].get("top_logprobs", []) if lp_content and isinstance(lp_content[0], dict) else []

        probs = _softmax_from_logprobs(top_lps, label_map)
        if not probs and raw_text:
            first_char = raw_text[:1]
            mapped = label_map.get(raw_text) or label_map.get(first_char)
            if mapped:
                # Server did not return logprobs; assign neutral uncalibrated confidence
                probs = {mapped: 0.70}

        in_tok = int(((payload.get("usage") or {}).get("prompt_tokens")) or 0)
        if not probs:
            return None, in_tok, f"Could not map token {raw_text!r} to options"

        if q.type == "noul":
            p_true = probs.get("true", 1.0 - probs.get("false", 0.5))
            full_probs = {"true": p_true, "false": 1.0 - p_true}
            return (
                DecisionAnswer(
                    type="noul",
                    noul=p_true,
                    confidence=max(p_true, 1.0 - p_true),
                    probabilities=full_probs,
                ),
                in_tok,
                None,
            )
        if q.type == "choice":
            winner = max(probs.items(), key=lambda kv: kv[1])[0]
            return (
                DecisionAnswer(
                    type="choice",
                    choice=winner,
                    confidence=probs[winner],
                    probabilities=probs,
                ),
                in_tok,
                None,
            )
        # score
        expected = sum(int(k) * v for k, v in probs.items() if k.isdigit())
        top_p = max(probs.values())
        return (
            DecisionAnswer(
                type="score",
                score=expected,
                confidence=top_p,
                probabilities=probs,
            ),
            in_tok,
            None,
        )

    def decide(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
    ) -> DecisionBatchResult:
        started = time.perf_counter()
        state_json = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        answers: dict[str, DecisionAnswer] = {}
        total_in = 0
        last_err: str | None = None
        for q_name, q_spec in questions.items():
            ans, in_tok, err = self._ask_one(state_json, q_spec)
            total_in += in_tok
            if ans is not None:
                answers[q_name] = ans
            elif err:
                last_err = err
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DecisionBatchResult(
            backend="local_logits",
            model=self.model,
            answers=answers,
            latency_ms=elapsed_ms,
            input_tokens=total_in,
            output_tokens=len(answers),
            error=None if answers else (last_err or "local_logits produced no answers"),
        )


class HermesAuxBackend:
    """Zero-config fallback using `ctx.llm.complete_structured` on `trismegistus_decider`."""

    def __init__(self, ctx: Any, *, timeout: float = 4.0) -> None:
        self.ctx = ctx
        self.timeout = max(1.0, float(timeout))

    def decide(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
    ) -> DecisionBatchResult:
        started = time.perf_counter()
        props: dict[str, Any] = {}
        required: list[str] = []
        instructions_lines = [
            "Evaluate every decision question strictly against the supplied state and return calibrated probabilities (0.0 to 1.0)."
        ]
        for name, q in questions.items():
            required.append(name)
            if q.type == "noul":
                props[name] = {
                    "type": "object",
                    "properties": {"noul": {"type": "number"}},
                    "required": ["noul"],
                }
                instructions_lines.append(f"- {name} (noul 0..1): {q.instructions}")
            elif q.type == "choice" and isinstance(q.criteria, dict):
                opts = list(q.criteria.keys())
                props[name] = {
                    "type": "object",
                    "properties": {
                        "choice": {"type": "string", "enum": opts},
                        "confidence": {"type": "number"},
                    },
                    "required": ["choice", "confidence"],
                }
                desc = "; ".join(f"{k}={v}" for k, v in q.criteria.items())
                instructions_lines.append(f"- {name} (choice from {opts}): {q.instructions} ({desc})")
            elif q.type == "score" and isinstance(q.criteria, list):
                max_idx = len(q.criteria) - 1
                props[name] = {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["score", "confidence"],
                }
                levels = ", ".join(f"{i}={lbl}" for i, lbl in enumerate(q.criteria))
                instructions_lines.append(f"- {name} (score 0..{max_idx}): {q.instructions} ({levels})")

        schema = {"type": "object", "properties": props, "required": required}
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        try:
            res = self.ctx.llm.complete_structured(
                instructions="\n".join(instructions_lines),
                input=[{"type": "text", "text": state_text}],
                json_schema=schema,
                schema_name="trismegistus.decisions",
                purpose="trismegistus.triage",
                temperature=0.0,
                max_tokens=256,
                timeout=self.timeout,
                task="trismegistus_decider",
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return DecisionBatchResult(
                backend="hermes_aux",
                model="trismegistus_decider",
                answers={},
                latency_ms=elapsed_ms,
                error=str(exc),
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        parsed = getattr(res, "parsed", None)
        if not isinstance(parsed, dict):
            return DecisionBatchResult(
                backend="hermes_aux",
                model=str(getattr(res, "model", "trismegistus_decider")),
                answers={},
                latency_ms=elapsed_ms,
                error="Structured output returned None",
            )

        wrapped_payload = {
            "model": getattr(res, "model", "trismegistus_decider"),
            "answers": {
                k: {**v, "type": questions[k].type}
                for k, v in parsed.items()
                if k in questions and isinstance(v, dict)
            },
            "usage": {
                "input_tokens": getattr(getattr(res, "usage", None), "input_tokens", 0) or 0,
                "output_tokens": getattr(getattr(res, "usage", None), "output_tokens", 0) or 0,
            },
        }
        return parse_systemone_payload(
            wrapped_payload,
            questions,
            backend_name="hermes_aux",
            fallback_model="trismegistus_decider",
            latency_ms=elapsed_ms,
        )


def build_decision_backend(ctx: Any, *, opener_factory: Callable[..., Any] | None = None):
    """Instantiate the active System-1 backend from `ctx.get_config(...)`."""
    kind = str(ctx.get_config("backend", default="foundry_systemone") or "foundry_systemone").strip()
    timeout = float(ctx.get_config("timeout_seconds", default=4.0) or 4.0)

    if kind == "foundry_systemone":
        endpoint = str(ctx.get_config("foundry_endpoint", default="") or "").strip()
        deployment = str(ctx.get_config("foundry_deployment", default="microsoft-decision-1.1") or "microsoft-decision-1.1").strip()
        auth_mode = str(ctx.get_config("foundry_auth_header", default="api-key") or "api-key").strip()
        api_key = read_secret("TRISMEGISTUS_FOUNDRY_API_KEY")
        return SystemOneHttpBackend(
            url=normalize_foundry_systemone_url(endpoint),
            model=deployment,
            api_key=api_key,
            auth_mode=auth_mode,
            backend_name="foundry_systemone",
            timeout=timeout,
            opener_factory=opener_factory,
        )

    if kind == "local_systemone":
        endpoint = str(ctx.get_config("local_endpoint", default="http://127.0.0.1:8000/v1") or "").strip()
        model = str(ctx.get_config("local_model", default="open-jev") or "open-jev").strip()
        api_key = read_secret("TRISMEGISTUS_LOCAL_API_KEY")
        return SystemOneHttpBackend(
            url=normalize_local_systemone_url(endpoint),
            model=model,
            api_key=api_key,
            auth_mode="bearer",
            backend_name="local_systemone",
            timeout=timeout,
            opener_factory=opener_factory,
        )

    if kind == "local_logits":
        endpoint = str(ctx.get_config("local_endpoint", default="http://127.0.0.1:8080/v1") or "").strip()
        model = str(ctx.get_config("local_model", default="gemma-4-26b-a4b") or "gemma-4-26b-a4b").strip()
        api_key = read_secret("TRISMEGISTUS_LOCAL_API_KEY")
        return LocalLogitsBackend(
            url=endpoint,
            model=model,
            api_key=api_key,
            timeout=timeout,
            opener_factory=opener_factory,
        )

    return HermesAuxBackend(ctx, timeout=timeout)
