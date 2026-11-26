"""System-1 decision backends for Trismegistus:

1. `SystemOneHttpBackend` (`foundry_systemone` / `local_systemone`):
   Speaks the native System One HTTP protocol (`POST .../providers/microsoft/v1/systemone` on
   Microsoft Foundry for `Microsoft-Decision-1.1`, or `.../v1/systemone` for `open-jev`).
   Uses persistent TLS Keep-Alive connections by default, sends explicit `Content-Length`,
   handles Foundry's `retry-after-ms` / `Retry-After` with pacing cooldowns, refreshes rotated
   profile secrets on 401 / expired Entra JWTs, and caches identical repeated triage payloads.
2. `LocalLogitsBackend` (`local_logits`):
   Calls any local OpenAI-compatible inference engine (`llama-server`, `vLLM`, `Ollama`) with
   `max_tokens=1, logprobs=True, top_logprobs=10, temperature=0.0` for calibrated softmax probabilities.
3. `HermesAuxBackend` (`hermes_aux`):
   Uses Hermes's host-owned `ctx.llm.complete_structured` on `auxiliary.trismegistus_decider`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import math
import random
import threading
import time
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request

from .compat import (
    is_jwt_token,
    jwt_seconds_until_expiry,
    open_safe_url,
    parse_retry_after,
    read_secret,
    user_agent,
)
from .contracts import DecisionAnswer, DecisionBatchResult, DecisionQuestion

_CACHE_LOCK = threading.Lock()
_DECISION_CACHE: OrderedDict[str, tuple[float, DecisionBatchResult]] = OrderedDict()
_DECISION_CACHE_MAX = 256

_RATE_LOCK = threading.Lock()
_RATE_LIMIT_UNTIL: dict[str, float] = {}
_AUTH_COOLDOWN_UNTIL: dict[tuple[str, str], tuple[float, str]] = {}


def clear_backend_caches() -> None:
    """Clear in-memory triage decision cache and endpoint cooldown state."""
    with _CACHE_LOCK:
        _DECISION_CACHE.clear()
    with _RATE_LOCK:
        _RATE_LIMIT_UNTIL.clear()
        _AUTH_COOLDOWN_UNTIL.clear()


def _cache_key(backend_name: str, model: str, body_bytes: bytes) -> str:
    digest = hashlib.sha256(body_bytes).hexdigest()
    return f"{backend_name}:{model}:{digest}"


def _cache_get(key: str, ttl_s: float) -> DecisionBatchResult | None:
    if ttl_s <= 0:
        return None
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _DECISION_CACHE.get(key)
        if entry is None:
            return None
        stored_at, batch = entry
        if now - stored_at > ttl_s:
            _DECISION_CACHE.pop(key, None)
            return None
        _DECISION_CACHE.move_to_end(key)
        return replace(batch, latency_ms=0.0, input_tokens=0, output_tokens=0, cached=True)


def _cache_put(key: str, batch: DecisionBatchResult, ttl_s: float) -> None:
    if ttl_s <= 0 or not batch.ok:
        return
    with _CACHE_LOCK:
        _DECISION_CACHE[key] = (time.monotonic(), batch)
        _DECISION_CACHE.move_to_end(key)
        while len(_DECISION_CACHE) > _DECISION_CACHE_MAX:
            _DECISION_CACHE.popitem(last=False)


def _secret_digest(secret: str) -> str:
    if not secret:
        return "empty"
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]


def normalize_foundry_systemone_url(endpoint: str) -> str:
    """Normalize a Microsoft Foundry resource URL to `/providers/microsoft/v1/systemone`."""
    raw = str(endpoint or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.endswith("/v1/systemone"):
        return raw
    if raw.endswith("/providers/microsoft"):
        return f"{raw}/v1/systemone"
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


def _apply_auth_header(headers: dict[str, str], api_key: str, auth_mode: str) -> None:
    headers.pop("Authorization", None)
    headers.pop("api-key", None)
    if not api_key:
        return
    mode = str(auth_mode or "auto").strip().lower()
    if mode == "bearer" or (mode in ("auto", "api-key") and is_jwt_token(api_key)):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["api-key"] = api_key


class SystemOneHttpBackend:
    """Client for Microsoft Foundry (`Microsoft-Decision-1.1`) and local `/v1/systemone` servers."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str = "",
        auth_mode: str = "auto",
        backend_name: str = "foundry_systemone",
        timeout: float = 1.2,
        max_attempts: int = 2,
        max_retry_wait: float = 2.0,
        cache_ttl: float = 0.0,
        keep_alive: bool = True,
        opener_factory: Callable[..., Any] | None = None,
        secret_refresher: Callable[[], str] | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.auth_mode = auth_mode
        self.backend_name = backend_name
        self.timeout = max(0.3, float(timeout))
        self.max_attempts = max(1, int(max_attempts))
        self.max_retry_wait = max(0.05, float(max_retry_wait))
        self.cache_ttl = max(0.0, float(cache_ttl))
        self.keep_alive = bool(keep_alive)
        self.opener_factory = opener_factory
        self.secret_refresher = secret_refresher

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

        current_key = self.api_key
        refreshed_once = False

        if self.backend_name == "foundry_systemone" and not current_key and self.secret_refresher is not None:
            current_key = str(self.secret_refresher() or "").strip()
            refreshed_once = True
            self.api_key = current_key

        if self.backend_name == "foundry_systemone" and not current_key:
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
            sort_keys=True,
        ).encode("utf-8")

        ckey = _cache_key(self.backend_name, self.model, body_bytes)
        cached = _cache_get(ckey, self.cache_ttl)
        if cached is not None:
            return cached

        # Pre-flight local JWT expiry check (prevents guaranteed-401 network round-trips)
        if current_key and is_jwt_token(current_key):
            rem = jwt_seconds_until_expiry(current_key)
            if rem is not None and rem <= 0:
                if not refreshed_once and self.secret_refresher is not None:
                    refreshed_once = True
                    candidate = str(self.secret_refresher() or "").strip()
                    if candidate and candidate != current_key:
                        current_key = candidate
                        self.api_key = current_key
                        rem = jwt_seconds_until_expiry(current_key)
                if rem is not None and rem <= 0:
                    return DecisionBatchResult(
                        backend=self.backend_name,
                        model=self.model,
                        answers={},
                        latency_ms=0.0,
                        error="HTTP 401: Foundry bearer token expired; awaiting external token refresh",
                    )

        # Check short per-(origin, credential-digest) 401 cooldown; a rotated secret changes digest and bypasses immediately
        auth_state_key = (self.url, _secret_digest(current_key))
        with _RATE_LOCK:
            auth_cd = _AUTH_COOLDOWN_UNTIL.get(auth_state_key)
            if auth_cd is not None:
                until_mono, cached_err = auth_cd
                if time.monotonic() < until_mono:
                    if not refreshed_once and self.secret_refresher is not None:
                        refreshed_once = True
                        candidate = str(self.secret_refresher() or "").strip()
                        if candidate and candidate != current_key:
                            current_key = candidate
                            self.api_key = current_key
                            auth_state_key = (self.url, _secret_digest(current_key))
                        else:
                            return DecisionBatchResult(
                                backend=self.backend_name,
                                model=self.model,
                                answers={},
                                latency_ms=0.0,
                                error=cached_err,
                            )
                    else:
                        return DecisionBatchResult(
                            backend=self.backend_name,
                            model=self.model,
                            answers={},
                            latency_ms=0.0,
                            error=cached_err,
                        )
                else:
                    _AUTH_COOLDOWN_UNTIL.pop(auth_state_key, None)

            rl_until = _RATE_LIMIT_UNTIL.get(self.url, 0.0)
            rl_wait = rl_until - time.monotonic()

        if rl_wait > 0:
            if rl_wait <= self.max_retry_wait:
                time.sleep(rl_wait)
            else:
                return DecisionBatchResult(
                    backend=self.backend_name,
                    model=self.model,
                    answers={},
                    latency_ms=0.0,
                    error=f"HTTP 429: Rate limit cooldown active ({rl_wait:.1f}s remaining)",
                )

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Content-Length": str(len(body_bytes)),
            "User-Agent": user_agent(),
        }
        _apply_auth_header(headers, current_key, self.auth_mode)

        last_error = "unknown error"
        max_loops = self.max_attempts + 1  # allows one extra pass if secret rotates on 401
        attempt = 0
        while attempt < max_loops:
            attempt += 1
            req = urllib.request.Request(
                self.url,
                data=body_bytes,
                headers=headers,
                method="POST",
            )
            try:
                with open_safe_url(
                    req,
                    timeout=self.timeout,
                    opener_factory=self.opener_factory,
                    keep_alive=self.keep_alive,
                ) as resp:
                    raw = resp.read()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                payload = json.loads(raw.decode("utf-8"))
                parsed = parse_systemone_payload(
                    payload,
                    questions,
                    backend_name=self.backend_name,
                    fallback_model=self.model,
                    latency_ms=elapsed_ms,
                )
                _cache_put(ckey, parsed, self.cache_ttl)
                return parsed
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                formatted_err = f"HTTP {exc.code}: {err_body or exc.reason}"

                if exc.code == 401:
                    if not refreshed_once and self.secret_refresher is not None:
                        refreshed_once = True
                        candidate = str(self.secret_refresher() or "").strip()
                        if candidate and candidate != current_key:
                            rem_new = jwt_seconds_until_expiry(candidate)
                            if rem_new is None or rem_new > 0:
                                current_key = candidate
                                self.api_key = current_key
                                _apply_auth_header(headers, current_key, self.auth_mode)
                                continue
                    with _RATE_LOCK:
                        _AUTH_COOLDOWN_UNTIL[(self.url, _secret_digest(current_key))] = (
                            time.monotonic() + 20.0,
                            formatted_err,
                        )
                    last_error = formatted_err
                    break

                if exc.code == 429:
                    ra = parse_retry_after(exc.headers)
                    base_wait = ra if ra is not None else (0.20 * attempt)
                    jitter = random.uniform(0.03, 0.12) if base_wait >= 0.10 else 0.0
                    wait_s = base_wait + jitter
                    with _RATE_LOCK:
                        prev_until = _RATE_LIMIT_UNTIL.get(self.url, 0.0)
                        _RATE_LIMIT_UNTIL[self.url] = max(prev_until, time.monotonic() + base_wait)
                    if attempt < self.max_attempts and wait_s <= self.max_retry_wait:
                        time.sleep(wait_s)
                        continue
                    last_error = formatted_err
                    break

                if exc.code in (502, 503, 504) and attempt < self.max_attempts:
                    time.sleep(0.08 * attempt)
                    continue

                last_error = formatted_err
                break
            except urllib.error.URLError as exc:
                last_error = str(exc)
                if attempt < self.max_attempts:
                    time.sleep(0.05 * attempt)
                    continue
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
        timeout: float = 1.2,
        keep_alive: bool = True,
        opener_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.url = normalize_chat_completions_url(url)
        self.model = model
        self.api_key = api_key
        self.timeout = max(0.3, float(timeout))
        self.keep_alive = bool(keep_alive)
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
            with open_safe_url(
                req,
                timeout=self.timeout,
                opener_factory=self.opener_factory,
                keep_alive=self.keep_alive,
            ) as resp:
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

    def __init__(self, ctx: Any, *, timeout: float = 2.0) -> None:
        self.ctx = ctx
        self.timeout = max(0.8, float(timeout))

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


class FallbackDecisionBackend:
    """Wraps a primary decision backend with an opt-in supported fallback backend on error."""

    def __init__(self, primary: Any, fallback: Any) -> None:
        self.primary = primary
        self.fallback = fallback

    def decide(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
    ) -> DecisionBatchResult:
        res = self.primary.decide(state, questions)
        if res.ok:
            return res
        fb_res = self.fallback.decide(state, questions)
        if fb_res.ok:
            return fb_res
        return res


def _build_single_backend(
    ctx: Any,
    kind: str,
    *,
    timeout: float,
    keep_alive: bool,
    max_retry_wait: float,
    cache_ttl: float,
    opener_factory: Callable[..., Any] | None = None,
):
    if kind == "foundry_systemone":
        endpoint = str(ctx.get_config("foundry_endpoint", default="") or "").strip()
        deployment = str(
            ctx.get_config("foundry_deployment", default="microsoft-decision-1.1")
            or "microsoft-decision-1.1"
        ).strip()
        auth_mode = str(ctx.get_config("foundry_auth_header", default="auto") or "auto").strip()
        api_key = read_secret("TRISMEGISTUS_FOUNDRY_API_KEY")
        return SystemOneHttpBackend(
            url=normalize_foundry_systemone_url(endpoint),
            model=deployment,
            api_key=api_key,
            auth_mode=auth_mode,
            backend_name="foundry_systemone",
            timeout=timeout,
            max_retry_wait=max_retry_wait,
            cache_ttl=cache_ttl,
            keep_alive=keep_alive,
            opener_factory=opener_factory,
            secret_refresher=lambda: read_secret("TRISMEGISTUS_FOUNDRY_API_KEY", refresh=True),
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
            max_retry_wait=max_retry_wait,
            cache_ttl=cache_ttl,
            keep_alive=keep_alive,
            opener_factory=opener_factory,
            secret_refresher=lambda: read_secret("TRISMEGISTUS_LOCAL_API_KEY", refresh=True),
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
            keep_alive=keep_alive,
            opener_factory=opener_factory,
        )

    return HermesAuxBackend(ctx, timeout=timeout)


def build_decision_backend(ctx: Any, *, opener_factory: Callable[..., Any] | None = None):
    """Instantiate the active System-1 backend (plus optional fallback) from `ctx.get_config(...)`."""
    kind = str(ctx.get_config("backend", default="foundry_systemone") or "foundry_systemone").strip()
    fallback_kind = str(ctx.get_config("fallback_backend", default="none") or "none").strip().lower()
    timeout = float(ctx.get_config("timeout_seconds", default=1.2) or 1.2)
    raw_retry_wait = ctx.get_config("max_retry_wait_seconds", default=2.0)
    max_retry_wait = 2.0 if raw_retry_wait is None else float(raw_retry_wait)
    raw_cache_ttl = ctx.get_config("triage_cache_ttl_seconds", default=180.0)
    cache_ttl = 180.0 if raw_cache_ttl is None else float(raw_cache_ttl)
    keep_alive = bool(ctx.get_config("keep_alive_enabled", default=True))

    primary = _build_single_backend(
        ctx,
        kind,
        timeout=timeout,
        keep_alive=keep_alive,
        max_retry_wait=max_retry_wait,
        cache_ttl=cache_ttl,
        opener_factory=opener_factory,
    )
    if fallback_kind not in ("", "none", kind.lower()):
        fallback = _build_single_backend(
            ctx,
            fallback_kind,
            timeout=timeout,
            keep_alive=keep_alive,
            max_retry_wait=max_retry_wait,
            cache_ttl=cache_ttl,
            opener_factory=opener_factory,
        )
        return FallbackDecisionBackend(primary, fallback)
    return primary
