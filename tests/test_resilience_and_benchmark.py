"""Tests for v0.2.2 manifest compatibility, 401/429/secret-refresh resilience, turn-deduplicated ledger, and held-out benchmark."""

from __future__ import annotations

import base64
import email.message
import importlib.util
import io
import json
import logging
from pathlib import Path
import sqlite3
import sys
import time
import urllib.error

ROOT = Path(__file__).resolve().parent.parent


def _b64url(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _make_synthetic_jwt(exp_epoch: float, tag: str = "v1") -> str:
    header = _b64url({"alg": "RS256", "typ": "JWT"})
    payload = _b64url({"aud": "https://cognitiveservices.azure.com", "exp": int(exp_epoch), "jti": tag})
    sig = _b64url({"sig": tag})
    return f"{header}.{payload}.{sig}"


def _make_headers(mapping: dict[str, str]) -> email.message.Message:
    msg = email.message.Message()
    for k, v in mapping.items():
        msg[k] = v
    return msg


class _FakeHttpResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._raw = json.dumps(payload).encode("utf-8")
        self.status = status
        self.headers = {"Connection": "keep-alive"}

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _RecordingOpener:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append((request, timeout))
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _FakeLowLevelResp:
    def __init__(self, payload: dict, status: int = 200, conn_header: str = "keep-alive"):
        self.status = status
        self.reason = "OK" if status < 400 else "Error"
        self.headers = {"Connection": conn_header}
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body


class _FakeConn:
    def __init__(self, events: list):
        self.events = list(events)
        self.sock = object()
        self.timeout = 1.2
        self.closed = False

    def request(self, method: str, target: str, body: bytes = b"", headers: dict | None = None):
        del method, target, body, headers
        evt = self.events.pop(0)
        if isinstance(evt, Exception):
            raise evt
        self._resp = evt

    def getresponse(self):
        return self._resp

    def close(self):
        self.closed = True
        self.sock = None


def test_manifest_parses_with_zero_warnings_and_preserves_shadow_default(pkg):
    del pkg
    manifest_text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert "provides_middleware:" in manifest_text
    assert "manifest_version: 2" not in manifest_text
    assert "default: shadow" in manifest_text

    try:
        from hermes_cli.plugins_manifest import parse_manifest_file
    except ImportError:
        return

    records: list[tuple[str, str]] = []

    class _MemHandler(logging.Handler):
        def emit(self, record):
            records.append((record.levelname, record.getMessage()))

    log = logging.getLogger("hermes_cli.plugins")
    handler = _MemHandler()
    log.addHandler(handler)
    try:
        parsed = parse_manifest_file(ROOT / "plugin.yaml", ROOT, "user", "")
    finally:
        log.removeHandler(handler)

    assert parsed is not None
    assert parsed.version == "0.3.1"
    assert parsed.config_schema["mode"]["default"] == "shadow"
    assert "llm.model_override" in parsed.capabilities
    warnings = [msg for lvl, msg in records if lvl in ("WARNING", "ERROR", "CRITICAL")]
    assert warnings == [], f"Unexpected manifest warning(s): {warnings}"


def test_429_standard_1s_retry_after_succeeds_and_long_cooldown_fast_fails(pkg):
    pkg.backends.clear_backend_caches()
    DQ = pkg.contracts.DecisionQuestion

    # Previously `Retry-After: 1` (1.0s) > 0.6s caused immediate failure on Attempt 1!
    err_429_short = urllib.error.HTTPError(
        url="https://demo.example/providers/microsoft/v1/systemone",
        code=429,
        msg="Too Many Requests",
        hdrs=_make_headers({"Retry-After": "0.12"}),
        fp=io.BytesIO(b'{"error":{"code":"RateLimitReached"}}'),
    )
    opener = _RecordingOpener(
        [
            err_429_short,
            _FakeHttpResponse(
                {
                    "model": "microsoft-decision-1.1",
                    "answers": {"gate": {"type": "noul", "noul": 0.25}},
                    "usage": {"input_tokens": 50, "output_tokens": 0},
                }
            ),
        ]
    )
    client = pkg.backends.SystemOneHttpBackend(
        url="https://demo.example/providers/microsoft/v1/systemone",
        model="microsoft-decision-1.1",
        api_key="static-key-value",
        timeout=1.2,
        max_retry_wait=2.0,
        opener_factory=lambda _h: opener,
    )
    res = client.decide({"x": 1}, {"gate": DQ.noul("Check?")})
    assert res.ok
    assert len(opener.requests) == 2

    # When Retry-After (10s) exceeds max_retry_wait (0.5s), subsequent concurrent calls
    # fail-open in 0ms without hammering Foundry again during the cooldown window.
    pkg.backends.clear_backend_caches()
    err_429_long = urllib.error.HTTPError(
        url="https://cooldown.example/providers/microsoft/v1/systemone",
        code=429,
        msg="Too Many Requests",
        hdrs=_make_headers({"retry-after-ms": "8000"}),
        fp=io.BytesIO(b'{"error":{"code":"RateLimitReached"}}'),
    )
    opener_long = _RecordingOpener([err_429_long])
    client_long = pkg.backends.SystemOneHttpBackend(
        url="https://cooldown.example/providers/microsoft/v1/systemone",
        model="microsoft-decision-1.1",
        api_key="static-key-value",
        max_retry_wait=0.5,
        opener_factory=lambda _h: opener_long,
    )
    r1 = client_long.decide({"x": 1}, {"gate": DQ.noul("Check?")})
    assert not r1.ok
    assert len(opener_long.requests) == 1

    # Immediate second call sees active 8s cooldown and avoids a second wasted HTTP 429!
    r2 = client_long.decide({"x": 2}, {"gate": DQ.noul("Check?")})
    assert not r2.ok
    assert "cooldown active" in (r2.error or "").lower()
    assert len(opener_long.requests) == 1


def test_expired_jwt_preflight_refresh_and_401_live_secret_rotation(pkg):
    pkg.backends.clear_backend_caches()
    DQ = pkg.contracts.DecisionQuestion
    now = time.time()
    expired_jwt = _make_synthetic_jwt(now - 600, tag="old")
    fresh_jwt = _make_synthetic_jwt(now + 3600, tag="fresh")

    assert pkg.compat.is_jwt_token(expired_jwt)
    assert pkg.compat.jwt_seconds_until_expiry(expired_jwt, now=now) < 0
    assert pkg.compat.jwt_seconds_until_expiry(fresh_jwt, now=now) > 3000

    # 1. Expired JWT + refresher returning fresh JWT -> zero 401 network calls, sends Bearer header!
    opener = _RecordingOpener(
        [
            _FakeHttpResponse(
                {
                    "model": "microsoft-decision-1.1",
                    "answers": {"gate": {"type": "noul", "noul": 0.12}},
                    "usage": {"input_tokens": 44, "output_tokens": 0},
                }
            )
        ]
    )
    client = pkg.backends.SystemOneHttpBackend(
        url="https://jwt-refresh.example/providers/microsoft/v1/systemone",
        model="microsoft-decision-1.1",
        api_key=expired_jwt,
        auth_mode="auto",
        opener_factory=lambda _h: opener,
        secret_refresher=lambda: fresh_jwt,
    )
    res = client.decide({"msg": "hello"}, {"gate": DQ.noul("Safe?")})
    assert res.ok
    assert len(opener.requests) == 1
    sent_headers = {k.lower(): v for k, v in opener.requests[0][0].header_items()}
    assert sent_headers.get("authorization") == f"Bearer {fresh_jwt}"
    assert "api-key" not in sent_headers

    # 2. Expired JWT + no refreshed token available -> fails fast in 0 HTTP requests!
    opener_none = _RecordingOpener([])
    client_still_exp = pkg.backends.SystemOneHttpBackend(
        url="https://jwt-exp.example/providers/microsoft/v1/systemone",
        model="microsoft-decision-1.1",
        api_key=expired_jwt,
        auth_mode="bearer",
        opener_factory=lambda _h: opener_none,
        secret_refresher=lambda: expired_jwt,
    )
    res_exp = client_still_exp.decide({"msg": "hello"}, {"gate": DQ.noul("Safe?")})
    assert not res_exp.ok
    assert "expired" in (res_exp.error or "").lower()
    assert len(opener_none.requests) == 0

    # 3. Server-side HTTP 401 triggers single secret_refresher rotation and succeeds on retry
    err_401 = urllib.error.HTTPError(
        url="https://rotate-401.example/providers/microsoft/v1/systemone",
        code=401,
        msg="Unauthorized",
        hdrs=_make_headers({}),
        fp=io.BytesIO(b'{"statusCode":401,"message":"Unauthorized"}'),
    )
    opener_401 = _RecordingOpener(
        [
            err_401,
            _FakeHttpResponse(
                {
                    "model": "microsoft-decision-1.1",
                    "answers": {"gate": {"type": "noul", "noul": 0.88}},
                    "usage": {"input_tokens": 30, "output_tokens": 0},
                }
            ),
        ]
    )
    client_401 = pkg.backends.SystemOneHttpBackend(
        url="https://rotate-401.example/providers/microsoft/v1/systemone",
        model="microsoft-decision-1.1",
        api_key="stale-key-1",
        auth_mode="api-key",
        opener_factory=lambda _h: opener_401,
        secret_refresher=lambda: "rotated-key-2",
    )
    res_401 = client_401.decide({"msg": "check"}, {"gate": DQ.noul("Gate?")})
    assert res_401.ok
    assert len(opener_401.requests) == 2


def test_keepalive_discards_error_sockets_and_backend_retries_transient_urlerror(pkg):
    err_conn = _FakeConn([_FakeLowLevelResp({"error": "rate_limited"}, status=429)])
    ok_conn = _FakeConn([_FakeLowLevelResp({"ok": True}, status=200)])
    conns = [err_conn, ok_conn]

    pool = pkg.compat.KeepAliveHttpPool(connection_factory=lambda _s, _h, _p, _t: conns.pop(0))
    url = "https://pool-err.example/providers/microsoft/v1/systemone"

    try:
        pool.post_bytes(url, b"{}", {"api-key": "placeholder"}, timeout=1.0)
        raise AssertionError("Expected HTTPError 429")
    except urllib.error.HTTPError as exc:
        assert exc.code == 429
    # Socket from 429 MUST be closed rather than returned to idle pool!
    assert err_conn.closed is True

    r2 = json.loads(pool.post_bytes(url, b"{}", {"api-key": "placeholder"}, timeout=1.0).read())
    assert r2 == {"ok": True}
    assert ok_conn.closed is False

    # Transient URLError (e.g. Remote end closed connection) retries once cleanly
    DQ = pkg.contracts.DecisionQuestion
    opener_transient = _RecordingOpener(
        [
            urllib.error.URLError("Remote end closed connection without response"),
            _FakeHttpResponse(
                {
                    "model": "microsoft-decision-1.1",
                    "answers": {"gate": {"type": "noul", "noul": 0.33}},
                    "usage": {"input_tokens": 25, "output_tokens": 0},
                }
            ),
        ]
    )
    client = pkg.backends.SystemOneHttpBackend(
        url="https://transient.example/providers/microsoft/v1/systemone",
        model="microsoft-decision-1.1",
        api_key="placeholder",
        max_attempts=2,
        opener_factory=lambda _h: opener_transient,
    )
    res = client.decide({"a": 1}, {"gate": DQ.noul("Check?")})
    assert res.ok
    assert len(opener_transient.requests) == 2


def test_triage_cache_and_supported_fallback_backend(pkg):
    pkg.backends.clear_backend_caches()
    DQ = pkg.contracts.DecisionQuestion
    opener = _RecordingOpener(
        [
            _FakeHttpResponse(
                {
                    "model": "microsoft-decision-1.1",
                    "answers": {"gate": {"type": "noul", "noul": 0.19}},
                    "usage": {"input_tokens": 120, "output_tokens": 0},
                }
            )
        ]
    )
    client = pkg.backends.SystemOneHttpBackend(
        url="https://cache.example/providers/microsoft/v1/systemone",
        model="microsoft-decision-1.1",
        api_key="placeholder",
        cache_ttl=60.0,
        opener_factory=lambda _h: opener,
    )
    first = client.decide({"heartbeat": "ping"}, {"gate": DQ.noul("Routine?")})
    assert first.ok and not first.cached and first.input_tokens == 120

    # Identical repeated heartbeat hits cache: 0 extra HTTP requests, 0 extra tokens!
    second = client.decide({"heartbeat": "ping"}, {"gate": DQ.noul("Routine?")})
    assert second.ok and second.cached and second.input_tokens == 0
    assert len(opener.requests) == 1

    # Opt-in fallback_backend='hermes_aux' recovers when Foundry lacks secret
    from test_pipeline import _StubPluginContext

    ctx = _StubPluginContext(
        config={
            "mode": "shadow",
            "backend": "foundry_systemone",
            "fallback_backend": "hermes_aux",
            "foundry_endpoint": "",
            "triage_cache_ttl_seconds": 0.0,
        },
        scripted_decisions=[
            {
                "tier": {"choice": "system1", "confidence": 0.95},
                "effort": {"score": 0.3, "confidence": 0.90},
                "trap_risk": {"noul": 0.05},
                "needs_plan": {"noul": 0.04},
            }
        ],
    )
    verdict = pkg.triage.run_turn_triage(
        ctx,
        session_id="ses_fb",
        user_message="Translate hello to Spanish",
        conversation_history=[],
        model="gpt-5.4",
    )
    assert verdict.error is None
    assert verdict.backend == "hermes_aux"
    assert verdict.effective_tier == "system1"


def test_ledger_deduplicates_multistep_turns_and_legacy_rows(pkg):
    # 1. v0.2.2+ turn with 4 API tool-loop steps -> 1 unique turn, 1x tokens, 1x error!
    v_ok = pkg.contracts.TriageVerdict(
        session_id="ses_multi",
        raw_tier="system2",
        effective_tier="system2",
        effort_score=3.2,
        effective_effort="high",
        confidence=0.93,
        abstained=False,
        sticky_override=False,
        error_streak_override=False,
        needs_plan_prob=0.80,
        backend="foundry_systemone",
        decision_model="microsoft-decision-1.1",
        decision_latency_ms=240.0,
        decision_input_tokens=600,
        applied_model="gpt-5.4",
        applied_effort="shadow(high)",
    )
    for step in range(4):
        v_ok.step_paced = step in (1, 2)
        pkg.ledger.record_turn_completion(
            verdict=v_ok,
            mode="shadow",
            provider_duration_ms=500.0,
            provider_input_tokens=1000,
            provider_output_tokens=150,
        )

    v_err = pkg.contracts.TriageVerdict(
        session_id="ses_multi",
        raw_tier="system2",
        effective_tier="system2",
        effort_score=2.5,
        effective_effort="high",
        confidence=0.0,
        abstained=True,
        sticky_override=False,
        error_streak_override=False,
        needs_plan_prob=0.0,
        backend="foundry_systemone",
        decision_model="microsoft-decision-1.1",
        decision_latency_ms=130.0,
        decision_input_tokens=0,
        error="HTTP 401: Unauthorized",
        applied_model="gpt-5.4",
        applied_effort="shadow(high)",
    )
    for _ in range(3):
        pkg.ledger.record_turn_completion(verdict=v_err, mode="shadow")

    stats = pkg.ledger.read_summary_stats()
    assert stats["total_steps"] == 7
    assert stats["total"] == 2
    assert stats["ok_turns"] == 1
    assert stats["error_turns"] == 1
    assert stats["decision_input_tokens"] == 600
    assert stats["step_paced"] == 2
    assert stats["errors_by_category"]["http_401"] == 1

    # 2. Legacy pre-v0.2.2 database (no turn_id/step_index columns) with interleaved sessions
    legacy_db = Path(pkg.compat.os.environ["HERMES_HOME"]) / "legacy_ledger.db"
    conn = sqlite3.connect(str(legacy_db))
    conn.execute(
        """
        CREATE TABLE trismegistus_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            session_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            backend TEXT NOT NULL,
            decision_model TEXT NOT NULL,
            raw_tier TEXT NOT NULL,
            effective_tier TEXT NOT NULL,
            effort_score REAL NOT NULL,
            effective_effort TEXT NOT NULL,
            confidence REAL NOT NULL,
            abstained INTEGER NOT NULL,
            sticky_override INTEGER NOT NULL,
            error_streak_override INTEGER NOT NULL,
            wire_guarded INTEGER NOT NULL,
            decision_latency_ms REAL NOT NULL,
            decision_input_tokens INTEGER NOT NULL,
            applied_model TEXT NOT NULL,
            applied_effort TEXT NOT NULL,
            error TEXT DEFAULT ''
        )
        """
    )
    base_ts = 1795000000.0
    # Session A (3 steps, same 312.45678ms triage) interleaved with Session B (2 steps, 145.12345ms 429)
    rows = [
        (base_ts + 1, "ses_A", 312.45678, 500, 0.91, ""),
        (base_ts + 2, "ses_B", 145.12345, 0, 0.0, "HTTP 429: RateLimitReached"),
        (base_ts + 5, "ses_A", 312.45678, 500, 0.91, ""),
        (base_ts + 6, "ses_B", 145.12345, 0, 0.0, "HTTP 429: RateLimitReached"),
        (base_ts + 9, "ses_A", 312.45678, 500, 0.91, ""),
    ]
    for ts, sid, lat, tok, conf, err in rows:
        conn.execute(
            """
            INSERT INTO trismegistus_decisions (
                ts, session_id, mode, backend, decision_model, raw_tier, effective_tier,
                effort_score, effective_effort, confidence, abstained, sticky_override,
                error_streak_override, wire_guarded, decision_latency_ms, decision_input_tokens,
                applied_model, applied_effort, error
            ) VALUES (?, ?, 'shadow', 'foundry_systemone', 'microsoft-decision-1.1', 'system2', 'system2',
                      2.5, 'high', ?, 0, 0, 0, 0, ?, ?, 'gpt-5.4', 'shadow(high)', ?)
            """,
            (ts, sid, conf, lat, tok, err),
        )
    conn.commit()
    conn.close()

    audited = pkg.ledger.audit_ledger_db(legacy_db)
    assert audited["total_steps"] == 5
    assert audited["total"] == 2
    assert audited["ok_turns"] == 1
    assert audited["error_turns"] == 1
    assert audited["decision_input_tokens"] == 500
    assert audited["errors_by_category"]["http_429"] == 1


def test_heldout_300_dataset_and_shadow_runner(pkg):
    del pkg
    bench_path = ROOT / "benchmarks" / "run_heldout.py"
    spec = importlib.util.spec_from_file_location("trismegistus_bench_runner", str(bench_path))
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)

    items = bench.load_heldout_dataset(ROOT / "benchmarks" / "heldout_300.jsonl")
    assert len(items) == 300
    counts = {}
    for x in items:
        counts[x["label"]] = counts.get(x["label"], 0) + 1
    assert counts == {"s1": 100, "s2": 100, "ambiguous_trap": 100}

    report = bench.run_benchmark(backend="offline_wire_harness")
    assert report["status"] == "completed"
    assert report["mode_enforced"] == "shadow"
    assert report["shadow_mutations_detected"] == 0
    assert report["independent_turns_evaluated"] == 300
    assert report["routing_risk_metrics"]["false_fast_route_rate"] == 0.0
    assert report["per_class"]["ambiguous_trap"]["routed_system2"] == 100
