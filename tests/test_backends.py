"""Tests for System-1 decision backends, Keep-Alive TLS pooling, and Logprobs calibration."""

from __future__ import annotations

import http.client
import io
import json
import urllib.error


class _FakeHttpResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._raw = json.dumps(payload).encode("utf-8")
        self.status = status

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


class _FakeKeepAliveConn:
    """Simulates http.client.HTTPSConnection with socket reuse & stale disconnect recovery."""

    def __init__(self, scripted_events: list):
        self.scripted_events = scripted_events
        self.sock = object()
        self.timeout = 1.2
        self.requests: list[tuple[str, str, bytes, dict]] = []
        self.closed = False

    def request(self, method: str, target: str, body: bytes = b"", headers: dict | None = None):
        self.requests.append((method, target, body, dict(headers or {})))
        evt = self.scripted_events.pop(0)
        if isinstance(evt, Exception):
            raise evt
        self._next_resp = evt

    def getresponse(self):
        return self._next_resp

    def close(self):
        self.closed = True
        self.sock = None


class _FakeLowLevelResp:
    def __init__(self, payload: dict, status: int = 200, conn_header: str = "keep-alive"):
        self.status = status
        self.reason = "OK" if status < 400 else "Error"
        self.headers = {"Connection": conn_header}
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body


def test_foundry_url_normalization(pkg):
    norm = pkg.backends.normalize_foundry_systemone_url
    assert (
        norm("https://demo-foundry.services.ai.azure.com/")
        == "https://demo-foundry.services.ai.azure.com/providers/microsoft/v1/systemone"
    )
    assert (
        norm("https://demo-foundry.services.ai.azure.com/openai/v1")
        == "https://demo-foundry.services.ai.azure.com/providers/microsoft/v1/systemone"
    )
    assert (
        norm("https://demo-foundry.services.ai.azure.com/providers/microsoft/v1/systemone")
        == "https://demo-foundry.services.ai.azure.com/providers/microsoft/v1/systemone"
    )


def test_keepalive_pool_reuses_warm_socket_and_recovers_stale_disconnect(pkg):
    created_conns: list[_FakeKeepAliveConn] = []
    # Conn 1 serves call #1, then raises RemoteDisconnected on call #2 (stale idle socket).
    # Pool must transparently discard Conn 1 and retry call #2 on fresh Conn 2!
    conn1 = _FakeKeepAliveConn(
        [
            _FakeLowLevelResp({"ok": 1}),
            http.client.RemoteDisconnected("Remote end closed connection without response"),
        ]
    )
    conn2 = _FakeKeepAliveConn([_FakeLowLevelResp({"ok": 2}), _FakeLowLevelResp({"ok": 3})])
    queue = [conn1, conn2]

    def factory(_scheme, _host, _port, timeout):
        c = queue.pop(0)
        c.timeout = timeout
        created_conns.append(c)
        return c

    pool = pkg.compat.KeepAliveHttpPool(connection_factory=factory)
    url = "https://demo-foundry.services.ai.azure.com/providers/microsoft/v1/systemone"

    r1 = json.loads(pool.post_bytes(url, b"{}", {"api-key": "k"}, timeout=1.2).read())
    assert r1 == {"ok": 1}
    assert len(created_conns) == 1

    # Second call hits stale conn1 -> transparently retries on conn2
    r2 = json.loads(pool.post_bytes(url, b"{}", {"api-key": "k"}, timeout=1.2).read())
    assert r2 == {"ok": 2}
    assert conn1.closed is True
    assert len(created_conns) == 2

    # Third call reuses warm conn2 with zero new socket creation!
    r3 = json.loads(pool.post_bytes(url, b"{}", {"api-key": "k"}, timeout=1.2).read())
    assert r3 == {"ok": 3}
    assert len(created_conns) == 2
    assert len(conn2.requests) == 2


def test_foundry_systemone_request_headers_and_multi_question_parse(pkg):
    DQ = pkg.contracts.DecisionQuestion
    opener = _RecordingOpener(
        [
            _FakeHttpResponse(
                {
                    "model": "microsoft-decision-1.1",
                    "answers": {
                        "tier": {
                            "type": "choice",
                            "choice": "system1",
                            "confidence": 0.93,
                            "probabilities": {"system1": 0.93, "system2": 0.07},
                        },
                        "effort": {
                            "type": "score",
                            "score": 0.42,
                            "confidence": 0.88,
                            "probabilities": {"0": 0.62, "1": 0.34, "2": 0.04, "3": 0.0, "4": 0.0},
                        },
                        "trap_risk": {
                            "type": "noul",
                            "noul": 0.08,
                        },
                    },
                    "usage": {"input_tokens": 184, "output_tokens": 0},
                }
            )
        ]
    )

    client = pkg.backends.SystemOneHttpBackend(
        url="https://demo-foundry.services.ai.azure.com/providers/microsoft/v1/systemone",
        model="microsoft-decision-1.1",
        api_key="test-credential-placeholder",
        auth_mode="api-key",
        opener_factory=lambda _handler: opener,
    )
    res = client.decide(
        {"msg": "Format this date as ISO-8601"},
        {
            "tier": DQ.choice("Pick tier", {"system1": "fast", "system2": "deep"}),
            "effort": DQ.score("Effort", ["Minimal", "Low", "Medium", "High", "XHigh"]),
            "trap_risk": DQ.noul("Hidden trap?"),
        },
    )

    assert res.ok
    assert res.model == "microsoft-decision-1.1"
    assert res.input_tokens == 184
    assert res.answers["tier"].choice == "system1"
    assert abs(res.answers["tier"].top_probability - 0.93) < 1e-6
    assert abs(res.answers["effort"].score - 0.42) < 1e-6
    assert abs(res.answers["trap_risk"].noul - 0.08) < 1e-6

    sent_req, _timeout = opener.requests[0]
    headers_lower = {k.lower(): v for k, v in sent_req.header_items()}
    assert int(headers_lower["content-length"]) == len(sent_req.data)
    assert headers_lower["api-key"] == "test-credential-placeholder"


def test_foundry_429_retry_after_ms(pkg):
    DQ = pkg.contracts.DecisionQuestion
    err_429 = urllib.error.HTTPError(
        url="https://demo.example/providers/microsoft/v1/systemone",
        code=429,
        msg="Too Many Requests",
        hdrs={"retry-after-ms": "10"},
        fp=io.BytesIO(b'{"error": "rate_limited"}'),
    )
    opener = _RecordingOpener(
        [
            err_429,
            _FakeHttpResponse(
                {
                    "model": "microsoft-decision-1.1",
                    "answers": {"gate": {"type": "noul", "noul": 0.91}},
                    "usage": {"input_tokens": 42, "output_tokens": 0},
                }
            ),
        ]
    )
    client = pkg.backends.SystemOneHttpBackend(
        url="https://demo.example/providers/microsoft/v1/systemone",
        model="microsoft-decision-1.1",
        api_key="test-credential-placeholder",
        max_attempts=2,
        opener_factory=lambda _handler: opener,
    )
    res = client.decide("drop database prod", {"gate": DQ.noul("Destructive?")})
    assert res.ok
    assert len(opener.requests) == 2
    assert abs(res.answers["gate"].noul - 0.91) < 1e-6


def test_local_logits_softmax_calibration(pkg):
    DQ = pkg.contracts.DecisionQuestion
    opener = _RecordingOpener(
        [
            _FakeHttpResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "B"},
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "B",
                                        "logprob": -0.10536,
                                        "top_logprobs": [
                                            {"token": "B", "logprob": -0.10536},
                                            {"token": "A", "logprob": -2.30258},
                                        ],
                                    }
                                ]
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 95},
                }
            )
        ]
    )
    backend = pkg.backends.LocalLogitsBackend(
        url="http://127.0.0.1:8080/v1",
        model="gemma-4-26b-a4b",
        opener_factory=lambda _handler: opener,
    )
    res = backend.decide(
        "Debug race condition in WAL checkpoint",
        {"tier": DQ.choice("Which tier?", {"system1": "fast", "system2": "deep"})},
    )
    assert res.ok
    assert res.answers["tier"].choice == "system2"
    assert 0.88 < res.answers["tier"].top_probability < 0.92
