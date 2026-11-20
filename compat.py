"""Public Hermes plugin API bridges + thread-safe TLS Keep-Alive connection pool.

Every Hermes import in this module comes strictly from the documented Public Plugin API
(`hermes_cli/data/plugin_api.json` / `developer-guide/plugins/public-api.md`), so
`hermes plugins validate` passes with zero private-import warnings while standalone
`pytest` runs cleanly even outside a Hermes checkout.
"""

from __future__ import annotations

from collections import deque
import http.client
import io
import os
from pathlib import Path
import re
import sqlite3
import ssl
import tempfile
import threading
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

_SECRET_PATTERNS = (
    re.compile(r"\b(gh[pousr]_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\b(github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,\"']{8,}"),
)


def redact_text(text: str) -> str:
    """Force-redact sensitive tokens before any external decision call or ledger write."""
    if not text:
        return ""
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(text, force=True)
    except ImportError:
        out = text
        for pat in _SECRET_PATTERNS:
            out = pat.sub(
                lambda m: (
                    m.group(1)
                    if m.lastindex and m.lastindex >= 1 and m.group(1).endswith((" ", "=", ":"))
                    else ""
                )
                + "[REDACTED]",
                out,
            )
        return out


def read_secret(env_name: str) -> str:
    """Read a plugin secret through Hermes's profile-scoped secret resolver."""
    try:
        from agent.secret_scope import get_secret

        val = get_secret(env_name)
        return str(val or "").strip()
    except ImportError:
        return str(os.getenv(env_name) or "").strip()


def user_agent() -> str:
    """Return the canonical Hermes User-Agent header when inside Hermes."""
    try:
        from hermes_cli.version_info import hermes_user_agent

        return hermes_user_agent("trismegistus/0.2.0")
    except ImportError:
        return "Hermes-Trismegistus/0.2.0"


class _BytesHttpResponse:
    """Context-manager response wrapper matching `urlopen(...).read()`."""

    def __init__(self, status: int, body: bytes, headers: Any) -> None:
        self.status = status
        self._body = body
        self.headers = headers

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_BytesHttpResponse":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


class KeepAliveHttpPool:
    """Thread-safe HTTP/1.1 Keep-Alive pool eliminating ~80-150 ms TLS handshakes to Foundry.

    Security properties:
    - Uses verified default `ssl.SSLContext` for `https://`.
    - Never forwards credentials across origins (redirects are not auto-followed).
    - Automatically discards stale idle sockets closed by upstream load balancers and retries
      transparently once on a fresh connection.
    """

    def __init__(self, max_per_origin: int = 4, connection_factory: Callable[..., Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._pools: dict[tuple[str, str, int], deque[Any]] = {}
        self._max_per_origin = max_per_origin
        self._ssl_ctx: ssl.SSLContext | None = None
        self._connection_factory = connection_factory

    def _get_ssl_context(self) -> ssl.SSLContext:
        if self._ssl_ctx is None:
            self._ssl_ctx = ssl.create_default_context()
        return self._ssl_ctx

    @staticmethod
    def _origin_and_target(url: str) -> tuple[tuple[str, str, int], str]:
        parsed = urllib.parse.urlparse(url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in ("https", "http"):
            raise urllib.error.URLError(f"Unsupported URL scheme: {scheme!r}")
        host = (parsed.hostname or "").lower().rstrip(".")
        if not host:
            raise urllib.error.URLError("Missing hostname in URL")
        port = parsed.port or (443 if scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return (scheme, host, port), path

    def _checkout(self, origin: tuple[str, str, int], timeout: float) -> tuple[Any, bool]:
        with self._lock:
            q = self._pools.get(origin)
            if q:
                conn = q.popleft()
                conn.timeout = timeout
                sock = getattr(conn, "sock", None)
                if sock is not None and hasattr(sock, "settimeout"):
                    sock.settimeout(timeout)
                return conn, True
        scheme, host, port = origin
        if self._connection_factory is not None:
            return self._connection_factory(scheme, host, port, timeout), False
        if scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=timeout, context=self._get_ssl_context()), False
        return http.client.HTTPConnection(host, port, timeout=timeout), False

    def _checkin(self, origin: tuple[str, str, int], conn: Any) -> None:
        if getattr(conn, "sock", None) is None:
            return
        with self._lock:
            q = self._pools.setdefault(origin, deque())
            if len(q) < self._max_per_origin:
                q.append(conn)
                return
        conn.close()

    def post_bytes(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        *,
        timeout: float,
    ) -> _BytesHttpResponse:
        origin, target = self._origin_and_target(url)
        req_headers = {**headers, "Connection": "keep-alive"}

        for try_idx in range(2):
            conn, was_reused = self._checkout(origin, timeout)
            try:
                conn.request("POST", target, body=body, headers=req_headers)
                resp = conn.getresponse()
                resp_body = resp.read()
                status = int(resp.status)
                resp_headers = resp.headers
                will_close = (
                    hasattr(resp_headers, "get")
                    and str(resp_headers.get("Connection") or "").lower() == "close"
                )
                if will_close:
                    conn.close()
                else:
                    self._checkin(origin, conn)

                if status >= 400:
                    raise urllib.error.HTTPError(
                        url=url,
                        code=status,
                        msg=getattr(resp, "reason", "HTTP Error"),
                        hdrs=resp_headers,
                        fp=io.BytesIO(resp_body),
                    )
                return _BytesHttpResponse(status=status, body=resp_body, headers=resp_headers)
            except urllib.error.HTTPError:
                raise
            except (
                http.client.RemoteDisconnected,
                http.client.BadStatusLine,
                BrokenPipeError,
                ConnectionResetError,
            ) as exc:
                conn.close()
                if was_reused and try_idx == 0:
                    continue
                raise urllib.error.URLError(str(exc)) from exc
            except Exception as exc:
                conn.close()
                if isinstance(exc, urllib.error.URLError):
                    raise
                raise urllib.error.URLError(str(exc)) from exc

        raise urllib.error.URLError("Keep-alive request exhausted")


_GLOBAL_KEEPALIVE_POOL = KeepAliveHttpPool()


def open_safe_url(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener_factory: Callable[..., Any] | None = None,
    keep_alive: bool = True,
):
    """Send an HTTP request with TLS keep-alive (default) or cross-origin-safe urllib."""
    if opener_factory is not None:
        try:
            from hermes_cli.urllib_security import open_credentialed_url

            return open_credentialed_url(request, timeout=timeout, opener_factory=opener_factory)
        except ImportError:
            opener = opener_factory(urllib.request.HTTPRedirectHandler())
            return opener.open(request, timeout=timeout)

    if (
        keep_alive
        and request.get_method() == "POST"
        and isinstance(request.data, (bytes, bytearray))
        and not os.environ.get("HTTPS_PROXY")
        and not os.environ.get("HTTP_PROXY")
        and not os.environ.get("https_proxy")
        and not os.environ.get("http_proxy")
    ):
        hdrs = {k: v for k, v in request.header_items()}
        return _GLOBAL_KEEPALIVE_POOL.post_bytes(
            request.full_url,
            bytes(request.data),
            hdrs,
            timeout=timeout,
        )

    try:
        from hermes_cli.urllib_security import open_credentialed_url

        return open_credentialed_url(request, timeout=timeout)
    except ImportError:
        return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - standalone fallback


def parse_retry_after(headers: Any) -> float | None:
    """Parse Foundry's non-standard `retry-after-ms` or standard `Retry-After`."""
    if headers is None:
        return None
    ms_val = headers.get("retry-after-ms") if hasattr(headers, "get") else None
    if ms_val is not None:
        try:
            ms = float(str(ms_val).strip())
            if ms >= 0:
                return min(ms / 1000.0, 2.0)
        except ValueError:
            pass
    ra_val = headers.get("Retry-After") if hasattr(headers, "get") else None
    if ra_val is not None:
        try:
            from agent.retry_utils import parse_retry_after_seconds

            parsed = parse_retry_after_seconds(str(ra_val))
            if parsed is not None:
                return min(float(parsed), 2.0)
        except ImportError:
            try:
                return min(float(str(ra_val).strip()), 2.0)
            except ValueError:
                return None
    return None


def open_plugin_db(plugin_name: str = "trismegistus", filename: str = "ledger.db") -> sqlite3.Connection:
    """Open a profile-scoped WAL SQLite database in `<HERMES_HOME>/plugin-data/<name>/`."""
    try:
        from plugins.plugin_storage import plugin_db

        return plugin_db(plugin_name, filename)
    except ImportError:
        base = Path(os.getenv("HERMES_HOME") or (Path(tempfile.gettempdir()) / "hermes-trismegistus-test"))
        target_dir = base / "plugin-data" / plugin_name
        target_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target_dir / filename), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


try:
    from agent.delegation_provider import DelegationProvider as BaseDelegationProvider
    from agent.delegation_provider import DelegationRouteError
except ImportError:

    class DelegationRouteError(RuntimeError):  # type: ignore[no-redef]
        """Fallback exception when running unit tests outside Hermes."""

    class BaseDelegationProvider:  # type: ignore[no-redef]
        """Minimal compatible base class for standalone unit tests."""

        display_name: str = ""
        summary: str = ""

        @property
        def name(self) -> str:
            raise NotImplementedError

        def settings(self, config: dict[str, Any] | None) -> dict[str, Any]:
            if not isinstance(config, dict):
                return {}
            backends = config.get("backends")
            if not isinstance(backends, dict):
                return {}
            own = backends.get(self.name)
            return dict(own) if isinstance(own, dict) else {}

        def is_available(self) -> bool:
            return True
