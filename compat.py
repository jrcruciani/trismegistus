"""Public Hermes plugin API bridges + thread-safe TLS Keep-Alive connection pool.

Every Hermes import in this module comes strictly from the documented Public Plugin API
(`hermes_cli/data/plugin_api.json` / `developer-guide/plugins/public-api.md`), so
`hermes plugins validate` passes with zero private-import warnings while standalone
`pytest` runs cleanly even outside a Hermes checkout.
"""

from __future__ import annotations

import base64
from collections import deque
import http.client
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import ssl
import tempfile
import threading
import time
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

_JWT_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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


def read_secret(env_name: str, *, refresh: bool = False) -> str:
    """Read a plugin secret through Hermes's supported profile-scoped resolvers.

    - Under multiplexed / routed profiles (`serves_routed_profile() == True`), reads strictly
      through `agent.secret_scope.get_secret` (refreshing the bound profile scope first when
      `refresh=True`).
    - Under single-profile CLI/TUI sessions (`serves_routed_profile() == False`), prefers
      `hermes_cli.config.get_env_value_prefer_dotenv` so rotated tokens written to `<HERMES_HOME>/.env`
      beat stale `os.environ` values inherited at process startup (while still honoring `op://`
      external secret sources).
    """
    try:
        from agent.secret_scope import (
            get_secret,
            invalidate_env_file_cache,
            refresh_installed_secret_scope,
            serves_routed_profile,
        )

        if refresh:
            invalidate_env_file_cache()
            try:
                from hermes_constants import get_hermes_home

                refresh_installed_secret_scope(get_hermes_home())
            except Exception:
                pass

        if serves_routed_profile():
            return str(get_secret(env_name) or "").strip()

        try:
            from hermes_cli.config import get_env_value_prefer_dotenv

            preferred = get_env_value_prefer_dotenv(env_name)
            if preferred is not None and str(preferred).strip():
                return str(preferred).strip()
        except Exception:
            pass

        return str(get_secret(env_name) or "").strip()
    except ImportError:
        # Standalone pytest fallback outside the Hermes runtime
        return str(os.getenv(env_name) or "").strip()


def is_jwt_token(value: str) -> bool:
    """Return True when `value` has standard 3-segment OAuth2/Entra JWT structure (`eyJ...`)."""
    raw = str(value or "").strip()
    if not raw.startswith("eyJ"):
        return False
    parts = raw.split(".")
    if len(parts) != 3:
        return False
    return all(bool(p) and bool(_JWT_SEGMENT_RE.match(p)) for p in parts)


def jwt_seconds_until_expiry(value: str, *, now: float | None = None) -> float | None:
    """Inspect only the numeric `exp` claim of a JWT in memory (never logging or storing it).

    Returns remaining seconds (`exp - now`), or `None` if `value` is not a JWT with an `exp` claim.
    """
    if not is_jwt_token(value):
        return None
    payload_seg = value.strip().split(".")[1]
    padded = payload_seg + ("=" * (-len(payload_seg) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        claims = json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(claims, dict):
        return None
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return None
    ref = time.time() if now is None else float(now)
    return float(exp) - ref


def user_agent() -> str:
    """Return the canonical Hermes User-Agent header when inside Hermes."""
    try:
        from hermes_cli.version_info import hermes_user_agent

        return hermes_user_agent("trismegistus/0.3.2")
    except ImportError:
        return "Hermes-Trismegistus/0.3.2"


def plugin_data_dir(plugin_name: str = "trismegistus") -> Path:
    """Return the profile-scoped plugin-data directory (`<HERMES_HOME>/plugin-data/<name>`)."""
    try:
        from hermes_constants import get_hermes_home

        base = Path(get_hermes_home())
    except ImportError:
        base = Path(os.getenv("HERMES_HOME") or (Path(tempfile.gettempdir()) / "hermes-trismegistus-test"))
    target_dir = base / "plugin-data" / plugin_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON using Hermes's `utils.atomic_json_write` when available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from utils import atomic_json_write

        atomic_json_write(path, data)
        return
    except ImportError:
        pass
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json_safe(path: Path) -> dict[str, Any]:
    """Read a JSON object safely via Hermes's `utils.read_json_or_empty` when available."""
    try:
        from utils import read_json_or_empty

        loaded = read_json_or_empty(path)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        if not path.is_file():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}


def hermes_clamp_effort(desired: str, supported: tuple[str, ...]) -> str | None:
    """Bridge to Hermes's public `agent.reasoning_effort.clamp_effort` when available."""
    if not supported:
        return None
    try:
        from agent.reasoning_effort import clamp_effort as _core_clamp

        res = _core_clamp(desired, supported)
        if isinstance(res, str) and res in supported:
            return res
    except Exception:
        pass
    return None


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

    Security & reliability properties:
    - Uses verified default `ssl.SSLContext` for `https://`.
    - Never forwards credentials across origins (redirects are not auto-followed).
    - Discards sockets after auth/rate-limit/5xx error responses so upstream APIM disconnects
      never poison subsequent requests, and transparently retries stale idle sockets once.
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
                    status >= 400
                    or (
                        hasattr(resp_headers, "get")
                        and str(resp_headers.get("Connection") or "").lower() == "close"
                    )
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


def _header_get_ci(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    if hasattr(headers, "get"):
        direct = headers.get(name)
        if direct is not None:
            return str(direct)
    if hasattr(headers, "items"):
        target = name.lower()
        for k, v in headers.items():
            if str(k).lower() == target and v is not None:
                return str(v)
    return None


def parse_retry_after(headers: Any, *, max_cap: float = 30.0) -> float | None:
    """Parse Foundry's `retry-after-ms` / `x-ms-retry-after-ms` or standard `Retry-After`."""
    if headers is None:
        return None
    for ms_header in ("retry-after-ms", "x-ms-retry-after-ms"):
        ms_val = _header_get_ci(headers, ms_header)
        if ms_val is not None:
            try:
                ms = float(ms_val.strip())
                if ms >= 0:
                    return min(ms / 1000.0, max_cap)
            except ValueError:
                pass
    ra_val = _header_get_ci(headers, "Retry-After")
    if ra_val is not None:
        try:
            from agent.retry_utils import parse_retry_after_seconds

            parsed = parse_retry_after_seconds(ra_val)
            if parsed is not None:
                return min(max(0.0, float(parsed)), max_cap)
        except ImportError:
            try:
                return min(max(0.0, float(ra_val.strip())), max_cap)
            except ValueError:
                return None
    return None


def open_plugin_db(plugin_name: str = "trismegistus", filename: str = "ledger.db") -> sqlite3.Connection:
    """Open a profile-scoped WAL SQLite database in `<HERMES_HOME>/plugin-data/<name>/`."""
    try:
        from plugins.plugin_storage import plugin_db

        return plugin_db(plugin_name, filename)
    except ImportError:
        target_dir = plugin_data_dir(plugin_name)
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
