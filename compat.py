"""Public Hermes plugin API bridges with clean standalone fallbacks for unit tests.

Every Hermes import in this module comes strictly from the documented Public Plugin API
(`hermes_cli/data/plugin_api.json` / `developer-guide/plugins/public-api.md`), so
`hermes plugins validate` passes with zero private-import warnings while standalone
`pytest` runs cleanly even outside a Hermes checkout.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sqlite3
import tempfile
import urllib.request
from typing import Any, Callable

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
            out = pat.sub(lambda m: (m.group(1) if m.lastindex and m.lastindex >= 1 and m.group(1).endswith((" ", "=", ":")) else "") + "[REDACTED]", out)
        return out


def read_secret(env_name: str) -> str:
    """Read a plugin secret through Hermes's profile-scoped secret resolver."""
    try:
        from agent.secret_scope import get_secret

        val = get_secret(env_name)
        return str(val or "").strip()
    except ImportError:
        # Standalone pytest fallback outside Hermes runtime
        return str(os.getenv(env_name) or "").strip()


def user_agent() -> str:
    """Return the canonical Hermes User-Agent header when inside Hermes."""
    try:
        from hermes_cli.version_info import hermes_user_agent

        return hermes_user_agent("trismegistus/0.1.0")
    except ImportError:
        return "Hermes-Trismegistus/0.1.0"


def open_safe_url(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener_factory: Callable[..., Any] | None = None,
):
    """Send an HTTP request without forwarding credentials across cross-origin redirects."""
    try:
        from hermes_cli.urllib_security import open_credentialed_url

        return open_credentialed_url(request, timeout=timeout, opener_factory=opener_factory)
    except ImportError:
        if opener_factory is not None:
            opener = opener_factory(urllib.request.HTTPRedirectHandler())
            return opener.open(request, timeout=timeout)
        return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - standalone test fallback


def parse_retry_after(headers: Any) -> float | None:
    """Parse Foundry's non-standard `retry-after-ms` or standard `Retry-After`."""
    if headers is None:
        return None
    ms_val = headers.get("retry-after-ms") if hasattr(headers, "get") else None
    if ms_val is not None:
        try:
            ms = float(str(ms_val).strip())
            if ms >= 0:
                return min(ms / 1000.0, 10.0)
        except ValueError:
            pass
    ra_val = headers.get("Retry-After") if hasattr(headers, "get") else None
    if ra_val is not None:
        try:
            from agent.retry_utils import parse_retry_after_seconds

            parsed = parse_retry_after_seconds(str(ra_val))
            if parsed is not None:
                return min(float(parsed), 10.0)
        except ImportError:
            try:
                return min(float(str(ra_val).strip()), 10.0)
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
