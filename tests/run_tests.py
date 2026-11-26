"""Zero-dependency test runner (works both standalone and inside Hermes's bundled Python)."""

from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if str(ROOT / "tests") not in sys.path:
        sys.path.insert(0, str(ROOT / "tests"))
    os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="trismegistus_test_")
    spec = importlib.util.spec_from_file_location(
        "trismegistus_pkg",
        str(ROOT / "__init__.py"),
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [str(ROOT)]
    sys.modules["trismegistus_pkg"] = pkg
    spec.loader.exec_module(pkg)

    passed = 0
    for test_file in sorted((ROOT / "tests").glob("test_*.py")):
        os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="trismegistus_case_")
        tspec = importlib.util.spec_from_file_location(test_file.stem, str(test_file))
        assert tspec is not None and tspec.loader is not None
        tmod = importlib.util.module_from_spec(tspec)
        sys.modules[test_file.stem] = tmod
        tspec.loader.exec_module(tmod)
        for name, fn in inspect.getmembers(tmod, inspect.isfunction):
            if name.startswith("test_"):
                os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="trismegistus_fn_")
                pkg.backends.clear_backend_caches()
                fn(pkg)
                passed += 1
                print(f"✓ {test_file.name}::{name}")

    print(f"\nOK: {passed} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
