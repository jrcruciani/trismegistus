"""Pytest bootstrap loading the root directory as package `trismegistus_pkg`."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_trismegistus_package():
    pkg_name = "trismegistus_pkg"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(ROOT / "__init__.py"),
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(ROOT)]
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def pkg():
    return load_trismegistus_package()


@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path
