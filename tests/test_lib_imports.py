"""Tests for library modules in scripts/lib/.

These modules were moved out of scripts/ so that test_scripts_smoke.py
can enforce a clean contract ("every .py in scripts/ has --help").
This file verifies that they still import and expose their expected
public API.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LIB_DIR = _REPO_ROOT / "scripts" / "lib"


def _discover_lib_modules() -> list[Path]:
    """Return all .py files in scripts/lib/, sorted by name.

    Excludes ``__init__.py`` and ``__pycache__``.
    """
    return sorted(
        p
        for p in _LIB_DIR.glob("*.py")
        if p.name != "__init__.py" and p.parent.name != "__pycache__"
    )


_LIB_MODULES = _discover_lib_modules()
LIB_NAMES = [p.stem for p in _LIB_MODULES]


# ---------------------------------------------------------------------------
# Import and API tests
# ---------------------------------------------------------------------------


class TestLibDirectory:
    """Verify scripts/lib/ structure."""

    def test_lib_dir_exists(self):
        assert _LIB_DIR.is_dir(), f"{_LIB_DIR} is not a directory"

    def test_init_py_exists(self):
        init = _LIB_DIR / "__init__.py"
        assert init.is_file(), f"{init} does not exist"

    def test_at_least_3_lib_modules(self):
        # We moved 5 modules; if this drops below 3, something is wrong.
        assert len(_LIB_MODULES) >= 3, (
            f"Only {len(_LIB_MODULES)} lib modules; expected >= 3"
        )


@pytest.mark.parametrize("module_path", _LIB_MODULES, ids=LIB_NAMES)
def test_lib_module_imports_cleanly(module_path: Path) -> None:
    """Each library module should import without errors."""
    # Insert scripts/lib/ into sys.path so the module can find its
    # sibling imports (e.g., mitre_tactic_lookup might import bucket_loader).
    sys_path_backup = sys.path.copy()
    sys.path.insert(0, str(module_path.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            f"_lib_{module_path.stem}", module_path
        )
        assert spec is not None, f"Could not build spec for {module_path}"
        module = importlib.util.module_from_spec(spec)
        assert module is not None
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        finally:
            sys.modules.pop(spec.name, None)
    finally:
        sys.path[:] = sys_path_backup


def test_mitre_tactic_lookup_api() -> None:
    """mitre_tactic_lookup should expose get_tactic_for_technique."""
    sys_path_backup = sys.path.copy()
    sys.path.insert(0, str(_LIB_DIR))
    try:
        from mitre_tactic_lookup import get_tactic_for_technique, get_tactic_name

        assert callable(get_tactic_for_technique)
        assert callable(get_tactic_name)
    finally:
        sys.path[:] = sys_path_backup


def test_bucket_loader_api() -> None:
    """bucket_loader should expose list_buckets."""
    sys_path_backup = sys.path.copy()
    sys.path.insert(0, str(_LIB_DIR))
    try:
        from bucket_loader import list_buckets

        assert callable(list_buckets)
    finally:
        sys.path[:] = sys_path_backup


def test_device_utils_api() -> None:
    """device_utils should expose hardware helper functions."""
    sys_path_backup = sys.path.copy()
    sys.path.insert(0, str(_LIB_DIR))
    try:
        import device_utils

        # device_utils exposes hardware detection and memory helpers
        assert hasattr(device_utils, "is_cuda") or hasattr(device_utils, "gpu_mem_info")
    finally:
        sys.path[:] = sys_path_backup


def test_evolved_mixer_api() -> None:
    """evolved_mixer should expose discover_evolved_files or mix_evolved."""
    sys_path_backup = sys.path.copy()
    sys.path.insert(0, str(_LIB_DIR))
    try:
        from evolved_mixer import discover_evolved_files, mix_evolved

        assert callable(discover_evolved_files)
        assert callable(mix_evolved)
    finally:
        sys.path[:] = sys_path_backup


def test_replay_mixer_api() -> None:
    """replay_mixer should expose discover_replay_files or mix_replay."""
    sys_path_backup = sys.path.copy()
    sys.path.insert(0, str(_LIB_DIR))
    try:
        from replay_mixer import discover_replay_files, mix_replay

        assert callable(discover_replay_files)
        assert callable(mix_replay)
    finally:
        sys.path[:] = sys_path_backup
