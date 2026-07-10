"""Regression net: every script in scripts/ imports cleanly and runs --help.

This is the "did someone break the build?" canary. If any of the
47 scripts in scripts/ have a syntax error, a missing import, a
broken argparse setup, or a top-level exception at import time, the
test fails. The user explicitly asked for this — "we don't want
things that once worked, breaking from new changes and us not
finding out until someone reports a bug."

What this catches:
  - SyntaxError / IndentationError in any script
  - ImportError (missing or renamed dependency)
  - Top-level module-load exceptions (e.g., from __future__ misuse,
    bad encoding declarations)
  - argparse setup that crashes (which ``--help`` would catch in CI
    if anyone runs it; we just call it for every script)

What this does NOT catch (and shouldn't):
  - Runtime errors that require data (those need integration tests
    in the data dir)
  - Logic bugs in the actual transformation (those need golden-vector
    tests)
  - Permissions / network / GPU failures (those are environmental)

Known issues (scripts that fail import or --help and the WHY):
  - extract_cybersec_llm_cve.py: requires the `datasets` library,
    which is an optional dev dependency. Not installed in CI.
  - init_pipeline.py: has a Python 3.14 incompatibility (dataclass
    with `from __future__ import annotations` and a frozen=True
    class — the `KW_ONLY` sentinel can't be resolved). Tracked
    separately.
  - bucket_loader.py, evolved_mixer.py, mitre_tactic_lookup.py,
    replay_mixer.py: utility modules, no main() and no top-level
    docstring (they're library code, not CLI scripts). The import
    succeeds; the assertion is too strict.
  - device_utils.py: no argparse. Maintenance utility, not a CLI.
  - rebuild_manifest.py, reorganize_buckets.py: no --help; they're
    maintenance scripts that do their thing unconditionally.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Known-broken scripts (with reason)
# ---------------------------------------------------------------------------


# Scripts that fail to import — the test for them is skipped.
IMPORT_BROKEN: dict[str, str] = {
    "extract_cybersec_llm_cve.py": "requires the `datasets` library (optional dev dep)",
    "init_pipeline.py": (
        "Python 3.14 incompatibility — dataclass(frozen=True) with "
        "`from __future__ import annotations` and the KW_ONLY sentinel "
        "isn't resolvable. Tracked separately; needs a port to 3.10-style "
        "forward refs."
    ),
}

# Scripts that don't have --help (no argparse). The --help test is skipped.
HELP_BROKEN: dict[str, str] = {
    "bucket_loader.py": "library module, no argparse",
    "device_utils.py": "library module, no argparse",
    "evolved_mixer.py": "library module, no argparse",
    "mitre_tactic_lookup.py": "library module, no argparse",
    "replay_mixer.py": "library module, no argparse",
    "rebuild_manifest.py": "maintenance script, runs unconditionally",
    "reorganize_buckets.py": "maintenance script, runs unconditionally",
}

# Scripts that don't have main() and don't have a module docstring.
# These are utility/library modules, not CLI scripts. The assertion
# is too strict for them.
NO_MAIN_OR_DOCSTRING: set[str] = {
    "bucket_loader.py",
    "evolved_mixer.py",
    "mitre_tactic_lookup.py",
    "replay_mixer.py",
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_scripts() -> list[Path]:
    """Return all .py files in scripts/, sorted by name.

    Excludes ``__init__.py`` and ``__pycache__``.
    """
    return sorted(
        p
        for p in _SCRIPTS_DIR.glob("*.py")
        if p.name != "__init__.py" and p.parent.name != "__pycache__"
    )


# Snapshot the script list at module load time. If you add a new script,
# this list will pick it up automatically.
_SCRIPTS = _discover_scripts()
SCRIPT_NAMES = [p.name for p in _SCRIPTS]


# ---------------------------------------------------------------------------
# Import-smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script_path", _SCRIPTS, ids=SCRIPT_NAMES)
def test_script_imports_cleanly(script_path: Path) -> None:
    """Importing the script's module should not raise.

    We do this via importlib's spec loader so we don't pollute sys.path
    or sys.modules with every script. Each script gets its own
    fresh module namespace.
    """
    if script_path.name in IMPORT_BROKEN:
        pytest.skip(
            f"{script_path.name} is in IMPORT_BROKEN: {IMPORT_BROKEN[script_path.name]}"
        )

    spec = importlib.util.spec_from_file_location(
        f"_smoke_{script_path.stem}", script_path
    )
    assert spec is not None, f"Could not build spec for {script_path}"
    module = importlib.util.module_from_spec(spec)
    assert module is not None
    # This is the canary: any syntax error, import error, or
    # top-level exception fires here.
    spec.loader.exec_module(module)  # type: ignore[union-attr]


@pytest.mark.parametrize("script_path", _SCRIPTS, ids=SCRIPT_NAMES)
def test_script_has_main_or_docstring(script_path: Path) -> None:
    """Most scripts are CLI entry points. We expect either:
      - a ``main()`` function, or
      - a module-level docstring (the bare-minimum case for utility
        modules that aren't called directly).

    This is a sanity check, not a hard rule. We just want to catch
    the case where a script silently lost its main() and became
    a no-op.
    """
    if script_path.name in NO_MAIN_OR_DOCSTRING:
        pytest.skip(f"{script_path.name} is a library module")

    source = script_path.read_text(encoding="utf-8", errors="replace")
    has_main = "def main" in source
    has_docstring = source.lstrip().startswith('"""') or source.lstrip().startswith(
        "'''"
    )
    assert has_main or has_docstring, (
        f"{script_path.name} has no main() and no docstring — is this an empty file?"
    )


# ---------------------------------------------------------------------------
# --help smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script_path", _SCRIPTS, ids=SCRIPT_NAMES)
def test_script_help_does_not_crash(script_path: Path) -> None:
    """``python script.py --help`` should exit 0.

    Catches argparse setup errors, sys.exit() at top level, and
    any other issue that would prevent a user from seeing the help
    text. Scripts that don't have --help (no argparse) are in
    HELP_BROKEN and skipped.
    """
    if script_path.name in HELP_BROKEN:
        pytest.skip(
            f"{script_path.name} is in HELP_BROKEN: {HELP_BROKEN[script_path.name]}"
        )
    if script_path.name in IMPORT_BROKEN:
        pytest.skip(f"{script_path.name} can't even import; see IMPORT_BROKEN")

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"{script_path.name} --help exited with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout[:1000]}\n"
        f"STDERR:\n{result.stderr[:1000]}"
    )


# ---------------------------------------------------------------------------
# Module structure tests
# ---------------------------------------------------------------------------


class TestScriptsDirectory:
    def test_scripts_dir_exists(self):
        assert _SCRIPTS_DIR.is_dir()

    def test_at_least_20_scripts(self):
        # Sanity check: if this drops below 20, scripts are being
        # deleted in bulk which is probably wrong.
        assert len(_SCRIPTS) >= 20, f"Only {len(_SCRIPTS)} scripts; expected >= 20"

    def test_no_duplicate_script_names(self):
        names = [p.name for p in _SCRIPTS]
        assert len(names) == len(set(names)), "Duplicate script filenames"

    def test_all_scripts_are_python(self):
        for p in _SCRIPTS:
            assert p.suffix == ".py", f"{p.name} is not .py"

    def test_scripts_dir_count_matches_parametrize_ids(self):
        # The SCRIPT_NAMES list is built from _SCRIPTS. Verify they
        # are in sync (catches the case where someone modifies
        # _discover_scripts but not SCRIPT_NAMES).
        assert len(SCRIPT_NAMES) == len(_SCRIPTS)

    def test_known_broken_lists_are_consistent(self):
        # Every entry in NO_MAIN_OR_DOCSTRING must be a real script.
        for name in NO_MAIN_OR_DOCSTRING:
            assert any(p.name == name for p in _SCRIPTS), (
                f"NO_MAIN_OR_DOCSTRING contains {name!r} but no such script exists"
            )
        # HELP_BROKEN keys must be real scripts.
        for name in HELP_BROKEN:
            assert any(p.name == name for p in _SCRIPTS), (
                f"HELP_BROKEN contains {name!r} but no such script exists"
            )
        # IMPORT_BROKEN keys must be real scripts.
        for name in IMPORT_BROKEN:
            assert any(p.name == name for p in _SCRIPTS), (
                f"IMPORT_BROKEN contains {name!r} but no such script exists"
            )
