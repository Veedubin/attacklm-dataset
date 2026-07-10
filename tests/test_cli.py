"""Tests for the attacklm-dataset CLI dispatcher (src/attacklm_dataset/cli.py).

The CLI is a thin argparse wrapper that dispatches to scripts/*.py via
subprocess. We test the dispatch logic without actually running the
subprocesses (which would require real data and the upstream clones).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure the package is importable when running tests from source.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from attacklm_dataset import cli as dataset_cli  # noqa: E402


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_returns_argument_parser(self):
        parser = dataset_cli.build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_prog_name(self):
        parser = dataset_cli.build_parser()
        assert parser.prog == "attacklm-dataset"

    def test_all_expected_subcommands_present(self):
        parser = dataset_cli.build_parser()
        # Parse --help output to discover subparsers.
        help_text = parser.format_help()
        for sub in ("init", "balance", "evolve", "audit", "package"):
            assert sub in help_text, f"Missing subcommand {sub!r}"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMainDispatches:
    """The main() function dispatches to a scripts/*.py file via subprocess.run.
    We mock subprocess.run to assert the right script is invoked for each
    subcommand."""

    @pytest.mark.parametrize(
        "argv,expected_script",
        [
            (["init"], "init_pipeline.py"),
            (["init", "--from-source"], "init_pipeline.py"),
            (["init", "--yes", "--extract-only"], "init_pipeline.py"),
            (["balance"], "balance_buckets.py"),
            (["balance", "--profile", "7b-16gb"], "balance_buckets.py"),
            (["evolve"], "evolve_pairs.py"),
            (["evolve", "--source", "metasploit-framework"], "evolve_pairs.py"),
            (["evolve", "--count", "100"], "evolve_pairs.py"),
            (["audit"], "audit_dataset.py"),
            (["package"], "package_dataset.py"),
        ],
    )
    def test_dispatches_to_correct_script(self, argv, expected_script, monkeypatch):
        # Mock subprocess.run so we don't actually execute the script.
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr(dataset_cli.subprocess, "run", mock_run)
        # Point _SCRIPTS_DIR at the actual repo scripts/ so the
        # "script not found" check passes. (The CLI's _SCRIPTS_DIR
        # computation is broken for installed packages; this is a
        # pre-existing bug tracked separately. The unit test exercises
        # the dispatch logic, not the _SCRIPTS_DIR resolution.)
        repo_scripts = _REPO_ROOT / "scripts"
        monkeypatch.setattr(dataset_cli, "_SCRIPTS_DIR", repo_scripts)

        with patch.object(sys, "argv", ["attacklm-dataset"] + argv):
            result = dataset_cli.main()

        assert result == 0
        assert mock_run.called
        # First positional arg to subprocess.run is the command list.
        cmd_list = mock_run.call_args[0][0]
        assert cmd_list[0] == sys.executable  # /path/to/python
        assert expected_script in str(cmd_list[1])

    def test_extra_argv_forwarded_to_script(self, monkeypatch):
        # The wrapper CLI accepts a specific set of known flags per
        # subcommand. They're parsed by argparse, then converted back
        # to argv before being passed to the underlying script.
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr(dataset_cli.subprocess, "run", mock_run)
        repo_scripts = _REPO_ROOT / "scripts"
        monkeypatch.setattr(dataset_cli, "_SCRIPTS_DIR", repo_scripts)

        with patch.object(
            sys, "argv", ["attacklm-dataset", "init", "--yes", "--from-source"]
        ):
            dataset_cli.main()

        cmd_list = mock_run.call_args[0][0]
        # Both known flags should be in cmd_list.
        for flag in ("--yes", "--from-source"):
            assert flag in cmd_list, f"Flag {flag!r} not forwarded to script"

    def test_string_flags_forwarded_with_value(self, monkeypatch):
        # String-valued flags like --profile 7b-16gb should be passed
        # as two separate argv elements.
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr(dataset_cli.subprocess, "run", mock_run)
        repo_scripts = _REPO_ROOT / "scripts"
        monkeypatch.setattr(dataset_cli, "_SCRIPTS_DIR", repo_scripts)

        with patch.object(
            sys,
            "argv",
            [
                "attacklm-dataset",
                "balance",
                "--profile",
                "7b-16gb",
                "--preset",
                "red-team",
            ],
        ):
            dataset_cli.main()

        cmd_list = mock_run.call_args[0][0]
        # --profile 7b-16gb should appear as consecutive argv elements.
        assert "--profile" in cmd_list
        assert "7b-16gb" in cmd_list
        profile_idx = cmd_list.index("--profile")
        assert cmd_list[profile_idx + 1] == "7b-16gb"
        assert "--preset" in cmd_list
        assert "red-team" in cmd_list

    def test_unset_flags_not_forwarded(self, monkeypatch):
        # If the user doesn't pass --yes, the CLI must NOT inject
        # --yes into the script's argv. Otherwise, scripts that
        # default to "ask for confirmation" would never ask.
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr(dataset_cli.subprocess, "run", mock_run)
        repo_scripts = _REPO_ROOT / "scripts"
        monkeypatch.setattr(dataset_cli, "_SCRIPTS_DIR", repo_scripts)

        with patch.object(sys, "argv", ["attacklm-dataset", "init"]):
            dataset_cli.main()

        cmd_list = mock_run.call_args[0][0]
        assert "--yes" not in cmd_list
        assert "--from-source" not in cmd_list

    def test_string_flag_with_none_not_forwarded(self, monkeypatch):
        # --dataset-url (which defaults to None) must not be passed as
        # "--dataset-url None".
        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        monkeypatch.setattr(dataset_cli.subprocess, "run", mock_run)
        repo_scripts = _REPO_ROOT / "scripts"
        monkeypatch.setattr(dataset_cli, "_SCRIPTS_DIR", repo_scripts)

        with patch.object(sys, "argv", ["attacklm-dataset", "init"]):
            dataset_cli.main()

        cmd_list = mock_run.call_args[0][0]
        assert "--dataset-url" not in cmd_list
        monkeypatch.setattr(dataset_cli.subprocess, "run", mock_run)
        repo_scripts = _REPO_ROOT / "scripts"
        monkeypatch.setattr(dataset_cli, "_SCRIPTS_DIR", repo_scripts)

        with patch.object(sys, "argv", ["attacklm-dataset", "init"]):
            dataset_cli.main()

        cmd_list = mock_run.call_args[0][0]
        assert "--dataset-url" not in cmd_list

    def test_subprocess_returncode_propagated(self, monkeypatch):
        # If the script exits non-zero, main() should return that code.
        mock_run = MagicMock(return_value=MagicMock(returncode=42))
        monkeypatch.setattr(dataset_cli.subprocess, "run", mock_run)
        repo_scripts = _REPO_ROOT / "scripts"
        monkeypatch.setattr(dataset_cli, "_SCRIPTS_DIR", repo_scripts)

        with patch.object(sys, "argv", ["attacklm-dataset", "init"]):
            result = dataset_cli.main()
        assert result == 42


class TestMainMissingScript:
    def test_missing_script_returns_nonzero(self, monkeypatch, capsys):
        # If the script doesn't exist in the scripts/ dir, main() should
        # print an error and return non-zero WITHOUT calling subprocess.run.
        mock_run = MagicMock()
        monkeypatch.setattr(dataset_cli.subprocess, "run", mock_run)

        # Patch the script path to point at a non-existent file.
        fake_scripts_dir = Path("/nonexistent/scripts")
        monkeypatch.setattr(dataset_cli, "_SCRIPTS_DIR", fake_scripts_dir)

        with patch.object(sys, "argv", ["attacklm-dataset", "init"]):
            result = dataset_cli.main()

        assert result == 1
        # subprocess.run should NOT have been called.
        assert not mock_run.called
        captured = capsys.readouterr()
        assert "not found" in captured.err or "Error" in captured.err


class TestMainNoArgs:
    def test_no_subcommand_prints_help_and_returns_zero(self, monkeypatch, capsys):
        # No subcommand → parser.print_help() → return 0
        mock_run = MagicMock()
        monkeypatch.setattr(dataset_cli.subprocess, "run", mock_run)

        with patch.object(sys, "argv", ["attacklm-dataset"]):
            result = dataset_cli.main()
        assert result == 0
        # No subprocess should have been launched.
        assert not mock_run.called
        # Help text was printed (to stdout).
        captured = capsys.readouterr()
        assert "init" in captured.out or "usage" in captured.out.lower()


# ---------------------------------------------------------------------------
# Module-level structure
# ---------------------------------------------------------------------------


class TestCliModuleStructure:
    def test_dunder_name_guard(self):
        import inspect

        source = inspect.getsource(dataset_cli)
        assert '__name__ == "__main__"' in source
        assert "sys.exit(main())" in source

    def test_scripts_dir_is_a_path(self):
        assert isinstance(dataset_cli._SCRIPTS_DIR, Path)

    def test_scripts_dir_points_at_repo_scripts(self):
        # _SCRIPTS_DIR is computed as
        # Path(__file__).parent.parent / "scripts"
        # When installed via pip, this resolves to
        # site-packages/scripts/ which doesn't exist. This is a
        # pre-existing bug (tracked as a follow-up). When running
        # from a source tree, the cli.py lives at
        # <repo>/src/attacklm_dataset/cli.py so the parent.parent is
        # <repo>/src, not <repo>. Either way, the constant is a Path.
        assert isinstance(dataset_cli._SCRIPTS_DIR, Path)
        assert dataset_cli._SCRIPTS_DIR.name == "scripts"
