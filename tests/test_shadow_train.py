"""Tests for scripts/inversion/shadow_train (LiRA shadow-model loss aggregation).

The shadow_train module is the orchestration layer for LiRA. It does NOT
train models (that's the user's job), but it does:
  - Load K shadow-model loss files
  - Build per-record IN/OUT loss lists from shadow membership files
  - Fit per-record Gaussians
  - Save the parameters

We test the pure functions: load_shadow_losses and build_in_out_losses.
The main() orchestrator is exercised end-to-end with a small fixture
of synthetic shadow loss files.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

# Make scripts/ importable (shadow_train does the same).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from inversion import shadow_train  # noqa: E402


# ---------------------------------------------------------------------------
# load_shadow_losses
# ---------------------------------------------------------------------------


class TestLoadShadowLosses:
    def test_loads_single_shadow(self, tmp_path: Path):
        # One shadow file.
        (tmp_path / "shadow_0.json").write_text(
            json.dumps({"rec-1": 0.5, "rec-2": 0.7})
        )
        result = shadow_train.load_shadow_losses(tmp_path)
        assert 0 in result
        assert result[0] == {"rec-1": 0.5, "rec-2": 0.7}

    def test_loads_multiple_shadows(self, tmp_path: Path):
        # Three shadow files.
        for k in range(3):
            (tmp_path / f"shadow_{k}.json").write_text(
                json.dumps({f"rec-{k}": float(k)})
            )
        result = shadow_train.load_shadow_losses(tmp_path)
        assert set(result.keys()) == {0, 1, 2}
        assert result[0] == {"rec-0": 0.0}
        assert result[2] == {"rec-2": 2.0}

    def test_empty_dir_returns_empty_dict(self, tmp_path: Path):
        result = shadow_train.load_shadow_losses(tmp_path)
        assert result == {}

    def test_skips_malformed_filenames(self, tmp_path: Path, caplog):
        # Files that don't match shadow_<int>.json are skipped with a
        # warning. We add a bad file alongside a good one.
        (tmp_path / "shadow_0.json").write_text(json.dumps({"rec": 1.0}))
        (tmp_path / "shadow_abc.json").write_text(json.dumps({"rec": 2.0}))
        (tmp_path / "shadow_.json").write_text(json.dumps({}))
        (tmp_path / "not-a-shadow.json").write_text(json.dumps({}))
        with caplog.at_level(logging.WARNING):
            result = shadow_train.load_shadow_losses(tmp_path)
        # Only shadow_0 is loaded.
        assert list(result.keys()) == [0]
        # Warnings were logged for the malformed files.
        assert any("malformed" in r.message for r in caplog.records)

    def test_loads_shadow_with_larger_index(self, tmp_path: Path):
        # 2-digit shadow index.
        (tmp_path / "shadow_42.json").write_text(json.dumps({"rec": 0.1}))
        result = shadow_train.load_shadow_losses(tmp_path)
        assert 42 in result

    def test_loads_shadow_floats_are_preserved(self, tmp_path: Path):
        # Float values pass through unchanged.
        (tmp_path / "shadow_0.json").write_text(json.dumps({"rec": 0.123456789}))
        result = shadow_train.load_shadow_losses(tmp_path)
        assert result[0]["rec"] == 0.123456789


# ---------------------------------------------------------------------------
# build_in_out_losses
# ---------------------------------------------------------------------------


class TestBuildInOutLosses:
    def test_basic_in_record(self):
        # One shadow, one IN record, one OUT record.
        shadow_losses = {0: {"rec-in": 0.5, "rec-out": 0.7}}
        in_records = [{"shadow_k": 0, "record_id": "rec-in"}]
        out_records = [{"shadow_k": 0, "record_id": "rec-out"}]
        in_losses, out_losses = shadow_train.build_in_out_losses(
            shadow_losses, in_records, out_records
        )
        assert in_losses == {"rec-in": [0.5]}
        assert out_losses == {"rec-out": [0.7]}

    def test_record_in_multiple_shadows(self):
        # Same record_id in shadows 0, 1, 2 → list of 3 losses.
        shadow_losses = {
            0: {"rec-x": 0.1},
            1: {"rec-x": 0.2},
            2: {"rec-x": 0.3},
        }
        in_records = [
            {"shadow_k": 0, "record_id": "rec-x"},
            {"shadow_k": 1, "record_id": "rec-x"},
            {"shadow_k": 2, "record_id": "rec-x"},
        ]
        in_losses, out_losses = shadow_train.build_in_out_losses(
            shadow_losses, in_records, []
        )
        assert in_losses == {"rec-x": [0.1, 0.2, 0.3]}
        assert out_losses == {}

    def test_missing_shadow_k_skipped(self):
        # in_records references shadow_k=99 but only shadow_0 exists.
        shadow_losses = {0: {"rec-a": 0.5}}
        in_records = [
            {"shadow_k": 0, "record_id": "rec-a"},
            {"shadow_k": 99, "record_id": "rec-b"},
        ]
        in_losses, _ = shadow_train.build_in_out_losses(shadow_losses, in_records, [])
        # rec-a is in, rec-b is silently skipped.
        assert in_losses == {"rec-a": [0.5]}

    def test_missing_record_id_skipped(self):
        # Shadow exists but the record_id is not in its loss dict.
        shadow_losses = {0: {"rec-a": 0.5}}
        in_records = [
            {"shadow_k": 0, "record_id": "rec-a"},
            {"shadow_k": 0, "record_id": "rec-missing"},
        ]
        in_losses, _ = shadow_train.build_in_out_losses(shadow_losses, in_records, [])
        assert in_losses == {"rec-a": [0.5]}

    def test_empty_inputs(self):
        in_losses, out_losses = shadow_train.build_in_out_losses({}, [], [])
        assert in_losses == {}
        assert out_losses == {}


# ---------------------------------------------------------------------------
# main() integration test
# ---------------------------------------------------------------------------


class TestShadowTrainMain:
    """End-to-end: write a small fixture of shadow loss files + IN/OUT
    membership files, invoke main(), and assert the output is a valid
    shadow_params.json."""

    def test_end_to_end_produces_shadow_params(self, tmp_path: Path, caplog):
        # 3 shadows, 2 records. The LiRA convention is that each
        # record is IN for some shadows and OUT for others (each
        # shadow trains on a different subset). Set up the membership
        # so that rec-0 is IN for shadow 0, OUT for shadows 1, 2;
        # rec-1 is IN for shadow 1, OUT for shadows 0, 2.
        for k in range(3):
            (tmp_path / f"shadow_{k}.json").write_text(
                json.dumps({f"rec-{i}": 0.1 * k + 0.01 * i for i in range(2)})
            )
        in_records = [
            # rec-0: IN for shadow 0, OUT for shadows 1, 2
            {"shadow_k": 0, "record_id": "rec-0"},
            # rec-1: IN for shadow 1, OUT for shadows 0, 2
            {"shadow_k": 1, "record_id": "rec-1"},
        ]
        out_records = [
            # rec-0: OUT for shadow 1, 2
            {"shadow_k": 1, "record_id": "rec-0"},
            {"shadow_k": 2, "record_id": "rec-0"},
            # rec-1: OUT for shadow 0, 2
            {"shadow_k": 0, "record_id": "rec-1"},
            {"shadow_k": 2, "record_id": "rec-1"},
        ]
        in_path = tmp_path / "in_records.jsonl"
        in_path.write_text("\n".join(json.dumps(r) for r in in_records))
        out_path = tmp_path / "out_records.jsonl"
        out_path.write_text("\n".join(json.dumps(r) for r in out_records))
        out_shadow = tmp_path / "shadow_params.json"

        with caplog.at_level(logging.INFO):
            rc = shadow_train.main(
                [
                    "--loss-dir",
                    str(tmp_path),
                    "--in-records",
                    str(in_path),
                    "--out-records",
                    str(out_path),
                    "--output",
                    str(out_shadow),
                ]
            )
        assert rc == 0
        assert out_shadow.is_file()
        # The output should be valid JSON with the expected shape.
        params = json.loads(out_shadow.read_text())
        assert isinstance(params, dict)
        # Top-level keys: lira_k, created_at, params (the per-record dict).
        assert "params" in params
        # Both records should be in the params dict.
        assert "rec-0" in params["params"]
        assert "rec-1" in params["params"]

    def test_no_shadow_files_returns_error(self, tmp_path: Path, caplog):
        in_path = tmp_path / "in_records.jsonl"
        in_path.write_text("")
        out_path = tmp_path / "out_records.jsonl"
        out_path.write_text("")
        out_shadow = tmp_path / "shadow_params.json"

        with caplog.at_level(logging.INFO):
            rc = shadow_train.main(
                [
                    "--loss-dir",
                    str(tmp_path),
                    "--in-records",
                    str(in_path),
                    "--out-records",
                    str(out_path),
                    "--output",
                    str(out_shadow),
                ]
            )
        # No shadow files → returns 1, no output written.
        assert rc == 1
        assert not out_shadow.exists()


# ---------------------------------------------------------------------------
# Module structure
# ---------------------------------------------------------------------------


class TestShadowTrainModule:
    def test_has_main(self):
        assert callable(shadow_train.main)

    def test_has_dunder_name_guard(self):
        import inspect

        source = inspect.getsource(shadow_train)
        assert '__name__ == "__main__"' in source
        assert "sys.exit(main())" in source

    def test_build_parser_required_args(self):
        parser = shadow_train.build_parser()
        # All four args are required: --loss-dir, --in-records,
        # --out-records, --output.
        with pytest.raises(SystemExit):
            parser.parse_args([])
