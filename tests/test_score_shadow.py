#!/usr/bin/env python3
"""Tests for score_shadow.py — LiRA shadow-model scoring (Step 2).

Hermetic tests using MagicMock and patching — no real model loads, no GPU required.

Run with:
    python -m pytest tests/test_score_shadow.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make scripts/ importable — must come before module imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from score_shadow import build_parser, main  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORD = {
    "messages": [
        {"role": "user", "content": "Explain MS17-010"},
        {
            "role": "assistant",
            "content": "This module exploits SMB vulnerability MS17-010.",
        },
    ],
    "source": "metasploit-framework",
    "license": "BSD-3-Clause",
    "id": "record_42",
}

SAMPLE_RECORD_NO_ASSISTANT = {
    "messages": [
        {"role": "user", "content": "Just a prompt, no response"},
    ],
    "source": "test-source",
    "license": "MIT",
    "id": "record_empty",
}


# ---------------------------------------------------------------------------
# Test: parser
# ---------------------------------------------------------------------------


class TestParser(unittest.TestCase):
    """Argument parser tests."""

    def test_parser_required_args(self):
        """--model, --records, --output-dir, --shadow-index are all required."""
        parser = build_parser()
        # Missing all required args
        with self.assertRaises(SystemExit):
            parser.parse_args([])

        # Missing --shadow-index
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--model",
                    "/fake/model",
                    "--records",
                    "/fake/records.jsonl",
                    "--output-dir",
                    "/tmp",
                ]
            )

    def test_parser_accepts_all_flags(self):
        """All flags accepted and parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/some/model",
                "--model-format",
                "hf",
                "--records",
                "/some/records.jsonl",
                "--output-dir",
                "/some/output",
                "--shadow-index",
                "3",
                "--max-records",
                "10",
                "--dry-run",
            ]
        )
        self.assertEqual(str(args.model), "/some/model")
        self.assertEqual(args.model_format, "hf")
        self.assertEqual(str(args.records), "/some/records.jsonl")
        self.assertEqual(str(args.output_dir), "/some/output")
        self.assertEqual(args.shadow_index, 3)
        self.assertEqual(args.max_records, 10)
        self.assertTrue(args.dry_run)


# ---------------------------------------------------------------------------
# Test: main() with mocked model
# ---------------------------------------------------------------------------


class TestScoreShadowMain(unittest.TestCase):
    """Main function tests with mocked model and scoring."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_records(self, records: list[dict]) -> Path:
        """Write records to a temp JSONL file."""
        path = self.tmpdir_path / "audit_set.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_dry_run_no_model_load(self):
        """--dry-run returns 0 without loading model or writing files."""
        records_path = self._write_records([SAMPLE_RECORD])
        output_dir = self.tmpdir_path / "losses"

        exit_code = main(
            [
                "--model",
                "/fake/model",
                "--records",
                str(records_path),
                "--output-dir",
                str(output_dir),
                "--shadow-index",
                "0",
                "--dry-run",
            ]
        )
        self.assertEqual(exit_code, 0)
        # No output file should exist
        self.assertFalse((output_dir / "shadow_0.json").exists())

    @patch("score_shadow.load_model")
    @patch("score_shadow.compute_nll")
    def test_writes_shadow_json_format(self, mock_compute_nll, mock_load):
        """shadow_0.json contains {record_id: nll} dict."""
        mock_compute_nll.return_value = (12.34, 10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        records_path = self._write_records([SAMPLE_RECORD])
        output_dir = self.tmpdir_path / "losses"

        exit_code = main(
            [
                "--model",
                str(self.tmpdir_path / "model"),
                "--records",
                str(records_path),
                "--output-dir",
                str(output_dir),
                "--shadow-index",
                "0",
            ]
        )
        self.assertEqual(exit_code, 0)

        output_path = output_dir / "shadow_0.json"
        self.assertTrue(output_path.exists())
        with open(output_path) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)
        self.assertIn("record_42", data)
        self.assertAlmostEqual(data["record_42"], 12.34, places=4)

    @patch("score_shadow.load_model")
    @patch("score_shadow.compute_nll")
    def test_shadow_index_in_filename(self, mock_compute_nll, mock_load):
        """--shadow-index 5 produces shadow_5.json."""
        mock_compute_nll.return_value = (1.0, 5)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        records_path = self._write_records([SAMPLE_RECORD])
        output_dir = self.tmpdir_path / "losses"

        exit_code = main(
            [
                "--model",
                str(self.tmpdir_path / "model"),
                "--records",
                str(records_path),
                "--output-dir",
                str(output_dir),
                "--shadow-index",
                "5",
            ]
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue((output_dir / "shadow_5.json").exists())
        self.assertFalse((output_dir / "shadow_0.json").exists())

    @patch("score_shadow.load_model")
    @patch("score_shadow.compute_nll")
    def test_empty_records_writes_sentinel(self, mock_compute_nll, mock_load):
        """Empty records file writes sentinel JSON with _meta.empty=True."""
        records_path = self._write_records([])
        output_dir = self.tmpdir_path / "losses"

        exit_code = main(
            [
                "--model",
                str(self.tmpdir_path / "model"),
                "--records",
                str(records_path),
                "--output-dir",
                str(output_dir),
                "--shadow-index",
                "0",
            ]
        )
        self.assertEqual(exit_code, 0)

        output_path = output_dir / "shadow_0.json"
        self.assertTrue(output_path.exists())
        with open(output_path) as f:
            data = json.load(f)
        self.assertIn("_meta", data)
        self.assertTrue(data["_meta"]["empty"])
        self.assertEqual(data["_meta"]["n_records"], 0)
        # load_model should NOT have been called
        mock_load.assert_not_called()

    @patch("score_shadow.load_model")
    @patch("score_shadow.compute_nll")
    def test_duplicate_record_ids_last_wins(self, mock_compute_nll, mock_load):
        """Duplicate record_ids: last write wins in output dict."""
        mock_compute_nll.side_effect = [(10.0, 5), (20.0, 5)]
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        records = [
            {
                "id": "dup_id",
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "first"},
                ],
            },
            {
                "id": "dup_id",
                "messages": [
                    {"role": "user", "content": "b"},
                    {"role": "assistant", "content": "second"},
                ],
            },
        ]
        records_path = self._write_records(records)
        output_dir = self.tmpdir_path / "losses"

        exit_code = main(
            [
                "--model",
                str(self.tmpdir_path / "model"),
                "--records",
                str(records_path),
                "--output-dir",
                str(output_dir),
                "--shadow-index",
                "0",
            ]
        )
        self.assertEqual(exit_code, 0)

        with open(output_dir / "shadow_0.json") as f:
            data = json.load(f)
        # Last write wins — should be 20.0, not 10.0
        self.assertIn("dup_id", data)
        self.assertAlmostEqual(data["dup_id"], 20.0, places=4)

    @patch("score_shadow.load_model")
    @patch("score_shadow.compute_nll")
    def test_missing_assistant_turn_writes_inf(self, mock_compute_nll, mock_load):
        """Record with no assistant turn gets nll=inf."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        records_path = self._write_records([SAMPLE_RECORD_NO_ASSISTANT])
        output_dir = self.tmpdir_path / "losses"

        exit_code = main(
            [
                "--model",
                str(self.tmpdir_path / "model"),
                "--records",
                str(records_path),
                "--output-dir",
                str(output_dir),
                "--shadow-index",
                "0",
            ]
        )
        self.assertEqual(exit_code, 0)

        with open(output_dir / "shadow_0.json") as f:
            data = json.load(f)
        self.assertIn("record_empty", data)
        self.assertEqual(data["record_empty"], float("inf"))
        # compute_nll should NOT have been called (no assistant turn)
        mock_compute_nll.assert_not_called()

    @patch("score_shadow.load_model")
    @patch("score_shadow.compute_nll")
    def test_atomic_write_via_tempfile(self, mock_compute_nll, mock_load):
        """Output is written via .tmp rename — no partial files on crash."""
        mock_compute_nll.return_value = (5.0, 10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        records_path = self._write_records([SAMPLE_RECORD])
        output_dir = self.tmpdir_path / "losses"

        exit_code = main(
            [
                "--model",
                str(self.tmpdir_path / "model"),
                "--records",
                str(records_path),
                "--output-dir",
                str(output_dir),
                "--shadow-index",
                "0",
            ]
        )
        self.assertEqual(exit_code, 0)

        # The .tmp file should NOT exist after successful write
        self.assertFalse((output_dir / "shadow_0.json.tmp").exists())
        # The real file should exist
        self.assertTrue((output_dir / "shadow_0.json").exists())

    @patch("score_shadow.load_model")
    @patch("score_shadow.compute_nll")
    def test_max_records_caps_processing(self, mock_compute_nll, mock_load):
        """--max-records caps the number of records scored."""
        mock_compute_nll.return_value = (3.0, 5)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        records = [
            {
                "id": f"r{i}",
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": f"resp{i}"},
                ],
            }
            for i in range(10)
        ]
        records_path = self._write_records(records)
        output_dir = self.tmpdir_path / "losses"

        exit_code = main(
            [
                "--model",
                str(self.tmpdir_path / "model"),
                "--records",
                str(records_path),
                "--output-dir",
                str(output_dir),
                "--shadow-index",
                "0",
                "--max-records",
                "3",
            ]
        )
        self.assertEqual(exit_code, 0)

        with open(output_dir / "shadow_0.json") as f:
            data = json.load(f)
        # Only 3 records should be in output
        self.assertEqual(len(data), 3)
        self.assertIn("r0", data)
        self.assertIn("r1", data)
        self.assertIn("r2", data)
        self.assertNotIn("r3", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
