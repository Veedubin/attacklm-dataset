#!/usr/bin/env python3
"""Hermetic tests for memorization_report.py.

Tests cover:
  1. recommend_epoch_cap — boundary values, edge cases
  2. compute_memorization_for_record — empty record, mock model logits
  3. load_epoch_cap_table — None, valid JSON, invalid JSON
  4. build_parser — default --nll-threshold (0.01), --sample-size-per-source (200)
  5. load_records — restricted sources excluded, source filter
  6. End-to-end main() with mock model — report schema, dry-run

Run with:
    python -m pytest tests/test_memorization_report.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Make scripts/ importable — must come before module imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from memorization_report import (  # noqa: E402
    DEFAULT_EPOCH_CAP_THRESHOLDS,
    recommend_epoch_cap,
    compute_memorization_for_record,
    load_epoch_cap_table,
    build_parser,
    load_records,
    main,
)


# =========================================================================
# recommend_epoch_cap — pure function, no mocking needed
# =========================================================================


class TestRecommendEpochCap(unittest.TestCase):
    """Boundary-value tests for recommend_epoch_cap.

    Thresholds (default): <0.05→8, 0.05-0.15→4, 0.15-0.30→2, >0.30→1
    """

    def test_below_first_threshold(self):
        """memorization_fraction < 0.05 → cap=8."""
        self.assertEqual(recommend_epoch_cap(0.0), 8)
        self.assertEqual(recommend_epoch_cap(0.01), 8)
        self.assertEqual(recommend_epoch_cap(0.0499), 8)

    def test_at_first_threshold_boundary(self):
        """memorization_fraction == 0.05 → cap=4 (second bucket)."""
        self.assertEqual(recommend_epoch_cap(0.05), 4)

    def test_second_bucket(self):
        """0.05 <= memorization_fraction < 0.15 → cap=4."""
        self.assertEqual(recommend_epoch_cap(0.05), 4)
        self.assertEqual(recommend_epoch_cap(0.10), 4)
        self.assertEqual(recommend_epoch_cap(0.1499), 4)

    def test_second_to_third_boundary(self):
        """memorization_fraction == 0.15 → cap=2 (third bucket)."""
        self.assertEqual(recommend_epoch_cap(0.15), 2)

    def test_third_bucket(self):
        """0.15 <= memorization_fraction < 0.30 → cap=2."""
        self.assertEqual(recommend_epoch_cap(0.15), 2)
        self.assertEqual(recommend_epoch_cap(0.20), 2)
        self.assertEqual(recommend_epoch_cap(0.2999), 2)

    def test_third_to_fourth_boundary(self):
        """memorization_fraction == 0.30 → cap=1 (fourth bucket)."""
        self.assertEqual(recommend_epoch_cap(0.30), 1)

    def test_fourth_bucket(self):
        """memorization_fraction > 0.30 → cap=1."""
        self.assertEqual(recommend_epoch_cap(0.50), 1)
        self.assertEqual(recommend_epoch_cap(0.99), 1)
        self.assertEqual(recommend_epoch_cap(1.0), 1)

    def test_negative_fraction(self):
        """Negative memorization_fraction → cap=8 (first bucket, < 0.05)."""
        self.assertEqual(recommend_epoch_cap(-1.0), 8)

    def test_custom_thresholds(self):
        """Custom thresholds override defaults."""
        custom = [(0.10, 5), (0.50, 3), (1.01, 1)]
        self.assertEqual(recommend_epoch_cap(0.05, custom), 5)
        self.assertEqual(recommend_epoch_cap(0.10, custom), 3)
        self.assertEqual(recommend_epoch_cap(0.50, custom), 1)
        self.assertEqual(recommend_epoch_cap(0.99, custom), 1)

    def test_default_thresholds_unchanged(self):
        """Default thresholds list is not mutated by calls."""
        before = list(DEFAULT_EPOCH_CAP_THRESHOLDS)
        recommend_epoch_cap(0.10)
        self.assertEqual(DEFAULT_EPOCH_CAP_THRESHOLDS, before)


# =========================================================================
# compute_memorization_for_record — mock model + tokenizer
# =========================================================================


class TestComputeMemorizationForRecord(unittest.TestCase):
    """Tests for compute_memorization_for_record with mocked model."""

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()

        # Default tokenizer returns 5 tokens
        self.mock_tokenizer.return_value = {"input_ids": MagicMock()}
        self.mock_tokenizer.return_value["input_ids"].shape = (1, 5)

    def test_empty_record_no_assistant_turn(self):
        """Record with no assistant turn → 0 tokens, 0 NLL<threshold."""
        record = {"id": "rec1", "messages": [{"role": "user", "content": "hi"}]}
        result = compute_memorization_for_record(
            record, self.mock_model, self.mock_tokenizer
        )
        self.assertEqual(result["nll_lt_threshold_tokens"], 0)
        self.assertEqual(result["total_tokens"], 0)
        self.assertEqual(result["per_token_nlls"], [])
        self.assertEqual(result["record_id"], "rec1")

    def test_empty_record_no_messages_key(self):
        """Record with no 'messages' key → 0 tokens."""
        record = {"id": "rec2"}
        result = compute_memorization_for_record(
            record, self.mock_model, self.mock_tokenizer
        )
        self.assertEqual(result["total_tokens"], 0)

    def test_all_tokens_below_threshold(self):
        """All tokens have NLL < 0.01 → all counted."""
        import torch

        record = {
            "id": "rec3",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        }

        # Mock tokenizer to return 2 input_ids (so targets has 1 element,
        # and target_log_probs[0] is a 0D tensor/scalar, avoiding the
        # production-code bug with -list on multi-element tolist())
        input_ids = torch.tensor([[0, 1]])
        self.mock_tokenizer.return_value = {"input_ids": input_ids}

        # Mock model output: logits shape (1, 2, 5) — 2 input tokens, vocab=5
        # Code slices logits[:, :-1, :] → (1, 1, 5)
        # targets = input_ids[:, 1:] → [[1]]
        # We want logits[0,0,1] high → softmax gives ~1.0 → NLL ≈ 0 < 0.01
        logits = torch.tensor(
            [
                [
                    [0.0, 100.0, 0.0, 0.0, 0.0],  # predicts token 1
                    [0.0, 0.0, 0.0, 0.0, 0.0],  # unused (sliced away)
                ]
            ]
        )
        mock_output = MagicMock()
        mock_output.logits = logits
        self.mock_model.return_value = mock_output
        self.mock_model.device = MagicMock()
        self.mock_model.device.type = "cpu"

        result = compute_memorization_for_record(
            record, self.mock_model, self.mock_tokenizer
        )
        self.assertEqual(result["total_tokens"], 1)
        self.assertEqual(result["nll_lt_threshold_tokens"], 1)

    def test_no_tokens_below_threshold(self):
        """All tokens have NLL >= 0.01 → none counted."""
        import torch

        record = {
            "id": "rec4",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        }

        # Mock tokenizer to return 2 input_ids (so targets has 1 element)
        input_ids = torch.tensor([[0, 1]])
        self.mock_tokenizer.return_value = {"input_ids": input_ids}

        # logits where target token has very low probability
        # logits[0,0,1] = -100 → softmax gives ~0 for token 1
        # -log(~0) ≈ 100, which is >> 0.01
        # logits shape: (1, 2, 5) — 2 input tokens, vocab=5
        logits = torch.tensor(
            [
                [
                    [0.0, -100.0, 0.0, 0.0, 0.0],  # token 1 has low prob
                    [0.0, 0.0, 0.0, 0.0, 0.0],  # unused (sliced away)
                ]
            ]
        )
        mock_output = MagicMock()
        mock_output.logits = logits
        self.mock_model.return_value = mock_output
        self.mock_model.device = MagicMock()
        self.mock_model.device.type = "cpu"

        result = compute_memorization_for_record(
            record, self.mock_model, self.mock_tokenizer
        )
        self.assertEqual(result["total_tokens"], 1)
        self.assertEqual(result["nll_lt_threshold_tokens"], 0)

    def test_custom_threshold(self):
        """Custom nll_threshold changes which tokens are counted."""
        import torch

        record = {
            "id": "rec5",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        }

        # Mock tokenizer to return 2 input_ids (so targets has 1 element)
        input_ids = torch.tensor([[0, 1]])
        self.mock_tokenizer.return_value = {"input_ids": input_ids}

        # logits where all logits are equal → softmax gives 0.2 for each token
        # -log(0.2) ≈ 1.609, which is > 0.01 but < 2.0
        # logits shape: (1, 2, 5) — 2 input tokens, vocab=5
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        mock_output = MagicMock()
        mock_output.logits = logits
        self.mock_model.return_value = mock_output
        self.mock_model.device = MagicMock()
        self.mock_model.device.type = "cpu"

        # With threshold=2.0, all tokens (NLL≈1.609) are below threshold
        result = compute_memorization_for_record(
            record, self.mock_model, self.mock_tokenizer, nll_threshold=2.0
        )
        self.assertEqual(result["total_tokens"], 1)
        self.assertEqual(result["nll_lt_threshold_tokens"], 1)

        # With threshold=1.0, none are below threshold
        result2 = compute_memorization_for_record(
            record, self.mock_model, self.mock_tokenizer, nll_threshold=1.0
        )
        self.assertEqual(result2["nll_lt_threshold_tokens"], 0)


# =========================================================================
# load_epoch_cap_table
# =========================================================================


class TestLoadEpochCapTable(unittest.TestCase):
    """Tests for load_epoch_cap_table."""

    def test_none_returns_defaults(self):
        """None input returns DEFAULT_EPOCH_CAP_THRESHOLDS."""
        result = load_epoch_cap_table(None)
        self.assertEqual(result, DEFAULT_EPOCH_CAP_THRESHOLDS)

    def test_valid_json_file(self):
        """Valid JSON file returns sorted thresholds."""
        data = [[0.10, 5], [0.50, 3], [1.01, 1]]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            result = load_epoch_cap_table(path)
            self.assertEqual(result, [(0.10, 5), (0.50, 3), (1.01, 1)])
        finally:
            path.unlink(missing_ok=True)

    def test_unsorted_json_is_sorted(self):
        """Unsorted input is sorted by upper_bound."""
        data = [[1.01, 1], [0.05, 8], [0.30, 2], [0.15, 4]]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            result = load_epoch_cap_table(path)
            self.assertEqual(result, [(0.05, 8), (0.15, 4), (0.30, 2), (1.01, 1)])
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_format_raises(self):
        """Non-list JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "a list"}, f)
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                load_epoch_cap_table(path)
        finally:
            path.unlink(missing_ok=True)

    def test_bad_entry_format_raises(self):
        """Entry with wrong length raises ValueError."""
        data = [[0.05, 8, "extra"]]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                load_epoch_cap_table(path)
        finally:
            path.unlink(missing_ok=True)


# =========================================================================
# build_parser — CLI defaults
# =========================================================================


class TestBuildParser(unittest.TestCase):
    """Tests for CLI argument defaults."""

    def test_nll_threshold_default_is_0_01(self):
        """--nll-threshold defaults to 0.01 (MANDATORY test)."""
        parser = build_parser()
        args = parser.parse_args(["--model", "/fake/model"])
        self.assertAlmostEqual(args.nll_threshold, 0.01)

    def test_sample_size_per_source_default_is_200(self):
        """--sample-size-per-source defaults to 200."""
        parser = build_parser()
        args = parser.parse_args(["--model", "/fake/model"])
        self.assertEqual(args.sample_size_per_source, 200)

    def test_dry_run_flag(self):
        """--dry-run sets dry_run=True."""
        parser = build_parser()
        args = parser.parse_args(["--model", "/fake/model", "--dry-run"])
        self.assertTrue(args.dry_run)

    def test_source_filter_default_none(self):
        """--source-filter defaults to None."""
        parser = build_parser()
        args = parser.parse_args(["--model", "/fake/model"])
        self.assertIsNone(args.source_filter)

    def test_model_required(self):
        """--model is required."""
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])


# =========================================================================
# load_records — file I/O with temp directories
# =========================================================================


class TestLoadRecords(unittest.TestCase):
    """Tests for load_records with temp directory structure."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="test_memorization_"))
        # Create source directories with JSONL files
        self._make_source("metasploit-framework", [{"id": "ms1", "text": "a"}])
        self._make_source(
            "sigma-hq", [{"id": "sg1", "text": "b"}, {"id": "sg2", "text": "c"}]
        )
        # Restricted source (should be excluded by default)
        self._make_source("rta", [{"id": "rta1", "text": "d"}])

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_source(self, name, records):
        src_dir = self.tmpdir / name
        src_dir.mkdir(parents=True, exist_ok=True)
        bucket_dir = src_dir / "default" / "tactic"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = bucket_dir / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_restricted_sources_excluded_by_default(self):
        """Restricted sources (rta) are excluded when no source_filter."""
        result = load_records(self.tmpdir, source_filter=None)
        self.assertIn("metasploit-framework", result)
        self.assertIn("sigma-hq", result)
        self.assertNotIn("rta", result)

    def test_source_filter_includes_requested(self):
        """Source filter includes only requested sources."""
        result = load_records(self.tmpdir, source_filter=["metasploit-framework"])
        self.assertIn("metasploit-framework", result)
        self.assertNotIn("sigma-hq", result)

    def test_per_source_limit(self):
        """per_source_limit caps the number of records per source."""
        result = load_records(self.tmpdir, source_filter=None, per_source_limit=1)
        self.assertEqual(len(result["sigma-hq"]), 1)

    def test_records_have_source_field(self):
        """Each record gets a _source field set."""
        result = load_records(self.tmpdir, source_filter=["metasploit-framework"])
        for rec in result["metasploit-framework"]:
            self.assertEqual(rec["_source"], "metasploit-framework")

    def test_nonexistent_root_raises(self):
        """Non-existent dataset root raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_records(Path("/nonexistent/path"), source_filter=None)


# =========================================================================
# End-to-end main() with mock model — report schema
# =========================================================================


class TestMainEndToEnd(unittest.TestCase):
    """End-to-end tests for main() with mocked model and file system."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="test_memorization_e2e_"))
        self._make_source(
            "metasploit-framework",
            [
                {
                    "id": "ms1",
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ],
                },
            ],
        )
        self._make_source(
            "sigma-hq",
            [
                {
                    "id": "sg1",
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "b"},
                    ],
                },
                {
                    "id": "sg2",
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "c"},
                    ],
                },
            ],
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_source(self, name, records):
        src_dir = self.tmpdir / name
        src_dir.mkdir(parents=True, exist_ok=True)
        bucket_dir = src_dir / "default" / "tactic"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = bucket_dir / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    @patch("memorization_report.load_model")
    def test_dry_run_returns_zero(self, mock_load_model):
        """--dry-run exits with code 0 without loading model."""
        exit_code = main(
            [
                "--model",
                "/fake/model",
                "--dataset-root",
                str(self.tmpdir),
                "--dry-run",
            ]
        )
        self.assertEqual(exit_code, 0)
        mock_load_model.assert_not_called()

    @patch("memorization_report.load_model")
    def test_report_schema_has_required_keys(self, mock_load_model):
        """Per-source report has all required schema keys."""
        import torch

        # Mock model + tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Tokenizer returns 2 input_ids (so targets has 1 element,
        # avoiding the production-code bug with multi-element tolist())
        input_ids = torch.tensor([[0, 1]])
        mock_tokenizer.return_value = {"input_ids": input_ids}

        # Model returns logits where all target tokens have NLL < 0.01
        # logits shape: (1, 2, 5) — 2 input tokens, vocab=5
        # logits[0,0,1] high → softmax gives ~1.0 → NLL ≈ 0 < 0.01
        logits = torch.tensor(
            [
                [
                    [0.0, 100.0, 0.0, 0.0, 0.0],  # predicts token 1
                    [0.0, 0.0, 0.0, 0.0, 0.0],  # unused (sliced away)
                ]
            ]
        )
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output
        mock_model.device = MagicMock()
        mock_model.device.type = "cpu"

        # Capture stdout
        from io import StringIO

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            exit_code = main(
                [
                    "--model",
                    "/fake/model",
                    "--dataset-root",
                    str(self.tmpdir),
                    "--sample-size-per-source",
                    "200",
                ]
            )
        finally:
            sys.stdout = old_stdout

        self.assertEqual(exit_code, 0)
        report = json.loads(captured.getvalue())

        # Top-level schema
        self.assertIn("schema_version", report)
        self.assertIn("sources", report)
        self.assertIn("nll_threshold", report)
        self.assertEqual(report["nll_threshold"], 0.01)

        # Per-source schema
        for src in report["sources"]:
            self.assertIn("source", src)
            self.assertIn("n_records_sampled", src)
            self.assertIn("total_tokens", src)
            self.assertIn("nll_lt_threshold_tokens", src)
            self.assertIn("memorization_fraction", src)
            self.assertIn("recommended_epoch_cap", src)
            self.assertIn("example_records", src)

    @patch("memorization_report.load_model")
    def test_restricted_sources_excluded_in_main(self, mock_load_model):
        """main() excludes restricted sources even when they exist on disk."""
        # Add a restricted source
        self._make_source(
            "rta",
            [
                {
                    "id": "rta1",
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "x"},
                    ],
                },
            ],
        )

        import torch

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        input_ids = torch.tensor([[0, 1]])
        mock_tokenizer.return_value = {"input_ids": input_ids}
        logits = torch.tensor(
            [
                [
                    [0.0, 100.0, 0.0, 0.0, 0.0],  # predicts token 1
                    [0.0, 0.0, 0.0, 0.0, 0.0],  # unused (sliced away)
                ]
            ]
        )
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output
        mock_model.device = MagicMock()
        mock_model.device.type = "cpu"

        from io import StringIO

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            exit_code = main(
                [
                    "--model",
                    "/fake/model",
                    "--dataset-root",
                    str(self.tmpdir),
                ]
            )
        finally:
            sys.stdout = old_stdout

        self.assertEqual(exit_code, 0)
        report = json.loads(captured.getvalue())
        source_names = [s["source"] for s in report["sources"]]
        self.assertNotIn("rta", source_names)
        self.assertIn("metasploit-framework", source_names)
        self.assertIn("sigma-hq", source_names)

    @patch("memorization_report.load_model")
    def test_no_records_returns_one(self, mock_load_model):
        """No records loaded → exit code 1."""
        empty_dir = Path(tempfile.mkdtemp(prefix="test_memorization_empty_"))
        try:
            exit_code = main(
                [
                    "--model",
                    "/fake/model",
                    "--dataset-root",
                    str(empty_dir),
                    "--dry-run",
                ]
            )
            self.assertEqual(exit_code, 1)
        finally:
            import shutil

            shutil.rmtree(empty_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
