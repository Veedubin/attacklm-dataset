#!/usr/bin/env python3
"""Hermetic tests for held_out_nll.py and split_held_out.py.

Tests cover:
  1. Per-record NLL aggregation (correct mean, edge cases)
  2. mimic_mai formula (weights sum to 1, bucket assignment, redistribution)
  3. equal and custom formulas
  4. End-to-end with mock model (output schema, dry-run)
  5. split_held_out.py idempotency

Run with:
    python -m pytest tests/test_held_out_nll.py -v
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

from held_out_nll import (  # noqa: E402
    BUCKET_WEIGHTS,
    source_to_bucket,
    compute_mean_nll_for_record,
    aggregate_mimic_mai,
    aggregate_equal,
    aggregate_custom,
    build_parser,
    main,
)
from split_held_out import (  # noqa: E402
    collect_jsonl_records,
    split_source,
    main as split_main,
)


# =========================================================================
# 1. Per-record NLL aggregation
# =========================================================================


class TestComputeMeanNLLForRecord(unittest.TestCase):
    """Tests for compute_mean_nll_for_record with mocked model."""

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()

    def test_correct_mean_nll(self):
        """compute_mean_nll_for_record returns correct mean NLL."""
        record = {
            "id": "rec1",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        }
        with patch("held_out_nll.compute_nll", return_value=(10.0, 5)):
            result = compute_mean_nll_for_record(
                record, self.mock_model, self.mock_tokenizer
            )
        self.assertAlmostEqual(result, 2.0)  # 10.0 / 5

    def test_zero_tokens_returns_inf(self):
        """No assistant turn → float('inf')."""
        record = {"id": "rec2", "messages": [{"role": "user", "content": "hello"}]}
        result = compute_mean_nll_for_record(
            record, self.mock_model, self.mock_tokenizer
        )
        self.assertEqual(result, float("inf"))

    def test_no_messages_key_returns_inf(self):
        """Record with no 'messages' key → float('inf')."""
        record = {"id": "rec3"}
        result = compute_mean_nll_for_record(
            record, self.mock_model, self.mock_tokenizer
        )
        self.assertEqual(result, float("inf"))

    def test_compute_nll_returns_zero_tokens(self):
        """compute_nll returns num_tokens=0 → float('inf')."""
        record = {
            "id": "rec4",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
        with patch("held_out_nll.compute_nll", return_value=(0.0, 0)):
            result = compute_mean_nll_for_record(
                record, self.mock_model, self.mock_tokenizer
            )
        self.assertEqual(result, float("inf"))


# =========================================================================
# 2. mimic_mai formula
# =========================================================================


class TestAggregateMimicMai(unittest.TestCase):
    """Tests for aggregate_mimic_mai — bucket weights, assignment, redistribution."""

    def test_weights_sum_to_one(self):
        """BUCKET_WEIGHTS sum to 1.0."""
        total = sum(BUCKET_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_bucket_assignment_known_sources(self):
        """Known sources map to correct buckets."""
        self.assertEqual(source_to_bucket("metasploit-framework"), "Code")
        self.assertEqual(source_to_bucket("sigma-hq"), "General")
        self.assertEqual(source_to_bucket("nist-ir"), "STEM")
        self.assertEqual(source_to_bucket("0xdf-writeups"), "Math")
        self.assertEqual(source_to_bucket("nyu-ctf-bench"), "Multilingual")

    def test_unknown_source_goes_to_general(self):
        """Unknown source → assigned to General bucket."""
        per_source = {"unknown-source": 2.5}
        result = aggregate_mimic_mai(per_source)
        self.assertIn("General", result["per_bucket"])
        self.assertAlmostEqual(result["per_bucket"]["General"], 2.5)

    def test_aggregate_with_all_buckets_present(self):
        """All 5 buckets present → correct weighted aggregate."""
        per_source = {
            "metasploit-framework": 2.0,
            "nist-ir": 3.0,
            "0xdf-writeups": 4.0,
            "sigma-hq": 5.0,
            "nyu-ctf-bench": 6.0,
        }
        result = aggregate_mimic_mai(per_source)
        # Code=2.0, STEM=3.0, Math=4.0, General=5.0, Multilingual=6.0
        expected = 0.50 * 2.0 + 0.175 * 3.0 + 0.175 * 4.0 + 0.10 * 5.0 + 0.05 * 6.0
        self.assertAlmostEqual(result["aggregate"], expected, places=6)

    def test_empty_bucket_redistributes_weight(self):
        """Empty bucket (e.g., no Math sources) → weight redistributed proportionally."""
        per_source = {
            "metasploit-framework": 2.0,  # Code
            "nist-ir": 3.0,  # STEM
            "sigma-hq": 5.0,  # General
            "nyu-ctf-bench": 6.0,  # Multilingual
        }
        result = aggregate_mimic_mai(per_source)
        # Math bucket is empty → its 0.175 weight is redistributed
        # Active weights: Code=0.50, STEM=0.175, General=0.10, Multilingual=0.05
        # Total active = 0.825
        # Redistributed: Code += 0.175 * (0.50/0.825) = 0.10606...
        # Code new = 0.60606..., STEM new = 0.21212..., General new = 0.12121..., Multilingual new = 0.06060...
        self.assertNotIn("Math", result["per_bucket"])
        # Aggregate should be computable
        self.assertGreater(result["aggregate"], 0)
        # Verify per_bucket has the 4 active buckets
        self.assertIn("Code", result["per_bucket"])
        self.assertIn("STEM", result["per_bucket"])
        self.assertIn("General", result["per_bucket"])
        self.assertIn("Multilingual", result["per_bucket"])

    def test_single_source_aggregate(self):
        """Single source → aggregate equals that source's NLL."""
        per_source = {"metasploit-framework": 2.5}
        result = aggregate_mimic_mai(per_source)
        self.assertAlmostEqual(result["aggregate"], 2.5, places=6)


# =========================================================================
# 3. equal and custom formulas
# =========================================================================


class TestAggregateEqual(unittest.TestCase):
    """Tests for aggregate_equal — simple mean across sources."""

    def test_equal_two_sources(self):
        """Simple mean across 2 sources."""
        per_source = {"src1": 2.0, "src2": 4.0}
        result = aggregate_equal(per_source)
        self.assertAlmostEqual(result["aggregate"], 3.0, places=6)
        self.assertEqual(result["per_bucket"], {})

    def test_equal_empty(self):
        """Empty per_source → aggregate=0.0."""
        result = aggregate_equal({})
        self.assertEqual(result["aggregate"], 0.0)

    def test_equal_single_source(self):
        """Single source → aggregate equals that source's NLL."""
        per_source = {"src1": 3.5}
        result = aggregate_equal(per_source)
        self.assertAlmostEqual(result["aggregate"], 3.5, places=6)


class TestAggregateCustom(unittest.TestCase):
    """Tests for aggregate_custom — custom weights from JSON."""

    def test_custom_bucket_weights(self):
        """Custom bucket-level weights produce correct weighted mean."""
        per_source = {
            "metasploit-framework": 2.0,
            "sigma-hq": 5.0,
        }
        weights = {"Code": 0.7, "General": 0.3}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(weights, f)
            weights_path = Path(f.name)
        try:
            result = aggregate_custom(per_source, weights_path)
            # Code=2.0, General=5.0
            # weighted: 0.7*2.0 + 0.3*5.0 = 1.4 + 1.5 = 2.9
            self.assertAlmostEqual(result["aggregate"], 2.9, places=6)
        finally:
            weights_path.unlink(missing_ok=True)

    def test_custom_source_weights(self):
        """Custom source-level weights produce correct weighted mean."""
        per_source = {"metasploit-framework": 2.0, "sigma-hq": 4.0}
        weights = {"metasploit-framework": 1.0, "sigma-hq": 3.0}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(weights, f)
            weights_path = Path(f.name)
        try:
            result = aggregate_custom(per_source, weights_path)
            # total_weight = 4.0
            # (2.0 * 1.0/4.0) + (4.0 * 3.0/4.0) = 0.5 + 3.0 = 3.5
            self.assertAlmostEqual(result["aggregate"], 3.5, places=6)
        finally:
            weights_path.unlink(missing_ok=True)

    def test_custom_missing_weight_fallback(self):
        """Source with no weight → weight treated as 0.0."""
        per_source = {"metasploit-framework": 2.0, "sigma-hq": 4.0}
        weights = {"metasploit-framework": 1.0}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(weights, f)
            weights_path = Path(f.name)
        try:
            result = aggregate_custom(per_source, weights_path)
            # total_weight = 1.0, sigma-hq weight = 0.0
            # (2.0 * 1.0/1.0) + (4.0 * 0.0/1.0) = 2.0
            self.assertAlmostEqual(result["aggregate"], 2.0, places=6)
        finally:
            weights_path.unlink(missing_ok=True)

    def test_custom_empty_weights_raises(self):
        """Empty weights file raises ValueError."""
        per_source = {"src1": 2.0}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            weights_path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                aggregate_custom(per_source, weights_path)
        finally:
            weights_path.unlink(missing_ok=True)


# =========================================================================
# 4. End-to-end with mock model
# =========================================================================


class TestMainEndToEnd(unittest.TestCase):
    """End-to-end tests for main() with mocked model and file system."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="test_held_out_nll_e2e_"))
        self.held_out_root = self.tmpdir / "held_out"
        self._make_held_out_source(
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
        self._make_held_out_source(
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

    def _make_held_out_source(self, name, records):
        src_dir = self.held_out_root / name / "default" / "tactic"
        src_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = src_dir / "data_held_out.jsonl"
        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    @patch("held_out_nll.load_model")
    def test_output_schema_has_required_keys(self, mock_load_model):
        """Output JSON has all required schema keys."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Mock compute_nll to return fixed values
        with patch("held_out_nll.compute_nll", return_value=(5.0, 2)):
            from io import StringIO

            captured = StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                exit_code = main(
                    [
                        "--model",
                        "/fake/model",
                        "--held-out-root",
                        str(self.held_out_root),
                    ]
                )
            finally:
                sys.stdout = old_stdout

        self.assertEqual(exit_code, 0)
        report = json.loads(captured.getvalue())

        # Required top-level keys
        self.assertIn("schema_version", report)
        self.assertIn("per_source", report)
        self.assertIn("per_bucket", report)
        self.assertIn("aggregate", report)
        self.assertIn("n_records_evaluated", report)
        self.assertIn("model", report)
        self.assertIn("date", report)
        self.assertIn("aggregation_formula", report)

        # Per-source should have both sources
        self.assertIn("metasploit-framework", report["per_source"])
        self.assertIn("sigma-hq", report["per_source"])

    @patch("held_out_nll.load_model")
    def test_dry_run_produces_output_without_model(self, mock_load_model):
        """--dry-run produces output without calling load_model."""
        from io import StringIO

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            exit_code = main(
                [
                    "--model",
                    "/fake/model",
                    "--held-out-root",
                    str(self.held_out_root),
                    "--dry-run",
                ]
            )
        finally:
            sys.stdout = old_stdout

        self.assertEqual(exit_code, 0)
        output = captured.getvalue()
        self.assertIn("Dry run", output)
        self.assertIn("metasploit-framework", output)
        self.assertIn("sigma-hq", output)
        mock_load_model.assert_not_called()

    @patch("held_out_nll.load_model")
    def test_restricted_source_filter_returns_one(self, mock_load_model):
        """--source-filter with restricted source → exit code 1."""
        exit_code = main(
            [
                "--model",
                "/fake/model",
                "--held-out-root",
                str(self.held_out_root),
                "--source-filter",
                "rta",
            ]
        )
        self.assertEqual(exit_code, 1)
        mock_load_model.assert_not_called()

    @patch("held_out_nll.load_model")
    def test_no_records_returns_one(self, mock_load_model):
        """No records loaded → exit code 1."""
        empty_dir = Path(tempfile.mkdtemp(prefix="test_held_out_empty_"))
        try:
            exit_code = main(
                [
                    "--model",
                    "/fake/model",
                    "--held-out-root",
                    str(empty_dir),
                    "--dry-run",
                ]
            )
            self.assertEqual(exit_code, 1)
        finally:
            import shutil

            shutil.rmtree(empty_dir, ignore_errors=True)


# =========================================================================
# 5. split_held_out.py idempotency
# =========================================================================


class TestSplitHeldOut(unittest.TestCase):
    """Tests for split_held_out.py — idempotency and correctness."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="test_split_held_out_"))
        self.dataset_root = self.tmpdir / "sources"
        self.held_out_root = self.tmpdir / "held_out"
        self._make_source("test-source", 10)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_source(self, name, num_records):
        src_dir = self.dataset_root / name / "base" / "execution"
        src_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = src_dir / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(num_records):
                rec = {
                    "id": f"{name}_{i}",
                    "messages": [
                        {"role": "user", "content": f"prompt_{i}"},
                        {"role": "assistant", "content": f"response_{i}"},
                    ],
                }
                f.write(json.dumps(rec) + "\n")

    def test_split_source_moves_correct_number(self):
        """split_source moves held_out_size records to held-out dir."""
        total, held_out = split_source(
            source_name="test-source",
            dataset_root=self.dataset_root,
            held_out_root=self.held_out_root,
            held_out_size=3,
        )
        self.assertEqual(total, 10)
        self.assertEqual(held_out, 3)

        # Verify destination file exists
        dest_file = (
            self.held_out_root
            / "test-source"
            / "base"
            / "execution"
            / "data_held_out.jsonl"
        )
        self.assertTrue(dest_file.exists())

        # Verify correct number of records in destination
        with open(dest_file) as f:
            dest_records = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(dest_records), 3)

        # Verify they are the LAST 3 records
        self.assertEqual(dest_records[0]["id"], "test-source_7")
        self.assertEqual(dest_records[1]["id"], "test-source_8")
        self.assertEqual(dest_records[2]["id"], "test-source_9")

    def test_split_source_idempotent(self):
        """Second call to split_source is a no-op (idempotent)."""
        # First call
        total1, held_out1 = split_source(
            source_name="test-source",
            dataset_root=self.dataset_root,
            held_out_root=self.held_out_root,
            held_out_size=3,
        )
        self.assertEqual(held_out1, 3)

        # Second call — should skip because held-out dir already exists
        total2, held_out2 = split_source(
            source_name="test-source",
            dataset_root=self.dataset_root,
            held_out_root=self.held_out_root,
            held_out_size=3,
        )
        self.assertEqual(total2, 0)
        self.assertEqual(held_out2, 0)

    def test_split_source_dry_run(self):
        """Dry run does not create any files."""
        total, held_out = split_source(
            source_name="test-source",
            dataset_root=self.dataset_root,
            held_out_root=self.held_out_root,
            held_out_size=3,
            dry_run=True,
        )
        self.assertEqual(total, 10)
        self.assertEqual(held_out, 3)

        # No files should have been created
        self.assertFalse(self.held_out_root.exists())

    def test_collect_jsonl_records(self):
        """collect_jsonl_records returns all records from a source directory."""
        records = collect_jsonl_records(self.dataset_root / "test-source")
        self.assertEqual(len(records), 10)
        self.assertEqual(records[0]["id"], "test-source_0")
        self.assertEqual(records[-1]["id"], "test-source_9")

    def test_split_main_idempotent(self):
        """split_held_out main() is idempotent — second run skips all sources."""
        # First run
        exit_code1 = split_main(
            dataset_root=self.dataset_root,
            held_out_root=self.held_out_root,
            held_out_size=3,
            yes=True,
        )
        self.assertEqual(exit_code1, 0)

        # Verify files were created
        dest_file = (
            self.held_out_root
            / "test-source"
            / "base"
            / "execution"
            / "data_held_out.jsonl"
        )
        self.assertTrue(dest_file.exists())

        # Second run — should be no-op
        exit_code2 = split_main(
            dataset_root=self.dataset_root,
            held_out_root=self.held_out_root,
            held_out_size=3,
            yes=True,
        )
        self.assertEqual(exit_code2, 0)

        # File should still exist and have same content
        with open(dest_file) as f:
            dest_records = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(dest_records), 3)


# =========================================================================
# CLI parser defaults
# =========================================================================


class TestBuildParser(unittest.TestCase):
    """Tests for CLI argument defaults."""

    def test_model_required(self):
        """--model is required."""
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_aggregation_formula_default(self):
        """--aggregation-formula defaults to mimic_mai."""
        parser = build_parser()
        args = parser.parse_args(["--model", "/fake/model"])
        self.assertEqual(args.aggregation_formula, "mimic_mai")

    def test_dry_run_flag(self):
        """--dry-run sets dry_run=True."""
        parser = build_parser()
        args = parser.parse_args(["--model", "/fake/model", "--dry-run"])
        self.assertTrue(args.dry_run)

    def test_held_out_root_default(self):
        """--held-out-root defaults to data/held_out/."""
        parser = build_parser()
        args = parser.parse_args(["--model", "/fake/model"])
        self.assertEqual(str(args.held_out_root), "data/held_out")


if __name__ == "__main__":
    unittest.main(verbosity=2)
