#!/usr/bin/env python3
"""Tests for offline MIA (K=0) — compute_offline_z and audit driver.

Hermetic tests using MagicMock and patching — no real model loads, no GPU required.

Run with:
    python -m pytest tests/test_offline_mia.py -v
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make scripts/ importable — must come before module imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from inversion.scoring import compute_offline_z  # noqa: E402


# ---------------------------------------------------------------------------
# Test: compute_offline_z
# ---------------------------------------------------------------------------


class TestComputeOfflineZ(unittest.TestCase):
    """Unit tests for compute_offline_z."""

    def test_compute_offline_z_basic(self):
        """Input [1, 2, ..., 30], check mu=15.5, sigma correct, z_scores length."""
        nlls = [float(i) for i in range(1, 31)]  # 1..30
        mu_out, sigma_out, z_scores = compute_offline_z(nlls)

        # mu = (1+30)/2 = 15.5
        self.assertAlmostEqual(mu_out, 15.5, places=4)

        # Population variance of 1..30:
        # mean = 15.5, variance = sum((x-15.5)^2)/30
        # sum of squares from -14.5 to 14.5 step 1
        # = 2 * sum_{k=1}^{14} k^2 + 0^2 + 14.5^2 + 15.5^2 ... wait let's just check
        # Actually: variance = (n^2 - 1) / 12 for uniform 1..n
        # For 1..30: (900 - 1) / 12 = 899/12 ≈ 74.9167
        # sigma = sqrt(74.9167) ≈ 8.655
        expected_variance = (30**2 - 1) / 12.0
        expected_sigma = math.sqrt(expected_variance)
        self.assertAlmostEqual(sigma_out, expected_sigma, places=4)

        # z_scores should have length 30
        self.assertEqual(len(z_scores), 30)

        # First z-score: (1 - 15.5) / sigma ≈ -14.5 / 8.655 ≈ -1.675
        expected_z0 = (1.0 - 15.5) / expected_sigma
        self.assertAlmostEqual(z_scores[0], expected_z0, places=4)

        # Last z-score: (30 - 15.5) / sigma ≈ 14.5 / 8.655 ≈ 1.675
        expected_z29 = (30.0 - 15.5) / expected_sigma
        self.assertAlmostEqual(z_scores[29], expected_z29, places=4)

    def test_compute_offline_z_constant_sigma_0(self):
        """Constant input [5.0]*30 → all z-scores = 0.0."""
        nlls = [5.0] * 30
        mu_out, sigma_out, z_scores = compute_offline_z(nlls)

        self.assertAlmostEqual(mu_out, 5.0, places=4)
        self.assertEqual(sigma_out, 0.0)
        self.assertEqual(len(z_scores), 30)
        for z in z_scores:
            self.assertEqual(z, 0.0)

    def test_compute_offline_z_too_few_records_raises(self):
        """N=3 raises ValueError mentioning N>=30."""
        nlls = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError) as ctx:
            compute_offline_z(nlls)
        self.assertIn("30", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))

    def test_compute_offline_z_exactly_30_ok(self):
        """Exactly 30 records works (boundary case)."""
        nlls = [float(i) * 0.1 for i in range(30)]
        mu_out, sigma_out, z_scores = compute_offline_z(nlls)
        self.assertAlmostEqual(mu_out, 1.45, places=4)
        self.assertEqual(len(z_scores), 30)


# ---------------------------------------------------------------------------
# Test: audit driver integration
# ---------------------------------------------------------------------------


class TestAuditOfflineMethod(unittest.TestCase):
    """Integration test: run inversion_audit.py with --mia-method offline."""

    def test_audit_offline_method_field_names(self):
        """Run the audit driver with --mia-method offline and verify offline_z field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a minimal dataset with 30 records (required by compute_offline_z)
            dataset_dir = tmpdir / "sources" / "test-source"
            dataset_dir.mkdir(parents=True)
            records = []
            for i in range(30):
                records.append(
                    json.dumps(
                        {
                            "id": str(i),
                            "messages": [
                                {"role": "user", "content": f"Test prompt {i}"},
                                {"role": "assistant", "content": f"Test response {i}"},
                            ],
                            "source": "test-source",
                            "license": "MIT",
                        }
                    )
                )
            jsonl_path = dataset_dir / "data.jsonl"
            jsonl_path.write_text("\n".join(records) + "\n")

            # Create _index.json
            index_path = tmpdir / "sources" / "_index.json"
            index_path.write_text(
                json.dumps([{"name": "test-source", "n_records": 30, "license": "MIT"}])
            )

            # Patch load_model and compute_nll to avoid real model loading
            with (
                patch("inversion_audit.load_model") as mock_load,
                patch("inversion_audit.detect_model_format") as mock_detect,
                patch("inversion.scoring.compute_nll") as mock_nll,
                patch("inversion_audit.check_output_dir_permissions"),
                patch("inversion_audit.create_audit_dir") as mock_create_dir,
            ):
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_load.return_value = (mock_model, mock_tokenizer)
                mock_detect.return_value = "hf"
                # Return varying NLLs so compute_offline_z gets non-constant input
                mock_nll.return_value = (5.0, 10)

                # Mock create_audit_dir to return a temp path
                audit_dir = tmpdir / "audit_output"
                audit_dir.mkdir(parents=True, exist_ok=True)
                mock_create_dir.return_value = audit_dir

                from inversion_audit import main as audit_main

                exit_code = audit_main(
                    [
                        "--model",
                        str(tmpdir / "model"),
                        "--dataset-root",
                        str(tmpdir / "sources"),
                        "--attack",
                        "mia",
                        "--mia-method",
                        "offline",
                        "--max-records",
                        "30",
                    ]
                )
                self.assertEqual(exit_code, 0)

            # Check that output JSON was written with offline_z field
            results_path = audit_dir / "inversion_results.jsonl"
            self.assertTrue(results_path.exists())
            with open(results_path) as f:
                lines = f.readlines()
            self.assertGreater(len(lines), 0)
            first = json.loads(lines[0])
            self.assertIn("offline_z", first)
            self.assertIn("offline_flagged", first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
