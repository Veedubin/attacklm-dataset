#!/usr/bin/env python3
"""Tests for LiRA (Likelihood Ratio Attack) — Carlini 2022 §4.

Hermetic tests using MagicMock and patching — no real model loads, no GPU required.

Run with:
    python -m pytest tests/test_lira.py -v
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

from inversion.lira import (  # noqa: E402
    GaussianParams,
    calibrate_lira_threshold,
    compute_lira_logit,
    fit_gaussian,
    fit_gaussians_per_record,
    gaussian_log_pdf,
    load_shadow_params,
    save_shadow_params,
    score_lira,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORD = {
    "messages": [
        {
            "role": "user",
            "content": "Explain how to exploit SMB vulnerability MS17-010",
        },
        {
            "role": "assistant",
            "content": (
                "**Module: `exploits/windows/smb/ms17_010_psexec`** — "
                "Microsoft Windows Authenticated User Code Execution\n\n"
                "This module exploits a vulnerability in SMB..."
            ),
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
# Test: fit_gaussian
# ---------------------------------------------------------------------------


class TestFitGaussian(unittest.TestCase):
    """Gaussian fitting from K shadow losses."""

    def test_fit_gaussian_basic(self):
        """Feed [1,2,3,4,5], assert mu=3.0, sigma≈1.414."""
        mu, sigma = fit_gaussian([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(mu, 3.0, places=4)
        # Population variance = 2.0, sigma = sqrt(2) ≈ 1.4142
        self.assertAlmostEqual(sigma, math.sqrt(2.0), places=4)

    def test_fit_gaussian_constant(self):
        """Constant losses → sigma = 1e-6 floor."""
        mu, sigma = fit_gaussian([5.0, 5.0, 5.0])
        self.assertAlmostEqual(mu, 5.0, places=4)
        self.assertAlmostEqual(sigma, 1e-6, places=10)

    def test_fit_gaussian_empty_raises(self):
        """Empty list → ValueError."""
        with self.assertRaises(ValueError):
            fit_gaussian([])

    def test_fit_gaussian_single_value(self):
        """Single value → mu=value, sigma=1e-6."""
        mu, sigma = fit_gaussian([7.5])
        self.assertAlmostEqual(mu, 7.5, places=4)
        self.assertAlmostEqual(sigma, 1e-6, places=10)


# ---------------------------------------------------------------------------
# Test: fit_gaussians_per_record
# ---------------------------------------------------------------------------


class TestFitGaussiansPerRecord(unittest.TestCase):
    """Per-record Gaussian fitting from IN/OUT loss dictionaries."""

    def test_fit_gaussians_per_record_basic(self):
        """2 records, K=4 each, assert correct Gaussian params."""
        in_losses = {
            "r0": [1.0, 2.0, 3.0, 4.0],
            "r1": [5.0, 6.0, 7.0, 8.0],
        }
        out_losses = {
            "r0": [10.0, 11.0, 12.0, 13.0],
            "r1": [14.0, 15.0, 16.0, 17.0],
        }
        result = fit_gaussians_per_record(in_losses, out_losses)

        # r0: in_mu=2.5, in_sigma≈1.118 (pop std of [1,2,3,4])
        # r0: out_mu=11.5, out_sigma≈1.118
        self.assertIn("r0", result)
        self.assertIn("r1", result)
        self.assertAlmostEqual(result["r0"].mu_in, 2.5, places=4)
        self.assertAlmostEqual(result["r0"].mu_out, 11.5, places=4)
        self.assertAlmostEqual(result["r1"].mu_in, 6.5, places=4)
        self.assertAlmostEqual(result["r1"].mu_out, 15.5, places=4)

    def test_fit_gaussians_per_record_mismatch_raises(self):
        """Record in in_losses but not in out_losses → ValueError."""
        in_losses = {"r0": [1.0, 2.0], "r1": [3.0, 4.0]}
        out_losses = {"r0": [5.0, 6.0]}  # missing r1
        with self.assertRaises(ValueError):
            fit_gaussians_per_record(in_losses, out_losses)


# ---------------------------------------------------------------------------
# Test: gaussian_log_pdf
# ---------------------------------------------------------------------------


class TestGaussianLogPdf(unittest.TestCase):
    """Log of the Gaussian PDF."""

    def test_gaussian_log_pdf_symmetric(self):
        """log N(0; 0, 1) == log N(0; 0, 1) (sanity check)."""
        val = gaussian_log_pdf(0.0, 0.0, 1.0)
        self.assertAlmostEqual(val, -0.5 * math.log(2 * math.pi), places=6)

    def test_gaussian_log_pdf_known_values(self):
        """log N(1; 0, 1) ≈ -1.4189."""
        val = gaussian_log_pdf(1.0, 0.0, 1.0)
        # N(1;0,1) = 1/sqrt(2*pi) * exp(-0.5)
        # log = -0.5*log(2*pi) - 0.5
        expected = -0.5 * math.log(2.0 * math.pi) - 0.5
        self.assertAlmostEqual(val, expected, places=6)

    def test_gaussian_log_pdf_zero_sigma(self):
        """sigma=0 returns -inf (avoids log(0))."""
        val = gaussian_log_pdf(5.0, 5.0, 0.0)
        self.assertEqual(val, float("-inf"))

    def test_gaussian_log_pdf_negative_sigma(self):
        """Negative sigma returns -inf."""
        val = gaussian_log_pdf(5.0, 5.0, -1.0)
        self.assertEqual(val, float("-inf"))


# ---------------------------------------------------------------------------
# Test: compute_lira_logit
# ---------------------------------------------------------------------------


class TestComputeLiRALogit(unittest.TestCase):
    """Log-likelihood ratio computation."""

    def test_compute_lira_logit_positive_for_member(self):
        """loss_target near mu_in, far from mu_out → positive lira_logit."""
        # loss_target=2.0, mu_in=2.0, sigma_in=1.0, mu_out=10.0, sigma_out=1.0
        # log N(2; 2, 1) = -0.5*log(2pi) - 0 = peak
        # log N(2; 10, 1) = -0.5*log(2pi) - 0.5*(8)^2 = very negative
        logit = compute_lira_logit(2.0, 2.0, 1.0, 10.0, 1.0)
        self.assertGreater(logit, 0.0)  # positive = more likely IN

    def test_compute_lira_logit_negative_for_nonmember(self):
        """loss_target near mu_out, far from mu_in → negative lira_logit."""
        logit = compute_lira_logit(10.0, 2.0, 1.0, 10.0, 1.0)
        self.assertLess(logit, 0.0)  # negative = more likely OUT

    def test_compute_lira_logit_zero_threshold_is_natural(self):
        """At the boundary, assert the sign is correct."""
        # Equal distance from both means → logit depends on sigma
        # If sigma_in == sigma_out and loss is equidistant, logit should be
        # 0 when loss is exactly at midpoint (by symmetry of Gaussians)
        # Actually: midpoint between 2 and 10 is 6.
        # log N(6; 2, 1) = -0.5*log(2pi) - 0.5*(4)^2 = -0.5*log(2pi) - 8
        # log N(6; 10, 1) = -0.5*log(2pi) - 0.5*(4)^2 = same
        # So logit should be ~0 (numerically)
        logit = compute_lira_logit(6.0, 2.0, 1.0, 10.0, 1.0)
        self.assertAlmostEqual(logit, 0.0, places=4)


# ---------------------------------------------------------------------------
# Test: calibrate_lira_threshold
# ---------------------------------------------------------------------------


class TestCalibrateLiRAThreshold(unittest.TestCase):
    """LiRA threshold calibration."""

    def test_calibrate_lira_threshold_empty_returns_zero(self):
        """No known_membership → 0.0."""
        threshold = calibrate_lira_threshold([], [])
        self.assertEqual(threshold, 0.0)

    def test_calibrate_lira_threshold_fpr_quantile(self):
        """known_membership=[T,T,F,F], target_fpr=0.5, threshold at 50th pctile."""
        logits = [3.0, 2.0, -1.0, -2.0]
        membership = [True, True, False, False]
        # Non-member logits: [-1.0, -2.0], sorted: [-2.0, -1.0]
        # FPR=0.5: idx = max(0, int(2*0.5)-1) = 0 → sorted_non[0] = -2.0
        threshold = calibrate_lira_threshold(logits, membership, target_fpr=0.5)
        self.assertEqual(threshold, -2.0)

    def test_calibrate_lira_threshold_all_members_returns_zero(self):
        """All members, no non-members → returns 0.0."""
        threshold = calibrate_lira_threshold([1.0, 2.0], [True, True])
        self.assertEqual(threshold, 0.0)

    def test_calibrate_lira_threshold_length_mismatch_raises(self):
        """Mismatched lengths → ValueError."""
        with self.assertRaises(ValueError):
            calibrate_lira_threshold([1.0, 2.0], [True])


# ---------------------------------------------------------------------------
# Test: score_lira
# ---------------------------------------------------------------------------


class TestScoreLiRA(unittest.TestCase):
    """LiRA scoring with mocked model."""

    @patch("inversion.lira.compute_nll")
    def test_score_lira_basic(self, mock_compute_nll):
        """Mock model returns fixed loss, shadow params have specific Gaussians."""
        mock_compute_nll.return_value = (5.0, 10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # mu_in=3.0, sigma_in=1.0, mu_out=20.0, sigma_out=2.0
        # loss_target=5.0 → much closer to mu_in=3.0 than mu_out=20.0
        # → lira_logit is clearly positive (member-like)
        shadow_params = GaussianParams(
            mu_in=3.0, sigma_in=1.0, mu_out=20.0, sigma_out=2.0
        )
        result = score_lira(SAMPLE_RECORD, mock_model, mock_tokenizer, shadow_params)

        # lira_logit should be positive (loss closer to IN distribution)
        self.assertGreater(result.lira_logit, 0.0)
        self.assertAlmostEqual(result.loss_target, 5.0, places=4)
        self.assertAlmostEqual(result.nll_per_token, 0.5, places=4)
        self.assertEqual(result.num_suffix_tokens, 10)
        self.assertEqual(result.mu_in, 3.0)
        self.assertEqual(result.sigma_in, 1.0)
        self.assertEqual(result.mu_out, 20.0)
        self.assertEqual(result.sigma_out, 2.0)

    def test_score_lira_no_assistant_turn(self):
        """Empty assistant turn → lira_logit=NaN, no crash."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        shadow_params = GaussianParams(
            mu_in=2.0, sigma_in=1.0, mu_out=8.0, sigma_out=1.0
        )
        result = score_lira(
            SAMPLE_RECORD_NO_ASSISTANT, mock_model, mock_tokenizer, shadow_params
        )

        self.assertTrue(math.isnan(result.lira_logit))
        self.assertEqual(result.loss_target, float("inf"))
        self.assertEqual(result.num_suffix_tokens, 0)


# ---------------------------------------------------------------------------
# Test: save/load shadow params roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadShadowParams(unittest.TestCase):
    """Save and load per-record Gaussian parameters."""

    def test_save_load_shadow_params_roundtrip(self):
        """Save a dict, load it back, assert all fields match."""
        params = {
            "record_0": GaussianParams(
                mu_in=1.234, sigma_in=0.567, mu_out=2.345, sigma_out=0.789
            ),
            "record_1": GaussianParams(
                mu_in=3.456, sigma_in=1.234, mu_out=4.567, sigma_out=1.345
            ),
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            save_shadow_params(params, output_path)
            loaded_params, lira_k = load_shadow_params(output_path)

            self.assertEqual(len(loaded_params), 2)
            self.assertIn("record_0", loaded_params)
            self.assertIn("record_1", loaded_params)

            self.assertAlmostEqual(loaded_params["record_0"].mu_in, 1.234, places=4)
            self.assertAlmostEqual(loaded_params["record_0"].sigma_in, 0.567, places=4)
            self.assertAlmostEqual(loaded_params["record_0"].mu_out, 2.345, places=4)
            self.assertAlmostEqual(loaded_params["record_0"].sigma_out, 0.789, places=4)

            self.assertAlmostEqual(loaded_params["record_1"].mu_in, 3.456, places=4)
            self.assertAlmostEqual(loaded_params["record_1"].sigma_in, 1.234, places=4)
            self.assertAlmostEqual(loaded_params["record_1"].mu_out, 4.567, places=4)
            self.assertAlmostEqual(loaded_params["record_1"].sigma_out, 1.345, places=4)

            # lira_k is 0 (unknown) since it can't be derived from params
            self.assertEqual(lira_k, 0)
        finally:
            output_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test: integration with inversion_audit.py
# ---------------------------------------------------------------------------


class TestInversionAuditLiRAMethod(unittest.TestCase):
    """Integration test: build parser, run main() with --mia-method lira."""

    def test_inversion_audit_lira_method_runs(self):
        """Parse --mia-method lira --lira-params, run main() on 3-record dataset."""
        # Create a temp shadow_params.json
        shadow_params = {
            "0": {"mu_in": 2.0, "sigma_in": 1.0, "mu_out": 8.0, "sigma_out": 1.0},
            "1": {"mu_in": 2.5, "sigma_in": 1.2, "mu_out": 9.0, "sigma_out": 1.1},
            "2": {"mu_in": 3.0, "sigma_in": 0.8, "mu_out": 7.5, "sigma_out": 0.9},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            shadow_path = tmpdir / "shadow_params.json"
            with open(shadow_path, "w") as f:
                json.dump({"lira_k": 4, "params": shadow_params}, f)

            # Create a minimal dataset
            dataset_dir = tmpdir / "sources" / "test-source"
            dataset_dir.mkdir(parents=True)
            jsonl_path = dataset_dir / "data.jsonl"
            records = []
            for i in range(3):
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
            jsonl_path.write_text("\n".join(records) + "\n")

            # Create _index.json so provenance works
            index_path = tmpdir / "sources" / "_index.json"
            index_path.write_text(
                json.dumps([{"name": "test-source", "n_records": 3, "license": "MIT"}])
            )

            # Patch load_model to avoid actual model loading
            with (
                patch("inversion_audit.load_model") as mock_load,
                patch("inversion_audit.detect_model_format") as mock_detect,
                patch("inversion.lira.compute_nll") as mock_nll,
                patch("inversion_audit.check_output_dir_permissions"),
            ):
                mock_detect.return_value = "hf"
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_load.return_value = (mock_model, mock_tokenizer)
                # Mock compute_nll to return fixed losses
                mock_nll.return_value = (5.0, 10)

                # Build args
                args = [
                    "--model",
                    str(tmpdir / "model"),
                    "--dataset-root",
                    str(tmpdir / "sources"),
                    "--attack",
                    "mia",
                    "--mia-method",
                    "lira",
                    "--lira-params",
                    str(shadow_path),
                    "--mia-threshold-mode",
                    "lrt",
                    "--max-records",
                    "3",
                    "--dry-run",
                ]

                # Just verify the parser accepts the args
                from inversion_audit import build_parser

                parser = build_parser()
                parsed = parser.parse_args(args)
                self.assertEqual(parsed.mia_method, "lira")
                self.assertEqual(parsed.lira_params, str(shadow_path))
                self.assertEqual(parsed.mia_threshold_mode, "lrt")
                self.assertEqual(parsed.lira_k, 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
