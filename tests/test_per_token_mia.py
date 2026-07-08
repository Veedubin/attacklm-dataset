#!/usr/bin/env python3
"""Tests for per-token MIA scoring (MUSE 2023 default).

Hermetic tests using MagicMock and patching — no real model loads, no GPU required.

Run with:
    python -m pytest tests/test_per_token_mia.py -v
"""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make scripts/ importable — must come before module imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from inversion.scoring import (  # noqa: E402
    compute_zlib_metrics,
    score_per_token,
    score_record,
    zscore_normalize,
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
}

SAMPLE_RECORD_LONG_PREAMBLE = {
    "messages": [
        {
            "role": "user",
            "content": "long preamble " * 100,
        },
        {
            "role": "assistant",
            "content": "short response",
        },
    ],
    "source": "test-source",
    "license": "MIT",
}

SAMPLE_RECORD_NO_ASSISTANT = {
    "messages": [
        {"role": "user", "content": "Just a prompt, no response"},
    ],
    "source": "test-source",
    "license": "MIT",
}


# ---------------------------------------------------------------------------
# Test: score_per_token
# ---------------------------------------------------------------------------


class TestScorePerToken(unittest.TestCase):
    """Per-token MIA scoring: MUSE 2023 default."""

    @patch("inversion.scoring.compute_nll")
    def test_score_per_token_basic(self, mock_compute_nll):
        """Mock model returns NLL=300, 100 tokens → nll_per_token=3.0."""
        mock_compute_nll.return_value = (300.0, 100)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = score_per_token(SAMPLE_RECORD, mock_model, mock_tokenizer)
        self.assertAlmostEqual(result.nll_per_token, 3.0, places=4)
        self.assertAlmostEqual(result.nll_total, 300.0, places=4)
        self.assertEqual(result.num_suffix_tokens, 100)
        self.assertEqual(result.alpha, 1.0)

    def test_score_per_token_zero_tokens(self):
        """Empty assistant turn → returns num_suffix_tokens=0, nll_per_token=inf, no crash."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = score_per_token(SAMPLE_RECORD_NO_ASSISTANT, mock_model, mock_tokenizer)
        self.assertEqual(result.num_suffix_tokens, 0)
        self.assertEqual(result.nll_per_token, float("inf"))
        self.assertEqual(result.membership_score, float("inf"))
        self.assertEqual(result.suffix_text, "")

    @patch("inversion.scoring.compute_nll")
    def test_score_per_token_zlib_ratio_computed(self, mock_compute_nll):
        """zlib_ratio field is the per-byte ratio (not per-token)."""
        mock_compute_nll.return_value = (30.0, 10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = score_per_token(SAMPLE_RECORD, mock_model, mock_tokenizer)
        # zlib_ratio should be > 0 and < 1 for meaningful text
        self.assertGreater(result.zlib_ratio, 0.0)
        self.assertLess(result.zlib_ratio, 1.0)
        # Verify it matches the per-byte ratio from compute_zlib_metrics
        suffix_text = SAMPLE_RECORD["messages"][1]["content"]
        _, expected_ratio = compute_zlib_metrics(suffix_text)
        self.assertAlmostEqual(result.zlib_ratio, expected_ratio, places=6)

    @patch("inversion.scoring.compute_nll")
    def test_score_per_token_long_suffix(self, mock_compute_nll):
        """500-token suffix scores same as 5-token suffix if per-token NLL is equal."""
        # 5-token suffix: NLL=15.0, tokens=5 → nll_per_token=3.0
        mock_compute_nll.return_value = (15.0, 5)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        result_5 = score_per_token(SAMPLE_RECORD, mock_model, mock_tokenizer)
        self.assertAlmostEqual(result_5.nll_per_token, 3.0, places=4)

        # 500-token suffix: NLL=1500.0, tokens=500 → nll_per_token=3.0
        mock_compute_nll.return_value = (1500.0, 500)
        result_500 = score_per_token(SAMPLE_RECORD, mock_model, mock_tokenizer)
        self.assertAlmostEqual(result_500.nll_per_token, 3.0, places=4)

        # nll_total differs but nll_per_token is the same — length bias removed
        self.assertAlmostEqual(result_5.nll_total, 15.0, places=4)
        self.assertAlmostEqual(result_500.nll_total, 1500.0, places=4)

    @patch("inversion.scoring.compute_nll")
    def test_score_per_token_full_record_separation(self, mock_compute_nll):
        """Per-token uses the assistant turn, not the full record (different input)."""
        # When compute_nll is called, it should receive only the assistant
        # turn text ("short response"), not the full record including preamble.
        # We'll verify by inspecting what text was passed to compute_nll.
        mock_compute_nll.return_value = (6.0, 1)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        score_per_token(SAMPLE_RECORD_LONG_PREAMBLE, mock_model, mock_tokenizer)

        # Verify compute_nll was called with "short response" (the assistant turn)
        call_args = mock_compute_nll.call_args
        # compute_nll(model, tokenizer, text) — text is 3rd positional arg
        self.assertEqual(call_args[0][2], "short response")

        # Also verify _extract_assistant_turn directly
        from inversion.scoring import _extract_assistant_turn

        assistant_turn = _extract_assistant_turn(SAMPLE_RECORD_LONG_PREAMBLE)
        self.assertEqual(assistant_turn, "short response")
        self.assertNotIn("long preamble", assistant_turn)

    @patch("inversion.scoring.compute_nll")
    def test_score_per_token_calibration_zscore(self, mock_compute_nll):
        """Feed a 50-element distribution, assert z-score normalization is correct."""
        # This test is about zscore_normalize, not score_per_token directly,
        # but we verify the scoring produces values that can be z-scored.
        mock_compute_nll.return_value = (30.0, 10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = score_per_token(SAMPLE_RECORD, mock_model, mock_tokenizer)
        self.assertAlmostEqual(result.nll_per_token, 3.0, places=4)

        # Now test z-score normalization on a distribution of membership_scores
        scores = [1.0, 2.0, 3.0, 4.0, 5.0] * 10  # 50 scores
        z_scores = zscore_normalize(scores)
        mean_z = sum(z_scores) / len(z_scores)
        self.assertAlmostEqual(mean_z, 0.0, places=10)
        # std should be ~1.0
        var_z = sum((z - mean_z) ** 2 for z in z_scores) / len(z_scores)
        self.assertAlmostEqual(var_z, 1.0, places=4)

    @patch("inversion.scoring.compute_nll")
    def test_score_per_token_alpha_zero(self, mock_compute_nll):
        """alpha=0 disables zlib term, membership_score == nll_per_token."""
        mock_compute_nll.return_value = (30.0, 10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = score_per_token(SAMPLE_RECORD, mock_model, mock_tokenizer, alpha=0.0)
        self.assertAlmostEqual(result.membership_score, result.nll_per_token, places=6)

    @patch("inversion.scoring.compute_nll")
    def test_score_per_token_alpha_one_default(self, mock_compute_nll):
        """alpha=1.0 default."""
        mock_compute_nll.return_value = (30.0, 10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = score_per_token(SAMPLE_RECORD, mock_model, mock_tokenizer)
        self.assertEqual(result.alpha, 1.0)
        # membership_score = nll_per_token - 1.0 * zlib_ratio
        expected = result.nll_per_token - 1.0 * result.zlib_ratio
        self.assertAlmostEqual(result.membership_score, expected, places=6)

    @patch("inversion.scoring.compute_nll")
    def test_score_per_token_does_not_modify_record(self, mock_compute_nll):
        """score_per_token should not mutate the input record."""
        mock_compute_nll.return_value = (30.0, 10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        record_copy = {
            "messages": [
                {"role": "user", "content": "Test prompt"},
                {"role": "assistant", "content": "Test response"},
            ],
            "source": "test",
            "license": "MIT",
        }
        import copy

        original = copy.deepcopy(record_copy)
        score_per_token(record_copy, mock_model, mock_tokenizer)
        self.assertEqual(record_copy, original)

    @patch("inversion.scoring.compute_nll")
    def test_score_per_token_integration_with_score_record(self, mock_compute_nll):
        """Score the same record both ways; assert both succeed and produce different keys."""
        mock_compute_nll.return_value = (30.0, 10)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        full_score = score_record(SAMPLE_RECORD, mock_model, mock_tokenizer)
        pt_score = score_per_token(SAMPLE_RECORD, mock_model, mock_tokenizer)

        # Both should succeed
        self.assertIsNotNone(full_score)
        self.assertIsNotNone(pt_score)

        # Both have membership_score but with different semantics
        self.assertIn("membership_score", full_score.__dataclass_fields__)
        self.assertIn("membership_score", pt_score.__dataclass_fields__)

        # Per-token score has fields that full-record doesn't
        self.assertIn("nll_per_token", pt_score.__dataclass_fields__)
        self.assertIn("num_suffix_tokens", pt_score.__dataclass_fields__)
        self.assertIn("suffix_text", pt_score.__dataclass_fields__)

        # Full-record score has fields that per-token doesn't
        self.assertIn("zlib_length", full_score.__dataclass_fields__)
        self.assertIn("perplexity", full_score.__dataclass_fields__)


# ---------------------------------------------------------------------------
# Test: zscore_normalize
# ---------------------------------------------------------------------------


class TestZscoreNormalize(unittest.TestCase):
    """Z-score normalization for cross-source MIA threshold calibration."""

    def test_zscore_typical_distribution(self):
        """Z-score of a typical distribution should have mean ~0 and std ~1."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = zscore_normalize(scores)
        # Mean of [1,2,3,4,5] = 3.0, variance = 2.0, std = sqrt(2) ≈ 1.414
        mean_result = sum(result) / len(result)
        self.assertAlmostEqual(mean_result, 0.0, places=10)
        # Check specific values
        self.assertAlmostEqual(result[2], 0.0, places=10)  # (3-3)/sqrt(2) = 0
        self.assertAlmostEqual(result[0], -2.0 / math.sqrt(2), places=4)

    def test_zscore_constant_distribution(self):
        """Constant distribution (std=0) returns all zeros."""
        scores = [5.0, 5.0, 5.0, 5.0]
        result = zscore_normalize(scores)
        self.assertEqual(result, [0.0, 0.0, 0.0, 0.0])

    def test_zscore_empty_list(self):
        """Empty list returns empty list."""
        result = zscore_normalize([])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
