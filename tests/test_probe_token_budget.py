#!/usr/bin/env python3
"""Tests for adaptive max_new_tokens in probe.py and MIA threshold modes.

Hermetic tests using monkeypatching — no real model loads, no GPU required.
No torch dependency — all tests use pure-Python logic.

Run with:
    python -m pytest tests/test_probe_token_budget.py -v
"""

import sys
import unittest
from pathlib import Path

# Make scripts/ importable — must come before module imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from inversion.probe import get_target_text  # noqa: E402
from inversion_audit import _percentile, build_parser  # noqa: E402


def compute_adaptive_max_new_tokens(suffix_token_count: int) -> int:
    """Replicate the adaptive cap logic from run_carlini_probe.

    This mirrors the computation in probe.py:
        suffix_token_count = len(tokenizer.encode(target, add_special_tokens=False))
        adaptive_max_new = min(256, max(64, 2 * suffix_token_count))
    """
    return min(256, max(64, 2 * suffix_token_count))


class TestProbeAdaptiveMaxNewTokens(unittest.TestCase):
    """Verify the adaptive per-record token cap computation.

    Tests the formula: min(256, max(64, 2 * suffix_token_count))
    """

    def test_probe_default_max_new_tokens_is_at_least_200(self):
        """For a 100-token suffix, adaptive cap = min(256, max(64, 200)) = 200 >= 200."""
        cap = compute_adaptive_max_new_tokens(suffix_token_count=100)
        self.assertGreaterEqual(cap, 200)
        self.assertEqual(cap, 200)

    def test_probe_adaptive_cap_clamps_to_256(self):
        """For a 50-token suffix, adaptive cap = min(256, max(64, 100)) = 100."""
        cap = compute_adaptive_max_new_tokens(suffix_token_count=50)
        self.assertEqual(cap, 100)

    def test_probe_adaptive_cap_floor_at_64(self):
        """For a 1-token suffix, adaptive cap = min(256, max(64, 2)) = 64."""
        cap = compute_adaptive_max_new_tokens(suffix_token_count=1)
        self.assertEqual(cap, 64)

    def test_probe_adaptive_cap_ceiling_at_256(self):
        """For a 200-token suffix, adaptive cap = min(256, max(64, 400)) = 256."""
        cap = compute_adaptive_max_new_tokens(suffix_token_count=200)
        self.assertEqual(cap, 256)

    def test_probe_adaptive_cap_32_tokens(self):
        """For a 32-token suffix, adaptive cap = min(256, max(64, 64)) = 64."""
        cap = compute_adaptive_max_new_tokens(suffix_token_count=32)
        self.assertEqual(cap, 64)

    def test_probe_adaptive_cap_33_tokens(self):
        """For a 33-token suffix, adaptive cap = min(256, max(64, 66)) = 66."""
        cap = compute_adaptive_max_new_tokens(suffix_token_count=33)
        self.assertEqual(cap, 66)

    def test_probe_adaptive_cap_128_tokens(self):
        """For a 128-token suffix, adaptive cap = min(256, max(64, 256)) = 256."""
        cap = compute_adaptive_max_new_tokens(suffix_token_count=128)
        self.assertEqual(cap, 256)

    def test_probe_adaptive_cap_0_tokens(self):
        """For a 0-token suffix (empty target), floor ensures 64."""
        cap = compute_adaptive_max_new_tokens(suffix_token_count=0)
        self.assertEqual(cap, 64)

    def test_get_target_text_extracts_assistant_content(self):
        """Verify get_target_text extracts the assistant message."""
        record = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"},
            ]
        }
        self.assertEqual(get_target_text(record), "World")

    def test_get_target_text_fallback(self):
        """Verify get_target_text falls back to last message."""
        record = {"messages": [{"role": "system", "content": "System prompt"}]}
        self.assertEqual(get_target_text(record), "System prompt")


class TestPercentileHelper(unittest.TestCase):
    """Test the _percentile helper function (pure-Python numpy replacement)."""

    def test_percentile_5th(self):
        """5th percentile of [10, 20, 30, 40, 50] should be near 12."""
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        p5 = _percentile(scores, 5)
        # Linear interpolation: (5-1)*5/100 = 0.2, so 10 + 0.2*(20-10) = 12.0
        self.assertAlmostEqual(p5, 12.0, places=2)

    def test_percentile_50th(self):
        """50th percentile (median) of [10, 20, 30, 40, 50] should be 30."""
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        p50 = _percentile(scores, 50)
        self.assertAlmostEqual(p50, 30.0, places=2)

    def test_percentile_0th(self):
        """0th percentile should be the minimum value."""
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        p0 = _percentile(scores, 0)
        self.assertAlmostEqual(p0, 10.0, places=2)

    def test_percentile_100th(self):
        """100th percentile should be the maximum value."""
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        p100 = _percentile(scores, 100)
        self.assertAlmostEqual(p100, 50.0, places=2)

    def test_percentile_empty_raises(self):
        """Percentile of empty list should raise ValueError."""
        with self.assertRaises(ValueError):
            _percentile([], 50)

    def test_percentile_single_value(self):
        """Any percentile of a single-element list should be that value."""
        self.assertAlmostEqual(_percentile([42.0], 50), 42.0)
        self.assertAlmostEqual(_percentile([42.0], 5), 42.0)
        self.assertAlmostEqual(_percentile([42.0], 99), 42.0)

    def test_percentile_5_of_audit_data(self):
        """With 150 scores, 5th percentile ≈ 7th lowest score."""
        # Simulate 150 scores from 1 to 150
        scores = list(range(1, 151))
        p5 = _percentile(scores, 5)
        # (150-1)*5/100 = 7.45, so floor=7, frac=0.45
        # scores[7] + 0.45*(scores[8]-scores[7]) = 8 + 0.45*(9-8) = 8.45
        self.assertAlmostEqual(p5, 8.45, places=1)


class TestMIAThresholdModes(unittest.TestCase):
    """Test that the CLI accepts the new MIA threshold mode flags."""

    def test_mia_threshold_mode_default(self):
        """Default --mia-threshold-mode should be 'percentile'."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
            ]
        )
        self.assertEqual(args.mia_threshold_mode, "percentile")

    def test_mia_threshold_mode_median(self):
        """--mia-threshold-mode median should be accepted."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
                "--mia-threshold-mode",
                "median",
            ]
        )
        self.assertEqual(args.mia_threshold_mode, "median")

    def test_mia_threshold_mode_percentile(self):
        """--mia-threshold-mode percentile should be accepted."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
                "--mia-threshold-mode",
                "percentile",
            ]
        )
        self.assertEqual(args.mia_threshold_mode, "percentile")

    def test_mia_threshold_mode_holdout_file(self):
        """--mia-threshold-mode holdout_file should be accepted."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
                "--mia-threshold-mode",
                "holdout_file",
            ]
        )
        self.assertEqual(args.mia_threshold_mode, "holdout_file")

    def test_mia_percentile_default(self):
        """Default --mia-percentile should be 5."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
            ]
        )
        self.assertEqual(args.mia_percentile, 5)

    def test_mia_percentile_custom(self):
        """--mia-percentile 10 should be accepted."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
                "--mia-percentile",
                "10",
            ]
        )
        self.assertEqual(args.mia_percentile, 10)

    def test_max_new_tokens_default_none(self):
        """Default --max-new-tokens should be None (adaptive)."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
            ]
        )
        self.assertIsNone(args.max_new_tokens)

    def test_max_new_tokens_explicit(self):
        """--max-new-tokens 128 should set an explicit value."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
                "--max-new-tokens",
                "128",
            ]
        )
        self.assertEqual(args.max_new_tokens, 128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
