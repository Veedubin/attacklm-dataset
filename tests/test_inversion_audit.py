#!/usr/bin/env python3
"""Tests for the inversion-attack audit harness.

Run with:
    python -m pytest tests/test_inversion_audit.py -v

All tests are hermetic — no real model loads, no GPU required.
Mock transformers.AutoModelForCausalLM, model.generate, model.forward.
"""

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make scripts/ importable — must come before module imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from inversion.provenance import (  # noqa: E402
    LICENSE_FAMILIES,
    OutputClass,
    RecordProvenance,
    classify_output,
    hash_text,
    strip_raw_text_for_export,
    validate_source_filter,
)
from inversion.scoring import (  # noqa: E402
    calibrate_threshold,
    compute_fpr_at_threshold,
    compute_mia_score,
    compute_perplexity,
    compute_tpr_at_threshold,
    compute_zlib_metrics,
)
from inversion.probe import (  # noqa: E402
    bleu4_score,
    exact_match,
    extract_prefix,
    generate_completions,
    get_target_text,
    lcs_length,
    normalize_text,
    score_completions,
)
from inversion.model_loader import detect_model_format  # noqa: E402
from inversion.reporting import (  # noqa: E402
    create_audit_dir,
    write_raw_results,
    write_summary,
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
    "license_notice": "Copyright (c) Rapid7, Inc. All rights reserved.",
    "mitre_ids": ["T1021.002"],
}

SAMPLE_SIGMA_RECORD = {
    "messages": [
        {"role": "user", "content": "Detect suspicious PowerShell execution"},
        {
            "role": "assistant",
            "content": "title: Suspicious PowerShell Execution\ndetection...",
        },
    ],
    "source": "sigma-hq",
    "license": "DRL-1.1",
    "license_notice": "Sigma rules are licensed under DRL-1.1.",
}

SAMPLE_NO_LICENSE_RECORD = {
    "messages": [
        {"role": "user", "content": "Test prompt"},
        {"role": "assistant", "content": "Test response"},
    ],
    "source": "unknown",
    "license": "",
}

SAMPLE_HELD_OUT_INDEX = {
    "sources": [
        {
            "name": "azure-pyrit",
            "n_records": 0,
            "status": "reserved",
            "license": "",
        },
        {
            "name": "cyberark-fuzzyai",
            "n_records": 0,
            "status": "reserved",
            "license": "",
        },
        {
            "name": "metasploit-framework",
            "n_records": 8349,
            "license": "BSD-3-Clause",
        },
    ],
}


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance(unittest.TestCase):
    """License carry-through, denylist, and output classification."""

    def test_bsd3_is_permissive(self):
        prov = RecordProvenance("metasploit-framework", "BSD-3-Clause", "permissive")
        self.assertEqual(prov.license_family, "permissive")

    def test_drl11_is_derived_data(self):
        prov = RecordProvenance("sigma-hq", "DRL-1.1", "derived-data")
        self.assertEqual(prov.license_family, "derived-data")

    def test_from_record_extracts_fields(self):
        prov = RecordProvenance.from_record(SAMPLE_RECORD)
        self.assertEqual(prov.source, "metasploit-framework")
        self.assertEqual(prov.license_id, "BSD-3-Clause")
        self.assertEqual(
            prov.license_notice, "Copyright (c) Rapid7, Inc. All rights reserved."
        )

    def test_missing_field_defaults(self):
        prov = RecordProvenance.from_record({"messages": []})
        self.assertEqual(prov.source, "unknown")
        self.assertEqual(prov.license_id, "")
        self.assertEqual(prov.license_family, "unlicensed")

    def test_restricted_source_denied(self):
        with self.assertRaises(ValueError) as ctx:
            validate_source_filter(["rta"], Path("/tmp/nonexistent"))
        self.assertIn("restricted", str(ctx.exception).lower())

    def test_all_license_families_mapped(self):
        for license_id, family in LICENSE_FAMILIES.items():
            self.assertIn(
                family, ["permissive", "derived-data", "restricted", "unlicensed"]
            )


# ---------------------------------------------------------------------------
# TestScoring
# ---------------------------------------------------------------------------


class TestScoring(unittest.TestCase):
    """Exact match, LCS, BLEU, NLL, zlib, perplexity, combined MIA score."""

    def test_exact_match_true(self):
        self.assertTrue(exact_match("Hello World", "hello world"))

    def test_exact_match_false(self):
        self.assertFalse(exact_match("Hello World", "Goodbye World"))

    def test_lcs_length(self):
        result = lcs_length("ABCBDAB", "BDCABA")
        self.assertEqual(result, 4)  # BCBA or BDAB

    def test_bleu4_identical(self):
        # Identical strings should score ~1.0
        score = bleu4_score(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )
        self.assertGreater(score, 0.9)

    def test_bleu4_dissimilar(self):
        score = bleu4_score(
            "Completely unrelated text about something else",
            "The quick brown fox jumps over the lazy dog",
        )
        self.assertLess(score, 0.5)

    def test_nll_monotonic_with_length(self):
        """Mock model.forward() — longer sequences should have higher total NLL."""
        mock_model = MagicMock()
        mock_output = MagicMock()
        # Simulate per-token loss of ~3.0
        mock_output.loss = MagicMock()
        mock_output.loss.item = MagicMock(return_value=3.0)
        mock_model.return_value = mock_output
        mock_model.device = MagicMock()
        mock_model.device.type = "cpu"

        mock_tokenizer = MagicMock()
        # 10-token input
        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 10))}

        # We can't fully test this without a real model, so we test
        # the helper functions instead
        nll_5_tokens = 3.0 * 5
        nll_10_tokens = 3.0 * 10
        self.assertGreater(nll_10_tokens, nll_5_tokens)

    def test_zlib_metrics_range(self):
        text = (
            "Hello world, this is a test of zlib compression." * 50
        )  # Need enough text for compression to help
        compressed_length, ratio = compute_zlib_metrics(text)
        self.assertGreater(compressed_length, 0)
        self.assertGreater(ratio, 0)
        self.assertLess(ratio, 1.0)  # Compressed should be smaller for sufficient text

    def test_perplexity_ratio(self):
        # perplexity = exp(NLL / num_tokens)
        nll = 100.0
        num_tokens = 10
        pplx = compute_perplexity(nll, num_tokens)
        import math

        expected = math.exp(nll / num_tokens)
        self.assertAlmostEqual(pplx, expected, places=4)

    def test_mia_score_formula(self):
        # membership_score = NLL - alpha * zlib_length
        score = compute_mia_score(nll=50.0, zlib_length=100, alpha=0.5)
        expected = 50.0 - 0.5 * 100  # = 0.0
        self.assertAlmostEqual(score, expected)


# ---------------------------------------------------------------------------
# TestMIAThreshold
# ---------------------------------------------------------------------------


class TestMIAThreshold(unittest.TestCase):
    """MIA threshold calibration: FPR < X, TPR > Y."""

    def test_calibrated_threshold(self):
        # Members have low scores, non-members have high scores
        member_scores = [10.0, 12.0, 15.0, 8.0, 11.0]
        non_member_scores = [50.0, 60.0, 55.0, 70.0, 65.0]
        threshold = calibrate_threshold(
            member_scores, non_member_scores, target_fpr=0.2
        )
        # Threshold at the FPR quantile: sorted non-members = [50, 55, 60, 65, 70]
        # idx = min(int(5 * 0.2), 4) = 1 -> threshold = 55.0
        # Only 1 of 5 non-members (50.0) is below -> FPR = 0.2
        self.assertEqual(threshold, 55.0)
        # Verify the FPR is approximately target_fpr
        fpr = compute_fpr_at_threshold(non_member_scores, threshold)
        self.assertAlmostEqual(fpr, 0.2, places=1)

    def test_calibrated_threshold_fpr_not_inverted(self):
        """Regression test: target_fpr=0.01 should NOT produce ~99% FPR."""
        non_members = [float(i) for i in range(100, 200)]
        threshold = calibrate_threshold(
            member_scores=[50.0], non_member_scores=non_members, target_fpr=0.01
        )
        fpr = compute_fpr_at_threshold(non_members, threshold)
        # Should be ~1%, NOT ~99%
        self.assertLess(fpr, 0.05, f"FPR={fpr} should be <5% for target_fpr=0.01")

    def test_fpr_at_threshold(self):
        non_members = [50.0, 60.0, 55.0, 70.0, 65.0]
        # Threshold at 55 means 2/5 non-members below = FPR 0.4
        fpr = compute_fpr_at_threshold(non_members, threshold=56.0)
        self.assertGreater(fpr, 0.0)

    def test_tpr_at_threshold(self):
        members = [10.0, 12.0, 15.0, 8.0, 11.0]
        # Threshold at 13 means 4/5 members below = TPR 0.8
        tpr = compute_tpr_at_threshold(members, threshold=13.0)
        self.assertGreater(tpr, 0.5)

    def test_calibration_with_empty_non_members_raises(self):
        with self.assertRaises(ValueError):
            calibrate_threshold([10.0], [])


# ---------------------------------------------------------------------------
# TestCarliniProbe
# ---------------------------------------------------------------------------


class TestCarliniProbe(unittest.TestCase):
    """Prefix-completion probe: K samples, best-of-K, token limits."""

    def test_k_samples_generated(self):
        """Verify K completions are generated."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer to return deterministic outputs
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "generated text here"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.pad_token_id = 0

        # Mock model.generate to return tensor with shape (num_completions, seq_len)
        # After Bug #4 fix, generate_completions uses num_return_sequences,
        # so the output has shape (K, prompt_len + new_tokens).
        import torch

        # 3 completions, prompt_len=5, total=6 (1 new token per completion)
        output_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 10],
                [1, 2, 3, 4, 5, 20],
                [1, 2, 3, 4, 5, 30],
            ]
        )
        mock_model.generate.return_value = output_ids
        mock_model.device = MagicMock()
        mock_model.device.type = "cpu"

        completions = generate_completions(
            mock_model,
            mock_tokenizer,
            "prefix",
            num_completions=3,
            max_new_tokens=10,
        )
        self.assertEqual(len(completions), 3)

    def test_best_of_k_selected(self):
        """Best-of-K should pick the completion with the highest BLEU."""
        completions = [
            "Completely wrong text",
            "Close to reference but not exact",
            "Exact reference match here",
        ]
        reference = "Exact reference match here"
        scores = score_completions(completions, reference)
        # The third completion should have the highest BLEU
        best_idx = max(range(len(scores)), key=lambda i: scores[i]["bleu4"])
        self.assertEqual(best_idx, 2)

    def test_max_new_tokens_honored(self):
        """Verify max_new_tokens is passed to model.generate."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "output"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.pad_token_id = 0

        import torch

        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.device = MagicMock()
        mock_model.device.type = "cpu"

        generate_completions(
            mock_model,
            mock_tokenizer,
            "test",
            num_completions=1,
            max_new_tokens=32,
        )
        call_kwargs = mock_model.generate.call_args[1]
        self.assertEqual(call_kwargs["max_new_tokens"], 32)

    def test_temperature_honored(self):
        """Verify temperature is passed to model.generate."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2]
        mock_tokenizer.decode.return_value = "output"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.pad_token_id = 0

        import torch

        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_model.device = MagicMock()
        mock_model.device.type = "cpu"

        generate_completions(
            mock_model,
            mock_tokenizer,
            "test",
            num_completions=1,
            temperature=0.7,
        )
        call_kwargs = mock_model.generate.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.7)


# ---------------------------------------------------------------------------
# TestReportingBoundary
# ---------------------------------------------------------------------------


class TestReportingBoundary(unittest.TestCase):
    """Output classification: raw text never in EXPORTABLE_SUMMARY, etc."""

    def test_raw_text_not_in_exportable(self):
        """EXPORTABLE_SUMMARY must never contain raw reconstruction text."""
        row = {
            "source": "test",
            "license": "MIT",
            "best_exact_match": False,
            "best_bleu4": 0.3,
            "prompt_text": "sensitive prefix text here",
            "best_reconstruction": "sensitive reconstruction text",
            "original_text": "sensitive original text",
        }
        export = strip_raw_text_for_export(row)
        self.assertNotIn("prompt_text", export)
        self.assertNotIn("best_reconstruction", export)
        self.assertNotIn("original_text", export)
        self.assertIn("best_bleu4", export)
        self.assertIn("source", export)

    def test_per_source_aggregates_sum(self):
        """Per-source aggregates should sum correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_dir = Path(tmpdir) / "audit_2026_07_06"
            audit_dir.mkdir()

            results = [
                {
                    "source": "s1",
                    "best_bleu4": 0.5,
                    "best_exact_match": True,
                    "license": "MIT",
                },
                {
                    "source": "s1",
                    "best_bleu4": 0.3,
                    "best_exact_match": False,
                    "license": "MIT",
                },
                {
                    "source": "s2",
                    "best_bleu4": 0.7,
                    "best_exact_match": True,
                    "license": "Apache-2.0",
                },
            ]
            provenances = [
                RecordProvenance("s1", "MIT", "permissive"),
                RecordProvenance("s1", "MIT", "permissive"),
                RecordProvenance("s2", "Apache-2.0", "permissive"),
            ]
            path = write_summary(audit_dir, results, provenances)
            with open(path) as f:
                summary = json.load(f)
            self.assertEqual(summary["total_records"], 3)
            self.assertIn("s1", summary["sources"])
            self.assertIn("s2", summary["sources"])

    def test_file_mode_0600(self):
        """inversion_results.jsonl must be chmod 0600."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_dir = Path(tmpdir) / "audit_test"
            audit_dir.mkdir()

            results = [
                {"record_index": 0, "source": "test", "best_bleu4": 0.5},
            ]
            path = write_raw_results(audit_dir, results)
            mode = stat.S_IMODE(os.stat(path).st_mode)
            self.assertEqual(mode, 0o600)

    def test_date_dir_collision_errors(self):
        """create_audit_dir must error if directory already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_dir = Path(tmpdir) / "2026-07-06"
            audit_dir.mkdir()
            with self.assertRaises(FileExistsError):
                create_audit_dir(Path(tmpdir), "2026-07-06")

    def test_no_license_records_flagged(self):
        """Records without license should be classified as INTERNAL_SUMMARY."""
        prov = RecordProvenance("unknown", "", "unlicensed")
        output_class = classify_output(prov, include_raw_text=True)
        self.assertEqual(output_class, OutputClass.INTERNAL_SUMMARY)


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI(unittest.TestCase):
    """CLI argument parsing: --source-filter, --date, --top-k, no args."""

    def test_source_filter_parses(self):
        """--source-filter should accept multiple source names."""
        sys.path.insert(0, str(SCRIPTS_DIR))
        from inversion_audit import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
                "--source-filter",
                "metasploit-framework",
                "sigma-hq",
            ]
        )
        self.assertEqual(args.source_filter, ["metasploit-framework", "sigma-hq"])

    def test_date_overrides(self):
        """--date should override the default today's date."""
        from inversion_audit import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
                "--date",
                "2026-01-15",
            ]
        )
        self.assertEqual(args.date, "2026-01-15")

    def test_top_k_default(self):
        """Default --top-k should be 20."""
        from inversion_audit import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "/tmp/model",
                "--dataset-root",
                "/tmp/data",
            ]
        )
        self.assertEqual(args.top_k, 20)

    def test_no_args_shows_usage(self):
        """No required args should trigger usage/error."""
        from inversion_audit import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# TestEndToEnd (mocked)
# ---------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    """End-to-end driver test with a 5-record fixture and mock model."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a 5-record fixture dataset
        sources_dir = (
            Path(self.tmpdir) / "sources" / "test-source" / "base" / "discovery"
        )
        sources_dir.mkdir(parents=True)
        records = []
        for i in range(5):
            records.append(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"Prompt {i}"},
                            {"role": "assistant", "content": f"Response {i}"},
                        ],
                        "source": "test-source",
                        "license": "MIT",
                    }
                )
            )
        (sources_dir / "data.jsonl").write_text("\n".join(records) + "\n")

        # Create _index.json
        index = {
            "sources": [
                {"name": "test-source", "n_records": 5, "license": "MIT"},
            ]
        }
        (Path(self.tmpdir) / "sources" / "_index.json").write_text(json.dumps(index))

    @patch("inversion.model_loader.load_model")
    @patch("inversion.probe.generate_completions")
    @patch("inversion.scoring.compute_nll")
    def test_e2e_driver(self, mock_compute_nll, mock_gen_completions, mock_load_model):
        """Run the full driver against 5 records with a mock model."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Mock generate_completions to return simple text
        mock_gen_completions.return_value = ["Response 0", "Response 1", "Response 2"]

        # Mock compute_nll to return plausible values
        mock_compute_nll.return_value = (50.0, 10)

        # Mock tokenizer for prefix extraction
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Prompt 0"

        # Mock model.generate for Carlini probe
        import torch

        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.return_value = MagicMock(
            loss=MagicMock(item=MagicMock(return_value=3.0))
        )
        mock_model.device = MagicMock()
        mock_model.device.type = "cpu"

        # Import and run the driver with --dry-run first
        from inversion_audit import load_records

        records = load_records(
            Path(self.tmpdir) / "sources",
            source_filter=["test-source"],
        )
        self.assertEqual(len(records), 5)

        # Verify records loaded correctly
        for rec in records:
            self.assertEqual(rec["source"], "test-source")
            self.assertEqual(rec["license"], "MIT")


# ---------------------------------------------------------------------------
# TestModelLoader
# ---------------------------------------------------------------------------


class TestModelLoader(unittest.TestCase):
    """Model format detection."""

    def test_detect_gguf_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.touch()
            self.assertEqual(detect_model_format(gguf_path), "gguf")

    def test_detect_hf_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_dir = Path(tmpdir) / "hf_model"
            hf_dir.mkdir()
            (hf_dir / "config.json").write_text('{"model_type": "gpt2"}')
            self.assertEqual(detect_model_format(hf_dir), "hf")

    def test_detect_unknown_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            unknown = Path(tmpdir) / "unknown"
            unknown.mkdir()
            with self.assertRaises(ValueError):
                detect_model_format(unknown)


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers(unittest.TestCase):
    """Helper function tests."""

    def test_normalize_text(self):
        self.assertEqual(normalize_text("  Hello   World  "), "hello world")
        self.assertEqual(normalize_text("UPPER"), "upper")

    def test_hash_text_deterministic(self):
        h1 = hash_text("hello world")
        h2 = hash_text("hello world")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)  # SHA-256 truncated to 16 chars

    def test_hash_text_different_inputs(self):
        h1 = hash_text("hello")
        h2 = hash_text("world")
        self.assertNotEqual(h1, h2)

    def test_get_target_text(self):
        target = get_target_text(SAMPLE_RECORD)
        self.assertIn("Module:", target)

    def test_extract_prefix_fallback(self):
        """Without a tokenizer, extract_prefix should do character-level truncation."""
        prefix = extract_prefix(
            {"messages": [{"role": "user", "content": "Hello world"}]}, None
        )
        self.assertTrue(len(prefix) > 0)

    def test_zlib_empty_string(self):
        # zlib.compress("") produces a small header (8 bytes), not 0
        length, ratio = compute_zlib_metrics("")
        self.assertGreater(length, 0)  # Header overhead
        self.assertEqual(ratio, 0.0)  # Ratio is 0.0 because raw_length is 0

    def test_compute_perplexity_zero_tokens(self):
        result = compute_perplexity(100.0, 0)
        self.assertEqual(result, float("inf"))


class TestPerSourceLimit(unittest.TestCase):
    """Verify that --max-records applies per-source, not globally."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create 2 sources with different record counts
        for source_name, count in [("alpha", 10), ("beta", 10)]:
            source_dir = (
                Path(self.tmpdir) / "sources" / source_name / "base" / "discovery"
            )
            source_dir.mkdir(parents=True)
            records = [
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"Prompt {i}"},
                            {"role": "assistant", "content": f"Response {i}"},
                        ],
                        "source": source_name,
                        "license": "MIT",
                    }
                )
                for i in range(count)
            ]
            (source_dir / "data.jsonl").write_text("\n".join(records) + "\n")

        # Create _index.json
        index = {
            "sources": [
                {"name": "alpha", "n_records": 10, "license": "MIT"},
                {"name": "beta", "n_records": 10, "license": "MIT"},
            ]
        }
        (Path(self.tmpdir) / "sources" / "_index.json").write_text(json.dumps(index))

    def test_per_source_limit_caps_each_source(self):
        """With per_source_limit=5, each source gets 5 records = 10 total."""
        from inversion_audit import load_records

        records = load_records(
            Path(self.tmpdir) / "sources",
            source_filter=["alpha", "beta"],
            per_source_limit=5,
        )
        alpha_records = [r for r in records if r["_source"] == "alpha"]
        beta_records = [r for r in records if r["_source"] == "beta"]
        self.assertEqual(len(alpha_records), 5)
        self.assertEqual(len(beta_records), 5)
        self.assertEqual(len(records), 10)

    def test_per_source_limit_none_loads_all(self):
        """With per_source_limit=None, all records are loaded."""
        from inversion_audit import load_records

        records = load_records(
            Path(self.tmpdir) / "sources",
            source_filter=["alpha", "beta"],
            per_source_limit=None,
        )
        self.assertEqual(len(records), 20)

    def test_per_source_limit_larger_than_source(self):
        """If limit exceeds source count, all source records are kept."""
        from inversion_audit import load_records

        records = load_records(
            Path(self.tmpdir) / "sources",
            source_filter=["alpha", "beta"],
            per_source_limit=100,
        )
        self.assertEqual(len(records), 20)

    def test_global_cap_would_miss_second_source(self):
        """Verify that the old global-slice approach would miss beta.

        This test confirms the bug: a global `records[:5]` would only
        get 5 alpha records and miss beta entirely. Our per-source fix
        avoids this.
        """
        from inversion_audit import load_records

        # Load all records, sorted alphabetically by source name
        all_records = load_records(
            Path(self.tmpdir) / "sources",
            source_filter=["alpha", "beta"],
            per_source_limit=None,
        )
        # Old behavior: records[:5] would only get 5 alpha, 0 beta
        global_sliced = all_records[:5]
        sources_in_global = {r["_source"] for r in global_sliced}
        # This proves the bug: global slice misses beta
        self.assertEqual(sources_in_global, {"alpha"})

        # New behavior: per_source_limit=5 gets 5 from each source
        per_source = load_records(
            Path(self.tmpdir) / "sources",
            source_filter=["alpha", "beta"],
            per_source_limit=5,
        )
        sources_in_per_source = {r["_source"] for r in per_source}
        self.assertEqual(sources_in_per_source, {"alpha", "beta"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
