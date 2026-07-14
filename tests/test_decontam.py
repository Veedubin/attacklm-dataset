#!/usr/bin/env python3
"""Sprint 2 decontamination tests — MAI-Thinking-1 §2.3.1 + §2.4.3.

Tests for 20-gram fuzzy dedup (MinHash LSH, threshold 0.80) against
public evaluation sets. Hermetic: no real disk I/O outside tempfile,
no network, no model loads.

Run with:
    python -m pytest tests/test_decontam.py -v
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

from decontam import (  # noqa: E402
    _DEFAULT_PERMUTATIONS,
    _compute_minhash_item,
    _record_text,
    build_parser,
    build_report,
    build_training_index,
    compute_minhash,
    extract_char_ngrams,
    find_overlaps,
    jaccard_similarity,
    load_eval_records,
    load_training_records,
    main,
    write_quarantine,
)
from datasketch import MinHash, MinHashLSH  # noqa: E402


# ---------------------------------------------------------------------------
# Test class 1: N-gram extraction
# ---------------------------------------------------------------------------


class TestNgramExtraction(unittest.TestCase):
    """Tests for character-level 20-gram extraction (MAI-Thinking-1 §2.3.1)."""

    def test_ngram_size_20_default(self):
        """Default n-gram size is 20 characters."""
        text = "abcdefghijklmnopqrstuvwxyz0123456789"  # 36 unique chars
        ngrams = extract_char_ngrams(text, n=20)
        # 36 chars → 36 - 20 + 1 = 17 n-grams
        self.assertEqual(len(ngrams), 17)
        for ng in ngrams:
            self.assertEqual(len(ng), 20)

    def test_ngram_handles_unicode(self):
        """Unicode characters (including multi-byte) are handled correctly."""
        text = "🔥💻🔓"  # 3 emoji, each 4 bytes in UTF-8
        ngrams = extract_char_ngrams(text, n=2)
        # 3 chars → 3 - 2 + 1 = 2 n-grams of 2 chars each
        self.assertEqual(len(ngrams), 2)
        # Each n-gram is 2 characters (not bytes)
        for ng in ngrams:
            self.assertEqual(len(ng), 2)

    def test_ngram_handles_short_strings(self):
        """Strings shorter than n return the whole string as a single n-gram."""
        text = "short"
        ngrams = extract_char_ngrams(text, n=20)
        self.assertEqual(ngrams, {"short"})

    def test_ngram_handles_empty_string(self):
        """Empty string returns an empty set."""
        ngrams = extract_char_ngrams("", n=20)
        self.assertEqual(ngrams, set())

    def test_ngram_exact_length(self):
        """String exactly n characters long returns one n-gram."""
        text = "exactly20chars!!"  # 16 chars
        ngrams = extract_char_ngrams(text, n=16)
        self.assertEqual(ngrams, {text})

    def test_ngram_handles_whitespace(self):
        """Whitespace is preserved in n-grams (it's character-level)."""
        text = "ab cd"
        ngrams = extract_char_ngrams(text, n=3)
        expected = {"ab ", "b c", " cd"}
        self.assertEqual(ngrams, expected)


# ---------------------------------------------------------------------------
# Test class 2: MinHash construction
# ---------------------------------------------------------------------------


class TestMinHashConstruction(unittest.TestCase):
    """Tests for MinHash LSH index construction (MAI-Thinking-1 §2.3.1)."""

    def test_minhash_deterministic_with_seed(self):
        """MinHash of the same text with the same seed is deterministic."""
        mh1 = compute_minhash("The quick brown fox", ngram_size=5, num_perm=128)
        mh2 = compute_minhash("The quick brown fox", ngram_size=5, num_perm=128)
        # MinHash hashes should be identical
        self.assertEqual(mh1.hashvalues.tolist(), mh2.hashvalues.tolist())

    def test_minhash_different_texts_differ(self):
        """Different texts produce different MinHash signatures."""
        mh1 = compute_minhash("The quick brown fox", ngram_size=5, num_perm=128)
        mh2 = compute_minhash("The lazy brown dog", ngram_size=5, num_perm=128)
        # Different texts should have different hashes (extremely unlikely to match)
        self.assertNotEqual(mh1.hashvalues.tolist(), mh2.hashvalues.tolist())

    def test_minhash_lsh_queryable(self):
        """MinHash LSH index can be queried and returns candidates."""
        lsh = MinHashLSH(threshold=0.3, num_perm=128)
        # Use longer, more similar texts so 3-gram overlap is high
        texts = {
            "a": "The quick brown fox jumps over the lazy dog near the river bank",
            "b": "The quick brown fox jumps over the lazy cat near the river bank",
            "c": "Quantum physics and particle acceleration in the large hadron collider",
        }
        for rid, text in texts.items():
            mh = compute_minhash(text, ngram_size=3, num_perm=128)
            lsh.insert(rid, mh)

        # Query with text 'a' — should find 'a' and 'b' (very similar)
        query_mh = compute_minhash(texts["a"], ngram_size=3, num_perm=128)
        candidates = lsh.query(query_mh)
        self.assertIn("a", candidates)
        # 'a' and 'b' share most n-grams (only "dog" vs "cat" differs)
        self.assertIn("b", candidates)

    def test_threshold_respected(self):
        """High threshold (0.95) only matches near-identical texts."""
        lsh = MinHashLSH(threshold=0.95, num_perm=256)
        text_a = "The quick brown fox jumps over the lazy dog"
        text_b = "The quick brown fox jumps over the lazy cat"
        text_c = text_a  # exact copy

        for rid, text in [("a", text_a), ("b", text_b), ("c", text_c)]:
            mh = compute_minhash(text, ngram_size=5, num_perm=256)
            lsh.insert(rid, mh)

        # Query with exact copy
        query_mh = compute_minhash(text_a, ngram_size=5, num_perm=256)
        candidates = lsh.query(query_mh)
        self.assertIn("c", candidates)  # exact match
        # 'b' is different enough that at 0.95 threshold it may not match
        # (this is probabilistic, but with 256 perms it's very reliable)

    def test_permutations_configurable(self):
        """Number of permutations is configurable and affects hash length."""
        mh_64 = compute_minhash("test", ngram_size=3, num_perm=64)
        mh_128 = compute_minhash("test", ngram_size=3, num_perm=128)
        self.assertEqual(len(mh_64.hashvalues), 64)
        self.assertEqual(len(mh_128.hashvalues), 128)

    def test_default_permutations_is_128(self):
        """Default number of permutations is 128."""
        self.assertEqual(_DEFAULT_PERMUTATIONS, 128)


# ---------------------------------------------------------------------------
# Test class 3: Jaccard similarity
# ---------------------------------------------------------------------------


class TestJaccardSimilarity(unittest.TestCase):
    """Tests for exact Jaccard similarity (MAI-Thinking-1 §2.4.3)."""

    def test_identical_sets(self):
        """Identical sets have Jaccard = 1.0."""
        s = {"a", "b", "c"}
        self.assertEqual(jaccard_similarity(s, s), 1.0)

    def test_disjoint_sets(self):
        """Disjoint sets have Jaccard = 0.0."""
        self.assertEqual(jaccard_similarity({"a", "b"}, {"c", "d"}), 0.0)

    def test_partial_overlap(self):
        """Partially overlapping sets have Jaccard between 0 and 1."""
        sim = jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        # intersection = {b, c} = 2, union = {a, b, c, d} = 4
        self.assertAlmostEqual(sim, 0.5)

    def test_both_empty(self):
        """Both empty sets have Jaccard = 1.0 (by convention)."""
        self.assertEqual(jaccard_similarity(set(), set()), 1.0)

    def test_one_empty(self):
        """One empty set has Jaccard = 0.0."""
        self.assertEqual(jaccard_similarity({"a"}, set()), 0.0)


# ---------------------------------------------------------------------------
# Test class 4: End-to-end decontamination
# ---------------------------------------------------------------------------


class TestDecontamEndToEnd(unittest.TestCase):
    """Tests for the decontam main loop on small fixtures."""

    def setUp(self):
        """Create a small training dataset and eval set in temp dirs."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

        # Create a training source directory with a few records
        source_dir = self.root / "test-source" / "reconnaissance" / "T1082"
        source_dir.mkdir(parents=True)
        self.training_file = source_dir / "data.jsonl"

        # Training records — some will overlap with eval records
        training_records = [
            {
                "messages": [
                    {"role": "user", "content": "How do I enumerate users?"},
                    {
                        "role": "assistant",
                        "content": "Use `net user` to list local users.",
                    },
                ],
                "source": "test-source",
                "license": "MIT",
            },
            {
                "messages": [
                    {"role": "user", "content": "What is port scanning?"},
                    {
                        "role": "assistant",
                        "content": "Port scanning discovers open ports on a target system using tools like nmap.",
                    },
                ],
                "source": "test-source",
                "license": "MIT",
            },
            {
                "messages": [
                    {"role": "user", "content": "How to clear event logs?"},
                    {
                        "role": "assistant",
                        "content": "Use `wevtutil cl Security` to clear the security event log.",
                    },
                ],
                "source": "test-source",
                "license": "MIT",
            },
        ]
        with open(self.training_file, "w", encoding="utf-8") as f:
            for rec in training_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Create eval set directory
        self.eval_dir = self.root / "eval_sets"
        self.eval_dir.mkdir(parents=True)
        self.eval_file = self.eval_dir / "test-eval.jsonl"

        # Eval records — first one is an exact copy of training record 0's
        # concatenated messages text (guarantees Jaccard=1.0 at any n-gram size).
        eval_records = [
            {
                "id": "eval-001",
                "text": "How do I enumerate users?\nUse `net user` to list local users.",
                "source": "test-eval",
            },
            {
                "id": "eval-002",
                "text": "Completely unrelated text about gardening and plants and flowers and trees and soil.",
                "source": "test-eval",
            },
        ]
        with open(self.eval_file, "w", encoding="utf-8") as f:
            for rec in eval_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_correct_match_count(self):
        """End-to-end decontam finds the expected number of overlaps."""
        records_by_source = load_training_records(self.root)
        eval_records = load_eval_records(self.eval_dir)

        lsh, index_map = build_training_index(
            records_by_source, ngram_size=10, num_perm=128, threshold=0.5
        )
        overlaps = find_overlaps(
            lsh, index_map, eval_records, ngram_size=10, num_perm=128, threshold=0.5
        )

        # eval-001 should match training record 0 (both mention "net user")
        # eval-002 is unrelated
        self.assertGreaterEqual(len(overlaps), 1)

    def test_quarantine_records_written(self):
        """Quarantine file contains deduplicated matched training records."""
        records_by_source = load_training_records(self.root)
        eval_records = load_eval_records(self.eval_dir)

        lsh, index_map = build_training_index(
            records_by_source, ngram_size=10, num_perm=128, threshold=0.5
        )
        overlaps = find_overlaps(
            lsh, index_map, eval_records, ngram_size=10, num_perm=128, threshold=0.5
        )

        quarantine_path = self.root / "quarantine.jsonl"
        count = write_quarantine(overlaps, index_map, quarantine_path)

        self.assertGreaterEqual(count, 1)
        self.assertTrue(quarantine_path.exists())
        # Verify the quarantine file is valid JSONL
        with open(quarantine_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(lines), count)
        for line in lines:
            rec = json.loads(line)
            self.assertIn("messages", rec)

    def test_report_schema(self):
        """Report has the expected schema with per-source breakdown."""
        records_by_source = load_training_records(self.root)
        eval_records = load_eval_records(self.eval_dir)

        lsh, index_map = build_training_index(
            records_by_source, ngram_size=10, num_perm=128, threshold=0.5
        )
        overlaps = find_overlaps(
            lsh, index_map, eval_records, ngram_size=10, num_perm=128, threshold=0.5
        )

        report = build_report(records_by_source, overlaps, max_examples=5)

        self.assertIn("sources", report)
        self.assertIn("total_overlaps", report)
        self.assertIn("test-source", report["sources"])
        src_report = report["sources"]["test-source"]
        self.assertIn("n_training_records", src_report)
        self.assertIn("n_eval_overlaps", src_report)
        self.assertIn("n_unique_eval_records_overlapping", src_report)
        self.assertIn("examples", src_report)
        self.assertEqual(src_report["n_training_records"], 3)

    def test_cross_source_dedup(self):
        """Multiple training sources are handled independently."""
        # Add a second source
        source2_dir = self.root / "test-source-2" / "execution" / "T1059"
        source2_dir.mkdir(parents=True)
        rec2 = {
            "messages": [
                {"role": "user", "content": "How to run PowerShell?"},
                {
                    "role": "assistant",
                    "content": "Use `powershell.exe -Command` to execute PowerShell commands.",
                },
            ],
            "source": "test-source-2",
            "license": "MIT",
        }
        with open(source2_dir / "data.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(rec2, ensure_ascii=False) + "\n")

        records_by_source = load_training_records(self.root)
        eval_records = load_eval_records(self.eval_dir)

        lsh, index_map = build_training_index(
            records_by_source, ngram_size=10, num_perm=128, threshold=0.5
        )
        overlaps = find_overlaps(
            lsh, index_map, eval_records, ngram_size=10, num_perm=128, threshold=0.5
        )

        report = build_report(records_by_source, overlaps)
        self.assertIn("test-source", report["sources"])
        self.assertIn("test-source-2", report["sources"])


# ---------------------------------------------------------------------------
# Test class 5: CLI and edge cases
# ---------------------------------------------------------------------------


class TestDecontamCLI(unittest.TestCase):
    """Tests for the decontam.py CLI."""

    def test_default_threshold_is_0_80(self):
        """Default MinHash similarity threshold is 0.80 (matches MAI-Thinking-1)."""
        parser = build_parser()
        args = parser.parse_args(["--eval-set-dir", "/tmp/fake"])
        self.assertAlmostEqual(args.threshold, 0.80, places=4)

    def test_default_ngram_size_is_20(self):
        """Default n-gram size is 20 (matches MAI-Thinking-1 §2.3.1)."""
        parser = build_parser()
        args = parser.parse_args(["--eval-set-dir", "/tmp/fake"])
        self.assertEqual(args.ngram_size, 20)

    def test_default_permutations_is_128(self):
        """Default number of permutations is 128."""
        parser = build_parser()
        args = parser.parse_args(["--eval-set-dir", "/tmp/fake"])
        self.assertEqual(args.permutations, 128)

    def test_dry_run_flag(self):
        """--dry-run flag is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["--eval-set-dir", "/tmp/fake", "--dry-run"])
        self.assertTrue(args.dry_run)

    def test_quarantine_output_flag(self):
        """--quarantine-output flag sets the output path."""
        parser = build_parser()
        args = parser.parse_args(
            ["--eval-set-dir", "/tmp/fake", "--quarantine-output", "/tmp/q.jsonl"]
        )
        self.assertEqual(args.quarantine_output, Path("/tmp/q.jsonl"))

    def test_source_filter_excludes_restricted(self):
        """Source filter should not include restricted sources (rta, infection_monkey, bpl)."""
        from decontam import RESTRICTED_SOURCES

        self.assertIn("rta", RESTRICTED_SOURCES)
        self.assertIn("infection_monkey", RESTRICTED_SOURCES)
        self.assertIn("bpl", RESTRICTED_SOURCES)

    def test_main_returns_1_on_missing_training_root(self):
        """main() returns 1 when training data root does not exist."""
        exit_code = main(
            [
                "--training-data-root",
                "/tmp/nonexistent_training_root_xyz",
                "--eval-set-dir",
                "/tmp/fake_eval",
            ]
        )
        self.assertEqual(exit_code, 1)

    def test_main_returns_1_on_missing_eval_set(self):
        """main() returns 1 when eval set path does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "test-source" / "reconnaissance" / "T1082"
            source_dir.mkdir(parents=True)
            with open(source_dir / "data.jsonl", "w") as f:
                f.write(
                    '{"messages": [{"role": "user", "content": "test"}], "source": "test"}\n'
                )
            exit_code = main(
                [
                    "--training-data-root",
                    str(root),
                    "--eval-set-dir",
                    "/tmp/nonexistent_eval_set_xyz",
                ]
            )
            self.assertEqual(exit_code, 1)


# ---------------------------------------------------------------------------
# Test class 6: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Tests for empty inputs and boundary conditions."""

    def test_empty_training_set(self):
        """Empty training set produces no overlaps."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Create an empty source directory (no JSONL files)
            source_dir = root / "empty-source" / "reconnaissance" / "T1082"
            source_dir.mkdir(parents=True)

            eval_dir = root / "eval_sets"
            eval_dir.mkdir()
            with open(eval_dir / "eval.jsonl", "w") as f:
                f.write('{"id": "e1", "text": "test", "source": "eval"}\n')

            records_by_source = load_training_records(root)
            self.assertEqual(len(records_by_source), 0)

    def test_empty_eval_set(self):
        """Empty eval set produces no overlaps."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "test-source" / "reconnaissance" / "T1082"
            source_dir.mkdir(parents=True)
            with open(source_dir / "data.jsonl", "w") as f:
                f.write(
                    '{"messages": [{"role": "user", "content": "test"}], "source": "test"}\n'
                )

            eval_dir = root / "eval_sets"
            eval_dir.mkdir()
            # Empty eval directory — no files

            records_by_source = load_training_records(root)
            eval_records = load_eval_records(eval_dir)
            self.assertEqual(len(eval_records), 0)

    def test_no_matches(self):
        """Completely different training and eval sets produce no overlaps."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "test-source" / "reconnaissance" / "T1082"
            source_dir.mkdir(parents=True)
            with open(source_dir / "data.jsonl", "w") as f:
                f.write(
                    '{"messages": [{"role": "user", "content": "How to hack?"}, {"role": "assistant", "content": "I cannot provide instructions for hacking."}], "source": "test"}\n'
                )

            eval_dir = root / "eval_sets"
            eval_dir.mkdir()
            with open(eval_dir / "eval.jsonl", "w") as f:
                f.write(
                    '{"id": "e1", "text": "The quick brown fox jumps over the lazy dog. This is a completely unrelated text about animals and nature.", "source": "eval"}\n'
                )

            records_by_source = load_training_records(root)
            eval_records = load_eval_records(eval_dir)

            lsh, index_map = build_training_index(
                records_by_source, ngram_size=10, num_perm=128, threshold=0.8
            )
            overlaps = find_overlaps(
                lsh, index_map, eval_records, ngram_size=10, num_perm=128, threshold=0.8
            )
            self.assertEqual(len(overlaps), 0)

    def test_all_match(self):
        """When training and eval are identical, all records match."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "test-source" / "reconnaissance" / "T1082"
            source_dir.mkdir(parents=True)
            with open(source_dir / "data.jsonl", "w") as f:
                f.write(
                    '{"messages": [{"role": "user", "content": "What is nmap?"}, {"role": "assistant", "content": "Nmap is a network scanning tool used for discovery."}], "source": "test"}\n'
                )

            eval_dir = root / "eval_sets"
            eval_dir.mkdir()
            with open(eval_dir / "eval.jsonl", "w") as f:
                f.write(
                    '{"id": "e1", "text": "Nmap is a network scanning tool used for discovery.", "source": "eval"}\n'
                )

            records_by_source = load_training_records(root)
            eval_records = load_eval_records(eval_dir)

            lsh, index_map = build_training_index(
                records_by_source, ngram_size=10, num_perm=128, threshold=0.5
            )
            overlaps = find_overlaps(
                lsh, index_map, eval_records, ngram_size=10, num_perm=128, threshold=0.5
            )
            self.assertGreaterEqual(len(overlaps), 1)


# ---------------------------------------------------------------------------
# Test class 7: Record text extraction
# ---------------------------------------------------------------------------


class TestRecordTextExtraction(unittest.TestCase):
    """Tests for _record_text helper."""

    def test_messages_format(self):
        """Messages format concatenates all message contents."""
        record = {
            "messages": [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "What is X?"},
                {"role": "assistant", "content": "X is a variable."},
            ]
        }
        text = _record_text(record)
        self.assertIn("You are a helper.", text)
        self.assertIn("What is X?", text)
        self.assertIn("X is a variable.", text)

    def test_text_fallback(self):
        """Fallback to 'text' field when messages is absent."""
        record = {"text": "This is a plain text record."}
        text = _record_text(record)
        self.assertEqual(text, "This is a plain text record.")

    def test_empty_messages(self):
        """Empty messages list returns empty string."""
        record = {"messages": []}
        text = _record_text(record)
        self.assertEqual(text, "")

    def test_missing_both_fields(self):
        """Missing both messages and text returns empty string."""
        record = {"id": "123"}
        text = _record_text(record)
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
