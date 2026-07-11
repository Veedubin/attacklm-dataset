#!/usr/bin/env python3
"""Tests for the 5 bug fixes in the inversion-audit harness.

Bug #1: scoring.py score_record uses _extract_assistant_turn (not _extract_full_text)
Bug #2: lira.py save_shadow_params accepts lira_k parameter, no crash on empty dict
Bug #3: probe.py ProbeResult includes prompt_text and best_reconstruction
Bug #4: probe.py generate_completions uses num_return_sequences batch call
Bug #5: lira.py LiRAScore no longer has dead alpha field

Run with:
    python -m pytest tests/test_audit_bugfixes.py -v
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

from inversion.scoring import (  # noqa: E402
    _extract_assistant_turn,
    _extract_full_text,
    score_record,
)
from inversion.lira import (  # noqa: E402
    GaussianParams,
    LiRAScore,
    save_shadow_params,
    load_shadow_params,
    score_lira,
)
from inversion.probe import (  # noqa: E402
    ProbeResult,
    generate_completions,
)


# ---------------------------------------------------------------------------
# Bug #1: score_record uses _extract_assistant_turn, not _extract_full_text
# ---------------------------------------------------------------------------


class TestBug1ScoringExtractsAssistantOnly(unittest.TestCase):
    """Bug #1: score_record should use _extract_assistant_turn per MUSE 2023."""

    def test_extract_assistant_turn_extracts_only_assistant(self):
        """_extract_assistant_turn returns only the assistant content."""
        record = {
            "messages": [
                {"role": "system", "content": "long system prompt " * 100},
                {"role": "user", "content": "What is MS17-010?"},
                {"role": "assistant", "content": "MS17-010 is an SMB vulnerability."},
            ]
        }
        full_text = _extract_full_text(record)
        assistant_text = _extract_assistant_turn(record)
        # Assistant text should be much shorter than full text
        self.assertLess(len(assistant_text), len(full_text))
        # Assistant text should equal the assistant content exactly
        self.assertEqual(assistant_text, "MS17-010 is an SMB vulnerability.")

    def test_two_records_same_assistant_different_prompt(self):
        """Two records with same assistant text but different-length prompts
        should produce the same NLL after Bug #1 fix (assistant-only scoring)."""
        record_short = {
            "messages": [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "Short answer."},
            ]
        }
        record_long = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. " * 50},
                {
                    "role": "user",
                    "content": "Please explain this concept in detail " * 20,
                },
                {"role": "assistant", "content": "Short answer."},
            ]
        }
        # After Bug #1 fix, _extract_assistant_turn returns the same text
        self.assertEqual(
            _extract_assistant_turn(record_short),
            _extract_assistant_turn(record_long),
        )

    @patch("inversion.scoring.compute_nll")
    @patch("inversion.scoring.compute_zlib_metrics")
    def test_score_record_calls_nll_on_assistant_only(self, mock_zlib, mock_nll):
        """score_record should pass only the assistant text to compute_nll."""
        mock_nll.return_value = (10.0, 5)
        mock_zlib.return_value = (50, 0.5)

        record = {
            "messages": [
                {"role": "system", "content": "long system prompt " * 100},
                {"role": "user", "content": "What is MS17-010?"},
                {"role": "assistant", "content": "MS17-010 is an SMB vulnerability."},
            ]
        }
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = score_record(record, mock_model, mock_tokenizer)

        # compute_nll should have been called with the assistant text only
        mock_nll.assert_called_once()
        call_args = mock_nll.call_args
        text_passed = (
            call_args[0][2]
            if len(call_args[0]) > 2
            else call_args[1].get("text", call_args[0][1])
        )
        # The text should be just the assistant content, NOT the full record
        self.assertEqual(text_passed, "MS17-010 is an SMB vulnerability.")


# ---------------------------------------------------------------------------
# Bug #2: save_shadow_params accepts lira_k, no crash on empty dict
# ---------------------------------------------------------------------------


class TestBug2ShadowParamsLiRAK(unittest.TestCase):
    """Bug #2: save_shadow_params takes explicit lira_k, no NameError on empty."""

    def test_save_empty_params_with_lira_k(self):
        """save_shadow_params({}, path, lira_k=16) should NOT raise."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)
        try:
            save_shadow_params({}, output_path, lira_k=16)
            data = json.loads(output_path.read_text())
            self.assertEqual(data["lira_k"], 16)
            self.assertEqual(data["params"], {})
        finally:
            output_path.unlink(missing_ok=True)

    def test_save_params_with_lira_k(self):
        """save_shadow_params with params and lira_k=8."""
        params = {
            "rec1": GaussianParams(0.0, 1.0, 2.0, 3.0),
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)
        try:
            save_shadow_params(params, output_path, lira_k=8)
            data = json.loads(output_path.read_text())
            self.assertEqual(data["lira_k"], 8)
            self.assertIn("rec1", data["params"])
        finally:
            output_path.unlink(missing_ok=True)

    def test_save_params_default_lira_k_is_zero(self):
        """Default lira_k=0 for backward compat."""
        params = {
            "rec1": GaussianParams(0.0, 1.0, 2.0, 3.0),
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)
        try:
            save_shadow_params(params, output_path)  # no lira_k
            data = json.loads(output_path.read_text())
            self.assertEqual(data["lira_k"], 0)
        finally:
            output_path.unlink(missing_ok=True)

    def test_save_empty_params_no_lira_k_no_crash(self):
        """save_shadow_params({}) without lira_k defaults to 0, no NameError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)
        try:
            save_shadow_params({}, output_path)
            data = json.loads(output_path.read_text())
            self.assertEqual(data["lira_k"], 0)
        finally:
            output_path.unlink(missing_ok=True)

    def test_roundtrip_with_lira_k(self):
        """Save with lira_k=16, load back, verify lira_k is preserved."""
        params = {
            "r0": GaussianParams(1.0, 0.5, 2.0, 0.7),
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)
        try:
            save_shadow_params(params, output_path, lira_k=16)
            loaded_params, lira_k = load_shadow_params(output_path)
            self.assertEqual(lira_k, 16)
            self.assertAlmostEqual(loaded_params["r0"].mu_in, 1.0, places=4)
        finally:
            output_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Bug #3: ProbeResult includes prompt_text and best_reconstruction
# ---------------------------------------------------------------------------


class TestBug3ProbeResultRawText(unittest.TestCase):
    """Bug #3: ProbeResult carries prompt_text and best_reconstruction."""

    def test_probe_result_dataclass_fields(self):
        """ProbeResult should have prompt_text and best_reconstruction fields."""
        result = ProbeResult(
            record_index=0,
            source="test",
            license_id="MIT",
            prompt_hash="abc123",
            reconstruction_hash="def456",
            best_exact_match=False,
            best_lcs_length=10,
            best_bleu4=0.35,
            original_length=50,
            best_completion_length=45,
            num_completions=20,
            prompt_text="What is MS17-010?",
            best_reconstruction="MS17-010 is an SMB vulnerability.",
        )
        self.assertEqual(result.prompt_text, "What is MS17-010?")
        self.assertEqual(
            result.best_reconstruction, "MS17-010 is an SMB vulnerability."
        )

    def test_probe_result_defaults_empty_strings(self):
        """Default values for prompt_text and best_reconstruction are empty strings."""
        result = ProbeResult(
            record_index=0,
            source="test",
            license_id="MIT",
            prompt_hash="abc",
            reconstruction_hash="def",
            best_exact_match=False,
            best_lcs_length=0,
            best_bleu4=0.0,
            original_length=0,
            best_completion_length=0,
            num_completions=0,
        )
        self.assertEqual(result.prompt_text, "")
        self.assertEqual(result.best_reconstruction, "")

    @patch("inversion.probe.generate_completions")
    def test_run_carlini_probe_populates_text_fields(self, mock_gen):
        """run_carlini_probe should populate prompt_text and best_reconstruction."""
        from inversion.probe import run_carlini_probe

        mock_gen.return_value = [
            "completion A",
            "completion B",
        ]
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        # Mock tokenizer.encode to return tokens for the prefix extraction
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "completion A"

        record = {
            "messages": [
                {"role": "user", "content": "What is MS17-010?"},
                {"role": "assistant", "content": "MS17-010 is an SMB vulnerability."},
            ],
            "source": "test",
            "license": "MIT",
        }
        result = run_carlini_probe(
            record,
            mock_model,
            mock_tokenizer,
            top_k=2,
            record_index=0,
        )
        # prompt_text should be set from the prefix
        self.assertIsInstance(result.prompt_text, str)
        self.assertGreater(len(result.prompt_text), 0)
        # best_reconstruction should be the best completion text
        self.assertIsInstance(result.best_reconstruction, str)


# ---------------------------------------------------------------------------
# Bug #4: generate_completions uses num_return_sequences
# ---------------------------------------------------------------------------


class TestBug4GenerateCompletionsBatch(unittest.TestCase):
    """Bug #4: generate_completions uses num_return_sequences for batched generation."""

    def test_generate_completions_calls_generate_once(self):
        """generate_completions should call model.generate exactly once with
        num_return_sequences=K, not K times sequentially."""
        import torch

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "generated text"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.pad_token_id = 0
        mock_model.device = MagicMock()
        mock_model.device.type = "cpu"

        # With num_return_sequences=3, the output shape is (3, prompt_len + new_tokens)
        # prompt_len = 5 (from tokenizer.encode return value)
        # total_len = 5 + 3 = 8 (prompt + 3 new tokens)
        output_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 10, 20, 30],
                [1, 2, 3, 4, 5, 40, 50, 60],
                [1, 2, 3, 4, 5, 70, 80, 90],
            ]
        )
        mock_model.generate.return_value = output_ids

        completions = generate_completions(
            mock_model,
            mock_tokenizer,
            "test prefix",
            num_completions=3,
            max_new_tokens=10,
        )

        # model.generate should be called exactly once (batched, not K times)
        self.assertEqual(mock_model.generate.call_count, 1)

        # Verify num_return_sequences was in the call kwargs
        call_kwargs = mock_model.generate.call_args[1]
        self.assertEqual(call_kwargs.get("num_return_sequences"), 3)

        # Should get 3 completions
        self.assertEqual(len(completions), 3)


# ---------------------------------------------------------------------------
# Bug #5: LiRAScore no longer has dead alpha field
# ---------------------------------------------------------------------------


class TestBug5LiRAScoreNoAlpha(unittest.TestCase):
    """Bug #5: LiRAScore should not have an alpha field."""

    def test_lira_score_has_no_alpha_field(self):
        """Creating a LiRAScore without alpha should work; alpha should not be
        an attribute."""
        score = LiRAScore(
            lira_logit=1.5,
            mu_in=1.0,
            sigma_in=0.5,
            mu_out=5.0,
            sigma_out=1.0,
            loss_target=3.0,
            nll_per_token=0.3,
            num_suffix_tokens=10,
        )
        # LiRAScore should NOT have an alpha attribute
        self.assertFalse(
            hasattr(score, "alpha"), "LiRAScore should not have alpha field"
        )

    def test_score_lira_no_alpha_param(self):
        """score_lira should not accept alpha parameter."""
        import inspect

        sig = inspect.signature(score_lira)
        param_names = list(sig.parameters.keys())
        self.assertNotIn("alpha", param_names, "score_lira should not have alpha param")


if __name__ == "__main__":
    unittest.main(verbosity=2)
