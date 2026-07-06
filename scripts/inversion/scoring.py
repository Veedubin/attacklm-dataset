"""Membership-inference attack (MIA) scoring.

Strategy 2 from Carlini et al. (2022): use NLL + zlib entropy to
distinguish training-set members from non-members. No shadow model
required — we calibrate the threshold on held-out sources.

Reference:
    Carlini, N., et al. "Membership Inference Attacks From First Principles."
    IEEE Symposium on Security and Privacy, 2022.

Scoring formula:
    membership_score = NLL - alpha * zlib_length
    Lower scores → more likely memorized (member).

The held-out sources (azure-pyrit, cyberark-fuzzyai) with n_records=0
in _index.json provide natural non-member calibration data.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass


@dataclass
class MIAScore:
    """Membership-inference score for a single record."""

    nll: float  # Negative log-likelihood
    zlib_length: int  # Compressed length of the text
    zlib_ratio: float  # zlib_length / raw_length
    perplexity: float  # exp(NLL / num_tokens)
    num_tokens: int
    membership_score: float  # NLL - alpha * zlib_length (lower = more memorized)
    alpha: float  # Weight used for the zlib term


def compute_nll(model, tokenizer, text: str, device: str = "auto") -> tuple[float, int]:
    """Compute the negative log-likelihood of text under the model.

    Uses model.forward() to get per-token log probabilities, which is
    the white-box path required for MIA. LMStudio/Ollama CANNOT be used
    here because they don't expose loss.

    Args:
        model: HuggingFace model with forward() returning loss.
        tokenizer: HuggingFace tokenizer.
        text: The text to score.
        device: Device for tensor placement.

    Returns:
        (nll, num_tokens) — total NLL and number of tokens scored.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt")
    if hasattr(model, "device") and model.device.type != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # outputs.loss is the mean NLL per token
    num_tokens = inputs["input_ids"].shape[1]
    total_nll = outputs.loss.item() * num_tokens

    return total_nll, num_tokens


def compute_zlib_metrics(text: str) -> tuple[int, float]:
    """Compute zlib compression metrics for text.

    Returns:
        (compressed_length, ratio) where ratio = compressed_length / len(raw_bytes)
    """
    raw_bytes = text.encode("utf-8")
    compressed = zlib.compress(raw_bytes)
    compressed_length = len(compressed)
    raw_length = len(raw_bytes)
    ratio = compressed_length / raw_length if raw_length > 0 else 0.0
    return compressed_length, ratio


def compute_perplexity(nll: float, num_tokens: int) -> float:
    """Compute perplexity from NLL and token count.

    perplexity = exp(NLL / num_tokens)
    """
    if num_tokens == 0:
        return float("inf")
    return math.exp(nll / num_tokens)


def compute_mia_score(
    nll: float,
    zlib_length: int,
    alpha: float = 1.0,
) -> float:
    """Compute the membership-inference score.

    membership_score = NLL - alpha * zlib_length

    Lower scores indicate the text is more likely memorized (member).
    """
    return nll - alpha * zlib_length


def score_record(
    record: dict,
    model,
    tokenizer,
    alpha: float = 1.0,
) -> MIAScore:
    """Score a single record for membership inference.

    Combines NLL from model forward pass with zlib entropy.
    """
    # Concatenate all message content for full-record scoring
    text = _extract_full_text(record)

    nll, num_tokens = compute_nll(model, tokenizer, text)
    zlib_length, zlib_ratio = compute_zlib_metrics(text)
    perplexity = compute_perplexity(nll, num_tokens)
    membership_score = compute_mia_score(nll, zlib_length, alpha)

    return MIAScore(
        nll=nll,
        zlib_length=zlib_length,
        zlib_ratio=zlib_ratio,
        perplexity=perplexity,
        num_tokens=num_tokens,
        membership_score=membership_score,
        alpha=alpha,
    )


def calibrate_threshold(
    member_scores: list[float],
    non_member_scores: list[float],
    target_fpr: float = 0.01,
) -> float:
    """Calibrate the MIA decision threshold on held-out data.

    Finds the threshold that achieves the target false positive rate
    (i.e., the fraction of non-members incorrectly classified as members).

    Args:
        member_scores: membership_score values for known training records.
        non_member_scores: membership_score values for known non-members.
        target_fpr: Target false positive rate (default 1%).

    Returns:
        Threshold value. Scores below this are classified as "member".
    """
    if not non_member_scores:
        raise ValueError("Need at least one non-member score for calibration")

    # Sort non-member scores; threshold at the (1 - target_fpr) quantile
    sorted_non = sorted(non_member_scores)
    idx = max(0, int(len(sorted_non) * (1 - target_fpr)) - 1)
    return sorted_non[idx]


def compute_tpr_at_threshold(
    member_scores: list[float],
    threshold: float,
) -> float:
    """Compute true positive rate at a given threshold.

    TPR = fraction of member scores below the threshold.
    """
    if not member_scores:
        return 0.0
    members_detected = sum(1 for s in member_scores if s < threshold)
    return members_detected / len(member_scores)


def compute_fpr_at_threshold(
    non_member_scores: list[float],
    threshold: float,
) -> float:
    """Compute false positive rate at a given threshold.

    FPR = fraction of non-member scores below the threshold.
    """
    if not non_member_scores:
        return 0.0
    false_positives = sum(1 for s in non_member_scores if s < threshold)
    return false_positives / len(non_member_scores)


def _extract_full_text(record: dict) -> str:
    """Extract the full text content from all messages in a record."""
    messages = record.get("messages", [])
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            parts.append(content)
    return "\n".join(parts)
