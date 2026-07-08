"""LiRA (Likelihood Ratio Attack) — Carlini 2022 §4.

The gold-standard MIA: trains K shadow models on disjoint subsets of the
same distribution, fits per-record Gaussian to IN/OUT loss distributions,
and computes the log-likelihood ratio at audit time.

Reference:
    Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramer, F.
    (2022). "Membership Inference Attacks From First Principles."
    IEEE Symposium on Security and Privacy. arXiv:2112.03570 §4.

This module implements the SCORING side of LiRA. The shadow-model
training is the user's responsibility (see docs/LIRA.md for the workflow
and attacklm/scripts/lira_train.py if/when that helper ships). What
ships in this module:

  - Per-record Gaussian fit from precomputed shadow losses
  - The log-likelihood ratio statistic
  - Threshold calibration (natural 0.0 for LiRA)
  - File I/O: load/save Gaussian parameters as a JSON artifact
  - The score_lira() function that takes a record + shadow stats dir
    and returns a LiRAScore dataclass

The shadow_stats directory is expected to be a flat directory of
{N_record_id: (μ_in, σ_in, μ_out, σ_out)} entries, one per record that
will be audited. The shadow training script (user-side) is responsible
for producing this artifact.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from inversion.scoring import _extract_assistant_turn, compute_nll


@dataclass
class LiRAScore:
    """LiRA score for a single record.

    Lower (more negative) lira_logit = more likely OUT (non-member).
    Higher (more positive) lira_logit = more likely IN (member).
    The natural decision threshold is 0.0.
    """

    lira_logit: float
    mu_in: float
    sigma_in: float
    mu_out: float
    sigma_out: float
    loss_target: float
    nll_per_token: float  # = loss_target / num_suffix_tokens (for cross-check)
    num_suffix_tokens: int
    alpha: float  # calibration weight (default 1.0; tune per attack)


@dataclass
class GaussianParams:
    """Per-record Gaussian parameters from K shadow models."""

    mu_in: float
    sigma_in: float
    mu_out: float
    sigma_out: float

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.mu_in, self.sigma_in, self.mu_out, self.sigma_out)


def fit_gaussian(losses: list[float]) -> tuple[float, float]:
    """Fit a 1-D Gaussian to a list of losses.

    Returns (mu, sigma). Uses sample variance (divide by N, not N-1) for
    consistency with Carlini 2022 §4.1. If sigma is 0 (constant losses),
    returns (mu, epsilon) where epsilon is a small floor to avoid log(0).

    Args:
        losses: List of K loss values from K shadow models.

    Returns:
        (mu, sigma) — the fitted Gaussian parameters.
    """
    if not losses:
        raise ValueError("fit_gaussian: empty losses list")
    n = len(losses)
    mu = sum(losses) / n
    variance = sum((x - mu) ** 2 for x in losses) / n
    sigma = variance**0.5
    if sigma == 0.0:
        sigma = 1e-6  # Floor to avoid log(0) in compute_lira_logit
    return mu, sigma


def fit_gaussians_per_record(
    in_losses: dict[str, list[float]],
    out_losses: dict[str, list[float]],
) -> dict[str, GaussianParams]:
    """Fit (μ_in, σ_in) and (μ_out, σ_out) per record.

    Args:
        in_losses: record_id -> list of K_IN losses from shadows where the
            record was IN the training set.
        out_losses: record_id -> list of K_OUT losses from shadows where
            the record was OUT of the training set.

    Returns:
        record_id -> GaussianParams

    The K for IN and K for OUT can differ. With a typical LiRA setup,
        K_IN = K_OUT = K/2 (half of the K shadows had this record IN).
    For K=16, K_IN = K_OUT = 8.

    Raises:
        ValueError: if a record is in one dict but not the other.
    """
    if set(in_losses.keys()) != set(out_losses.keys()):
        missing_in = set(out_losses.keys()) - set(in_losses.keys())
        missing_out = set(in_losses.keys()) - set(out_losses.keys())
        raise ValueError(
            f"Record sets differ: missing from in_losses={list(missing_in)[:5]}..., "
            f"missing from out_losses={list(missing_out)[:5]}..."
        )
    result: dict[str, GaussianParams] = {}
    for record_id in in_losses:
        mu_in, sigma_in = fit_gaussian(in_losses[record_id])
        mu_out, sigma_out = fit_gaussian(out_losses[record_id])
        result[record_id] = GaussianParams(mu_in, sigma_in, mu_out, sigma_out)
    return result


def gaussian_log_pdf(x: float, mu: float, sigma: float) -> float:
    """Log of the Gaussian PDF at x: log N(x; mu, sigma).

    Returns:
        -0.5 * log(2*pi) - log(sigma) - 0.5 * ((x - mu) / sigma) ** 2
    """
    if sigma <= 0:
        return float("-inf")
    z = (x - mu) / sigma
    return -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - 0.5 * z * z


def compute_lira_logit(
    loss_target: float,
    mu_in: float,
    sigma_in: float,
    mu_out: float,
    sigma_out: float,
) -> float:
    """Compute the LiRA log-likelihood ratio.

    lira_logit = log N(loss_target; mu_in, sigma_in) - log N(loss_target; mu_out, sigma_out)

    Returns:
        The LRT statistic. Positive = more likely IN (member).
        Negative = more likely OUT (non-member).
    """
    log_in = gaussian_log_pdf(loss_target, mu_in, sigma_in)
    log_out = gaussian_log_pdf(loss_target, mu_out, sigma_out)
    return log_in - log_out


def calibrate_lira_threshold(
    lira_logits: list[float],
    known_membership: list[bool],
    target_fpr: float = 0.01,
) -> float:
    """Calibrate the LiRA threshold for a target FPR.

    For LiRA, the natural threshold is 0.0 (positive = IN). This function
    finds a threshold that achieves the target FPR on a labeled calibration
    set, IF the user provides known membership labels.

    Args:
        lira_logits: List of LiRA logit values for the calibration set.
        known_membership: List of booleans (True = member, False = non-member).
        target_fpr: Target false positive rate (default 0.01 = 1%).

    Returns:
        The threshold value. lira_logit >= threshold → predicted member.

    If known_membership is empty, returns 0.0 (the natural threshold).
    """
    if not known_membership or not lira_logits:
        return 0.0
    if len(lira_logits) != len(known_membership):
        raise ValueError(
            f"lira_logits ({len(lira_logits)}) and known_membership "
            f"({len(known_membership)}) must have the same length"
        )
    # Get the logits of known non-members
    non_member_logits = [
        logit
        for logit, is_member in zip(lira_logits, known_membership)
        if not is_member
    ]
    if not non_member_logits:
        return 0.0
    # Threshold at the (1 - target_fpr) quantile of non-member logits
    sorted_non = sorted(non_member_logits)
    idx = max(0, int(len(sorted_non) * (1 - target_fpr)) - 1)
    return sorted_non[idx]


def score_lira(
    record: dict,
    model,
    tokenizer,
    shadow_params: GaussianParams,
    alpha: float = 1.0,
) -> LiRAScore:
    """Score a single record with LiRA.

    The record's `record_id` (from record["id"] or record["record_index"])
    must have a corresponding entry in shadow_params. The target model's
    loss on the record is computed, and the LiRA logit is computed from
    the shadow IN/OUT Gaussians.

    Args:
        record: The training record to score. Must have a "messages" key.
        model: HuggingFace model (used for the target loss computation).
        tokenizer: HuggingFace tokenizer.
        shadow_params: The fitted (μ_in, σ_in, μ_out, σ_out) for this record.
        alpha: Calibration weight (reserved for future use; default 1.0).
    """
    # Extract the assistant turn (per the M1 pattern; reuse the helper)
    suffix_text = _extract_assistant_turn(record)
    if not suffix_text:
        return LiRAScore(
            lira_logit=float("nan"),
            mu_in=shadow_params.mu_in,
            sigma_in=shadow_params.sigma_in,
            mu_out=shadow_params.mu_out,
            sigma_out=shadow_params.sigma_out,
            loss_target=float("inf"),
            nll_per_token=float("inf"),
            num_suffix_tokens=0,
            alpha=alpha,
        )

    loss_target, num_suffix_tokens = compute_nll(model, tokenizer, suffix_text)
    nll_per_token = (
        loss_target / num_suffix_tokens if num_suffix_tokens > 0 else float("inf")
    )

    lira_logit = compute_lira_logit(
        loss_target,
        shadow_params.mu_in,
        shadow_params.sigma_in,
        shadow_params.mu_out,
        shadow_params.sigma_out,
    )

    return LiRAScore(
        lira_logit=lira_logit,
        mu_in=shadow_params.mu_in,
        sigma_in=shadow_params.sigma_in,
        mu_out=shadow_params.mu_out,
        sigma_out=shadow_params.sigma_out,
        loss_target=loss_target,
        nll_per_token=nll_per_token,
        num_suffix_tokens=num_suffix_tokens,
        alpha=alpha,
    )


def save_shadow_params(
    params: dict[str, GaussianParams],
    output_path: Path | str,
) -> None:
    """Save per-record Gaussian parameters to a JSON file.

    Format:
    {
        "lira_k": 16,
        "created_at": "2026-07-08T...",
        "params": {
            "<record_id>": {
                "mu_in": 1.234,
                "sigma_in": 0.567,
                "mu_out": 2.345,
                "sigma_out": 0.789
            },
            ...
        }
    }
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive lira_k from params if available (best effort)
    if params:
        # We can't know K from the Gaussian params alone (it's the number
        # of shadow models, not derivable from 4 floats). The caller should
        # set this externally. Default to 0 (unknown).
        lira_k = 0

    data = {
        "lira_k": lira_k,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            rid: {
                "mu_in": p.mu_in,
                "sigma_in": p.sigma_in,
                "mu_out": p.mu_out,
                "sigma_out": p.sigma_out,
            }
            for rid, p in params.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_shadow_params(
    input_path: Path | str,
) -> tuple[dict[str, GaussianParams], int]:
    """Load per-record Gaussian parameters from a JSON file.

    Returns:
        (params, lira_k) where lira_k is the number of shadow models used.
    """

    with open(input_path) as f:
        data = json.load(f)
    lira_k = data.get("lira_k", 0)
    params = {
        rid: GaussianParams(
            mu_in=d["mu_in"],
            sigma_in=d["sigma_in"],
            mu_out=d["mu_out"],
            sigma_out=d["sigma_out"],
        )
        for rid, d in data["params"].items()
    }
    return params, lira_k
