from __future__ import annotations

# PROVENANCE METADATA — scripts/inversion/shadow_train.py
# ================================================================================
# Attack class:        LiRA shadow-model training (Carlini 2022 §4)
# Original authors:    Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song,
#                      Andreas Terzis, Florian Tramer
# Paper title:         Membership Inference Attacks From First Principles
# Year / venue:        2022 / IEEE Symposium on Security and Privacy
# Paper URL:           https://arxiv.org/abs/2112.03570
# Canonical repo:      N/A (no official code release by the authors)
#
# Implementation:
#   Type:              CLEAN_ROOM_REIMPLEMENTATION (orchestration only;
#                      this file does NOT train models — it produces the
#                      per-record (μ_in, σ_in, μ_out, σ_out) from shadow
#                      loss logs that the user provides out-of-band)
#   Lines of port:     N/A
#   Upstream license:  N/A
#
# Foundational work this builds on:
#   - Shokri et al. 2017 (https://arxiv.org/abs/1610.05820) — original
#     shadow-model MIA paradigm that LiRA refines.
#
# Data sources: N/A (this file orchestrates training, it does not ingest data)
#
# Rights claim contact: veedubin.legal@example.com
# See:                  RIGHTS.md (root), data/LEGAL.md, data/REMOVAL.md
# ================================================================================
"""Shadow-model training orchestrator for LiRA.

This module does NOT train models — that cost is the user's. What it does:
  1. Document the shadow-model training recipe in the docstring
     (so a user running `python -m inversion.shadow_train --help` sees it).
  2. Provide a CLI entry point that, given a list of "shadow" record
     subsets (one per shadow model), invokes the existing attacklm train
     command for each shadow.
  3. After all K shadows are trained, compute the per-record (μ_in, σ_in,
     μ_out, σ_out) from the K IN losses and K OUT losses.
  4. Save the result to a JSON file (see lira.save_shadow_params).

The shadow-model training is OUT of scope for this PR. The CLI scaffold
is here so a future release can run the full pipeline end-to-end.

For the v0.5.0 audit, the user produces shadow models OUT-OF-BAND and
provides the resulting loss logs (one JSON per shadow model, with
record_id -> loss). This module reads those logs and produces the
shadow_params JSON for use with --mia-method lira.

Usage:
  # Step 1: Train K shadow models (user-side, ~K hours on RTX 4080)
  for k in range(K):
      attacklm train \\
          --dataset data/shadows/shadow_${k}/train.jsonl \\
          --output models/shadow_${k} \\
          ...

  # Step 2: Score each shadow model on the audit set
  for k in range(K):
      attacklm-dataset/scripts/score_shadow.py \\
          --model models/shadow_${k} \\
          --records data/audit_set.jsonl \\
          --output losses/shadow_${k}.json

  # Step 3: Produce the per-record Gaussian parameters
  python -m inversion.shadow_train \\
      --loss-dir losses/ \\
      --in-records data/shadows/in_set.jsonl \\
      --out-records data/shadows/out_set.jsonl \\
      --output shadow_params.json

Step 3 is what this module implements. Steps 1-2 are user-side.
"""



import argparse
import json
import logging
import sys
from pathlib import Path

# Make the inversion package importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inversion.lira import (
    fit_gaussians_per_record,
    save_shadow_params,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Shadow-model loss aggregation for LiRA. "
            "Reads precomputed shadow loss files and produces "
            "per-record Gaussian parameters (μ_in, σ_in, μ_out, σ_out) "
            "for use with --mia-method lira."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--loss-dir",
        type=Path,
        required=True,
        help="Directory containing shadow_0.json, shadow_1.json, ..., shadow_K.json. "
        "Each JSON is {record_id: loss_value, ...}.",
    )
    parser.add_argument(
        "--in-records",
        type=Path,
        required=True,
        help="JSONL file of record IDs that are IN the training set for each shadow. "
        'Each line: {"shadow_k": int, "record_id": str}.',
    )
    parser.add_argument(
        "--out-records",
        type=Path,
        required=True,
        help="JSONL file of record IDs that are OUT of the training set for each shadow. "
        'Each line: {"shadow_k": int, "record_id": str}.',
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for shadow_params.json.",
    )
    return parser


def load_shadow_losses(loss_dir: Path) -> dict[int, dict[str, float]]:
    """Load shadow loss files from a directory.

    Each file is shadow_0.json, shadow_1.json, etc.
    Each contains {record_id: loss_value, ...}.
    """
    shadow_losses: dict[int, dict[str, float]] = {}
    for json_path in sorted(loss_dir.glob("shadow_*.json")):
        # Extract shadow index from filename
        stem = json_path.stem  # e.g., "shadow_0"
        try:
            shadow_idx = int(stem.split("_")[1])
        except (IndexError, ValueError):
            logger.warning("Skipping malformed shadow loss file: %s", json_path)
            continue
        with open(json_path) as f:
            losses = json.load(f)
        shadow_losses[shadow_idx] = losses
        logger.info("Loaded shadow %d: %d records", shadow_idx, len(losses))
    return shadow_losses


def build_in_out_losses(
    shadow_losses: dict[int, dict[str, float]],
    in_records: list[dict],
    out_records: list[dict],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Build per-record IN and OUT loss lists from shadow losses.

    Args:
        shadow_losses: shadow_index -> {record_id: loss_value}
        in_records: list of {shadow_k: int, record_id: str}
        out_records: list of {shadow_k: int, record_id: str}

    Returns:
        (in_losses, out_losses) where each is record_id -> list of losses.
    """
    in_losses: dict[str, list[float]] = {}
    out_losses: dict[str, list[float]] = {}

    for rec in in_records:
        shadow_k = rec["shadow_k"]
        record_id = rec["record_id"]
        if shadow_k in shadow_losses and record_id in shadow_losses[shadow_k]:
            in_losses.setdefault(record_id, []).append(
                shadow_losses[shadow_k][record_id]
            )

    for rec in out_records:
        shadow_k = rec["shadow_k"]
        record_id = rec["record_id"]
        if shadow_k in shadow_losses and record_id in shadow_losses[shadow_k]:
            out_losses.setdefault(record_id, []).append(
                shadow_losses[shadow_k][record_id]
            )

    return in_losses, out_losses


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load shadow losses
    logger.info("Loading shadow losses from %s", args.loss_dir)
    shadow_losses = load_shadow_losses(args.loss_dir)
    if not shadow_losses:
        logger.error("No shadow loss files found in %s", args.loss_dir)
        return 1

    # Load IN/OUT records
    logger.info("Loading IN records from %s", args.in_records)
    in_records: list[dict] = []
    with open(args.in_records) as f:
        for line in f:
            line = line.strip()
            if line:
                in_records.append(json.loads(line))

    logger.info("Loading OUT records from %s", args.out_records)
    out_records: list[dict] = []
    with open(args.out_records) as f:
        for line in f:
            line = line.strip()
            if line:
                out_records.append(json.loads(line))

    # Build per-record IN/OUT loss lists
    logger.info("Building per-record IN/OUT loss lists")
    in_losses, out_losses = build_in_out_losses(shadow_losses, in_records, out_records)

    logger.info("Records with IN losses: %d", len(in_losses))
    logger.info("Records with OUT losses: %d", len(out_losses))

    # Fit Gaussians
    logger.info("Fitting per-record Gaussians")
    try:
        params = fit_gaussians_per_record(in_losses, out_losses)
    except ValueError as e:
        logger.error("Error fitting Gaussians: %s", e)
        return 1

    logger.info("Fitted Gaussians for %d records", len(params))

    # Save
    save_shadow_params(params, args.output)
    logger.info("Saved shadow params to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
