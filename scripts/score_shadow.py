#!/usr/bin/env python3
from __future__ import annotations

# PROVENANCE METADATA — scripts/score_shadow.py
# ================================================================================
# Attack class:        LiRA shadow-model scoring (Carlini 2022 §4)
# Original authors:    Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song,
#                      Andreas Terzis, Florian Tramer (for the LiRA methodology)
# Paper title:         Membership Inference Attacks From First Principles
# Year / venue:        2022 / IEEE Symposium on Security and Privacy
# Paper URL:           https://arxiv.org/abs/2112.03570
# Canonical repo:      N/A (no official code release by the authors)
#
# Implementation:
#   Type:              ORIGINAL_WORK (CLI driver only; uses compute_nll from
#                      inversion.scoring and load_model from inversion.model_loader)
#   Lines of port:     N/A
#   Upstream license:  N/A
#
# This is Step 2 in the LiRA workflow (docs/LIRA.md §3): compute each shadow
# model's loss on the audit set. The NLL computation reuses the same
# compute_nll + _extract_assistant_turn helpers that inversion_audit.py uses
# for the reference attack (Carlini 2022 §3.2).
#
# Data sources: N/A (this file reads user-supplied JSONL and writes JSON)
#
# Rights claim contact: veedubin.legal@example.com
# See:                  RIGHTS.md (root), data/LEGAL.md, data/REMOVAL.md
# ================================================================================
"""Score each shadow model on the audit set — Step 2 of the LiRA workflow.

Loads a HuggingFace model, reads a JSONL of training records, computes the
total NLL for each record's assistant turn, and writes one JSON file
(shadow_{K}.json) per shadow model. This output is consumed by
scripts/inversion/shadow_train.py (Step 3) to fit per-record Gaussians.

Usage:
    python scripts/score_shadow.py \
        --model models/shadow_0 \
        --records data/audit_set.jsonl \
        --output-dir losses/ \
        --shadow-index 0

Output format (shadow_K.json):
    {"record_id": total_nll, ...}

This matches the format expected by load_shadow_losses() in
shadow_train.py:140-147.
"""


import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make the inversion package importable — score_shadow.py lives in scripts/,
# not scripts/inversion/, so we add scripts/ to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inversion import __version__
from inversion.model_loader import load_model
from inversion.scoring import compute_nll, _extract_assistant_turn

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score a shadow model on the audit set (LiRA Step 2). "
            "Loads a HuggingFace model, computes total NLL per record, "
            "and writes shadow_{K}.json to the output directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to HuggingFace model directory.",
    )
    parser.add_argument(
        "--model-format",
        choices=["hf"],
        default="hf",
        help="Model format. Only 'hf' is supported (MIA needs loss). Default: hf.",
    )
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="JSONL file of training records to score.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where shadow_{K}.json is written.",
    )
    parser.add_argument(
        "--shadow-index",
        type=int,
        required=True,
        help="The K index for this shadow model (0..K-1). Used in the "
        "output filename shadow_{K}.json.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap on number of records to score (for testing). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done, no model load, no file write.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"inversion-audit {__version__}",
    )
    return parser


def load_records_from_jsonl(
    records_path: Path, max_records: int | None = None
) -> list[dict]:
    """Load records from a JSONL file.

    Args:
        records_path: Path to the JSONL file.
        max_records: If set, cap at this many records.

    Returns:
        List of record dicts.
    """
    records: list[dict] = []
    if not records_path.exists():
        logger.error("Records file not found: %s", records_path)
        return records

    with open(records_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Bad JSON in %s: %s", records_path, e)

    if max_records is not None and len(records) > max_records:
        logger.info(
            "Capping records: %d -> %d (--max-records)",
            len(records),
            max_records,
        )
        records = records[:max_records]

    logger.info("Loaded %d records from %s", len(records), records_path)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load records
    records = load_records_from_jsonl(args.records, args.max_records)

    # Empty records file — write sentinel and exit 0
    if not records:
        logger.warning(
            "No records loaded from %s; writing empty sentinel", args.records
        )
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"shadow_{args.shadow_index}.json"
        sentinel = {"_meta": {"empty": True, "n_records": 0}}
        tmp_path = str(output_path) + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(sentinel, f, indent=2)
        os.replace(tmp_path, str(output_path))
        logger.info("Wrote empty sentinel to %s", output_path)
        return 0

    if args.dry_run:
        print(f"Dry run: would score {len(records)} records")
        print(f"  Model: {args.model}")
        print(f"  Records: {args.records}")
        print(f"  Output: {args.output_dir / f'shadow_{args.shadow_index}.json'}")
        print(f"  Shadow index: {args.shadow_index}")
        return 0

    # Load model (HF only — GGUF doesn't expose loss)
    logger.info("Loading model from %s", args.model)
    model, tokenizer = load_model(args.model, args.model_format)

    # Score each record
    losses: dict[str, float] = {}
    skipped = 0
    for i, record in enumerate(records):
        record_id = record.get("id", str(i))
        suffix_text = _extract_assistant_turn(record)

        if not suffix_text:
            logger.warning("Record %s: no assistant turn; skipping", record_id)
            skipped += 1
            # Convention: missing assistant turn → nll=inf (record present but unscored)
            losses[record_id] = float("inf")
            continue

        try:
            total_nll, _num_tokens = compute_nll(model, tokenizer, suffix_text)
            losses[record_id] = total_nll
        except Exception as e:
            logger.warning("Record %s: compute_nll failed: %s", record_id, e)
            losses[record_id] = float("inf")
            skipped += 1

        if (i + 1) % 100 == 0:
            logger.info("Scored %d/%d records", i + 1, len(records))

    # Check for duplicate record_ids (last write wins)
    seen_ids: dict[str, int] = {}
    for record in records:
        rid = record.get("id", str(records.index(record)))
        seen_ids[rid] = seen_ids.get(rid, 0) + 1
    duplicates = {rid: count for rid, count in seen_ids.items() if count > 1}
    if duplicates:
        logger.warning(
            "Duplicate record_ids detected (last write wins): %s",
            {rid: count for rid, count in duplicates.items()},
        )

    # Atomic write: shadow_K.json.tmp → shadow_K.json
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shadow_{args.shadow_index}.json"
    tmp_path = str(output_path) + ".tmp"

    logger.info("Writing %d scores to %s", len(losses), output_path)
    with open(tmp_path, "w") as f:
        json.dump(losses, f, indent=2)
    os.replace(tmp_path, str(output_path))

    logger.info(
        "Shadow %d complete: %d records scored, %d skipped",
        args.shadow_index,
        len(losses),
        skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
