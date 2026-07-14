#!/usr/bin/env python3
from __future__ import annotations

# Implements MAI-Thinking-1 §2.5.4 (Memorization-aware epoch capping)
# by The Microsoft AI Team, June 2026. See docs/MEMORIZATION.md.

"""Memorization-aware epoch capping report.

Implements MAI-Thinking-1 §2.5.4 NLL<0.01 fraction proxy.
See docs/MEMORIZATION.md for the full spec.

For each per-source training dataset, loads a small held-out sample,
runs per-token NLL on each record through the model, computes the
fraction of NLL<0.01 tokens (the memorization proxy), and emits a
per-source report with the recommended epoch cap.

Usage:
    python scripts/memorization_report.py \
        --model /path/to/model \
        --dataset-root data/datasets/buckets/sources/ \
        --source-filter metasploit-framework sigma-hq \
        --sample-size-per-source 200

All flags are documented in the argument parser below.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

# Make the inversion package importable (E402 pattern consistent with inversion_audit.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inversion import __version__
from inversion.model_loader import load_model, detect_model_format
from inversion.provenance import (
    RESTRICTED_SOURCES,
    get_available_sources,
    validate_source_filter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default epoch-cap table (from MAI-Thinking-1 §2.5.4)
# ---------------------------------------------------------------------------
# Implements MAI-Thinking-1 §2.5.4 memorization proxy: fraction of NLL<0.01 tokens.
# See docs/MEMORIZATION.md.

DEFAULT_EPOCH_CAP_THRESHOLDS: list[tuple[float, int]] = [
    # (memorization_fraction_upper_bound, recommended_epoch_cap)
    (0.05, 8),  # < 0.05 → high exposure, low memorization → 8 epochs
    (0.15, 4),  # 0.05–0.15 → moderate → 4 epochs
    (0.30, 2),  # 0.15–0.30 → elevated memorization → 2 epochs
    (1.01, 1),  # > 0.30 → high memorization → 1 epoch (low exposure)
]


def recommend_epoch_cap(
    memorization_fraction: float,
    thresholds: list[tuple[float, int]] | None = None,
) -> int:
    """Map a memorization fraction to a recommended epoch cap.

    Implements MAI-Thinking-1 §2.5.4 memorization proxy: fraction of NLL<0.01 tokens.
    See docs/MEMORIZATION.md.

    Args:
        memorization_fraction: Fraction of tokens with NLL < threshold.
        thresholds: List of (upper_bound, cap) tuples, sorted ascending.
            Defaults to DEFAULT_EPOCH_CAP_THRESHOLDS.

    Returns:
        Recommended epoch cap (integer).
    """
    if thresholds is None:
        thresholds = DEFAULT_EPOCH_CAP_THRESHOLDS
    for upper_bound, cap in thresholds:
        if memorization_fraction < upper_bound:
            return cap
    # Fallback: return the lowest cap (most conservative)
    return thresholds[-1][1]


def load_records(
    dataset_root: Path,
    source_filter: list[str] | None,
    per_source_limit: int | None = None,
) -> dict[str, list[dict]]:
    """Load training records from the per-source dataset layout.

    Reads JSONL files from sources/<source>/<bucket>/<tactic>/data*.jsonl.
    Applies source filtering and skips restricted sources.

    Returns:
        Dict mapping source name to list of records.
    """
    sources_dir = dataset_root

    if not sources_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {sources_dir}")

    available = get_available_sources(sources_dir)
    if source_filter:
        validate_source_filter(source_filter, sources_dir)
        selected = [s for s in source_filter if s in available]
        if not selected:
            logger.warning("No matching sources found for filter: %s", source_filter)
    else:
        selected = [s for s in available if s not in RESTRICTED_SOURCES]

    records_by_source: dict[str, list[dict]] = {}

    for source_name in sorted(selected):
        source_records: list[dict] = []
        source_dir = sources_dir / source_name
        if not source_dir.is_dir():
            logger.warning("Source directory missing: %s", source_dir)
            continue
        for jsonl_path in sorted(source_dir.rglob("*.jsonl")):
            logger.info("Loading %s", jsonl_path.relative_to(sources_dir))
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            record.setdefault("_source", source_name)
                            source_records.append(record)
                        except json.JSONDecodeError as e:
                            logger.warning("Bad JSON in %s: %s", jsonl_path, e)

        if per_source_limit is not None and len(source_records) > per_source_limit:
            logger.info(
                "Capping source '%s': %d -> %d records (per_source_limit=%d)",
                source_name,
                len(source_records),
                per_source_limit,
                per_source_limit,
            )
            # Use deterministic sampling for reproducibility
            rng = random.Random(42)
            source_records = rng.sample(source_records, per_source_limit)

        records_by_source[source_name] = source_records
        logger.info("Source '%s': %d records loaded", source_name, len(source_records))

    return records_by_source


def compute_memorization_for_record(
    record: dict,
    model: Any,
    tokenizer: Any,
    nll_threshold: float = 0.01,
) -> dict[str, Any]:
    """Compute per-token NLL metrics for a single record.

    Returns dict with keys:
        - nll_lt_threshold_tokens: count of tokens with NLL < threshold
        - total_tokens: total tokens scored
        - per_token_nlls: list of per-token NLL values (for analysis)
        - record_id: the record's id field (or index-based fallback)
    """
    # Extract the assistant turn for scoring (same as scoring.py)
    from inversion.scoring import _extract_assistant_turn

    text = _extract_assistant_turn(record)
    if not text:
        return {
            "nll_lt_threshold_tokens": 0,
            "total_tokens": 0,
            "per_token_nlls": [],
            "record_id": record.get("id", "unknown"),
        }

    # Compute per-token NLL via model forward pass
    import torch

    inputs = tokenizer(text, return_tensors="pt")
    if hasattr(model, "device") and model.device.type != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # outputs.loss is the mean NLL per token; get per-token log probs
    # for the fraction computation
    num_tokens = inputs["input_ids"].shape[1]

    # Get per-token NLL from the logit outputs
    # Shift: predict token t+1 from logits at position t
    logits = outputs.logits[:, :-1, :]  # (batch, seq_len-1, vocab)
    targets = inputs["input_ids"][:, 1:]  # (batch, seq_len-1)

    # Per-token NLL: -log softmax probability of the correct token
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # Gather the log prob of the target token at each position
    target_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(
        -1
    )  # (batch, seq_len-1)

    per_token_nlls = (-target_log_probs[0]).cpu().tolist()  # negative log prob = NLL

    nll_lt_threshold = sum(1 for nll in per_token_nlls if nll < nll_threshold)

    return {
        "nll_lt_threshold_tokens": nll_lt_threshold,
        "total_tokens": len(per_token_nlls),
        "per_token_nlls": per_token_nlls[:10],  # Keep top-10 for brevity in report
        "record_id": record.get("id", "unknown"),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for memorization_report."""
    parser = argparse.ArgumentParser(
        description="Memorization-aware epoch capping report. "
        "Computes the NLL<0.01 fraction proxy (MAI-Thinking-1 §2.5.4) "
        "for each source and recommends training epoch caps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Methodology: implements MAI-Thinking-1 §2.5.4 NLL<0.01 fraction "
        "proxy (Microsoft AI Team, June 2026). See docs/MEMORIZATION.md.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model directory (HF safetensors) or .gguf file.",
    )
    parser.add_argument(
        "--model-format",
        choices=["hf", "gguf"],
        default=None,
        help="Model format. Default: auto-detect.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/datasets/buckets/sources/"),
        help="Path to data/datasets/buckets/sources/ directory "
        "(default: data/datasets/buckets/sources/).",
    )
    parser.add_argument(
        "--source-filter",
        nargs="+",
        default=None,
        help="Restrict to these source names (default: all non-restricted).",
    )
    parser.add_argument(
        "--sample-size-per-source",
        type=int,
        default=200,
        help="Number of records to sample per source (default: 200).",
    )
    parser.add_argument(
        "--nll-threshold",
        type=float,
        default=0.01,
        help='The "near-certain" NLL threshold for memorization proxy '
        "(default: 0.01). Tokens with NLL below this threshold are "
        "counted as memorized.",
    )
    parser.add_argument(
        "--epoch-cap-table",
        type=Path,
        default=None,
        help="Path to a JSON file defining custom epoch-cap thresholds. "
        "Format: [[upper_bound, epoch_cap], ...]. "
        "Default: [[0.05, 8], [0.15, 4], [0.30, 2], [1.01, 1]].",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Path to write the JSON report (default: stdout).",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processes (default: 1; GPU-bound, parallelism "
        "rarely helps unless you have multiple GPUs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without loading model or computing NLL.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"memorization-report {__version__}",
    )
    return parser


def load_epoch_cap_table(path: Path | None) -> list[tuple[float, int]]:
    """Load a custom epoch-cap threshold table from a JSON file.

    The JSON file should contain a list of [upper_bound, epoch_cap] pairs,
    e.g. [[0.05, 8], [0.15, 4], [0.30, 2], [1.01, 1]].

    Returns:
        Sorted list of (upper_bound, epoch_cap) tuples.
    """
    if path is None:
        return DEFAULT_EPOCH_CAP_THRESHOLDS

    with open(path) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in {path}, got {type(raw).__name__}")

    thresholds: list[tuple[float, int]] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"Each entry must be [upper_bound, epoch_cap], got {entry}"
            )
        thresholds.append((float(entry[0]), int(entry[1])))

    # Sort by upper bound ascending
    thresholds.sort(key=lambda x: x[0])
    return thresholds


def main(argv: list[str] | None = None) -> int:
    """Main entry point for memorization_report."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load custom epoch-cap table if provided
    thresholds = load_epoch_cap_table(args.epoch_cap_table)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dataset_root = args.dataset_root.resolve()

    # Validate source filter
    source_filter = args.source_filter
    if source_filter:
        try:
            validate_source_filter(source_filter, dataset_root)
        except ValueError as e:
            logger.error("%s", e)
            return 1

    # Load records grouped by source
    logger.info("Loading records from %s", dataset_root)
    records_by_source = load_records(
        dataset_root, source_filter, per_source_limit=args.sample_size_per_source
    )

    if not records_by_source:
        logger.error("No records loaded. Check --dataset-root and --source-filter.")
        return 1

    total_records = sum(len(r) for r in records_by_source.values())
    logger.info(
        "Loaded %d records from %d sources (sample_size_per_source=%d)",
        total_records,
        len(records_by_source),
        args.sample_size_per_source,
    )

    if args.dry_run:
        print(
            f"Dry run: would evaluate {total_records} records across {len(records_by_source)} sources"
        )
        print(f"  Model: {args.model}")
        print(f"  Dataset root: {dataset_root}")
        print(f"  Source filter: {source_filter or 'all'}")
        print(f"  NLL threshold: {args.nll_threshold}")
        print(f"  Sample size per source: {args.sample_size_per_source}")
        print(f"  Epoch cap thresholds: {thresholds}")
        for src, recs in records_by_source.items():
            print(f"  Source '{src}': {len(recs)} records")
        return 0

    # Load model
    logger.info("Loading model from %s", args.model)
    try:
        model_format = args.model_format or detect_model_format(args.model)
    except ValueError:
        logger.warning(
            "Cannot auto-detect model format at %s; defaulting to 'hf'. "
            "Specify --model-format explicitly to suppress this warning.",
            args.model,
        )
        model_format = "hf"

    if model_format == "gguf":
        logger.error(
            "GGUF models do not expose per-token loss. "
            "Memorization fraction requires white-box access to model logits. "
            "Use --model-format hf with a HuggingFace model."
        )
        return 1

    model, tokenizer = load_model(args.model, args.model_format)

    # Compute memorization metrics per source
    report_results: list[dict[str, Any]] = []

    for source_name, records in sorted(records_by_source.items()):
        logger.info("Evaluating source '%s': %d records", source_name, len(records))

        source_nll_lt_threshold = 0
        source_total_tokens = 0
        record_scores: list[dict[str, Any]] = []

        for i, record in enumerate(records):
            try:
                result = compute_memorization_for_record(
                    record, model, tokenizer, args.nll_threshold
                )
                source_nll_lt_threshold += result["nll_lt_threshold_tokens"]
                source_total_tokens += result["total_tokens"]

                # Track per-record memorization for example_records
                if result["total_tokens"] > 0:
                    frac = result["nll_lt_threshold_tokens"] / result["total_tokens"]
                    record_scores.append(
                        {
                            "record_id": result["record_id"],
                            "memorization_fraction": frac,
                            "total_tokens": result["total_tokens"],
                            "nll_lt_threshold_tokens": result[
                                "nll_lt_threshold_tokens"
                            ],
                        }
                    )
            except Exception as e:
                logger.warning(
                    "Failed to score record %d from source '%s': %s",
                    i,
                    source_name,
                    e,
                )

        # Compute source-level memorization fraction
        if source_total_tokens > 0:
            memorization_fraction = source_nll_lt_threshold / source_total_tokens
        else:
            memorization_fraction = 0.0

        # Map to recommended epoch cap
        # Implements MAI-Thinking-1 §2.5.4 memorization proxy: fraction of NLL<0.01 tokens.
        # See docs/MEMORIZATION.md.
        epoch_cap = recommend_epoch_cap(memorization_fraction, thresholds)

        # Find top-5 most-memorized records (highest NLL<0.01 fraction)
        record_scores.sort(key=lambda r: r["memorization_fraction"], reverse=True)
        example_records = [r["record_id"] for r in record_scores[:5]]

        source_result = {
            "source": source_name,
            "n_records_sampled": len(records),
            "total_tokens": source_total_tokens,
            "nll_lt_threshold_tokens": source_nll_lt_threshold,
            "memorization_fraction": round(memorization_fraction, 4),
            "recommended_epoch_cap": epoch_cap,
            "example_records": example_records,
        }
        report_results.append(source_result)

        logger.info(
            "Source '%s': memorization_fraction=%.4f, epoch_cap=%d, "
            "tokens=%d (nll<%.2f=%d)",
            source_name,
            memorization_fraction,
            epoch_cap,
            source_total_tokens,
            args.nll_threshold,
            source_nll_lt_threshold,
        )

    # Build the full report
    report = {
        "schema_version": "1.0",
        "methodology": "MAI-Thinking-1 §2.5.4 NLL<0.01 fraction proxy",
        "nll_threshold": args.nll_threshold,
        "model_path": str(args.model),
        "model_format": model_format,
        "dataset_root": str(dataset_root),
        "source_filter": source_filter or ["all"],
        "sample_size_per_source": args.sample_size_per_source,
        "epoch_cap_thresholds": [[ub, cap] for ub, cap in thresholds],
        "sources": report_results,
    }

    # Write report
    report_json = json.dumps(report, indent=2)

    if args.report_output:
        output_path = args.report_output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_json + "\n")
        logger.info("Report written to %s", output_path)
    else:
        print(report_json)

    # Also write the epoch cap table to the default path if it doesn't exist
    default_cap_path = (
        dataset_root.parent.parent.parent / "data" / "memorization_epoch_caps.json"
    )
    try:
        cap_data = {
            "schema_version": "1.0",
            "methodology": "MAI-Thinking-1 §2.5.4 NLL<0.01 fraction proxy",
            "nll_threshold": args.nll_threshold,
            "epoch_caps": {
                r["source"]: {
                    "memorization_fraction": r["memorization_fraction"],
                    "recommended_epoch_cap": r["recommended_epoch_cap"],
                }
                for r in report_results
            },
        }
        default_cap_path.parent.mkdir(parents=True, exist_ok=True)
        default_cap_path.write_text(json.dumps(cap_data, indent=2) + "\n")
        logger.info("Epoch cap table written to %s", default_cap_path)
    except OSError as e:
        logger.warning("Could not write epoch cap table to %s: %s", default_cap_path, e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
