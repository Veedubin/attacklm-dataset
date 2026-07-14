#!/usr/bin/env python3
from __future__ import annotations

# Implements MAI-Thinking-1 §2.3 (Evaluation Methodology) + §2.3.2
# (Comparison of Accuracy and NLL Evaluations) by The Microsoft AI Team, June 2026.
# See docs/HELD_OUT_NLL.md.

import argparse
import json
import logging
import sys  # noqa: E402 — needed before local imports for sys.path
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the inversion package importable
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

from inversion.model_loader import detect_model_format, load_model  # noqa: E402
from inversion.provenance import RESTRICTED_SOURCES  # noqa: E402
from inversion.scoring import compute_nll  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bucket mapping for MAI-Thinking-1 §2.3 mimic_mai formula.
# See docs/HELD_OUT_NLL.md.
# ---------------------------------------------------------------------------

BUCKET_MAP: dict[str, list[str]] = {
    "Code": [
        "metasploit-framework",
        "atomic-red-team",
        "llm-generated",
    ],
    "STEM": [
        "nist-ir",
        "nvidia-garak",
    ],
    "Math": [
        "0xdf-writeups",
        "ctf-dojo",
    ],
    "General": [
        "sigma-hq",
        "splunk-security-content",
        "elastic-detection-rules",
        "manx",
        "threathunter-playbook",
        "stockpile",
        "arsenal",
        "mordor",
    ],
    "Multilingual": [
        "nyu-ctf-bench",
        "attacklm-synthetic",
    ],
}

# Weighted aggregate per MAI-Thinking-1 §2.3 Eq 3.
# See docs/HELD_OUT_NLL.md.
BUCKET_WEIGHTS: dict[str, float] = {
    "Code": 0.50,
    "STEM": 0.175,
    "Math": 0.175,
    "General": 0.10,
    "Multilingual": 0.05,
}


def source_to_bucket(source: str) -> str | None:
    """Map a source name to its bucket. Returns None if unknown."""
    for bucket, sources in BUCKET_MAP.items():
        if source in sources:
            return bucket
    return None


def load_held_out_records(
    held_out_root: Path,
    source_filter: list[str] | None = None,
    max_records_per_source: int | None = None,
) -> dict[str, list[dict]]:
    """Load held-out records from data/held_out/<source>/.../data_held_out.jsonl.

    Skips restricted sources (RTA, infection_monkey, BPL).

    Returns:
        Dict mapping source name to list of records.
    """
    if not held_out_root.exists():
        raise FileNotFoundError(
            f"Held-out root not found: {held_out_root}. "
            f"Run scripts/split_held_out.py first to create held-out splits."
        )

    available_dirs = sorted(
        d.name
        for d in held_out_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    # Filter out restricted sources and log warnings
    sources_to_load = []
    for src in available_dirs:
        if src in RESTRICTED_SOURCES:
            logger.warning("Skipping restricted source '%s' (denylisted).", src)
            continue
        if source_filter is not None and src not in source_filter:
            continue
        sources_to_load.append(src)

    records_by_source: dict[str, list[dict]] = {}

    for source_name in sources_to_load:
        source_dir = held_out_root / source_name
        source_records: list[dict] = []

        if not source_dir.is_dir():
            logger.warning("Source directory missing: %s", source_dir)
            continue

        for jsonl_path in sorted(source_dir.rglob("data_held_out.jsonl")):
            logger.info("Loading %s", jsonl_path.relative_to(held_out_root))
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

        if (
            max_records_per_source is not None
            and len(source_records) > max_records_per_source
        ):
            logger.info(
                "Capping source '%s': %d -> %d records (max_records_per_source=%d)",
                source_name,
                len(source_records),
                max_records_per_source,
                max_records_per_source,
            )
            # Deterministic truncation (first N records)
            source_records = source_records[:max_records_per_source]

        records_by_source[source_name] = source_records
        logger.info(
            "Source '%s': %d held-out records loaded", source_name, len(source_records)
        )

    return records_by_source


def compute_mean_nll_for_record(
    record: dict,
    model: Any,
    tokenizer: Any,
) -> float:
    """Compute mean per-token NLL for a single record.

    Uses the model's forward pass to get the total NLL and divides by
    the number of tokens scored.

    Returns:
        Mean per-token NLL (float). Returns float('inf') if no assistant
        turn or no tokens.
    """
    from inversion.scoring import _extract_assistant_turn

    text = _extract_assistant_turn(record)
    if not text:
        return float("inf")

    total_nll, num_tokens = compute_nll(model, tokenizer, text)

    if num_tokens == 0:
        return float("inf")

    return total_nll / num_tokens


# Implements MAI-Thinking-1 §2.3 Eq-3 weighted aggregate.
# See docs/HELD_OUT_NLL.md.
def aggregate_mimic_mai(per_source: dict[str, float]) -> dict[str, Any]:
    """Aggregate per-source NLLs using the MAI-Thinking-1 §2.3 mimic_mai formula.

    Groups sources into 5 buckets with predefined weights, computes
    each bucket's NLL as the mean of its member sources' NLLs, then
    computes the weighted aggregate.

    If a bucket has no sources present, its weight is redistributed
    proportionally to the other buckets.

    Returns:
        Dict with 'per_bucket' and 'aggregate' keys.
    """
    # Assign sources to buckets
    bucket_nlls: dict[str, list[float]] = {b: [] for b in BUCKET_MAP}

    for source, nll in per_source.items():
        bucket = source_to_bucket(source)
        if bucket is not None:
            bucket_nlls[bucket].append(nll)
        else:
            # Unknown source -> assign to General bucket
            logger.info("Source '%s' not in any bucket; assigning to General.", source)
            bucket_nlls["General"].append(nll)

    # Compute per-bucket NLL (mean of member sources)
    per_bucket: dict[str, float] = {}
    for bucket, nlls in bucket_nlls.items():
        if nlls:
            per_bucket[bucket] = sum(nlls) / len(nlls)
        # Empty buckets are not included in per_bucket

    # Redistribute weights for empty buckets
    active_weights = {}
    inactive_weight = 0.0
    for bucket, weight in BUCKET_WEIGHTS.items():
        if bucket in per_bucket:
            active_weights[bucket] = weight
        else:
            inactive_weight += weight

    # Redistribute inactive weight proportionally
    if inactive_weight > 0 and active_weights:
        total_active = sum(active_weights.values())
        if total_active > 0:
            for bucket in active_weights:
                active_weights[bucket] += inactive_weight * (
                    active_weights[bucket] / total_active
                )

    # Compute weighted aggregate
    # Implements MAI-Thinking-1 §2.3 Eq-3 weighted aggregate.
    # See docs/HELD_OUT_NLL.md.
    aggregate = 0.0
    if active_weights:
        aggregate = sum(
            per_bucket[bucket] * active_weights[bucket] for bucket in active_weights
        )

    return {"per_bucket": per_bucket, "aggregate": round(aggregate, 6)}


def aggregate_equal(per_source: dict[str, float]) -> dict[str, Any]:
    """Simple mean across all sources (the 'I don't want to weight' option)."""
    if not per_source:
        return {"per_bucket": {}, "aggregate": 0.0}

    values = list(per_source.values())
    aggregate = sum(values) / len(values)
    return {"per_bucket": {}, "aggregate": round(aggregate, 6)}


def aggregate_custom(
    per_source: dict[str, float],
    weights_path: Path,
) -> dict[str, Any]:
    """Aggregate using custom weights from a JSON file.

    The JSON file can map either:
    - source names to weights (source-level)
    - bucket names to weights (bucket-level)

    Returns:
        Dict with 'per_bucket' and 'aggregate' keys.
    """
    with open(weights_path) as f:
        weights = json.load(f)

    if not weights:
        raise ValueError(f"Empty weights file: {weights_path}")

    # Check if keys look like bucket names or source names
    all_keys = set(weights.keys())
    bucket_keys = set(BUCKET_MAP.keys())
    all_sources = set()
    for sources in BUCKET_MAP.values():
        all_sources.update(sources)

    if all_keys <= bucket_keys:
        # Bucket-level weights
        per_bucket: dict[str, float] = {}
        bucket_nlls: dict[str, list[float]] = {b: [] for b in BUCKET_MAP}

        for source, nll in per_source.items():
            bucket = source_to_bucket(source)
            if bucket is not None:
                bucket_nlls[bucket].append(nll)
            else:
                bucket_nlls["General"].append(nll)

        for bucket, nlls in bucket_nlls.items():
            if nlls:
                per_bucket[bucket] = sum(nlls) / len(nlls)

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError(
                f"Weights must sum to a positive number, got {total_weight}"
            )

        aggregate = sum(
            per_bucket.get(b, 0.0) * (weights.get(b, 0.0) / total_weight)
            for b in weights
            if b in per_bucket
        )
        return {"per_bucket": per_bucket, "aggregate": round(aggregate, 6)}

    elif all_keys <= all_sources:
        # Source-level weights
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError(
                f"Weights must sum to a positive number, got {total_weight}"
            )

        aggregate = sum(
            nll * (weights.get(src, 0.0) / total_weight)
            for src, nll in per_source.items()
        )
        return {"per_bucket": {}, "aggregate": round(aggregate, 6)}

    else:
        raise ValueError(
            f"Weights file keys must be either all bucket names "
            f"({sorted(bucket_keys)}) or all source names. Got: {sorted(all_keys)}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for held_out_nll."""
    parser = argparse.ArgumentParser(
        description="Held-out NLL evaluation suite. "
        "Computes per-source held-out NLL and aggregates via a weighted "
        "formula inspired by MAI-Thinking-1 §2.3 Equation 3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Methodology: implements MAI-Thinking-1 §2.3 (Evaluation Methodology) "
        "+ §2.3.2 (Comparison of Accuracy and NLL Evaluations). Uses the paper's "
        "weighted Eq-3 aggregate across 5 buckets. See docs/HELD_OUT_NLL.md and "
        "'MAI-Thinking-1: Building a Hill-Climbing Machine' (Microsoft AI Team, "
        "June 2026).",
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
        "--held-out-root",
        type=Path,
        default=Path("data/held_out/"),
        help="Path to data/held_out/ directory (default: data/held_out/).",
    )
    parser.add_argument(
        "--source-filter",
        nargs="+",
        default=None,
        help="Restrict to these source names (default: all non-restricted).",
    )
    parser.add_argument(
        "--aggregation-formula",
        choices=["mimic_mai", "equal", "custom"],
        default="mimic_mai",
        help="Aggregation formula (default: mimic_mai). "
        "'mimic_mai' uses MAI-Thinking-1 §2.3 Eq-3 weights. "
        "'equal' uses simple mean across sources. "
        "'custom' loads weights from --weights file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("data/held_out_weights.json"),
        help="Path to weights JSON file (only used with --aggregation-formula=custom). "
        "Default: data/held_out_weights.json.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Path to write the JSON report (default: stdout).",
    )
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="Maximum number of held-out records per source (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without loading model or computing NLL.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="held-out-nll 0.8.0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for held_out NLL evaluation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    held_out_root = args.held_out_root.resolve()

    # Validate source filter - check for restricted sources
    if args.source_filter:
        for src in args.source_filter:
            if src in RESTRICTED_SOURCES:
                logger.error(
                    "Source '%s' is in the restricted-source denylist. "
                    "Its license constrains redistribution and it is excluded "
                    "from evaluation. Restricted sources: %s",
                    src,
                    sorted(RESTRICTED_SOURCES),
                )
                return 1

    # Load held-out records
    logger.info("Loading held-out records from %s", held_out_root)
    records_by_source = load_held_out_records(
        held_out_root,
        source_filter=args.source_filter,
        max_records_per_source=args.max_records_per_source,
    )

    if not records_by_source:
        logger.error("No records loaded. Check --held-out-root and --source-filter.")
        return 1

    total_records = sum(len(r) for r in records_by_source.values())
    logger.info(
        "Loaded %d records from %d sources",
        total_records,
        len(records_by_source),
    )

    if args.dry_run:
        print(
            f"Dry run: would evaluate {total_records} records "
            f"across {len(records_by_source)} sources"
        )
        print(f"  Model: {args.model}")
        print(f"  Held-out root: {held_out_root}")
        print(f"  Source filter: {args.source_filter or 'all'}")
        print(f"  Aggregation formula: {args.aggregation_formula}")
        print(f"  Max records per source: {args.max_records_per_source or 'all'}")
        for src, recs in sorted(records_by_source.items()):
            bucket = source_to_bucket(src) or "General (unassigned)"
            print(f"  Source '{src}': {len(recs)} records (bucket: {bucket})")
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
            "NLL evaluation requires white-box access to model logits. "
            "Use --model-format hf with a HuggingFace model."
        )
        return 1

    model, tokenizer = load_model(args.model, args.model_format)

    # Compute per-source mean NLL
    per_source: dict[str, float] = {}
    total_evaluated = 0

    for source_name, records in sorted(records_by_source.items()):
        logger.info("Evaluating source '%s': %d records", source_name, len(records))

        source_nlls: list[float] = []
        for i, record in enumerate(records):
            try:
                mean_nll = compute_mean_nll_for_record(record, model, tokenizer)
                if mean_nll != float("inf"):
                    source_nlls.append(mean_nll)
                    total_evaluated += 1
            except Exception as e:
                logger.warning(
                    "Failed to score record %d from source '%s': %s",
                    i,
                    source_name,
                    e,
                )

        if source_nlls:
            source_mean_nll = sum(source_nlls) / len(source_nlls)
        else:
            source_mean_nll = float("inf")
            logger.warning("Source '%s': no valid NLL scores computed.", source_name)

        per_source[source_name] = round(source_mean_nll, 6)
        logger.info(
            "Source '%s': mean_nll=%.6f (%d/%d records scored)",
            source_name,
            source_mean_nll,
            len(source_nlls),
            len(records),
        )

    # Apply aggregation formula
    if args.aggregation_formula == "mimic_mai":
        result = aggregate_mimic_mai(per_source)
    elif args.aggregation_formula == "equal":
        result = aggregate_equal(per_source)
    elif args.aggregation_formula == "custom":
        result = aggregate_custom(per_source, args.weights)
    else:
        logger.error("Unknown aggregation formula: %s", args.aggregation_formula)
        return 1

    # Build the report
    report = {
        "schema_version": "1.0",
        "methodology": "MAI-Thinking-1 §2.3 Eq-3 weighted aggregate",
        "per_source": per_source,
        "per_bucket": result["per_bucket"],
        "aggregate": result["aggregate"],
        "aggregation_formula": args.aggregation_formula,
        "n_records_evaluated": total_evaluated,
        "model": str(args.model),
        "model_format": model_format,
        "held_out_root": str(held_out_root),
        "source_filter": args.source_filter or ["all"],
        "date": datetime.now(timezone.utc).isoformat(),
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
