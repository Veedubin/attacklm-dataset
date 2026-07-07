#!/usr/bin/env python3
"""Inversion-attack audit harness for AttackLM — CLI driver.

Defensive audit tooling to probe the user's OWN models for memorized
training data. Two strategies:

1. Carlini prefix-completion extraction (2021)
2. Membership-inference scoring via loss + zlib (2022, no shadow model)

Usage:
    python scripts/inversion_audit.py \
        --model /path/to/model \
        --dataset-root /path/to/data/datasets/buckets/sources \
        --source-filter metasploit-framework sigma-hq atomic-red-team \
        --probe-carlini --probe-mia \
        --top-k 20 --max-new-tokens 64 --temperature 1.0

All flags are documented in the argument parser below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path

# Make the inversion package importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inversion import __version__
from inversion.model_loader import load_model, detect_model_format
from inversion.probe import run_carlini_probe
from inversion.scoring import (
    score_record,
)
from inversion.provenance import (
    RESTRICTED_SOURCES,
    RecordProvenance,
    get_available_sources,
    validate_source_filter,
)
from inversion.reporting import (
    check_output_dir_permissions,
    create_audit_dir,
    write_raw_results,
    write_summary,
    write_exportable_summary,
    write_run_log,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inversion-attack audit harness for AttackLM. "
        "Probes your OWN models for memorized training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        required=True,
        help="Path to data/datasets/buckets/sources/ directory.",
    )
    parser.add_argument(
        "--source-filter",
        nargs="+",
        default=None,
        help="Restrict to these source names (default: all non-restricted).",
    )
    parser.add_argument(
        "--probe-carlini",
        action="store_true",
        default=True,
        help="Enable Carlini prefix-completion extraction probe (default: on).",
    )
    parser.add_argument(
        "--no-probe-carlini",
        action="store_true",
        help="Disable Carlini prefix-completion probe.",
    )
    parser.add_argument(
        "--probe-mia",
        action="store_true",
        default=True,
        help="Enable MIA loss+zlib scoring (default: on).",
    )
    parser.add_argument(
        "--no-probe-mia",
        action="store_true",
        help="Disable MIA scoring.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of completions per prefix for Carlini probe (default: 20).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens per completion (default: adaptive — "
        "min(256, max(64, 2*suffix_token_count)) per record). "
        "Set explicitly to override the adaptive cap.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for Carlini probe (default: 1.0).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date for audit directory (YYYY-MM-DD). Default: today. "
        "Errors if directory already exists.",
    )
    parser.add_argument(
        "--audit-output-root",
        type=Path,
        default=None,
        help="Root directory for audit output. Default: derived from dataset-root.",
    )
    parser.add_argument(
        "--member-threshold",
        type=float,
        default=None,
        help="MIA membership threshold. Default: auto-calibrate on held-out set.",
    )
    parser.add_argument(
        "--mia-threshold-mode",
        choices=["median", "percentile", "holdout_file"],
        default="percentile",
        help="MIA threshold calibration mode. 'median': median of probed scores "
        "(calibration artifact — WARNING logged). 'percentile': Nth percentile "
        "of probed scores (default). 'holdout_file': read threshold from a JSON "
        "file (Track 2 placeholder).",
    )
    parser.add_argument(
        "--mia-percentile",
        type=int,
        default=5,
        help="Percentile for MIA threshold when --mia-threshold-mode=percentile "
        "(default: 5, meaning only the bottom 5%% of scores are flagged).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without loading model or writing files.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of records per source to probe (for testing). "
        "Applied per-source, not globally — each source gets up to N records. "
        "Default: all records from each source.",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=None,
        help="Alias for --max-records. Maximum records per source to probe.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"inversion-audit {__version__}",
    )
    return parser


def load_records(
    dataset_root: Path,
    source_filter: list[str] | None,
    per_source_limit: int | None = None,
) -> list[dict]:
    """Load training records from the per-source dataset layout.

    Reads JSONL files from sources/<source>/<bucket>/<tactic>/data*.jsonl.
    Applies source filtering and skips restricted sources.

    Args:
        dataset_root: Path to data/datasets/buckets/sources/.
        source_filter: List of source names to include (None = all non-restricted).
        per_source_limit: If set, cap each source to this many records.
            This ensures all requested sources get probed equally, rather than
            a global cap that only reaches the first source alphabetically.
    """
    records = []
    sources_dir = dataset_root

    if not sources_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {sources_dir}")

    available = get_available_sources(sources_dir)
    if source_filter:
        # Validate no restricted sources
        validate_source_filter(source_filter, sources_dir)
        selected = [s for s in source_filter if s in available]
        if not selected:
            logger.warning("No matching sources found for filter: %s", source_filter)
    else:
        # Use all available sources except restricted ones
        selected = [s for s in available if s not in RESTRICTED_SOURCES]

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
            source_records = source_records[:per_source_limit]
        records.extend(source_records)

    logger.info("Loaded %d records from %d sources", len(records), len(selected))
    return records


def compute_manifest_hash(dataset_root: Path) -> str:
    """Hash the _index.json manifest for audit trail."""
    index_path = dataset_root / "_index.json"
    if index_path.exists():
        content = index_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    return "no-manifest"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Merge --probe-count into --max-records (alias)
    if args.probe_count is not None:
        if args.max_records is not None:
            logger.warning(
                "Both --max-records and --probe-count specified; using --max-records=%d",
                args.max_records,
            )
        else:
            args.max_records = args.probe_count

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve flags
    probe_carlini = args.probe_carlini and not args.no_probe_carlini
    probe_mia = args.probe_mia and not args.no_probe_mia

    audit_date = args.date or date.today().isoformat()
    dataset_root = args.dataset_root.resolve()

    # Determine output root
    if args.audit_output_root:
        audit_output_root = args.audit_output_root.resolve()
    else:
        audit_output_root = dataset_root.parent.parent.parent / "audit"

    # Validate source filter
    source_filter = args.source_filter
    if source_filter:
        try:
            validate_source_filter(source_filter, dataset_root)
        except ValueError as e:
            logger.error("%s", e)
            return 1

    # Pre-flight checks
    check_output_dir_permissions(audit_output_root)

    # Load records — per_source_limit ensures all sources get probed
    per_source_limit = args.max_records  # --max-records now applies per-source
    logger.info("Loading records from %s", dataset_root)
    records = load_records(
        dataset_root, source_filter, per_source_limit=per_source_limit
    )

    if not records:
        logger.error("No records loaded. Check --dataset-root and --source-filter.")
        return 1

    if per_source_limit:
        logger.info(
            "Per-source limit: %d records (--max-records), total loaded: %d",
            per_source_limit,
            len(records),
        )

    if args.dry_run:
        print(f"Dry run: would probe {len(records)} records")
        print(f"  Model: {args.model}")
        print(f"  Dataset root: {dataset_root}")
        print(f"  Source filter: {source_filter or 'all'}")
        print(f"  Carlini probe: {probe_carlini}")
        print(f"  MIA probe: {probe_mia}")
        print(f"  Top-K: {args.top_k}")
        print(
            f"  Max new tokens: {args.max_new_tokens or 'adaptive (min(256, max(64, 2*suffix_tokens)))'}"
        )
        print(f"  Temperature: {args.temperature}")
        print(f"  Audit date: {audit_date}")
        print(f"  Output root: {audit_output_root}")
        return 0

    # Load model
    logger.info("Loading model from %s", args.model)
    model_format = args.model_format or detect_model_format(args.model)

    if model_format == "gguf" and probe_mia:
        logger.error(
            "MIA scoring requires white-box access to model loss. "
            "GGUF models do not expose loss. Use --no-probe-mia or "
            "provide a HuggingFace model with --model-format hf."
        )
        return 1

    model, tokenizer = load_model(args.model, args.model_format)

    # Create audit directory
    audit_dir = create_audit_dir(audit_output_root, audit_date)
    logger.info("Created audit directory: %s", audit_dir)

    # Run probes
    results = []
    provenances = []

    for i, record in enumerate(records):
        provenance = RecordProvenance.from_record(record)
        provenances.append(provenance)
        row = {
            "record_index": i,
            "source": provenance.source,
            "license": provenance.license_id,
            "license_family": provenance.license_family,
            "prompt_hash": "",
            "reconstruction_hash": "",
        }

        if probe_carlini and model_format == "hf":
            logger.info("Carlini probe: record %d/%d", i + 1, len(records))
            try:
                probe_result = run_carlini_probe(
                    record,
                    model,
                    tokenizer,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    record_index=i,
                )
                row.update(
                    {
                        "best_exact_match": probe_result.best_exact_match,
                        "best_lcs_length": probe_result.best_lcs_length,
                        "best_bleu4": probe_result.best_bleu4,
                        "original_length": probe_result.original_length,
                        "best_completion_length": probe_result.best_completion_length,
                        "num_completions": probe_result.num_completions,
                        "prompt_hash": probe_result.prompt_hash,
                        "reconstruction_hash": probe_result.reconstruction_hash,
                    }
                )
            except Exception as e:
                logger.warning("Carlini probe failed for record %d: %s", i, e)

        if probe_mia and model_format == "hf":
            logger.info("MIA scoring: record %d/%d", i + 1, len(records))
            try:
                mia_score = score_record(record, model, tokenizer)
                row.update(
                    {
                        "nll": mia_score.nll,
                        "zlib_length": mia_score.zlib_length,
                        "zlib_ratio": mia_score.zlib_ratio,
                        "perplexity": mia_score.perplexity,
                        "num_tokens": mia_score.num_tokens,
                        "membership_score": mia_score.membership_score,
                    }
                )
            except Exception as e:
                logger.warning("MIA scoring failed for record %d: %s", i, e)

        results.append(row)

    # MIA threshold calibration
    mia_threshold_derivation = ""
    if probe_mia and model_format == "hf":
        member_scores = [
            r["membership_score"] for r in results if "membership_score" in r
        ]
        if member_scores:
            if args.member_threshold is not None:
                # Explicit threshold from CLI
                member_threshold = args.member_threshold
                mia_threshold_derivation = f"explicit={args.member_threshold}"
                threshold_mode = "explicit"
            elif args.mia_threshold_mode == "median":
                import statistics

                member_threshold = statistics.median(member_scores)
                threshold_mode = "median"
                mia_threshold_derivation = (
                    f"median of {len(member_scores)} probed scores "
                    f"(calibration artifact)"
                )
                logger.warning(
                    "MIA threshold is median; this is a calibration artifact. "
                    "Interpretation as a memorization signal is not supported. "
                    "Threshold=%.4f, flagged=%d/%d",
                    member_threshold,
                    sum(1 for s in member_scores if s < member_threshold),
                    len(member_scores),
                )
            elif args.mia_threshold_mode == "percentile":
                member_threshold = _percentile(member_scores, args.mia_percentile)
                threshold_mode = "percentile"
                n_flagged = sum(1 for s in member_scores if s < member_threshold)
                mia_threshold_derivation = (
                    f"percentile:{args.mia_percentile} of "
                    f"{len(member_scores)} probed scores "
                    f"(threshold={member_threshold:.4f}, "
                    f"flagged={n_flagged}/{len(member_scores)})"
                )
                logger.info(
                    "MIA threshold=%.4f (%sth percentile of %d scores, flagged=%d/%d)",
                    member_threshold,
                    args.mia_percentile,
                    len(member_scores),
                    n_flagged,
                    len(member_scores),
                )
            elif args.mia_threshold_mode == "holdout_file":
                # Track 2 placeholder: read threshold from a JSON file
                threshold_mode = "holdout_file"
                holdout_path = getattr(args, "mia_holdout_file", None)
                if not holdout_path:
                    # Try default location
                    holdout_path = dataset_root / "_holdout" / "threshold.json"
                if not Path(holdout_path).exists():
                    logger.error(
                        "Holdout file not found: %s. "
                        "Create a JSON file with "
                        '{"threshold": float, "source": str, '
                        '"note": str} or use --mia-threshold-mode percentile.',
                        holdout_path,
                    )
                    return 1
                with open(holdout_path) as f:
                    holdout_data = json.load(f)
                member_threshold = float(holdout_data["threshold"])
                mia_threshold_derivation = (
                    f"holdout_file={holdout_path} "
                    f"(source={holdout_data.get('source', 'unknown')})"
                )
                logger.info(
                    "MIA threshold=%.4f from holdout file %s",
                    member_threshold,
                    holdout_path,
                )
            else:
                # Should not reach here due to argparse choices
                raise ValueError(
                    f"Unknown mia_threshold_mode: {args.mia_threshold_mode}"
                )

            # Classify records
            for row in results:
                if "membership_score" in row:
                    row["mia_member"] = row["membership_score"] < member_threshold

    # Write threshold.md
    if mia_threshold_derivation:
        _write_threshold_md(
            audit_dir,
            mode=threshold_mode,
            percentile=args.mia_percentile,
            threshold=member_threshold,
            derivation=mia_threshold_derivation,
            n_flagged=sum(1 for r in results if r.get("mia_member", False)),
            n_total=len(results),
        )

    # Write results
    write_raw_results(audit_dir, results)
    write_summary(audit_dir, results, provenances)
    write_exportable_summary(audit_dir, results, provenances)

    # Write run log
    config = {
        "date": audit_date,
        "model_path": str(args.model),
        "model_format": model_format,
        "dataset_root": str(dataset_root),
        "source_filter": source_filter or ["all"],
        "probe_carlini": probe_carlini,
        "probe_mia": probe_mia,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens or "adaptive",
        "temperature": args.temperature,
        "model_git_sha": _get_model_git_sha(args.model),
        "dataset_manifest_hash": compute_manifest_hash(dataset_root),
    }
    write_run_log(audit_dir, config)

    logger.info("Audit complete. Results in %s", audit_dir)
    return 0


def _percentile(scores: list[float], pct: int) -> float:
    """Compute the pct-th percentile of scores using linear interpolation.

    Pure-Python equivalent of numpy.percentile(scores, pct) with
    linear interpolation, avoiding a numpy dependency.
    """
    if not scores:
        raise ValueError("Cannot compute percentile of empty list")
    sorted_scores = sorted(scores)
    # Linear interpolation (matches numpy default method='linear')
    k = (len(sorted_scores) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_scores):
        return sorted_scores[-1]
    d = k - f
    return sorted_scores[f] + d * (sorted_scores[c] - sorted_scores[f])


def _write_threshold_md(
    audit_dir: Path,
    mode: str,
    percentile: int,
    threshold: float,
    derivation: str,
    n_flagged: int,
    n_total: int,
) -> Path:
    """Write threshold.md documenting the MIA threshold derivation."""
    lines = [
        "# MIA Threshold Documentation",
        "",
        f"- **Mode**: {mode}",
    ]
    if mode == "percentile":
        lines.append(f"- **Percentile**: {percentile}")
    elif mode == "median":
        lines.append(
            "- **WARNING**: threshold is median; this is a calibration artifact"
        )
        lines.append("  Interpretation as a memorization signal is not supported.")
    elif mode == "holdout_file":
        lines.append("- **Holdout file**: see derivation below")
    lines.extend(
        [
            f"- **Threshold value**: {threshold:.4f}",
            f"- **Records flagged as mia_member=True**: {n_flagged}/{n_total}",
            f"- **Derivation**: {derivation}",
            "",
            "Per docs/MIA_THRESHOLD_CALIBRATION.md Track 1.",
        ]
    )
    output_path = audit_dir / "threshold.md"
    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote threshold documentation to %s", output_path)
    return output_path


def _get_model_git_sha(model_path: Path) -> str:
    """Try to get the git SHA from the model directory."""
    git_dir = model_path / ".git"
    if git_dir.exists():
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(model_path),
            )
            if result.returncode == 0:
                return result.stdout.strip()[:16]
        except Exception:
            pass
    return "unknown"


if __name__ == "__main__":
    sys.exit(main())
