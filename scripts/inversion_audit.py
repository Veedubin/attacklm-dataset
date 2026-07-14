#!/usr/bin/env python3
from __future__ import annotations

# PROVENANCE METADATA — scripts/inversion_audit.py
# ================================================================================
# Attack class:        CLI driver for Carlini 2021 prefix-completion +
#                      Carlini 2022 MIA (loss + zlib) + LiRA (2022 §4) +
#                      per-token MIA (MUSE 2023)
# Original authors:    CLI: Veedubin (in-repo author)
#                      Attacks: see scripts/inversion/probe.py, scoring.py,
#                               lira.py for per-attack paper provenance
# Paper title:         CLI driver — see per-attack files for the papers
# Year / venue:        CLI 2026 / in-repo
# Paper URL:           CLI N/A; see per-attack files
# Canonical repo:      N/A
#
# Implementation:
#   Type:              ORIGINAL_WORK (orchestration only; the attack code
#                      lives in the modules imported below)
#   Lines of port:     N/A
#   Upstream license:  N/A
#
# Data sources: N/A (this file is a CLI driver; it does not ingest data
# itself — it operates on whatever the user passes via --dataset-root)
#
# Rights claim contact: veedubin.legal@example.com
# See:                  RIGHTS.md (root), data/LEGAL.md, data/REMOVAL.md
# ================================================================================

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


import argparse
import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable

# Make the inversion package importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inversion import __version__
from inversion.model_loader import load_model, detect_model_format
from inversion.probe import run_carlini_probe
from inversion.lira import (
    GaussianParams,
    load_shadow_params,
    score_lira,
)
from inversion.scoring import (
    compute_offline_z,
    score_per_token,
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
from inversion.variant_generator import (
    generate_suffix_injection,
    generate_prompt_template,
    generate_paraphrase,
)
from inversion.attack_success_curve import compute_success_curve, write_curve

logger = logging.getLogger(__name__)


class _ExpandVariantStrategies(argparse.Action):
    """Custom argparse action that expands 'all' to all strategies."""

    def __call__(self, parser, namespace, values, option_string=None):
        if "all" in values:
            values = ["suffix", "template", "paraphrase"]
        setattr(namespace, self.dest, values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inversion-attack audit harness for AttackLM. "
        "Probes your OWN models for memorized training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Methodology: implements MAI-Thinking-1 §5.2 closed-loop "
        "adversarial audit pattern (Independent Red Teaming — TAP). "
        "See docs/AUDIT_ITER.md and 'MAI-Thinking-1: Building a "
        "Hill-Climbing Machine' (Microsoft AI Team, June 2026).",
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
        help="DEPRECATED v0.4.0; use --attack. Will be removed in v0.6.0. "
        "Enable Carlini prefix-completion extraction probe (default: on).",
    )
    parser.add_argument(
        "--no-probe-carlini",
        action="store_true",
        help="DEPRECATED v0.4.0; use --attack. Will be removed in v0.6.0. "
        "Disable Carlini prefix-completion probe.",
    )
    parser.add_argument(
        "--probe-mia",
        action="store_true",
        default=True,
        help="DEPRECATED v0.4.0; use --attack. Will be removed in v0.6.0. "
        "Enable MIA loss+zlib scoring (default: on).",
    )
    parser.add_argument(
        "--no-probe-mia",
        action="store_true",
        help="DEPRECATED v0.4.0; use --attack. Will be removed in v0.6.0. "
        "Disable MIA scoring.",
    )
    parser.add_argument(
        "--attack",
        choices=["extraction", "mia", "all"],
        default="all",
        help="Which attack class(es) to run. 'extraction' = Carlini 2021 "
        "prefix-completion probing (LLM model inversion / TDE). "
        "'mia' = membership inference attack. 'all' (default) = both. "
        "Use --mia-method to pick a specific MIA technique.",
    )
    parser.add_argument(
        "--mia-method",
        choices=["reference", "zlib", "per_token", "lira", "offline", "all"],
        default="reference",
        help="MIA scoring method. 'reference' = NLL only (Carlini 2022 §3.2, "
        "Loss attack). 'zlib' = NLL - zlib_length (Carlini 2022 §3.2, "
        "zlib-entropy calibration). 'per_token' = NLL normalized by "
        "suffix token count (MUSE 2023 default). 'lira' = "
        "likelihood-ratio test from K shadow models (v0.5.0+). "
        "'offline' = white-box baseline using sample mean/std of audit-set "
        "NLL as the OUT distribution (no shadow models needed; requires "
        "N >= 30 records). 'all' = run all available methods and report each.",
    )
    parser.add_argument(
        "--offline-z-threshold",
        type=float,
        default=-1.5,
        help="Z-score threshold for offline MIA. Records with z < threshold "
        "are flagged as potential members. Default: -1.5.",
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
        choices=["median", "percentile", "holdout_file", "lrt"],
        default="percentile",
        help="MIA threshold calibration mode. 'median': median of probed scores "
        "(calibration artifact — WARNING logged). 'percentile': Nth percentile "
        "of probed scores (default). 'holdout_file': read threshold from a JSON "
        "file. 'lrt': natural 0.0 threshold for LiRA (positive logit = member).",
    )
    parser.add_argument(
        "--mia-percentile",
        type=int,
        default=5,
        help="Percentile for MIA threshold when --mia-threshold-mode=percentile "
        "(default: 5, meaning only the bottom 5%% of scores are flagged).",
    )
    parser.add_argument(
        "--lira-k",
        type=int,
        default=16,
        help="Number of shadow models for LiRA (default: 16). "
        "K=1 = reference-model MIA, K=4 = cheap LiRA, K=16 = gold standard. "
        "Only used when --mia-method lira is specified.",
    )
    parser.add_argument(
        "--lira-params",
        type=str,
        default=None,
        help="Path to shadow_params.json (output of inversion.shadow_train). "
        "Required when --mia-method lira is used.",
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
    parser.add_argument(
        "--audit-iter",
        type=int,
        default=1,
        help="Number of closed-loop iterations (default: 1 = single-pass, "
        "identical to v0.4.3). Set >1 to enable iterative variant-based "
        "auditing per MAI-Thinking-1 §5.2.",
    )
    parser.add_argument(
        "--variant-strategies",
        nargs="+",
        default=["suffix", "template"],
        choices=["paraphrase", "suffix", "template", "all"],
        action=_ExpandVariantStrategies,
        help="Variant generation strategies for closed-loop audit. "
        "Default: suffix template. 'paraphrase' requires a model and "
        "is opt-in. 'all' enables suffix, template, and paraphrase.",
    )
    parser.add_argument(
        "--variant-count-per-iter",
        type=int,
        default=5,
        help="Number of top-K fooling records to vary per iteration (default: 5).",
    )
    parser.add_argument(
        "--iter-output-dir",
        type=Path,
        default=None,
        help="Directory for per-iteration JSONL dumps (default: derived "
        "from audit date).",
    )
    parser.add_argument(
        "--iter-curve-output",
        type=Path,
        default=None,
        help="Path for the final attack-success curve JSON "
        "(default: <audit_date>/attack_success_curve.json).",
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

    # Detect legacy flag usage and map to --attack
    legacy_used = (
        getattr(args, "probe_carlini", None) is not None
        or getattr(args, "no_probe_carlini", False)
        or getattr(args, "probe_mia", None) is not None
        or getattr(args, "no_probe_mia", False)
    )
    if legacy_used:
        import warnings

        warnings.warn(
            "--probe-carlini / --probe-mia / --no-probe-* are deprecated since "
            "v0.4.0; use --attack {extraction,mia,all} and --mia-method. "
            "Will be removed in v0.6.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Map legacy flags → new --attack value (only if user explicitly
        # passed --no-probe-* flags; default True means they didn't)
        if args.no_probe_carlini and args.no_probe_mia:
            args.attack = "none"  # sentinel: skip everything
        elif args.no_probe_carlini:
            args.attack = "mia"
        elif args.no_probe_mia:
            args.attack = "extraction"
        # else: keep --attack as user set it (or default 'all')

    # Validate LiRA requirements
    lira_shadow_params: dict[str, GaussianParams] | None = None
    if args.mia_method in ("lira", "all"):
        if not args.lira_params:
            print(
                "ERROR: --mia-method lira requires --lira-params "
                "(path to shadow_params.json).",
                file=sys.stderr,
            )
            print(
                "Produce shadow_params.json with: "
                "python -m inversion.shadow_train --loss-dir ... --output shadow_params.json",
                file=sys.stderr,
            )
            sys.exit(1)
        lira_shadow_params_dict, lira_k_loaded = load_shadow_params(args.lira_params)
        lira_shadow_params = lira_shadow_params_dict
        logger.info(
            "Loaded LiRA shadow params for %d records (K=%d from file)",
            len(lira_shadow_params),
            lira_k_loaded,
        )

    # Derive run flags from --attack
    run_extraction = args.attack in ("extraction", "all")
    run_mia = args.attack in ("mia", "all")
    # Sentinel: "none" means user explicitly disabled both — skip
    if args.attack == "none":
        print(
            "No probes requested (legacy --no-probe-carlini --no-probe-mia). Exiting.",
        )
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve flags (derived from --attack above)
    # probe_carlini/probe_mia are now set via --attack; legacy flags were
    # mapped in the deprecation block above.

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
        print(f"  Carlini probe: {run_extraction}")
        print(f"  MIA probe: {run_mia}")
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
    try:
        model_format = args.model_format or detect_model_format(args.model)
    except ValueError:
        # Auto-detection failed; default to hf for mock/CI environments.
        # Users hitting this in production should specify --model-format.
        logger.warning(
            "Cannot auto-detect model format at %s; defaulting to 'hf'. "
            "Specify --model-format explicitly to suppress this warning.",
            args.model,
        )
        model_format = "hf"

    if model_format == "gguf" and run_mia:
        logger.error(
            "MIA scoring requires white-box access to model loss. "
            "GGUF models do not expose loss. Use --attack extraction or "
            "provide a HuggingFace model with --model-format hf."
        )
        return 1

    model, tokenizer = load_model(args.model, args.model_format)

    # Create audit directory
    audit_dir = create_audit_dir(audit_output_root, audit_date)
    logger.info("Created audit directory: %s", audit_dir)

    # Closed-loop audit path (--audit-iter > 1)
    if args.audit_iter > 1:
        iter_output_dir = args.iter_output_dir or audit_dir / "iter"
        iter_curve_output = (
            args.iter_curve_output or audit_dir / "attack_success_curve.json"
        )

        # Build an attack function that wraps the existing MIA scoring.
        # For closed-loop audit, we compute membership scores then classify
        # using the percentile threshold (same as single-pass, but per iteration).
        def _attack_fn(
            records: list[dict], model_handle: Any, **kwargs: Any
        ) -> list[dict]:
            results = []
            scores = []
            for i, rec in enumerate(records):
                rec_id = rec.get("id", str(i))
                row: dict = {
                    "record_index": i,
                    "id": rec_id,
                }
                if run_mia and model_format == "hf":
                    try:
                        mia_score = score_record(rec, model_handle, tokenizer)
                        row["nll"] = mia_score.nll
                        row["membership_score"] = mia_score.membership_score
                        scores.append(mia_score.membership_score)
                    except Exception:
                        row["nll"] = float("inf")
                        row["membership_score"] = float("inf")
                        scores.append(float("inf"))
                else:
                    row["nll"] = float("inf")
                    row["membership_score"] = float("inf")
                    scores.append(float("inf"))
                results.append(row)

            # Classify using percentile threshold (same logic as single-pass)
            if scores and run_mia:
                valid_scores = [s for s in scores if s != float("inf")]
                if valid_scores:
                    threshold = _percentile(valid_scores, args.mia_percentile)
                    for row, score in zip(results, scores):
                        row["fooling"] = score < threshold
                else:
                    for row in results:
                        row["fooling"] = False
            else:
                for row in results:
                    row["fooling"] = False

            return results

        curve = run_closed_loop_audit(
            records=records,
            model_handle=model,
            attack_fn=_attack_fn,
            n_iter=args.audit_iter,
            variant_strategies=args.variant_strategies,
            k_per_iter=args.variant_count_per_iter,
            iter_output_dir=iter_output_dir,
        )

        # Write the attack-success curve
        write_curve(curve, iter_curve_output)
        logger.info("Attack-success curve written to %s", iter_curve_output)

        # Write run log for closed-loop audit
        config = {
            "date": audit_date,
            "model_path": str(args.model),
            "model_format": model_format,
            "dataset_root": str(dataset_root),
            "source_filter": source_filter or ["all"],
            "audit_iter": args.audit_iter,
            "variant_strategies": args.variant_strategies,
            "variant_count_per_iter": args.variant_count_per_iter,
            "attack": args.attack,
            "mia_method": args.mia_method,
        }
        write_run_log(audit_dir, config)

        logger.info("Closed-loop audit complete. Results in %s", audit_dir)
        return 0

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

        if run_extraction and model_format == "hf":
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
                        "prompt_text": probe_result.prompt_text,
                        "best_reconstruction": probe_result.best_reconstruction,
                    }
                )
            except Exception as e:
                logger.warning("Carlini probe failed for record %d: %s", i, e)

        if run_mia and model_format == "hf":
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

            # Per-token MIA scoring (MUSE 2023 default)
            if args.mia_method in ("per_token", "all"):
                try:
                    pt_score = score_per_token(record, model, tokenizer)
                    row.update(
                        {
                            "nll_per_token": pt_score.nll_per_token,
                            "nll_total_per_token": pt_score.nll_total,
                            "num_suffix_tokens": pt_score.num_suffix_tokens,
                            "suffix_text": pt_score.suffix_text,
                            "membership_score_per_token": pt_score.membership_score,
                            "zlib_ratio_per_token": pt_score.zlib_ratio,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        "Per-token MIA scoring failed for record %d: %s", i, e
                    )

            # Offline (K=0) MIA — white-box baseline, no shadow models needed
            # This is handled as a batch operation AFTER the per-record loop,
            # not per-record. See the offline MIA block below.

            # LiRA scoring (Carlini 2022 §4)
            if args.mia_method in ("lira", "all") and lira_shadow_params is not None:
                record_id = record.get("id", str(i))
                if record_id in lira_shadow_params:
                    try:
                        lira_result = score_lira(
                            record, model, tokenizer, lira_shadow_params[record_id]
                        )
                        row.update(
                            {
                                "lira_logit": lira_result.lira_logit,
                                "lira_mu_in": lira_result.mu_in,
                                "lira_sigma_in": lira_result.sigma_in,
                                "lira_mu_out": lira_result.mu_out,
                                "lira_sigma_out": lira_result.sigma_out,
                                "lira_loss_target": lira_result.loss_target,
                                "lira_nll_per_token": lira_result.nll_per_token,
                                "lira_num_suffix_tokens": lira_result.num_suffix_tokens,
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            "LiRA scoring failed for record %d (id=%s): %s",
                            i,
                            record_id,
                            e,
                        )
                else:
                    logger.warning(
                        "No shadow params for record %d (id=%s); skipping LiRA",
                        i,
                        record_id,
                    )

        results.append(row)

    # Offline (K=0) MIA — white-box baseline, no shadow models needed
    # Uses sample mean/std of audit-set NLL as the OUT distribution.
    # Equivalent to Carlini 2022 reference attack with sample-std normalization.
    if run_mia and args.mia_method in ("offline", "all") and model_format == "hf":
        if len(results) < 30:
            logger.error("offline MIA requires N >= 30 records; got N=%d", len(results))
            return 1
        nlls = [r["nll"] for r in results if "nll" in r]
        if not nlls:
            logger.warning("No NLLs available for offline MIA; skipping")
        else:
            mu_out, sigma_out, z_scores = compute_offline_z(nlls)
            for r, z in zip(results, z_scores):
                r["offline_z"] = z
                r["offline_mu_out"] = mu_out
                r["offline_sigma_out"] = sigma_out
                r["offline_flagged"] = z < args.offline_z_threshold
            logger.info(
                "offline MIA: mu_out=%.4f sigma_out=%.4f threshold=%.2f flagged=%d/%d",
                mu_out,
                sigma_out,
                args.offline_z_threshold,
                sum(1 for r in results if r.get("offline_flagged", False)),
                len(results),
            )

    # MIA threshold calibration
    mia_threshold_derivation = ""
    if run_mia and model_format == "hf":
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
            elif args.mia_threshold_mode == "lrt":
                # LiRA natural threshold: 0.0 (positive logit = member)
                threshold_mode = "lrt"
                member_threshold = 0.0
                mia_threshold_derivation = (
                    "lrt: natural 0.0 threshold for LiRA (positive logit = likely IN)"
                )
                logger.info(
                    "MIA threshold=0.0 (LiRA natural threshold; "
                    "positive lira_logit = likely member)"
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
                # LiRA classification (positive logit = member, threshold = 0.0)
                if "lira_logit" in row:
                    row["lira_member"] = (
                        row["lira_logit"] >= member_threshold
                        if args.mia_threshold_mode == "lrt"
                        else row["lira_logit"] >= 0.0
                    )

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
        "probe_carlini": run_extraction,
        "probe_mia": run_mia,
        "attack": args.attack,
        "mia_method": args.mia_method,
        "lira_k": args.lira_k,
        "lira_params": args.lira_params or "none",
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
    elif mode == "lrt":
        lines.append(
            "- **LRT**: LiRA natural threshold (0.0); positive logit = likely member"
        )
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


# Implements MAI-Thinking-1 §5.2 closed-loop audit. See docs/AUDIT_ITER.md.


def run_closed_loop_audit(
    records: list[dict],
    model_handle: Any,
    attack_fn: Callable,
    n_iter: int,
    variant_strategies: list[str],
    k_per_iter: int,
    iter_output_dir: Path,
) -> dict:
    """Run a closed-loop adversarial audit for n_iter iterations.

    For each iteration:
    1. Run the attack on the current record set
    2. Identify fooling records (where the attack succeeded)
    3. Generate variants of the top-K fooling records
    4. Use variants as the next iteration's input

    Returns the attack-success curve dict {iter_idx: {attack: {probed, fooling, success_rate}}}.
    """

    iter_output_dir.mkdir(parents=True, exist_ok=True)

    current_records = records
    all_per_iter_results: list[dict] = []
    all_fooling_records_by_iter: dict[str, list[str]] = {}

    for iter_idx in range(n_iter):
        logger.info("Closed-loop audit iteration %d/%d", iter_idx + 1, n_iter)

        # Run the attack on current records
        attack_results = attack_fn(current_records, model_handle)

        # Count probed and fooling
        probed_count = len(attack_results)
        fooling_count = sum(1 for r in attack_results if r.get("fooling", False))
        # Cap fooling_count at k_per_iter for the curve (top-K selection)
        capped_fooling = min(fooling_count, k_per_iter)
        fooling_ids = [
            r.get("id", str(r.get("record_index", i)))
            for i, r in enumerate(attack_results)
            if r.get("fooling", False)
        ]

        iter_result = {
            "iteration": iter_idx,
            "probed_count": probed_count,
            "fooling_count": capped_fooling,
            "fooling_record_ids": fooling_ids[:k_per_iter],
            "attack": "mia",
        }
        all_per_iter_results.append(iter_result)
        all_fooling_records_by_iter[str(iter_idx)] = fooling_ids[:k_per_iter]

        # If not the last iteration, generate variants for next round
        if iter_idx < n_iter - 1 and fooling_count > 0:
            # Select top-K fooling records
            fooling_records = [
                current_records[i]
                for i, r in enumerate(attack_results)
                if r.get("fooling", False)
            ]
            top_k_records = fooling_records[:k_per_iter]

            # Generate variants
            variant_records: list[dict] = []
            for rec in top_k_records:
                if "suffix" in variant_strategies:
                    variant_records.extend(generate_suffix_injection(rec))
                if "template" in variant_strategies:
                    variant_records.extend(generate_prompt_template(rec))
                if "paraphrase" in variant_strategies:
                    variant_records.extend(generate_paraphrase(rec, model_handle))

            if variant_records:
                current_records = variant_records
            else:
                # No variants generated; reuse current records
                logger.info(
                    "No variants generated at iteration %d; re-probing current records.",
                    iter_idx,
                )
        elif iter_idx < n_iter - 1 and fooling_count == 0:
            # No fooling records; re-probe current records (no variants to generate)
            logger.info(
                "No fooling records at iteration %d; re-probing current records.",
                iter_idx,
            )

    curve = compute_success_curve(all_per_iter_results)

    # Write per-iteration results to disk
    iter_jsonl = iter_output_dir / "iter_results.jsonl"
    with open(iter_jsonl, "w") as f:
        for result in all_per_iter_results:
            f.write(json.dumps(result) + "\n")

    return curve


if __name__ == "__main__":
    sys.exit(main())
