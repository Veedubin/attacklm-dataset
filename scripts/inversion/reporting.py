"""Audit report generation for the inversion harness.

Writes structured output to data/audit/<date>/ with three tiers:
    - INTERNAL_RAW (inversion_results.jsonl, chmod 0600)
    - INTERNAL_SUMMARY (aggregate statistics)
    - EXPORTABLE_SUMMARY (per-source aggregates, no raw text)

Pre-flight checks:
    1. Refuse to run if output dir mode is wider than 0700.
    2. Refuse if date directory already exists (prevents accidental overwrite).
    3. All raw reconstruction text stays in inversion_results.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path

from .provenance import (
    RecordProvenance,
    OutputClass,
    classify_output,
    strip_raw_text_for_export,
)

logger = logging.getLogger(__name__)


def check_output_dir_permissions(path: Path) -> None:
    """Pre-flight check: refuse if output dir mode is wider than 0700.

    This prevents accidental exposure of raw reconstruction data.
    """
    if path.exists():
        mode = stat.S_IMODE(os.stat(path).st_mode)
        if mode > 0o0700:
            raise PermissionError(
                f"Output directory {path} has mode {oct(mode)}, which is wider "
                f"than 0700. Raw reconstruction data must not be world-readable. "
                f"Run: chmod 0700 {path}"
            )


def create_audit_dir(
    audit_output_root: Path,
    audit_date: str,
) -> Path:
    """Create the audit output directory with appropriate permissions.

    Structure:
        <audit_output_root>/<date>/
            inversion_results.jsonl   (INTERNAL_RAW, chmod 0600)
            summary.json              (INTERNAL_SUMMARY)
            exportable_summary.json    (EXPORTABLE_SUMMARY)
            run.log                    (audit trail)
    """
    audit_dir = audit_output_root / audit_date
    if audit_dir.exists():
        raise FileExistsError(
            f"Audit directory {audit_dir} already exists. "
            f"Use a different --date or remove the existing directory."
        )
    audit_dir.mkdir(parents=True, exist_ok=False)
    # Set directory to 0700 (owner-only)
    audit_dir.chmod(0o700)
    return audit_dir


def write_raw_results(
    audit_dir: Path,
    results: list[dict],
) -> Path:
    """Write INTERNAL_RAW results to inversion_results.jsonl.

    Each line is a JSON object with:
        - All probe scores
        - Provenance (source, license, license_family)
        - prompt_hash and reconstruction_hash (never raw text)
        - For INTERNAL_RAW: also includes prompt_text and best_reconstruction
          (the actual generated text — owner-only)

    File permissions are set to 0600 (owner read/write only).
    """
    output_path = audit_dir / "inversion_results.jsonl"
    with open(output_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    # Set file permissions to owner-only
    output_path.chmod(0o600)
    logger.info("Wrote %d raw results to %s", len(results), output_path)
    return output_path


def write_summary(
    audit_dir: Path,
    results: list[dict],
    provenances: list[RecordProvenance],
) -> Path:
    """Write INTERNAL_SUMMARY with aggregate statistics.

    Includes per-source counts, mean scores, and license-stamped counts.
    No raw text, but record-level scores are present.
    """
    import collections

    by_source = collections.defaultdict(list)
    for row, prov in zip(results, provenances):
        by_source[prov.source].append(row)

    summary = {
        "total_records": len(results),
        "sources": {},
    }
    for source, rows in sorted(by_source.items()):
        scores = {
            "exact_matches": sum(1 for r in rows if r.get("best_exact_match")),
            "mean_bleu4": _safe_mean([r.get("best_bleu4", 0) for r in rows]),
            "mean_lcs_length": _safe_mean([r.get("best_lcs_length", 0) for r in rows]),
            "mean_membership_score": _safe_mean(
                [r.get("membership_score", 0) for r in rows]
            ),
            "mean_nll": _safe_mean([r.get("nll", 0) for r in rows]),
            "mean_perplexity": _safe_mean([r.get("perplexity", 0) for r in rows]),
            "total_records": len(rows),
            "licenses": {},
        }
        # License-stamped counts
        license_counts = collections.Counter(r.get("license", "unknown") for r in rows)
        scores["licenses"] = dict(license_counts)
        summary["sources"][source] = scores

    output_path = audit_dir / "summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return output_path


def write_exportable_summary(
    audit_dir: Path,
    results: list[dict],
    provenances: list[RecordProvenance],
) -> Path:
    """Write EXPORTABLE_SUMMARY — per-source aggregates with NO raw text
    and NO re-identifiable record IDs.

    Each row is stripped via strip_raw_text_for_export() before writing.
    """
    exportable_rows = []
    for row, prov in zip(results, provenances):
        # Classify: only exportable rows go in this file
        output_class = classify_output(prov, include_raw_text=False)
        if output_class == OutputClass.EXPORTABLE_SUMMARY:
            exportable_rows.append(strip_raw_text_for_export(row))
        elif output_class == OutputClass.INTERNAL_SUMMARY:
            exportable_rows.append(strip_raw_text_for_export(row))
        # INTERNAL_RAW-only records (DRL-1.1 with raw text) are excluded

    output_path = audit_dir / "exportable_summary.json"
    with open(output_path, "w") as f:
        json.dump(exportable_rows, f, indent=2)
    logger.info("Wrote %d exportable rows to %s", len(exportable_rows), output_path)
    return output_path


def write_run_log(
    audit_dir: Path,
    config: dict,
) -> Path:
    """Write the run.log with model SHA, dataset manifest hash, and config."""
    output_path = audit_dir / "run.log"
    lines = [
        f"Inversion Audit Run — {config.get('date', 'unknown')}",
        f"Model: {config.get('model_path', 'unknown')}",
        f"Model format: {config.get('model_format', 'auto')}",
        f"Dataset root: {config.get('dataset_root', 'unknown')}",
        f"Sources: {config.get('source_filter', 'all')}",
        f"Probes: carlini={config.get('probe_carlini', True)}, mia={config.get('probe_mia', True)}",
        f"Top-K: {config.get('top_k', 20)}",
        f"Max new tokens: {config.get('max_new_tokens', 64)}",
        f"Temperature: {config.get('temperature', 1.0)}",
        f"Model git SHA: {config.get('model_git_sha', 'unknown')}",
        f"Dataset manifest hash: {config.get('dataset_manifest_hash', 'unknown')}",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


def _safe_mean(values: list[float]) -> float:
    """Compute mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)
