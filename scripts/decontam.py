#!/usr/bin/env python3
"""20-gram fuzzy decontamination against public evaluation sets.

Implements MAI-Thinking-1 §2.3.1 (Public Evaluation Decontamination)
and §2.4.3 (Deduplication, including cross-dataset drop-order)
by The Microsoft AI Team, June 2026. See docs/DECONTAM.md.

Compares training records against evaluation-set benchmarks using
character-level 20-gram MinHash LSH at 0.80 similarity threshold.
Reports overlapping record pairs per source and optionally writes a
quarantine JSONL file for manual review.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# datasketch is the only new dep. Add to pyproject.toml [inversion] optional group.
from datasketch import MinHash, MinHashLSH

# Implements MAI-Thinking-1 §2.3.1 MinHash LSH fuzzy dedup, n-gram=20, threshold=0.80.
# See docs/DECONTAM.md.

logger = logging.getLogger(__name__)

# Restricted sources that should never be processed (same denylist as
# inversion_audit.py).
RESTRICTED_SOURCES: frozenset[str] = frozenset(
    {
        "rta",  # Red Team Automation — GPL-3.0
        "infection_monkey",  # Infection Monkey — AGPL-3.0
        "bpl",  # BloodHound Payload Library — scrubbed from git
    }
)


# ---------------------------------------------------------------------------
# N-gram extraction
# ---------------------------------------------------------------------------


def extract_char_ngrams(text: str, n: int = 20) -> set[str]:
    """Extract character-level n-grams from text.

    Implements MAI-Thinking-1 §2.3.1 character-level n-gram extraction.
    See docs/DECONTAM.md.
    """
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


# ---------------------------------------------------------------------------
# MinHash construction
# ---------------------------------------------------------------------------

_DEFAULT_PERMUTATIONS = 128


def _compute_minhash_item(
    item: tuple[str, str, dict[str, Any]],
    ngram_size: int,
    num_perm: int,
) -> tuple[str, MinHash, set[str], dict[str, Any]]:
    """Compute MinHash for a single training record.

    Must be module-level to be picklable for ProcessPoolExecutor.
    Implements MAI-Thinking-1 §2.3.1 MinHash signature computation.
    See docs/DECONTAM.md.
    """
    rid, text, rec = item
    ngrams = extract_char_ngrams(text, ngram_size)
    mh = MinHash(num_perm=num_perm)
    for ng in ngrams:
        mh.update(ng.encode("utf-8"))
    return (rid, mh, ngrams, rec)


def compute_minhash(
    text: str,
    ngram_size: int = 20,
    num_perm: int = _DEFAULT_PERMUTATIONS,
) -> MinHash:
    """Compute MinHash signature for a text string.

    Implements MAI-Thinking-1 §2.3.1 MinHash signature computation.
    See docs/DECONTAM.md.
    """
    ngrams = extract_char_ngrams(text, ngram_size)
    mh = MinHash(num_perm=num_perm)
    for ng in ngrams:
        mh.update(ng.encode("utf-8"))
    return mh


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute exact Jaccard similarity between two sets.

    Implements MAI-Thinking-1 §2.4.3 exact Jaccard verification step.
    See docs/DECONTAM.md.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _record_text(record: dict[str, Any]) -> str:
    """Extract concatenated text content from a training record.

    Training records use the messages format (list of {role, content}
    dicts). We concatenate all message contents to form the comparison
    text, matching how the model sees the record.
    """
    messages = record.get("messages")
    if messages and isinstance(messages, list):
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                parts.append(content)
        return "\n".join(parts)
    # Fallback: top-level text field (for eval-set fixtures).
    return record.get("text", "")


def load_training_records(
    dataset_root: Path,
    source_filter: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load all training records grouped by source.

    Scans dataset_root/<source>/<bucket>/<tactic>/data*.jsonl for each
    non-restricted source. Returns {source_name: [record, ...]}.
    """
    available_sources: list[str] = []
    for entry in sorted(dataset_root.iterdir()):
        if entry.is_dir() and entry.name not in RESTRICTED_SOURCES:
            available_sources.append(entry.name)

    if source_filter is not None:
        sources = [s for s in source_filter if s not in RESTRICTED_SOURCES]
    else:
        sources = available_sources

    records_by_source: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        source_dir = dataset_root / source
        if not source_dir.is_dir():
            logger.warning("Source directory not found: %s", source_dir)
            continue
        source_records: list[dict[str, Any]] = []
        for jsonl_path in sorted(source_dir.rglob("data*.jsonl")):
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        source_records.append(rec)
                    except json.JSONDecodeError:
                        logger.warning("Bad JSON in %s: %.80s", jsonl_path, line)
        if source_records:
            records_by_source[source] = source_records
            logger.info(
                "Loaded %d records from source '%s'", len(source_records), source
            )

    return records_by_source


def load_eval_records(eval_set_dir: Path) -> list[dict[str, Any]]:
    """Load all eval-set JSONL files from a directory.

    Each file may contain one or more JSONL records. Returns a flat
    list of records with an '_eval_file' key added for traceability.
    """
    records: list[dict[str, Any]] = []
    if eval_set_dir.is_file():
        # Single file path.
        jsonl_files = [eval_set_dir]
    else:
        jsonl_files = sorted(eval_set_dir.glob("*.jsonl"))

    for jsonl_path in jsonl_files:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Skip comment lines (license headers start with #).
                if line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                    rec["_eval_file"] = str(jsonl_path)
                    records.append(rec)
                except json.JSONDecodeError:
                    logger.warning("Bad JSON in %s: %.80s", jsonl_path, line)
    return records


# ---------------------------------------------------------------------------
# Decontamination algorithm
# ---------------------------------------------------------------------------


def build_training_index(
    records_by_source: dict[str, list[dict[str, Any]]],
    ngram_size: int = 20,
    num_perm: int = _DEFAULT_PERMUTATIONS,
    threshold: float = 0.80,
    num_proc: int = 4,
) -> tuple[MinHashLSH, dict[str, tuple[MinHash, set[str], dict[str, Any]]]]:
    """Build MinHash LSH index over the training corpus.

    Implements MAI-Thinking-1 §2.3.1 index construction.
    See docs/DECONTAM.md.

    Returns:
        (lsh, index_map) where index_map maps record_id -> (minhash, ngrams, record).
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Collect all training records with a unique ID.
    all_items: list[tuple[str, str, dict[str, Any]]] = []
    for source, records in records_by_source.items():
        for idx, rec in enumerate(records):
            record_id = f"{source}::{idx}"
            text = _record_text(rec)
            all_items.append((record_id, text, rec))

    # Compute MinHash signatures.
    index_map: dict[str, tuple[MinHash, set[str], dict[str, Any]]] = {}

    if num_proc > 1 and len(all_items) > 100:
        with ProcessPoolExecutor(max_workers=num_proc) as executor:
            results = list(
                executor.map(
                    _compute_minhash_item,
                    all_items,
                    [ngram_size] * len(all_items),
                    [num_perm] * len(all_items),
                )
            )
    else:
        results = [
            _compute_minhash_item(item, ngram_size, num_perm) for item in all_items
        ]

    for rid, mh, ngrams, rec in results:
        lsh.insert(rid, mh)
        index_map[rid] = (mh, ngrams, rec)

    logger.info("Built LSH index with %d training records", len(index_map))
    return lsh, index_map


def find_overlaps(
    lsh: MinHashLSH,
    index_map: dict[str, tuple[MinHash, set[str], dict[str, Any]]],
    eval_records: list[dict[str, Any]],
    ngram_size: int = 20,
    num_perm: int = _DEFAULT_PERMUTATIONS,
    threshold: float = 0.80,
) -> list[dict[str, Any]]:
    """Query the LSH index for each eval record and verify with exact Jaccard.

    Implements MAI-Thinking-1 §2.3.1 candidate retrieval + §2.4.3 exact
    verification. See docs/DECONTAM.md.

    Returns a list of overlap dicts:
        {
            "training_id": str,
            "training_source": str,
            "eval_id": str,
            "eval_source": str,
            "similarity": float,
        }
    """
    overlaps: list[dict[str, Any]] = []
    for eval_idx, eval_rec in enumerate(eval_records):
        eval_text = _record_text(eval_rec)
        eval_id = eval_rec.get("id", f"eval_{eval_idx}")
        eval_source = eval_rec.get("source", "unknown")
        eval_ngrams = extract_char_ngrams(eval_text, ngram_size)

        if not eval_ngrams:
            continue

        eval_mh = MinHash(num_perm=num_perm)
        for ng in eval_ngrams:
            eval_mh.update(ng.encode("utf-8"))

        # Step 4: query LSH for candidates.
        candidates = lsh.query(eval_mh)

        # Step 5: exact Jaccard verification.
        for cand_id in candidates:
            if cand_id not in index_map:
                continue
            _mh, cand_ngrams, cand_rec = index_map[cand_id]
            sim = jaccard_similarity(eval_ngrams, cand_ngrams)
            if sim >= threshold:
                # Extract source from the composite ID.
                training_source = cand_id.split("::", 1)[0]
                overlaps.append(
                    {
                        "training_id": cand_id,
                        "training_source": training_source,
                        "eval_id": str(eval_id),
                        "eval_source": eval_source,
                        "similarity": round(sim, 6),
                    }
                )

    return overlaps


def build_report(
    records_by_source: dict[str, list[dict[str, Any]]],
    overlaps: list[dict[str, Any]],
    max_examples: int = 10,
) -> dict[str, Any]:
    """Build a per-source contamination report from the overlap list.

    Implements MAI-Thinking-1 §2.4.3 per-source reporting.
    See docs/DECONTAM.md.
    """
    # Group overlaps by training source.
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for overlap in overlaps:
        by_source[overlap["training_source"]].append(overlap)

    report: dict[str, Any] = {"sources": {}}
    for source, records in records_by_source.items():
        source_overlaps = by_source.get(source, [])
        # Unique eval records that overlap with this source.
        unique_eval_ids = {o["eval_id"] for o in source_overlaps}
        examples = source_overlaps[:max_examples]
        report["sources"][source] = {
            "n_training_records": len(records),
            "n_eval_overlaps": len(source_overlaps),
            "n_unique_eval_records_overlapping": len(unique_eval_ids),
            "examples": examples,
        }

    # Also include sources with no overlaps.
    total_overlaps = len(overlaps)
    report["total_overlaps"] = total_overlaps
    return report


def write_quarantine(
    overlaps: list[dict[str, Any]],
    index_map: dict[str, tuple[MinHash, set[str], dict[str, Any]]],
    output_path: Path,
) -> int:
    """Write matched training records to a quarantine JSONL file.

    Records are deduplicated by training_id. Returns the number of
    records written.
    """
    seen_ids: set[str] = set()
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for overlap in overlaps:
            tid = overlap["training_id"]
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
            if tid in index_map:
                _mh, _ng, rec = index_map[tid]
                # Remove internal keys before writing.
                clean_rec = {k: v for k, v in rec.items() if not k.startswith("_")}
                f.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")
                count += 1
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="20-gram fuzzy decontamination against public evaluation sets. "
        "Compares training data to eval benchmarks using MinHash LSH and "
        "reports overlapping records. Implements MAI-Thinking-1 §2.3.1 + §2.4.3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Methodology: implements MAI-Thinking-1 §2.3.1 (Public Evaluation "
        "Decontamination) + §2.4.3 (Deduplication). 20-gram MinHash LSH at "
        "0.80 similarity threshold. See docs/DECONTAM.md and "
        "'MAI-Thinking-1: Building a Hill-Climbing Machine' "
        "(Microsoft AI Team, June 2026).",
    )
    parser.add_argument(
        "--training-data-root",
        type=Path,
        default=Path("data/datasets/buckets/sources"),
        help="Path to the per-source training data root directory. "
        "Default: data/datasets/buckets/sources/",
    )
    parser.add_argument(
        "--eval-set-dir",
        type=Path,
        required=True,
        help="Path to a directory of eval-set JSONL files, or a single JSONL file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="MinHash similarity threshold for overlap detection. Default: 0.80.",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=20,
        help="Character-level n-gram size (matches MAI-Thinking-1 §2.3.1). Default: 20.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=128,
        help="Number of MinHash permutations. Default: 128.",
    )
    parser.add_argument(
        "--source-filter",
        nargs="+",
        default=None,
        help="Restrict to these source names (default: all non-restricted).",
    )
    parser.add_argument(
        "--quarantine-output",
        type=Path,
        default=None,
        help="Path to write quarantine JSONL file with matched training records. "
        "If not set, no quarantine file is written.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Path to write the JSON report. Default: stdout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and exit without writing any output files.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of parallel processes for MinHash construction. Default: 4.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum number of overlap examples per source in the report. Default: 10.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate paths.
    if not args.training_data_root.is_dir():
        logger.error("Training data root does not exist: %s", args.training_data_root)
        return 1

    if not args.eval_set_dir.exists():
        logger.error("Eval set path does not exist: %s", args.eval_set_dir)
        return 1

    # Step 1: Load training records per source.
    logger.info("Loading training records from %s", args.training_data_root)
    records_by_source = load_training_records(
        args.training_data_root,
        source_filter=args.source_filter,
    )
    total_training = sum(len(recs) for recs in records_by_source.values())
    logger.info(
        "Loaded %d training records across %d sources",
        total_training,
        len(records_by_source),
    )

    if not records_by_source:
        logger.error(
            "No training records found. Check --training-data-root and --source-filter."
        )
        return 1

    # Step 2: Load eval records.
    logger.info("Loading eval records from %s", args.eval_set_dir)
    eval_records = load_eval_records(args.eval_set_dir)
    logger.info("Loaded %d eval records", len(eval_records))

    if not eval_records:
        logger.error("No eval records found. Check --eval-set-dir.")
        return 1

    # Step 3: Build MinHash LSH index over training corpus.
    logger.info(
        "Building MinHash LSH index (ngram=%d, threshold=%.2f, permutations=%d)",
        args.ngram_size,
        args.threshold,
        args.permutations,
    )
    lsh, index_map = build_training_index(
        records_by_source,
        ngram_size=args.ngram_size,
        num_perm=args.permutations,
        threshold=args.threshold,
        num_proc=args.num_proc,
    )

    # Step 4-6: Find overlaps via LSH query + exact Jaccard verification.
    logger.info("Querying %d eval records against training index", len(eval_records))
    overlaps = find_overlaps(
        lsh,
        index_map,
        eval_records,
        ngram_size=args.ngram_size,
        num_perm=args.permutations,
        threshold=args.threshold,
    )
    logger.info("Found %d overlapping pairs", len(overlaps))

    # Step 7: Build per-source report.
    report = build_report(
        records_by_source,
        overlaps,
        max_examples=args.max_examples,
    )

    # Step 8: Write quarantine if requested.
    quarantine_count = 0
    if args.quarantine_output and not args.dry_run:
        quarantine_count = write_quarantine(overlaps, index_map, args.quarantine_output)
        logger.info(
            "Wrote %d quarantine records to %s",
            quarantine_count,
            args.quarantine_output,
        )

    # Output the report.
    report_json = json.dumps(report, indent=2, ensure_ascii=False)

    if args.dry_run:
        logger.info("DRY RUN — no files written")
        print(report_json)
        return 0

    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_output, "w", encoding="utf-8") as f:
            f.write(report_json + "\n")
        logger.info("Report written to %s", args.report_output)
    else:
        print(report_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
