#!/usr/bin/env python3
"""Split held-out evaluation records from each source dataset.

Reads each source under data/datasets/buckets/sources/<source>/<bucket>/<tactic>/data*.jsonl,
takes the last N records (configurable, default 300), and writes them to
data/held_out/<source>/<bucket>/<tactic>/data_held_out.jsonl.

The remaining (training) records stay in place — this script only copies,
never modifies the original files.

Idempotent: if data/held_out/<source>/ already exists, that source is skipped
with a warning log.
"""

import argparse
import json
import sys
from pathlib import Path

DATASET_ROOT = Path("data/datasets/buckets/sources")
HELD_OUT_ROOT = Path("data/held_out")

# Companion to MAI-Thinking-1 §2.3 held-out evaluation. See docs/HELD_OUT_NLL.md.


def collect_jsonl_records(source_dir: Path) -> list[dict]:
    """Load all records from data*.jsonl files under a source directory.

    Walks the source directory tree, finds every file matching data*.jsonl,
    and returns the combined list of parsed JSON records (preserving file order,
    then line order within each file).
    """
    records: list[dict] = []
    jsonl_files = sorted(source_dir.rglob("data*.jsonl"))
    for jsonl_path in jsonl_files:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines — upstream scripts handle this
                        pass
    return records


def split_source(
    source_name: str,
    dataset_root: Path,
    held_out_root: Path,
    held_out_size: int,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Split the last N records from a source into the held-out directory.

    Returns (total_records, held_out_count).
    """
    source_dir = dataset_root / source_name
    if not source_dir.is_dir():
        print(f"  WARNING: source directory not found: {source_dir}", file=sys.stderr)
        return 0, 0

    # Check idempotency: if held_out/<source>/ already exists, skip
    held_out_source_dir = held_out_root / source_name
    if held_out_source_dir.exists():
        print(
            f"  SKIP: {source_name} — held-out directory already exists at {held_out_source_dir}"
        )
        return 0, 0

    records = collect_jsonl_records(source_dir)
    total = len(records)

    if total == 0:
        print(f"  SKIP: {source_name} — no records found")
        return 0, 0

    # Take the last N records for held-out (computed later with file tracking)
    held_out_count = min(held_out_size, total)

    if dry_run:
        print(
            f"  [DRY RUN] would split {source_name}: {total} total → {held_out_count} held-out, {total - held_out_count} training"
        )
        return total, held_out_count

    # Write held-out records, preserving the bucket/tactic subdirectory structure.
    # We need to figure out which bucket/tactic each record came from.
    # Strategy: walk the source tree, find all data*.jsonl, split per-file,
    # and write to corresponding held-out paths.
    jsonl_files = sorted(source_dir.rglob("data*.jsonl"))
    if not jsonl_files:
        return total, held_out_count

    # Build a mapping from records to their source files for proper routing.
    # Since we take the LAST N records globally, we need to distribute them
    # back to their original files. We do this by tracking line offsets.
    global_records: list[tuple[dict, Path]] = []
    for jsonl_path in jsonl_files:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        global_records.append((json.loads(line), jsonl_path))
                    except json.JSONDecodeError:
                        pass

    total_global = len(global_records)
    held_out_count = min(held_out_size, total_global)

    # Last N records for held-out
    held_out_items = global_records[-held_out_count:]

    # Group held-out records by their destination subdirectory
    # dest_path: held_out_root / source_name / <bucket> / <tactic> / data_held_out.jsonl
    dest_groups: dict[Path, list[dict]] = {}
    for record, src_path in held_out_items:
        # src_path is like: dataset_root/source_name/<bucket>/<tactic>/data*.jsonl
        # We need to extract the <bucket>/<tactic> part
        rel = src_path.relative_to(source_dir)
        # rel is like: <bucket>/<tactic>/data.jsonl or base/execution/data.jsonl
        # The parent of the data file is <bucket>/<tactic>/
        bucket_tactic = (
            rel.parent
        )  # e.g., Path("base/execution") or Path("tools/metasploit")
        dest_dir = held_out_root / source_name / bucket_tactic
        dest_groups.setdefault(dest_dir, []).append(record)

    # Write each group
    for dest_dir, group_records in dest_groups.items():
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "data_held_out.jsonl"
        with open(dest_file, "w") as f:
            for rec in group_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"  {source_name}: {total_global} total → {held_out_count} held-out across {len(dest_groups)} file(s)"
    )
    return total_global, held_out_count


def main(
    dataset_root: Path | None = None,
    held_out_root: Path | None = None,
    held_out_size: int = 300,
    dry_run: bool = False,
    yes: bool = False,
) -> int:
    """Split held-out evaluation records from each source dataset."""
    if dataset_root is None:
        dataset_root = DATASET_ROOT
    if held_out_root is None:
        held_out_root = HELD_OUT_ROOT

    dataset_root = Path(dataset_root)
    held_out_root = Path(held_out_root)

    if not dataset_root.exists():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        return 1

    # Discover source directories (immediate children of dataset_root that contain data*.jsonl)
    source_dirs = sorted(
        d.name
        for d in dataset_root.iterdir()
        if d.is_dir() and any(d.rglob("data*.jsonl"))
    )

    if not source_dirs:
        print("No source directories with data*.jsonl found.", file=sys.stderr)
        return 1

    print("=" * 70)
    print("HELD-OUT SPLIT" + (" [DRY RUN]" if dry_run else ""))
    print("=" * 70)
    print()
    print(f"Dataset root:  {dataset_root}")
    print(f"Held-out root: {held_out_root}")
    print(f"Held-out size: {held_out_size} records per source")
    print(f"Sources found:  {len(source_dirs)}")
    print()

    # Confirmation prompt (unless --yes)
    if not dry_run and not yes:
        print(f"About to split {held_out_size} records per source into {held_out_root}")
        print("Type 'yes' to continue: ", end="", flush=True)
        response = input().strip().lower()
        if response != "yes":
            print("Aborted.")
            return 0
        print()

    total_records = 0
    total_held_out = 0
    sources_processed = 0
    sources_skipped = 0

    for source_name in source_dirs:
        total, held_out = split_source(
            source_name=source_name,
            dataset_root=dataset_root,
            held_out_root=held_out_root,
            held_out_size=held_out_size,
            dry_run=dry_run,
        )
        total_records += total
        total_held_out += held_out
        if held_out > 0 or (total > 0 and dry_run):
            sources_processed += 1
        else:
            sources_skipped += 1

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Sources processed: {sources_processed}")
    print(f"  Sources skipped:   {sources_skipped}")
    print(f"  Total records:     {total_records:,}")
    print(f"  Held-out records:  {total_held_out:,}")
    if not dry_run:
        print(f"  Output directory:  {held_out_root}")
    else:
        print("  [DRY RUN] no files written")
    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split held-out evaluation records from each source dataset. "
        "Takes the last N records from each source and writes them to a "
        "held-out directory for NLL evaluation.",
        epilog="Methodology: companion to MAI-Thinking-1 §2.3 held-out evaluation — "
        "splits 300 records per source at extract time. See docs/HELD_OUT_NLL.md.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Path to the per-source dataset directory (default: {DATASET_ROOT}).",
    )
    parser.add_argument(
        "--held-out-root",
        type=Path,
        default=HELD_OUT_ROOT,
        help=f"Path to the held-out output directory (default: {HELD_OUT_ROOT}).",
    )
    parser.add_argument(
        "--held-out-size",
        type=int,
        default=300,
        help="Number of records to hold out from each source (default: 300).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be done without writing any files.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()
    raise SystemExit(
        main(
            dataset_root=args.dataset_root,
            held_out_root=args.held_out_root,
            held_out_size=args.held_out_size,
            dry_run=args.dry_run,
            yes=args.yes,
        )
    )
