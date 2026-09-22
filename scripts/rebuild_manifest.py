#!/usr/bin/env python3
"""Rebuild data/datasets/buckets/manifest.json from the per-source layout.

Usage:
    python scripts/rebuild_manifest.py [--dry-run] [--buckets-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from manifest_builder import write_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--buckets-dir",
        type=Path,
        default=Path("data/datasets/buckets"),
    )
    args = parser.parse_args()
    manifest = write_manifest(args.buckets_dir, dry_run=args.dry_run)
    print(
        f"{'[dry-run] ' if args.dry_run else ''}manifest: "
        f"{manifest['total_buckets']} buckets, {manifest['total_pairs']:,} pairs, "
        f"{len(manifest['sources'])} sources"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
