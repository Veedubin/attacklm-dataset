"""manifest_builder — Rebuild the AttackLM bucket manifest from disk.

Library module resurrected from the deleted v5 ``scripts/rebuild_manifest.py``
(commit b620703, removed in a1017d2). Walks the per-source layout
(``data/datasets/buckets/sources/<source>/<bucket>/<tactic>/data*.jsonl``);
the legacy flat layout (``data/datasets/buckets/<bucket>/data.jsonl``) is
still discoverable via :func:`discover_from_flat` for back-compat sanity
checks but is no longer the source of truth.

Buckets in ``archive/restricted-sources/`` are NEVER included in the
public manifest.

CLI entry point: ``scripts/rebuild_manifest.py``.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Sub-file names for three-tier provenance
# ---------------------------------------------------------------------------
SUB_FILES = {
    "human": "data_human.jsonl",
    "llm": "data_llm.jsonl",
    "synth": "data_synth.jsonl",
}


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def discover_from_sources(sources_dir: Path) -> tuple[list[dict], dict]:
    """
    Walk the per-source layout under ``sources_dir``. Returns (buckets, source_meta).

    buckets: list of bucket entries (tactic-level), each carrying a
             ``sources`` field with per-source record counts.
    source_meta: per-source summary { name: {license, n_records, buckets: [...] } }
    """
    buckets_map: dict[str, dict] = {}  # bucket_path -> entry
    source_meta: dict[str, dict] = {}

    if not sources_dir.exists():
        return [], {}

    # Load _index.json if present for license/display info
    index_path = sources_dir / "_index.json"
    index_data = {}
    if index_path.exists():
        try:
            with index_path.open() as f:
                idx = json.load(f)
            for s in idx.get("sources", []):
                index_data[s["name"]] = s
        except Exception:
            pass

    for src_dir in sorted(sources_dir.iterdir()):
        if not src_dir.is_dir() or src_dir.name.startswith("_"):
            continue
        source_name = src_dir.name
        meta = index_data.get(source_name, {})
        src_entry = {
            "name": source_name,
            "display": meta.get("display", source_name),
            "license": meta.get("license", "unknown"),
            "license_uri": meta.get("license_uri", ""),
            "upstream_url": meta.get("upstream_url", ""),
            "risk": meta.get("risk", "unknown"),
            "n_records": 0,
            "buckets": [],
        }

        for jsonl in sorted(src_dir.rglob("*.jsonl")):
            parts = jsonl.relative_to(src_dir).parts
            # Layout variants:
            #   <bucket>/<tactic>/<file>.jsonl    (e.g. base/execution/data.jsonl)
            #   <bucket>/<file>.jsonl              (e.g. orchestrator/data.jsonl)
            if len(parts) == 3:
                bucket_path = f"{parts[0]}/{parts[1]}"
            elif len(parts) == 2:
                bucket_path = parts[0]
            else:
                continue
            n = count_jsonl_lines(jsonl)
            src_entry["n_records"] += n
            src_entry["buckets"].append(
                {
                    "bucket": bucket_path,
                    "file": jsonl.name,
                    "records": n,
                }
            )

            # Aggregate into tactic-level bucket
            if bucket_path not in buckets_map:
                buckets_map[bucket_path] = {
                    "name": bucket_path.replace("/", "_"),
                    "path": bucket_path,
                    "sources": {},
                    "files": {},
                }
            entry = buckets_map[bucket_path]
            entry["sources"][source_name] = entry["sources"].get(source_name, 0) + n
            entry["files"][f"{source_name}:{jsonl.name}"] = n

        source_meta[source_name] = src_entry

    buckets = []
    for path, entry in buckets_map.items():
        # Derive category from first path component
        first = path.split("/")[0]
        cat_map = {
            "base": "tactic",
            "ai": "ai_redteam",
            "atlas": "atlas",
            "tools": "tools",
            "orchestrator": "meta",
            "attack_tactics": "attack_tactics",
            "web_app": "web_app",
            "cloud": "cloud",
            "social_engineering": "social_engineering",
            "supply_chain": "supply_chain",
            "ics": "ics",
            "wireless": "wireless",
        }
        category = cat_map.get(first, first)
        # Determine dominant source for this bucket
        dominant_source = max(entry["sources"], key=entry["sources"].get)
        dom_meta = source_meta.get(dominant_source, {})

        # Determine sub_sources tier for the bucket based on filename
        sub_sources = {"human": 0, "llm": 0, "synth": 0}
        for key, n in entry["files"].items():
            src, fname = key.split(":", 1)
            if fname == "data_llm.jsonl":
                sub_sources["llm"] += n
            elif fname == "data_synth.jsonl":
                sub_sources["synth"] += n
            else:
                # data.jsonl is treated as "human" if it comes from a real
                # upstream source, "synth" if from attacklm-synthetic.
                if src == "attacklm-synthetic" or src == "llm-generated":
                    sub_sources["synth"] += n
                else:
                    sub_sources["human"] += n

        buckets.append(
            {
                "name": entry["name"],
                "path": entry["path"],
                "category": category,
                "count": sum(entry["sources"].values()),
                "sub_sources": sub_sources,
                "sources": entry["sources"],
                "dominant_source": dominant_source,
                "license": dom_meta.get("license", "unknown"),
            }
        )

    return buckets, source_meta


def discover_from_flat(buckets_dir: Path) -> list[dict]:
    """
    Walk the legacy flat layout (buckets_dir/<bucket>/data*.jsonl).
    Used as a fallback / sanity check.
    """
    buckets: list[dict] = []
    for metadata_path in sorted(buckets_dir.rglob("metadata.json")):
        # Skip metadata under sources/ (handled by discover_from_sources)
        if "sources/" in str(metadata_path.relative_to(buckets_dir)):
            continue
        bucket_dir = metadata_path.parent
        rel_path = bucket_dir.relative_to(buckets_dir)
        # Skip root
        if str(rel_path) == ".":
            continue
        try:
            with open(metadata_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        n = 0
        for fname in SUB_FILES.values():
            n += count_jsonl_lines(bucket_dir / fname)
        if n == 0:
            legacy = bucket_dir / "data.jsonl"
            n = count_jsonl_lines(legacy)
        if n == 0:
            continue
        buckets.append(
            {
                "name": meta.get("name", bucket_dir.name),
                "path": str(rel_path),
                "count": n,
            }
        )
    return buckets


def sort_buckets(buckets: list[dict]) -> list[dict]:
    CATEGORY_ORDER = [
        "tactic",
        "tools",
        "ai_redteam",
        "atlas",
        "attack_tactics",
        "web_app",
        "cloud",
        "social_engineering",
        "supply_chain",
        "ics",
        "wireless",
        "meta",
    ]
    cat_index = {c: i for i, c in enumerate(CATEGORY_ORDER)}

    def key(b: dict) -> tuple[int, str]:
        return (
            cat_index.get(b.get("category", "zzz"), len(CATEGORY_ORDER)),
            b.get("name", ""),
        )

    return sorted(buckets, key=key)


def build_manifest(buckets_dir: Path) -> dict:
    """Walk sources/ and return the v5 manifest dict (not written)."""
    buckets, source_meta = discover_from_sources(buckets_dir / "sources")
    buckets = sort_buckets(buckets)
    tier: Counter = Counter()
    for b in buckets:
        for k, v in b.get("sub_sources", {}).items():
            tier[k] += v
    return {
        "version": 5,
        "layout": "per-source (sources/<source>/<bucket>/<tactic>/data*.jsonl)",
        "legacy_layout": "flat (data/datasets/buckets/<bucket>/data.jsonl) — kept for back-compat",
        "provenance": "per-source + per-record fields (source, source_uri, license, license_uri, rights_contact, plus per-license attribution fields)",
        "excluded_sources": [
            "endgameinc/RTA (AGPL-3.0)",
            "guardicore/infection_monkey (GPL-3.0)",
            "TheBigPromptLibrary (mixed/unclear)",
        ],
        "created": datetime.now(timezone.utc).isoformat(),
        "total_buckets": len(buckets),
        "total_pairs": sum(b["count"] for b in buckets),
        "tier_totals": dict(tier),
        "source_totals": {k: v["n_records"] for k, v in source_meta.items()},
        "sources": source_meta,
        "buckets": buckets,
    }


def write_manifest(buckets_dir: Path, dry_run: bool = False) -> dict:
    manifest = build_manifest(buckets_dir)
    if not dry_run:
        path = buckets_dir / "manifest.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest
