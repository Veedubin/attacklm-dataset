"""Hermetic tests for scripts/lib/manifest_builder.py (resurrected v5)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mb = _load("manifest_builder", "scripts/lib/manifest_builder.py")


def _make_sources_tree(tmp_path: Path) -> Path:
    """Minimal per-source layout: 2 sources, 3 buckets."""
    buckets = tmp_path / "buckets"
    # source-a: base/execution (2 records), atlas/reconnaissance (1 record)
    d1 = buckets / "sources" / "source-a" / "base" / "execution"
    d1.mkdir(parents=True)
    d1.joinpath("data.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    d2 = buckets / "sources" / "source-a" / "atlas" / "reconnaissance"
    d2.mkdir(parents=True)
    d2.joinpath("data.jsonl").write_text('{"a": 3}\n')
    # source-b: base/execution (1 record)
    d3 = buckets / "sources" / "source-b" / "base" / "execution"
    d3.mkdir(parents=True)
    d3.joinpath("data.jsonl").write_text('{"b": 1}\n')
    return buckets


def test_discover_aggregates_across_sources(tmp_path):
    buckets_dir = _make_sources_tree(tmp_path)
    buckets, source_meta = mb.discover_from_sources(buckets_dir / "sources")
    by_path = {b["path"]: b for b in buckets}
    assert by_path["base/execution"]["count"] == 3
    assert by_path["base/execution"]["sources"] == {"source-a": 2, "source-b": 1}
    assert by_path["base/execution"]["dominant_source"] == "source-a"
    assert by_path["base/execution"]["category"] == "tactic"
    assert by_path["atlas/reconnaissance"]["count"] == 1
    assert by_path["atlas/reconnaissance"]["category"] == "atlas"
    assert source_meta["source-a"]["n_records"] == 3
    assert source_meta["source-b"]["n_records"] == 1


def test_build_manifest_writes_v5_schema(tmp_path):
    buckets_dir = _make_sources_tree(tmp_path)
    manifest = mb.write_manifest(buckets_dir)
    written = json.loads((buckets_dir / "manifest.json").read_text())
    assert written["version"] == 5
    assert written["total_pairs"] == 4
    assert written["source_totals"] == {"source-a": 3, "source-b": 1}
    assert manifest == written


def test_dry_run_does_not_write(tmp_path):
    buckets_dir = _make_sources_tree(tmp_path)
    mb.write_manifest(buckets_dir, dry_run=True)
    assert not (buckets_dir / "manifest.json").exists()
