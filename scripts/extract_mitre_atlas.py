#!/usr/bin/env python3
# CREDITS — DATA SOURCE ATTRIBUTION
# ----------------------------------
# This script transforms data from: MITRE ATLAS
# Repository: https://github.com/mitre-atlas/atlas-data
# License:    Apache-2.0
# Copyright:  (c) 2021-2026 The MITRE Corporation. All rights reserved.
# ----------------------------------
"""Deterministic extraction of MITRE ATLAS into AttackLM JSONL training pairs.

Reads the vendored ``data/mitre-atlas/ATLAS-2026.09.yaml`` and writes
per-tactic buckets to
``data/datasets/buckets/sources/mitre-atlas/atlas/<tactic>/data.jsonl``.

Usage:
    python scripts/extract_mitre_atlas.py
    python scripts/extract_mitre_atlas.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
ATLAS_VERSION = "2026.09"
DEFAULT_INPUT = BASE_DIR / "data" / "mitre-atlas" / f"ATLAS-{ATLAS_VERSION}.yaml"
DEFAULT_OUTPUT_DIR = (
    BASE_DIR / "data" / "datasets" / "buckets" / "sources" / "mitre-atlas"
)

ATTRIBUTION = {
    "source": "mitre-atlas",
    "source_uri": "https://github.com/mitre-atlas/atlas-data",
    "license": "Apache-2.0",
    "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
    "rights_contact": "see data/REMOVAL.md",
}

BASE_TAGS = ["atlas", "ai_ml", f"atlas-{ATLAS_VERSION}"]


def tactic_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def load_atlas(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_indexes(data: dict) -> dict:
    tech_to_tactics: dict[str, list[str]] = defaultdict(list)
    sub_to_parent: dict[str, str] = {}
    tech_to_mitigations: dict[str, list[str]] = defaultdict(list)
    case_steps: dict[str, list[dict]] = defaultdict(list)
    for rels in data.get("relationships", {}).values():
        for r in rels.get("achieves", []):
            tech_to_tactics[r["source"]].append(r["target"])
        for r in rels.get("specializes", []):
            sub_to_parent[r["source"]] = r["target"]
        for r in rels.get("mitigates", []):
            tech_to_mitigations[r["target"]].append(r["source"])
        for r in rels.get("employs", []):
            case_steps[r["source"]].append(r)
    return {
        "tech_to_tactics": dict(tech_to_tactics),
        "sub_to_parent": sub_to_parent,
        "tech_to_mitigations": dict(tech_to_mitigations),
        "case_steps": dict(case_steps),
    }


def _system_msg(tactic_name: str) -> str:
    return (
        "You are an authorized AI red team specialist with deep knowledge of "
        "the MITRE ATLAS framework, focused on the "
        f"{tactic_name} tactic against AI-enabled systems. Provide precise, "
        "technically accurate guidance for security validation testing."
    )


def _tactic_info(tech_id: str, data: dict, idx: dict) -> tuple[str, str]:
    """Return (tactic_id, tactic_name) for a technique; defaults if unmapped."""
    tactics = data.get("tactics", {})
    for ta_id in idx["tech_to_tactics"].get(tech_id, []):
        if ta_id in tactics:
            return ta_id, tactics[ta_id]["name"]
    return "AML.TA0000", "AI Model Access"


def _record(user: str, assistant: str, tactic_name: str, mitre_ids: list[str],
            extra: dict | None = None) -> dict:
    rec = {
        "messages": [
            {"role": "system", "content": _system_msg(tactic_name)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "mitre_ids": mitre_ids,
        "tactic": tactic_slug(tactic_name),
        "tags": list(BASE_TAGS),
        **ATTRIBUTION,
    }
    if extra:
        rec.update(extra)
    return rec


def technique_pairs(tech: dict, data: dict, idx: dict) -> list[dict]:
    tid = tech["id"]
    name = tech["name"]
    desc = tech.get("description", "").strip()
    platforms = ", ".join(tech.get("platforms", [])) or "Not specified"
    maturity = tech.get("maturity", "Unknown")
    _, tactic_name = _tactic_info(tid, data, idx)

    parent_note = ""
    parent_id = idx["sub_to_parent"].get(tid)
    if parent_id and parent_id in data["techniques"]:
        parent_note = (
            f"\n\nThis is a sub-technique of **{parent_id} "
            f"({data['techniques'][parent_id]['name']})**."
        )

    body = (
        f"## {name}\n**ATLAS:** {tid} | **Tactic:** {tactic_name}\n\n"
        f"{desc}{parent_note}\n\n"
        f"**Platforms:** {platforms}\n**Maturity:** {maturity}"
    )

    pairs = [
        _record(
            f"Describe ATLAS technique {tid} ({name}).",
            body, tactic_name, [tid],
        ),
        _record(
            f"How does {name} ({tid}) work in attacks against AI systems?",
            body, tactic_name, [tid],
        ),
        _record(
            f"Which AI platforms are affected by {tid}, and what is its "
            f"maturity level?",
            f"**{tid} ({name})** applies to: {platforms}.\n"
            f"Maturity: **{maturity}**.",
            tactic_name, [tid],
        ),
    ]

    attack_ref = tech.get("attack-reference")
    if attack_ref and attack_ref.get("id"):
        att_id = attack_ref["id"]
        pairs.append(
            _record(
                f"How does ATLAS {tid} relate to MITRE ATT&CK?",
                f"**{tid} ({name})** maps to ATT&CK Enterprise "
                f"**{att_id}** ({attack_ref.get('url', '')}).",
                tactic_name, [tid, att_id],
            )
        )
    return pairs


def mitigation_pairs(mit: dict, data: dict, idx: dict) -> list[dict]:
    mid = mit["id"]
    name = mit["name"]
    desc = mit.get("description", "").strip()
    phases = ", ".join(mit.get("lifecycle-phases", [])) or "Not specified"
    categories = ", ".join(mit.get("categories", [])) or "Not specified"

    # One pair per technique this mitigation mitigates (invert mitigates
    # index) — computed first so the description pair routes to the same
    # tactic as the first linked technique.
    techs = data.get("techniques", {})
    mitigates = [
        tid for tid, mids in idx["tech_to_mitigations"].items() if mid in mids
    ]
    linked_tactic = "Defense Evasion"
    for tid in sorted(mitigates):
        if tid in techs:
            _, linked_tactic = _tactic_info(tid, data, idx)
            break

    pairs = [
        _record(
            f"Describe ATLAS mitigation {mid} ({name}).",
            f"## {name}\n**ATLAS Mitigation:** {mid}\n\n{desc}\n\n"
            f"**Lifecycle phases:** {phases}\n**Categories:** {categories}",
            linked_tactic,
            [mid],
        )
    ]

    for tid in sorted(mitigates):
        tech = techs.get(tid)
        if not tech:
            continue
        _, tactic_name = _tactic_info(tid, data, idx)
        pairs.append(
            _record(
                f"How can you mitigate ATLAS technique {tid} "
                f"({tech['name']})?",
                f"**{mid} ({name})** mitigates {tid}:\n\n{desc}",
                tactic_name,
                [mid, tid],
            )
        )
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract MITRE ATLAS into AttackLM JSONL training pairs."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Count pairs without writing files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to the vendored ATLAS YAML.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Bucket output root.")
    args = parser.parse_args()

    data = load_atlas(args.input)
    idx = build_indexes(data)

    # Group technique pairs per tactic bucket.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for tech in data.get("techniques", {}).values():
        for rec in technique_pairs(tech, data, idx):
            buckets[rec["tactic"]].append(rec)

    if args.dry_run:
        total = sum(len(v) for v in buckets.values())
        print(f"dry-run: {total} pairs across {len(buckets)} tactic buckets")
        return 0

    for tactic, recs in sorted(buckets.items()):
        out = args.output_dir / "atlas" / tactic
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "data.jsonl", "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
