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


def order_steps(steps: list[dict]) -> list[dict]:
    """Order case-study steps by walking the leads-to graph.

    Falls back to step-id sort, then document order, when the graph is
    ambiguous (no unique head) or contains cycles/unreachable nodes.
    """
    by_id = {s.get("step-id"): s for s in steps if s.get("step-id")}
    if len(by_id) != len(steps):
        return list(steps)
    referenced = {t for s in by_id.values() for t in (s.get("leads-to") or [])}
    heads = [sid for sid in by_id if sid not in referenced]
    if len(heads) != 1:
        return sorted(steps, key=lambda s: s.get("step-id", ""))
    ordered: list[dict] = []
    seen: set[str] = set()
    cur: str | None = heads[0]
    while cur and cur not in seen and cur in by_id:
        seen.add(cur)
        step = by_id[cur]
        ordered.append(step)
        nxt = [t for t in (step.get("leads-to") or []) if t not in seen]
        cur = nxt[0] if nxt else None
    for sid in sorted(by_id):
        if sid not in seen:
            ordered.append(by_id[sid])
    return ordered


def case_study_pairs(cs: dict, data: dict, idx: dict) -> list[dict]:
    cid = cs["id"]
    name = cs["name"]
    actor = cs.get("actor", "the adversary")
    techs = data.get("techniques", {})
    tactics = data.get("tactics", {})
    steps = order_steps(idx["case_steps"].get(cid, []))

    pairs: list[dict] = []
    flow_lines: list[str] = []
    for i, step in enumerate(steps, 1):
        tid = step["target"]
        tech = techs.get(tid, {})
        tech_name = tech.get("name", tid)
        ta_id = step.get("tactic", "")
        tactic_name = tactics.get(ta_id, {}).get("name", "AI Model Access")
        step_desc = (step.get("description") or "").strip()
        flow_lines.append(f"{i}. **{tid} ({tech_name})** [{tactic_name}]")
        pairs.append(
            _record(
                f"In the {name} case study, how did {actor} perform "
                f"{tech_name} ({tid})?",
                f"**{name} ({cid}) — step {step.get('step-id', i)}:**\n\n"
                f"{step_desc}",
                tactic_name,
                [tid],
                extra={"case_study": cid},
            )
        )

    if steps:
        first_ta = steps[0].get("tactic", "")
        tactic_name = tactics.get(first_ta, {}).get("name", "AI Model Access")
        pairs.append(
            _record(
                f"Walk through the full attack sequence of the {name} "
                f"case study ({cid}).",
                f"## {name} ({cid})\n**Actor:** {actor} | "
                f"**Target:** {cs.get('target', 'unknown')}\n\n"
                + "\n".join(flow_lines),
                tactic_name,
                [s["target"] for s in steps],
                extra={"case_study": cid},
            )
        )
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract MITRE ATLAS into AttackLM JSONL training pairs."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts and 2 sample pairs; do not write")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to the vendored ATLAS YAML.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Bucket output root.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ATLAS YAML not found: {args.input}", file=sys.stderr)
        return 1

    data = load_atlas(args.input)
    idx = build_indexes(data)

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    n = 0
    for tech in data.get("techniques", {}).values():
        for p in technique_pairs(tech, data, idx):
            by_bucket[f"atlas/{p['tactic']}"].append(p)
            n += 1
    for mit in data.get("mitigations", {}).values():
        for p in mitigation_pairs(mit, data, idx):
            by_bucket[f"atlas/{p['tactic']}"].append(p)
            n += 1
    for cs in data.get("case-studies", {}).values():
        for p in case_study_pairs(cs, data, idx):
            by_bucket[f"atlas/{p['tactic']}"].append(p)
            n += 1

    print(f"Generated {n} pairs across {len(by_bucket)} buckets",
          file=sys.stderr)
    for bucket in sorted(by_bucket):
        print(f"  {bucket:35s} {len(by_bucket[bucket]):>5d}", file=sys.stderr)

    if args.dry_run:
        for p in (by_bucket[sorted(by_bucket)[0]])[:2]:
            print(json.dumps(p, indent=2))
        return 0

    for bucket, pairs in sorted(by_bucket.items()):
        out_dir = args.output_dir / bucket
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "data.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {n} pairs under {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
