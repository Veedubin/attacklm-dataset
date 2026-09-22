# MITRE ATLAS

Adversarial Threat Landscape for Artificial-Intelligence Systems — a knowledge base of adversary tactics, techniques, and case studies for attacks against AI-enabled systems.

## What it's used for

ATLAS tactics, techniques, mitigations, and case studies converted to training triples across 16 `atlas/*` tactic buckets.

## Provenance

| Field | Value |
|---|---|
| **Source name** | `mitre-atlas` |
| **Display name** | MITRE ATLAS |
| **Upstream** | <https://github.com/mitre-atlas/atlas-data> |
| **License** | Apache-2.0 |
| **Risk level** | low |

## How it's ingested

The records in this directory are produced by
`scripts/extract_mitre_atlas.py`, which reads the vendored
`data/mitre-atlas/ATLAS-2026.09.yaml` and writes here.

To re-run: `uv run python scripts/extract_mitre_atlas.py`

## Rights-holder contact

If you are a rights holder for `MITRE ATLAS` and would like any of
these records removed, see **`data/REMOVAL.md`** at the repository root.
Removal is fast and unconditional.
