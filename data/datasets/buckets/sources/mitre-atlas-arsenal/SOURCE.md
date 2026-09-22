# MITRE ATLAS Arsenal

Adversary tactics for AI systems — case studies and evaluations of real attacks against machine learning systems.

## What it's used for

Adversarial ML case studies and evaluation harnesses.

## Provenance

| Field | Value |
|---|---|
| **Source name** | `mitre-atlas-arsenal` |
| **Display name** | MITRE ATLAS Arsenal |
| **Upstream** | <https://github.com/mitre-atlas/arsenal> |
| **License** | Apache-2.0 |
| **Risk level** | low |

## How it's ingested

The records in this directory are produced by
`scripts/extract_caldera_plugins_to_jsonl.py` (the arsenal plugin
configuration), which reads the vendored `data/arsenal/` checkout and
writes here.

To re-run: `uv run python scripts/extract_caldera_plugins_to_jsonl.py`

## Rights-holder contact

If you are a rights holder for `MITRE ATLAS Arsenal` and would like any of
these records removed, see **`data/REMOVAL.md`** at the repository root.
Removal is fast and unconditional.
