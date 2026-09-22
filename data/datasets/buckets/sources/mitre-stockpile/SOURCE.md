# MITRE Caldera — Stockpile

Plugin-based TTP descriptors for the MITRE Caldera adversary emulation platform. YAML-defined abilities with executor, command, and parser sections.

## What it's used for

Adversary emulation ability triples (YAML-defined TTP descriptors).

## Provenance

| Field | Value |
|---|---|
| **Source name** | `mitre-stockpile` |
| **Display name** | MITRE Caldera — Stockpile |
| **Upstream** | <https://github.com/mitre/stockpile> |
| **License** | Apache-2.0 |
| **Risk level** | low |

## How it's ingested

The records in this directory are produced by
`scripts/extract_caldera_plugins_to_jsonl.py`, which reads the vendored
`data/stockpile/` plugin checkout and writes here.

To re-run: `uv run python scripts/extract_caldera_plugins_to_jsonl.py`

## Rights-holder contact

If you are a rights holder for `MITRE Caldera — Stockpile` and would like any of
these records removed, see **`data/REMOVAL.md`** at the repository root.
Removal is fast and unconditional.
