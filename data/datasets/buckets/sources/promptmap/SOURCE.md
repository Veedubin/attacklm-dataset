# promptmap (utkusen)

Prompt injection scanner with a YAML rule file format. Tests an LLM endpoint against a curated set of injection attacks.

## What it's used for

Prompt injection YAML rule files.

## Provenance

| Field | Value |
|---|---|
| **Source name** | `promptmap` |
| **Display name** | promptmap (utkusen) |
| **Upstream** | <https://github.com/utkusen/promptmap> |
| **License** | MIT |
| **Risk level** | low |

## How it's ingested

The records in this directory are produced by
`scripts/extract_ai_tools_to_jsonl.py`, which reads the cloned
`data/ai_tools/promptmap/` repository and writes here.

To re-run: `uv run python scripts/extract_ai_tools_to_jsonl.py`

## Rights-holder contact

If you are a rights holder for `promptmap (utkusen)` and would like any of
these records removed, see **`data/REMOVAL.md`** at the repository root.
Removal is fast and unconditional.
