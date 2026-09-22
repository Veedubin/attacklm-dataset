# garak (NVIDIA)

LLM vulnerability scanner. Probes for hallucinations, data leakage, prompt injection, misinformation, toxicity, and jailbreaks.

## What it's used for

DAN/probe JSON & TXT resources for AI red-teaming.

## Provenance

| Field | Value |
|---|---|
| **Source name** | `nvidia-garak` |
| **Display name** | garak (NVIDIA) |
| **Upstream** | <https://github.com/NVIDIA/garak> |
| **License** | Apache-2.0 |
| **Risk level** | low |

## How it's ingested

The records in this directory are produced by
`scripts/extract_ai_tools_to_jsonl.py`, which reads the cloned
`data/ai_tools/garak/` repository and writes here.

To re-run: `uv run python scripts/extract_ai_tools_to_jsonl.py`

## Rights-holder contact

If you are a rights holder for `garak (NVIDIA)` and would like any of
these records removed, see **`data/REMOVAL.md`** at the repository root.
Removal is fast and unconditional.
