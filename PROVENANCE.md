# PROVENANCE.md — Per-file attribution template

This file is a **template** for the per-file provenance block that should appear at the top of every Python source file in this repository that contains attack code, data-ingest code, or significant derived logic.

The block lives as a Python docstring immediately after the `from __future__ import annotations` import (if present) and before the rest of the module-level code.

---

## Template (copy-paste and fill)

```python
"""
PROVENANCE METADATA — {FILE_PATH}
================================================================================
Attack class:        {ATTACK_CLASS_OR_N/A}
Original authors:    {AUTHORS_FULL_NAMES}
Paper title:         {PAPER_TITLE}
Year / venue:        {YEAR} / {VENUE}
Paper URL:           {ARXIV_OR_DOI_URL}
Canonical repo:      {CANONICAL_REPO_URL_OR_N/A}

Implementation:
  Type:              {PORT | CLEAN_ROOM_REIMPLEMENTATION | ORIGINAL_WORK_INSPIRED_BY_PAPER}
  Lines of port:     {N_OR_N/A}  (set if Type == PORT)
  Upstream license:  {UPSTREAM_LICENSE_OR_N/A}

Data sources (only if this file ingests data, not for pure-attack code):
  Upstream repo:     {UPSTREAM_REPO_URL_OR_N/A}
  Upstream license:  {UPSTREAM_DATA_LICENSE}
  Per-record:        see data/ATTRIBUTION.md

Rights claim contact: veedubin.legal@example.com
See:                  RIGHTS.md (root), data/LEGAL.md, data/REMOVAL.md
================================================================================
"""
```

---

## Field reference

| Field                 | Required | Examples                                                                                |
| --------------------- | -------- | --------------------------------------------------------------------------------------- |
| `FILE_PATH`           | yes      | `scripts/inversion/probe.py`                                                            |
| `ATTACK_CLASS`        | no       | `Carlini 2021 Strategy 1: prefix-completion extraction`                                 |
| `AUTHORS_FULL_NAMES`  | yes      | `Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, ...` (12 authors)   |
| `PAPER_TITLE`         | yes      | `Extracting Training Data from Large Language Models`                                   |
| `YEAR`                | yes      | `2021`                                                                                  |
| `VENUE`               | yes      | `USENIX Security Symposium` or `IEEE Symposium on Security and Privacy`                 |
| `ARXIV_OR_DOI_URL`    | yes      | `https://arxiv.org/abs/2012.07805`                                                      |
| `CANONICAL_REPO_URL`  | no       | `https://github.com/woooooda/MUSE_unlearning` (MUSE has one; Carlini 2021/2022 do not)   |
| `PORT`                | yes      | one of: PORT, CLEAN_ROOM_REIMPLEMENTATION, ORIGINAL_WORK_INSPIRED_BY_PAPER              |
| `N_OR_N/A`            | no       | `~200` or `N/A`                                                                         |
| `UPSTREAM_LICENSE`    | no       | `MIT` or `Apache-2.0` or `N/A`                                                          |
| `UPSTREAM_REPO_URL`   | no       | `https://github.com/rapid7/metasploit-framework`                                         |
| `UPSTREAM_DATA_LICENSE` | no     | `BSD-3-Clause`                                                                          |

---

## Notes

1. The provenance block is intended to be **machine-parseable** for a future audit script. The format above (the 80-character `=` border, the colons aligned, the kebab-case labels) is intentional. Do not reformat.

2. If the file's code is purely original (no upstream paper or source), set `ATTACK_CLASS` to `N/A`, `AUTHORS_FULL_NAMES` to the in-repo author, and the `Implementation` type to `ORIGINAL_WORK`. The `Paper title` / `Paper URL` fields can be left as `N/A` or set to `(internal design doc)` with a link to a `docs/` file.

3. If a file mixes multiple attack classes (e.g., a CLI driver that calls both Carlini 2021 extraction and Carlini 2022 MIA), list both in `ATTACK_CLASS` separated by `+` and include both paper citations. The PROVENANCE block can be longer than the template in this case.

4. The "canonical repo" line is intentionally separate from the "upstream data" line. A Carlini 2021 implementation has no canonical repo (no official code release), but a Metasploit ingest script has GitHub `https://github.com/rapid7/metasploit-framework` as its canonical data repo. Do not conflate these.

5. If a paper's authors are not human (e.g., a corporate publication), set `AUTHORS_FULL_NAMES` to the corporation name (e.g., `Microsoft Security Response Center`).

---

## Example — fully filled

```python
"""
PROVENANCE METADATA — scripts/inversion/probe.py
================================================================================
Attack class:        Carlini 2021 Strategy 1: prefix-completion extraction
Original authors:    Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew
                     Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts,
                     Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea,
                     Colin Raffel
Paper title:         Extracting Training Data from Large Language Models
Year / venue:        2021 / USENIX Security Symposium
Paper URL:           https://arxiv.org/abs/2012.07805
Canonical repo:      N/A (no official code release by the authors)

Implementation:
  Type:              CLEAN_ROOM_REIMPLEMENTATION
  Lines of port:     N/A
  Upstream license:  N/A

Data sources: N/A (this file attacks a model, it does not ingest data)

Rights claim contact: veedubin.legal@example.com
See:                  RIGHTS.md (root), data/LEGAL.md, data/REMOVAL.md
================================================================================
"""
```

---

## How to update this template

If you need a new field (e.g., `Citation count: N`), add a row to the "Field reference" table above AND update the template in this file. The intent is for all PROVENANCE blocks across the repo to be in sync — any future audit script can assume the template is the single source of truth.

**Last template revision:** 2026-07-09
