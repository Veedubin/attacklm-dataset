# attacklm-dataset

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Provenance: 100%](https://img.shields.io/badge/provenance-100%25-brightgreen.svg)](data/ATTRIBUTION.md)
[![Sources: 11 active](https://img.shields.io/badge/sources-11_active-blue.svg)](#dataset-composition)
[![Tests: 475+](https://img.shields.io/badge/tests-475%2B-brightgreen.svg)](#testing)
[![Distribution: GH-only](https://img.shields.io/badge/distribution-GH--only-yellow.svg)](#distribution)

**A MITRE ATT&CK-grounded security fine-tuning dataset, an extraction
pipeline from 11+ upstream security sources, and a privacy-audit
harness (Carlini 2021 extraction + 4 MIA methods).**

This repo is the **data + research-toolkit side** of the AttackLM
project. The trainer/tuner lives in the companion package
[Veedubin/AttackLM](https://github.com/Veedubin/AttackLM). The two
split in v0.11.0 to keep the audit code (defensive research) and
the data (license-aware) isolated from the trainer (general-purpose
infrastructure).

---

## What's in here

Three things, in one repo:

1. **A 24,652-record dataset** of MITRE ATT&CK-grounded training
   pairs. Every record carries full provenance (source, source URI,
   license, license URI, rights contact). 18 source directories,
   11 active after the v0.3.0 security review (3 high-risk sources
   excluded: RTA, infection_monkey, BPL).

2. **An extraction pipeline** — 21 `extract_*.py` scripts that read
   upstream security tools (Metasploit, Atomic Red Team, Sigma,
   Elastic, Splunk, Mordor, etc.) and write per-bucket JSONL files
   in the `data/datasets/buckets/sources/<source>/<bucket>/<tactic>/`
   layout. Re-runnable from source for upstream updates; the
   pre-built tarball is shipped as a GitHub Release.

3. **A privacy-audit harness** (`scripts/inversion/`) for owner-
   side model security testing. Implements:

    - **MIA offline baseline** (sample z-score, no shadow models — `--mia-method offline`)
    - **Shadow scoring** (`scripts/score_shadow.py` — LiRA Step 2 helper)
    - **MIA LiRA** (likelihood ratio, Carlini 2022 §4)
    - **MIA per-token loss** (MUSE 2024 default)
    - **MIA reference attack** (loss on assistant turn + zlib entropy, Carlini 2022)


---

## Distribution

**This is a GitHub-only distribution. It is not on PyPI.**

The reason: this is a **data bundle** (8,147 tracked JSONL files,
~50MB compressed), not a Python library. A 50MB `pip install
attacklm-dataset` goes against the typical "small Python library"
expectation of PyPI, and the data distribution is already handled
by `attacklm init` in the AttackLM package (which downloads
`attacklm-dataset.tar.gz` from GitHub Releases).

Users get the data in one of three ways:

```bash
# (Recommended) Through the AttackLM trainer
pip install "attacklm[all]"
attacklm init --yes   # downloads the GitHub Releases tarball

# Or as a git clone (gives you the data + the extractors + the audit harness)
git clone https://github.com/Veedubin/attacklm-dataset.git
cd attacklm-dataset

# Or as a Python wrapper only (no data — use this for the CLI tools)
pip install -e .
attacklm-dataset --help
```

The `attacklm-dataset` Python wrapper is small (3 source files, ~8KB).
The data is what makes the package large, and the data lives at
`data/datasets/buckets/sources/<source>/<bucket>/<tactic>/data.jsonl`
in this repo.

---

## Quickstart

### As a data consumer (most users)

Just use the AttackLM trainer. It downloads the pre-built tarball
and sets up the bucket layout for you:

```bash
pip install "attacklm[all]"
attacklm init --yes                          # downloads + organizes
attacklm balance --profile 7b-16gb          # builds a balanced subset
attacklm train -- --dataset data/datasets/balanced/balanced_7b-16gb.jsonl --epochs 10 --train
```

### As a researcher (audit harness)

```bash
git clone https://github.com/Veedubin/attacklm-dataset.git
cd attacklm-dataset
pip install -e ".[inversion]"

# Audit a model (e.g., your trained AttackLM)
attacklm-dataset audit --model /path/to/model --attack all --mia-method per_token
```

### As a data contributor (extractors)

```bash
git clone https://github.com/Veedubin/attacklm-dataset.git
cd attacklm-dataset
pip install -e ".[extract]"

# Build from upstream sources (clones ~6GB of Metasploit, Sigma, etc.)
attacklm-dataset init --from-source
```

See [`scripts/`](scripts/) for the 23 per-source extractors. Each
has a docstring with the source URI, license, and a usage example.

---

## CLI reference

`attacklm-dataset` is a thin CLI wrapper that dispatches to the
scripts in [`scripts/`](scripts/):

| Command | What it runs | Purpose |
| :--- | :--- | :--- |
| `attacklm-dataset init` | `scripts/init_pipeline.py` | Download pre-built tarball OR build from upstream sources |
| `attacklm-dataset balance` | `scripts/balance_buckets.py` | Build a balanced training subset (anti-source-bias) |
| `attacklm-dataset evolve` | `scripts/evolve_pairs.py` | Synthetically expand short pairs into complex reasoning examples |
| `attacklm-dataset audit` | `scripts/inversion_audit.py` | Run the privacy-audit harness on a model |
| `attacklm-dataset package` | `scripts/package_dataset.py` | Package the dataset for distribution (the GitHub Release tarball) |

Each command has its own flag set; `attacklm-dataset <cmd> --help`
shows them.

---

## Dataset composition

| Category | Source examples | Approx. pairs | License |
| :--- | :--- | :--- | :--- |
| **Offensive** | Metasploit, Atomic Red Team, MITRE Stockpile | 15,000+ | BSD-3 / MIT / Apache-2.0 |
| **Defensive** | Sigma, Elastic, Splunk, Mordor, ThreatHunter | 7,000+ | DRL-1.1 / Apache-2.0 |
| **AI Security** | Garak, Promptfoo, PromptMap | 100+ | MIT / Apache-2.0 |
| **Meta/IR** | NIST IR, Orchestrator | 500+ | Public Domain / MIT |
| **Synthetic** | LLM-generated, AttackLM synthetic, Replay | 2,000+ | GPL-3.0 / MIT |

**Total**: 24,652 records across 11 active sources (18 directories,
2 reserved for future; 3 high-risk sources in the
`archive/restricted-sources/` dir are gitignored and never
re-ingested).

### Directory layout

```
data/datasets/buckets/sources/
  <source>/                    # 18 source directories
    LICENSE.md                 # License excerpt + URI
    SOURCE.md                  # Source description + URI
    <bucket>/                  # Training bucket
      <tactic>/                # MITRE TAxxxx
        data.jsonl             # Human-sourced pairs
        data_llm.jsonl         # LLM-generated pairs
        data_synth.jsonl       # Deterministic templates
```

### Per-record provenance

Every record in the dataset carries these fields:

```json
{
  "source": "atomic-red-team",
  "source_uri": "https://github.com/redcanaryco/atomic-red-team",
  "license": "MIT",
  "license_uri": "https://opensource.org/licenses/MIT",
  "rights_contact": "see data/REMOVAL.md"
}
```

For the full per-record attribution (which record came from which
file in which upstream repo), see
[data/ATTRIBUTION.md](data/ATTRIBUTION.md).

## Decontamination

> Inspired by MAI-Thinking-1 §2.3.1 + §2.4.3 (Public Evaluation Decontamination + Deduplication) by The Microsoft AI Team, June 2026. Full doc: [docs/DECONTAM.md](docs/DECONTAM.md)

Identify and isolate training data that overlaps with public evaluation benchmarks using 20-gram fuzzy matching to prevent data leakage.

```bash
attacklm-dataset decontam --eval-set-dir data/eval_sets/ --quarantine-output data/quarantine.jsonl
```

### Memorization-aware epoch capping

> Inspired by MAI-Thinking-1 §2.5.4 (Mid-training Data Mixture — memorization-aware epoch capping) by The Microsoft AI Team, June 2026. Full doc: [docs/MEMORIZATION.md](docs/MEMORIZATION.md)

Analyze training data for verbatim memorization and structural repetition using a per-token NLL proxy to recommend optimal epoch caps per source.

```bash
attacklm-dataset memorization-report --model /path/to/model
```

## Held-out NLL evaluation

> Inspired by MAI-Thinking-1 §2.3 + §2.3.2 (Evaluation Methodology + Comparison of Accuracy and NLL Evaluations) by The Microsoft AI Team, June 2026. See [docs/HELD_OUT_NLL.md](docs/HELD_OUT_NLL.md).

Compute a cheap, contamination-resistant signal for model improvement using held-out Negative Log-Likelihood (NLL) across 5 weighted buckets.

```bash
python scripts/held_out_nll.py --model /path/to/model --aggregation-formula mimic_mai
```

---

## Privacy audit (research toolkit)

The `scripts/inversion/` package is the **owner-side model security
test**. The question is "if I ship this model, what can an attacker
extract from it?" — which the model owner wants to know *before*
shipping.

### Attack classes

| Attack class | Paper | What it measures |
| :--- | :--- | :--- |
| **Prefix-completion extraction** | Carlini et al. 2021 ([arXiv:2012.07805](https://arxiv.org/abs/2012.07805)) | Whether the model can regenerate verbatim training data given a prefix. |
| **MIA reference attack (loss on assistant turn + zlib)** | Carlini et al. 2022 ([arXiv:2112.03570](https://arxiv.org/abs/2112.03570)) | Whether per-record loss is lower on members than on non-members. |
| **MIA per-token loss** | Shi et al. (MUSE) 2024 ([arXiv:2407.06460](https://arxiv.org/abs/2407.06460)) | Same idea, normalized by suffix-token count (removes length bias). |
| **MIA offline baseline (sample z-score)** | Carlini et al. 2022 §3.2 ([arXiv:2112.03570](https://arxiv.org/abs/2112.03570)) | No shadow models; uses sample mean/std of audit-set NLL. Requires N ≥ 30. |
| **MIA LiRA (likelihood ratio)** | Carlini et al. 2022 §4 ([arXiv:2112.03570](https://arxiv.org/abs/2112.03570)) | The "10× more powerful at low FPR" MIA. Requires K shadow-model loss files. |

### Closed-loop audit

> Inspired by MAI-Thinking-1 §5.2 (TAP closed-loop) by The Microsoft AI Team, June 202 la 2026. Full doc: [docs/AUDIT_ITER.md](docs/AUDIT_ITER.md)

Perform iterative adversarial auditing to detect brittle memorization by generating semantic variants of fooling records.

```bash
attacklm-dataset audit --model <path> --audit-iter 3 --variant-strategies suffix,template
```

```bash
# From the AttackLM trainer
attacklm audit --attack all --mia-method per_token --model <path>

# Or directly from this repo
attacklm-dataset audit --model <path> --attack all --mia-method per_token

# Just prefix-completion extraction, 100 probes
attacklm audit --attack extraction --max-records 100

# Just LiRA MIA (requires pre-computed shadow loss files)
attacklm audit --attack mia --mia-method lira --lira-params shadow_params.json

# Quick offline MIA baseline (no shadow models, needs N >= 30 records)
attacklm audit --attack mia --mia-method offline --offline-z-threshold -1.5

# Score a shadow model on the audit set (LiRA Step 2)
python scripts/score_shadow.py --model models/shadow_0 --records data/audit_set.jsonl --output-dir losses/ --shadow-index 0
```

**Output structure** is `data/audit/<date>/` with:
- `summary.json` — high-level aggregate metrics (safe to share)
- `threshold.md` — documentation of the MIA threshold derivation
- `inversion_results.jsonl` — raw record-level reconstructions, including `prompt_text` and `best_reconstruction` fields for a self-contained evidence chain (**chmod 0600**, stay workspace-internal; training data carries
  BSD-3, DRL-1.1, and other terms that may not allow redistribution
  of raw samples)

**Design docs** (the "why" behind each design decision):
- [docs/ATTACK_TAXONOMY.md](docs/ATTACK_TAXONOMY.md) — the 3-attack
  taxonomy, the LLM MI = TDE collapse argument, and the CLI flag
  mapping
- [docs/LIRA.md](docs/LIRA.md) — LiRA design, K parameter guide,
  compute cost, threshold calibration
- [docs/MIA_THRESHOLD_CALIBRATION.md](docs/MIA_THRESHOLD_CALIBRATION.md) —
  threshold derivation
- [docs/PROBE_TOKEN_BUDGET.md](docs/PROBE_TOKEN_BUDGET.md) — probe
  length rationale
- [docs/AUDIT_RUNNER.md](docs/AUDIT_RUNNER.md) — overnight-runner plan

**Hermetic design.** The audit harness is hermetic — no network
calls, no GPU required, runs on a CPU laptop in minutes. Mocked
model loaders mean you can test the audit pipeline in CI without
owning a real model.

---

## Legal & provenance

This project implements privacy auditing techniques (training-data
extraction and membership-inference attacks) derived from published
academic research. All attack code is for **defensive, audit, and
academic-research use only**.

- **[RIGHTS.md](RIGHTS.md)** — full rights statement, the canonical
  paper list (8 papers), 11-source data attribution table, and the
  takedown-request process. This is the "trend" DMCA-style notice
  the user requested.
- **[PROVENANCE.md](PROVENANCE.md)** — per-file attribution template
  used by every Python file in this repo. Every attack code file
  has a `PROVENANCE` block at the top naming the paper, full author
  list, year/venue, arXiv URL, and rights-claim contact.
- **[data/ATTRIBUTION.md](data/ATTRIBUTION.md)** — per-record
  attribution for every record in the dataset.
- **[data/REMOVAL.md](data/REMOVAL.md)** — how to file a removal
  request if you're a rights-holder of one of the upstream sources.

**Rights-claim contact:** `veedubin.legal@example.com` (placeholder
— replace before public release).

---

## Testing

As of v0.5.0 there are 475+ tests across 10 test files, all hermetic:

```
tests/test_inversion_audit.py      (47 tests — full audit harness, MIA + extraction)
tests/test_lira.py                 (21 tests — LiRA scoring + shadow params)
tests/test_per_token_mia.py        (13 tests — per-token MIA scoring)
tests/test_probe_token_budget.py   (25 tests — probe length calibration)
tests/test_audit_bugfixes.py       (14 tests — regression for commit 4386995)
... (and 5 other utility/pipeline tests)
```

Run them all:

```bash
git clone https://github.com/Veedubin/attacklm-dataset.git
cd attacklm-dataset
pip install -e ".[inversion]"
pytest tests/ -v
```

The tests use `MagicMock` for the model and tokenizer, so the full
audit pipeline (Carlini probe, MIA scoring, LiRA scoring, threshold
derivation, JSONL output) can be exercised without a real model
or GPU. This is the regression net for "did someone break the audit
harness?".

---

## Related

- **[Veedubin/AttackLM](https://github.com/Veedubin/AttackLM)** —
  the trainer/tuner/TUI that consumes this dataset
- **[RIGHTS.md](RIGHTS.md)** — full rights statement + canonical paper list
- **[PROVENANCE.md](PROVENANCE.md)** — per-file attribution template
- **[CHANGELOG.md](CHANGELOG.md)** — full version history
