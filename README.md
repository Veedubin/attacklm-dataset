# attacklm-dataset

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Provenance: 100%](https://img.shields.io/badge/provenance-100%25-brightgreen.svg)](data/ATTRIBUTION.md)
[![Sources: 19](https://img.shields.io/badge/sources-19-blue.svg)](#dataset-composition)
[![Tests: 722](https://img.shields.io/badge/tests-722-brightgreen.svg)](#testing)
[![Distribution: GitHub](https://img.shields.io/badge/distribution-GitHub-yellow.svg)](#distribution)

**A MITRE ATT&CK- and ATLAS-grounded security fine-tuning dataset, the
extraction pipeline that builds it from 19 upstream security sources, and a
privacy-audit harness (Carlini 2021 training-data extraction + four
membership-inference methods).**

This repository is the **data and research-toolkit** half of the AttackLM
project. The trainer, tuner, and evaluation suite live in the companion
package [Veedubin/AttackLM](https://github.com/Veedubin/AttackLM). The two were
split in v0.11.0 so the license-aware data and the defensive audit code stay
isolated from the general-purpose training infrastructure.

Every record carries complete provenance, every attack implementation cites the
paper it derives from, and the whole audit harness is hermetic — no network, no
GPU, runs on a CPU laptop in minutes.

---

## Contents

The repository ships three things:

1. **The dataset** — 26,459 instruction/response pairs grounded in MITRE ATT&CK
   and ATLAS, drawn from 19 vetted source directories. Every record carries
   `source`, `source_uri`, `license`, `license_uri`, and a rights-contact
   pointer. Three high-risk upstreams (RTA, Infection Monkey, BPL) are
   deliberately excluded and never re-ingested.

2. **The extraction pipeline** — 20 `extract_*.py` scripts that read upstream
   security projects (Metasploit, Atomic Red Team, Sigma, Elastic, Splunk,
   Mordor, MITRE ATLAS, and others) and emit per-bucket JSONL under
   `data/datasets/buckets/sources/<source>/<bucket>/<tactic>/`. The pipeline is
   re-runnable to track upstream changes; the pre-built corpus ships as a
   GitHub Release tarball.

3. **The privacy-audit harness** (`scripts/inversion/`) — owner-side model
   security testing that answers "if I ship this model, what can an attacker
   recover from it?" It implements training-data extraction plus four
   membership-inference attacks (offline baseline, reference, per-token, and
   LiRA).

---

## Distribution

**GitHub-only. This package is intentionally not published to PyPI.**

The payload is a ~50 MB data bundle (8,147 tracked JSONL files), not a small
Python library — the profile PyPI expects. Data distribution is instead handled
by `attacklm init`, which pulls the release tarball from GitHub Releases. You
can obtain the dataset three ways:

```bash
# 1. Through the AttackLM trainer (recommended) — downloads the release tarball
pip install "attacklm[all]"
attacklm init --yes

# 2. Git clone — data + extractors + audit harness
git clone https://github.com/Veedubin/attacklm-dataset.git

# 3. Python wrapper only (CLI tools, no data)
pip install -e .
attacklm-dataset --help
```

The Python wrapper is tiny (three source files, ~8 KB). The data is what makes
the package large; it lives at
`data/datasets/buckets/sources/<source>/<bucket>/<tactic>/data.jsonl`.

---

## Quickstart

**As a data consumer (most users).** Let the trainer fetch and organize
everything:

```bash
pip install "attacklm[all]"
attacklm init --yes                                # download + organize
attacklm balance --profile 7b-16gb                 # build a balanced subset
attacklm train -- --dataset data/datasets/balanced/balanced_7b-16gb.jsonl --epochs 10 --train
```

**As a researcher (audit harness):**

```bash
git clone https://github.com/Veedubin/attacklm-dataset.git && cd attacklm-dataset
pip install -e ".[inversion]"
attacklm-dataset audit --model /path/to/model --attack all \
    --mia-method per_token --mia-threshold-mode percentile --mia-percentile 5
```

**As a data contributor (extractors):**

```bash
git clone https://github.com/Veedubin/attacklm-dataset.git && cd attacklm-dataset
pip install -e ".[extract]"
attacklm-dataset init --from-source                # clones ~6 GB of upstreams
```

Each of the 20 extractors in [`scripts/`](scripts/) documents its source URI,
license, and a usage example in its module docstring.

---

## CLI reference

`attacklm-dataset` is a thin dispatcher over the scripts in [`scripts/`](scripts/):

| Command | Runs | Purpose |
| :--- | :--- | :--- |
| `attacklm-dataset init` | `scripts/init_pipeline.py` | Download the pre-built tarball, or build from upstream sources |
| `attacklm-dataset balance` | `scripts/balance_buckets.py` | Build a source-balanced training subset |
| `attacklm-dataset evolve` | `scripts/evolve_pairs.py` | Expand short pairs into richer reasoning examples |
| `attacklm-dataset audit` | `scripts/audit_dataset.py` | Run the privacy-audit harness against a model |
| `attacklm-dataset package` | `scripts/package_dataset.py` | Build the GitHub Release distribution tarball |

Run `attacklm-dataset <command> --help` for each command's flags. The analysis
tools — decontamination, memorization reporting, held-out NLL, shadow scoring —
run directly as scripts (`python scripts/<tool>.py ...`) and are documented in
the sections below.

---

## Dataset composition

| Category | Source examples | Approx. pairs | License |
| :--- | :--- | :--- | :--- |
| **Offensive** | Metasploit, Atomic Red Team, MITRE Stockpile | 15,000+ | BSD-3 / MIT / Apache-2.0 |
| **Defensive** | Sigma, Elastic, Splunk, Mordor, ThreatHunter | 7,000+ | DRL-1.1 / Apache-2.0 |
| **AI security** | Garak, Promptfoo, PromptMap | 100+ | MIT / Apache-2.0 |
| **AI security (ATLAS)** | MITRE ATLAS, ATLAS Arsenal | 1,800+ | Apache-2.0 |
| **Meta / IR** | NIST IR, Orchestrator | 500+ | Public Domain / MIT |
| **Synthetic** | LLM-generated, AttackLM synthetic, Replay | 2,000+ | GPL-3.0 / MIT |

**Total: 26,459 records across 19 source directories.** (Two directories,
`azure-pyrit` and `cyberark-fuzzyai`, are reserved and currently empty; three
high-risk sources under `archive/restricted-sources/` are gitignored and never
ingested.)

### MITRE ATLAS (since v0.10.0)

[MITRE ATLAS](https://atlas.mitre.org/) — the Adversarial Threat Landscape for
Artificial-Intelligence Systems — is MITRE's threat matrix for AI, the
AI-security sibling of ATT&CK. The `mitre-atlas` source contributes **1,807
case-study-grounded pairs** across 16 ATLAS tactics (from
`ai_attack_adaptation` and `ai_model_access` through `impact` and
`resource_development`), extracted from
[mitre-atlas/atlas-data](https://github.com/mitre-atlas/atlas-data) (Apache-2.0,
© The MITRE Corporation) by `scripts/extract_mitre_atlas.py`. The companion
`mitre-atlas-arsenal` source adds 20 pairs from ATLAS Arsenal, MITRE's catalogue
of real-world AI-attack tools. In the bucket loader, `atlas` is its own category
alongside the ATT&CK-tactic and defensive categories.

### Directory layout

```
data/datasets/buckets/sources/
  <source>/                    # 19 source directories
    LICENSE.md                 # license excerpt + URI
    SOURCE.md                  # source description + URI
    <bucket>/                  # training bucket
      <tactic>/                # MITRE tactic, e.g. TA0001
        data.jsonl             # human-sourced pairs
        data_llm.jsonl         # LLM-generated pairs
        data_synth.jsonl       # deterministic templates
```

### Per-record provenance

Every record carries its origin inline:

```json
{
  "source": "atomic-red-team",
  "source_uri": "https://github.com/redcanaryco/atomic-red-team",
  "license": "MIT",
  "license_uri": "https://opensource.org/licenses/MIT",
  "rights_contact": "see data/REMOVAL.md"
}
```

The full record-to-upstream-file mapping is in
[data/ATTRIBUTION.md](data/ATTRIBUTION.md).

---

## Data-quality tooling

Three analysis tools keep the corpus honest. Each is a standalone script.

**Decontamination** — isolate training records that overlap public evaluation
benchmarks (20-gram fuzzy matching) so they cannot leak into your eval scores:

```bash
python scripts/decontam.py --eval-set-dir data/eval_sets/ --quarantine-output data/quarantine.jsonl
```

**Memorization-aware epoch capping** — flag verbatim memorization and structural
repetition via a per-token NLL proxy, and recommend a per-source epoch cap:

```bash
python scripts/memorization_report.py --model /path/to/model
```

**Held-out NLL evaluation** — a cheap, contamination-resistant training signal:
held-out negative log-likelihood across five weighted buckets:

```bash
python scripts/held_out_nll.py --model /path/to/model --aggregation-formula mimic_mai
```

*(The decontamination, epoch-capping, and NLL methodologies are adapted from the
public MAI-Thinking-1 report, The Microsoft AI Team, June 2026.)*

---

## Privacy audit (research toolkit)

`scripts/inversion/` is an **owner-side** model security test: it quantifies how
much of the training data a shipped model would leak, so the owner can measure
and mitigate it before release. All attack code is a clean-room reimplementation
of published research, carries a per-file provenance block, and is intended for
**defensive, audit, and academic use only** against models the auditor owns or
is authorized to test.

| Attack class | Paper | What it measures |
| :--- | :--- | :--- |
| **Prefix-completion extraction** | Carlini et al. 2021 ([2012.07805](https://arxiv.org/abs/2012.07805)) | Whether the model regenerates verbatim training data from a prefix |
| **MIA — reference (loss + zlib)** | Carlini et al. 2022 ([2112.03570](https://arxiv.org/abs/2112.03570)) | Whether per-record loss is lower on members than non-members |
| **MIA — per-token loss** | Shi et al. (MUSE) 2024 ([2407.06460](https://arxiv.org/abs/2407.06460)) | The same signal, normalized by suffix length to remove length bias |
| **MIA — offline baseline (z-score)** | Carlini et al. 2022 §3.2 | Sample mean/std of audit-set NLL; no shadow models (needs N ≥ 30) |
| **MIA — LiRA (likelihood ratio)** | Carlini et al. 2022 §4 | The high-power, low-FPR attack; needs K shadow-model loss files |

Run the harness end to end, or a single attack:

```bash
# Full audit
attacklm-dataset audit --model <path> --attack all \
    --mia-method per_token --mia-threshold-mode percentile --mia-percentile 5

# Prefix-completion extraction, 100 probes
attacklm-dataset audit --model <path> --attack extraction --max-records 100

# LiRA MIA (requires pre-computed shadow loss files)
attacklm-dataset audit --model <path> --attack mia --mia-method lira --lira-params shadow_params.json

# Offline MIA baseline (no shadow models, needs N ≥ 30)
attacklm-dataset audit --model <path> --attack mia --mia-method offline --offline-z-threshold -1.5

# Iterative closed-loop audit (semantic variants surface brittle memorization)
attacklm-dataset audit --model <path> --audit-iter 3 --variant-strategies suffix,template

# Score a shadow model on the audit set (LiRA step 2)
python scripts/score_shadow.py --model models/shadow_0 --records data/audit_set.jsonl --output-dir losses/ --shadow-index 0
```

Results land in `data/audit/<date>/`:

- `summary.json` — aggregate metrics, safe to share
- `threshold.md` — how the MIA decision threshold was derived
- `inversion_results.jsonl` — raw record-level reconstructions (`prompt_text`,
  `best_reconstruction`). This stays **workspace-internal** (mode `0600`):
  training data carries BSD-3, DRL-1.1, and other terms that may forbid
  redistribution of raw samples.

*(The closed-loop / TAP auditing approach is adapted from the public
MAI-Thinking-1 report, The Microsoft AI Team, June 2026.)*

---

## Testing

722 tests across 19 files, all hermetic — the model and tokenizer are mocked, so
the full pipeline (Carlini probe, MIA scoring, LiRA, threshold derivation, JSONL
output) runs in CI without a real model or GPU.

```bash
git clone https://github.com/Veedubin/attacklm-dataset.git && cd attacklm-dataset
pip install -e ".[inversion]"
pytest tests/ -v
```

Coverage spans the audit harness and MIA/extraction attacks, the closed-loop
auditor, LiRA shadow models, the ATLAS extractor, the bucket loader and manifest
builder, decontamination, memorization reporting, held-out NLL, and the schema /
import / smoke / CLI checks.

---

## Documentation & legal

Internal methodology and design notes (attack taxonomy, audit-runner design, MIA
threshold calibration, LiRA design, decontamination, memorization capping,
held-out NLL, closed-loop audit) are maintainer-local and not distributed. The
public documents are:

| Document | Covers |
| :--- | :--- |
| [data/ATTRIBUTION.md](data/ATTRIBUTION.md) | Per-record attribution and counts |
| [data/LEGAL.md](data/LEGAL.md) | License terms per source |
| [data/REMOVAL.md](data/REMOVAL.md) | Takedown / removal process |
| [RIGHTS.md](RIGHTS.md) | Rights and usage statement, upstream authors, canonical paper list |
| [PROVENANCE.md](PROVENANCE.md) | Per-file provenance template for the audit code |
| [SECURITY.md](SECURITY.md) | Security policy and vulnerability reporting |

This project implements privacy-auditing techniques (training-data extraction
and membership inference) derived from published academic research, for
defensive and research use only. Every attack file carries a `PROVENANCE` block
naming the paper, authors, venue, and arXiv URL.

**Rights, licensing, or takedown requests:** open an issue at
<https://github.com/Veedubin/attacklm-dataset/issues> (see
[data/REMOVAL.md](data/REMOVAL.md) for the takedown process). Security reports go
through a private advisory — see [SECURITY.md](SECURITY.md).

---

## Related

- **[Veedubin/AttackLM](https://github.com/Veedubin/AttackLM)** — the trainer,
  tuner, TUI, and evaluation suite that consume this dataset
- **[CHANGELOG.md](CHANGELOG.md)** — full version history
