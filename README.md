# AttackLM Dataset

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A MITRE ATT&CK-grounded security fine-tuning dataset with 100% per-record provenance and license attribution.**

---

## Overview

The AttackLM Dataset provides 24,652 high-quality training pairs across 16 security sources, organized by source → bucket → MITRE tactic. Every record carries full provenance: source, source URI, license, license URI, and rights contact.

## Dataset Composition

| Category | Source Examples | Approx. Pairs | License |
| :--- | :--- | :--- | :--- |
| **Offensive** | Metasploit, Atomic Red Team, MITRE Stockpile | 15,000+ | BSD-3 / MIT / Apache-2.0 |
| **Defensive** | Sigma, Elastic, Splunk, Mordor, ThreatHunter | 7,000+ | DRL-1.1 / Apache-2.0 |
| **AI Security** | Garak, Promptfoo, PromptMap | 100+ | MIT / Apache-2.0 |
| **Meta/IR** | NIST IR, Orchestrator | 500+ | Public Domain / MIT |
| **Synthetic** | LLM-generated, AttackLM synthetic, Replay | 2,000+ | GPL-3.0 / MIT |

**Total**: 24,652 records across 16 active sources (18 directories, 2 reserved for future)

## Quickstart

```bash
# Install
pip install attacklm-dataset

# Initialize the dataset (downloads pre-built tarball)
attacklm-dataset init --yes

# Or build from source
attacklm-dataset init --from-source

# Build a balanced training subset
attacklm-dataset balance --profile 7b-16gb --preset red-team
```

## Inversion Audit

Audit your own model for memorized training data using Carlini's prefix-completion extraction + MIA loss+zlib scoring.

```bash
# Example audit run
attacklm-dataset audit --model <path_to_model> --probe-count 50
```

**Key Flags:**
- `--model`: Path to the model being audited.
- `--source-filter`: Filter probes to specific sources.
- `--probe-count`: Number of probes per source.
- `--top-k`: Number of top-k candidates to evaluate.
- `--mia-threshold-mode`: Threshold method (`median`, `percentile`, `holdout_file`).
- `--mia-percentile`: Percentile for thresholding (default: 5).

**Output Structure:**
Results are stored in `data/audit/<date>/` with the following files:
- `summary.json`: High-level aggregate metrics.
- `threshold.md`: Documentation of the MIA threshold derivation.
- `inversion_results.jsonl`: Raw record-level reconstructions.

**⚠️ WARNING**: Raw reconstructions are sensitive and must stay workspace-internal (`chmod 0600`). Only aggregate metrics should be exported. Training data carries various licenses (BSD-3, DRL-1.1, etc.); exporting raw samples may violate these terms.

**Further Reading:**
- See [docs/AUDIT_RUNNER.md](docs/AUDIT_RUNNER.md) for the overnight-runner plan.
- See [docs/PROBE_TOKEN_BUDGET.md](docs/PROBE_TOKEN_BUDGET.md) for the probe-length rationale.
- See [docs/MIA_THRESHOLD_CALIBRATION.md](docs/MIA_THRESHOLD_CALIBRATION.md) for the threshold calibration design.

## Per-Record Provenance

Every record in this dataset carries these fields:

```json
{
  "source": "atomic-red-team",
  "source_uri": "https://github.com/redcanaryco/atomic-red-team",
  "license": "MIT",
  "license_uri": "https://opensource.org/licenses/MIT",
  "rights_contact": "see data/REMOVAL.md"
}
```

## Directory Layout

```
data/datasets/buckets/sources/
  <source>/                    # 18 source directories
    LICENSE.md                 # License excerpt + URI
    SOURCE.md                  # Source description + URI
    <bucket>/                  # Training bucket
      <tactic>/               # MITRE TAxxxx
        data.jsonl             # Human-sourced pairs
        data_llm.jsonl         # LLM-generated pairs
        data_synth.jsonl       # Deterministic templates
```

## License

- **Project**: MIT
- **Data**: Mixed — see per-source `LICENSE.md` and [ATTRIBUTION.md](ATTRIBUTION.md) for full details
- **Rights holders**: See [data/REMOVAL.md](data/REMOVAL.md) for removal requests

## Related

- [AttackLM](https://github.com/Veedubin/AttackLM) — Training pipeline that consumes this dataset

## Documentation

- [PROBE_TOKEN_BUDGET.md](docs/PROBE_TOKEN_BUDGET.md) — Rationale for adaptive probe length.
- [MIA_THRESHOLD_CALIBRATION.md](docs/MIA_THRESHOLD_CALIBRATION.md) — Design of the MIA thresholding system.
- [AUDIT_RUNNER.md](docs/AUDIT_RUNNER.md) — Execution plan for large-scale audits.
