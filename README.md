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