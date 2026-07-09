# Rights and Usage Statement — `attacklm-dataset`

**Effective:** 2026-07-09
**Repository:** https://github.com/Veedubin/attacklm-dataset
**Maintainer:** Veedubin (GitHub) — `veedubin.legal@example.com` (replace before any public release; current address is a placeholder)

This document establishes the legal and ethical framework for the `attacklm-dataset` repository and its associated software. The primary purpose of this project is to facilitate **defensive privacy auditing** and **membership-inference attack (MIA) research** on language models that the user owns or has explicit authorization to audit. By providing standardized implementations of known attack vectors, this project enables model developers to quantify the leakage of training data and implement robust defenses to protect user privacy.

This statement supplements (does not replace) the existing `data/ATTRIBUTION.md`, `data/LEGAL.md`, and `data/REMOVAL.md` files. In case of conflict, the per-source attribution in `data/ATTRIBUTION.md` takes precedence for the specific record(s) in question.

---

## 1. What is in this repository

This repository contains a suite of tools, data, and documentation designed for AI security research, specifically:

- **Attack implementations** — Python code implementing membership-inference and training-data extraction attacks. The two primary attack classes are (a) **training-data extraction** (Carlini et al. 2021) and (b) **membership-inference attacks** (Carlini et al. 2022). See Section 2 for the canonical citations.
- **Dataset ingest pipelines** — Scripts that collect, clean, deduplicate, bucketize, and license-tag data from openly-licensed upstream security corpora.
- **Audit harnesses** — Evaluation frameworks that score the success of attacks against a target model and emit three-tier reporting (internal-raw, internal-summary, exportable-summary).
- **Provenance metadata** — Per-record attribution records for every JSONL training pair, every audit-output record, and every source-file in `scripts/`.

---

## 2. Authoritative sources (canonical papers)

The attack implementations in this repository are derived from the following academic research:

| Attack class                | Paper                                                                                                                 | Year | Authors                                                                                                                                |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Training-data extraction    | [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)                                | 2021 | N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, U. Erlingsson, A. Oprea, C. Raffel |
| Membership-inference (loss) | [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570) (IEEE S&P 2022)                | 2022 | N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramer                                                                              |
| LiRA (likelihood-ratio MIA) | Same paper, §4                                                                                                        | 2022 | (same as above)                                                                                                                        |
| zlib baseline               | Same paper, §3                                                                                                        | 2022 | (same as above)                                                                                                                        |
| Reference attack baseline   | Same paper, §3                                                                                                        | 2022 | (same as above)                                                                                                                        |
| Production extraction       | [Scalable Extraction of Training Data from (Production) LLMs](https://arxiv.org/abs/2311.17035)                       | 2023 | M. Nasr, N. Carlini, J. Hayase, M. Jagielski, A. F. Cooper, D. Ippolito, C. A. Choquette-Choo, E. Wallace, F. Tramèr, K. Lee             |
| MUSE eval framework         | [MUSE: Machine Unlearning Six-Way Evaluation](https://arxiv.org/abs/2407.06460) (ICLR 2025)                           | 2024 | W. Shi, J. Lee, Y. Huang, S. Malladi, J. Zhao, A. Holtzman, D. Liu, L. Zettlemoyer, N. A. Smith, C. Zhang                              |
| Original MIA (shadow model) | [Membership Inference Attacks against Machine Learning Models](https://arxiv.org/abs/1610.05820) (IEEE S&P 2017)       | 2017 | R. Shokri, M. Stronati, C. Song, V. Shmatikov                                                                                            |
| Per-example loss attack     | [Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting](https://arxiv.org/abs/1709.01604)         | 2018 | S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha                                                                                            |

**Note on official code:** None of the four Carlini/Nasr papers above released official author-maintained code. The implementations in this repository are **clean-room reimplementations** based on the published paper text. The MUSE repository is the exception (Shi et al. 2024 has an official repo at https://github.com/woooooda/MUSE_unlearning); we reuse the Carlini 2021 verbatim-memorization test as defined in that repo.

**Note on the original papers' license terms:** Academic papers themselves are typically licensed under the default academic-copyright regime (all rights reserved by the authors / their publishers). The *ideas and algorithms* in the papers are not copyrightable in the US/EU; the *text and figures* are. The implementations in this repository do not reproduce paper text or figures — they translate the algorithms from pseudocode into Python.

---

## 3. Third-party data sources

All data ingested by this project is drawn from **openly-licensed** upstream sources. Each source has its own `LICENSE.md` and `SOURCE.md` in its per-source directory under `data/datasets/buckets/sources/<source>/`. The full list lives in `data/ATTRIBUTION.md` and is regenerated by `scripts/rebuild_manifest.py`.

The eleven primary sources, grouped by license, are:

- **BSD-3-Clause**: Metasploit Framework (https://github.com/rapid7/metasploit-framework)
- **DRL-1.1** (Detection Rule License): Sigma Rules (https://github.com/SigmaHQ/sigma)
- **Apache-2.0**: Atomic Red Team (https://github.com/redcanaryco/atomic-red-team), Elastic Detection Rules (https://github.com/elastic/detection-rules), Splunk Security Content (https://github.com/splunk/security_content), Mordor (https://github.com/OTRF/Security-Datasets), ThreatHunter-Playbook (https://github.com/hunters-forge/ThreatHunter-Playbook), Caldera Plugins (https://github.com/mitre/caldera-plugins), Stockpile (https://github.com/mitre/stockpile), AI Tools corpus, CyberLLMInstruct, NYU CTF Bench
- **MIT**: Primus-Seed (https://huggingface.co/datasets/trendmicro-ailab/Primus-Seed), AI tools (Garak, Promptfoo, PromptMap)
- **Public Domain**: NIST IR documents
- **Restricted (NOT publicly distributed)**: RTA (Red Team Automation), Infection Monkey, BPL (Binary Playground Library) — these are denylisted from the public dataset and live only at `archive/restricted-sources/` (gitignored)

The user must comply with each source's license when redistributing the resulting derived data. See `data/ATTRIBUTION.md` and `data/LEGAL.md` for the per-record and per-source license metadata.

---

## 4. Permitted use

The software and datasets provided here are intended strictly for **defensive, audit, and academic-research** purposes.

- **Permitted:**
  - Auditing your own models for training-data leakage.
  - Testing defenses against membership-inference and extraction attacks.
  - Academic study of LLM privacy, memorization, and unlearning.
  - Building benchmark datasets for unlearning research (the MUSE 2024 use case).
  - Reproducing published research results in a controlled lab environment.

- **Prohibited:**
  - Using these tools to attack models you do not own or do not have explicit written authorization to audit.
  - Attempting to extract private data from third-party proprietary services.
  - Any offensive application of these techniques against production systems.
  - Removing attribution or license metadata from derived datasets.
  - Republishing data from sources with `no-redistribution` clauses (e.g., RTA, Infection Monkey, BPL — see Section 3).

The maintainer disclaims all liability for misuse of the code or data in this repository.

---

## 5. Rights claims and takedown requests

The maintainers of this project respect the intellectual property rights of all contributors and data providers. If you are a rights-holder and believe that any material in this repository violates your copyright or license terms, please contact us immediately.

**Contact:** `veedubin.legal@example.com` (placeholder — replace before public release)

Upon receipt of a valid request, the maintainer commits to:

1. **Acknowledge** the request within 7 calendar days.
2. **Investigate** the claim and verify it against the per-record provenance in `data/ATTRIBUTION.md`.
3. **Act** within 30 calendar days:
   - If the claim is valid, **remove** the offending records from the public dataset.
   - If the claim is valid, **re-attribute** the records to the correct author/license (if the issue is mis-attribution, not wrongful inclusion).
   - If the claim is invalid, **document** the reason for the rejection and offer to discuss further.
4. **Audit** the removal across all derived artifacts (audit reports, training checkpoints if any, model weights if any).

If you are unable to reach the maintainer by email, please open a GitHub issue at https://github.com/Veedubin/attacklm-dataset/issues with the tag `rights-claim`. GitHub issues are public by default — if your claim is sensitive, please use the email channel.

---

## 6. License

This repository's own code is licensed under the **MIT License** (see `LICENSE`). The third-party data ingested by this project is licensed under its original upstream license (see `data/ATTRIBUTION.md`).

The maintainer makes no claim of authorship over:
- The attack algorithms themselves (they are mathematical ideas, not copyrightable expression).
- The upstream data ingested by the pipeline (each source has its own author and license).
- The academic papers cited in Section 2 (those are the works of their respective authors and publishers).

The maintainer claims authorship only over:
- The specific Python implementation of the algorithms (the code in `scripts/` and `src/`).
- The audit harness, the three-tier reporting logic, and the threshold-calibration logic.
- The documentation, design decisions, and architectural choices captured in `docs/`.

---

## 7. Disclaimer of warranty

THE SOFTWARE AND DATASETS IN THIS REPOSITORY ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 8. Contact

For questions about this document, contact the maintainer via the email address above or open a GitHub issue.

**Last reviewed:** 2026-07-09
**Next review:** 2026-10-09 (quarterly)
