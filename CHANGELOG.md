## [0.5.0] — 2026-07-08 — LiRA Shadow-Model MIA (Track 2)

### Added
- **`scripts/inversion/lira.py`** — LiRA (Likelihood Ratio Attack) scoring module (Carlini 2022 §4). Implements `LiRAScore`, `GaussianParams`, `fit_gaussian`, `fit_gaussians_per_record`, `compute_lira_logit`, `score_lira`, `calibrate_lira_threshold`, `save_shadow_params`, `load_shadow_params`. LiRA is "10× more powerful at low FPR" than reference MIA per Carlini 2022 §4.4.
- **`scripts/inversion/shadow_train.py`** — CLI scaffold for the 3-step LiRA workflow (train K shadows, score each, fit Gaussians). Reads precomputed shadow loss files and produces `shadow_params.json` for use with `--mia-method lira`.
- **`--mia-method lira`** in `inversion_audit.py` — Replaces the "not yet implemented" error with working LiRA scoring. Requires `--lira-params <path>` pointing to the shadow_params.json file.
- **`--lira-k`** flag (default 16) — Number of shadow models for LiRA (K=1 = reference-model MIA, K=4 = cheap LiRA, K=16 = gold standard).
- **`--lira-params`** flag — Path to shadow_params.json (output of `inversion.shadow_train`). Required when `--mia-method lira` is used.
- **`--mia-threshold-mode lrt`** — Natural 0.0 threshold for LiRA (positive logit = member). No `holdout_file` needed.
- 17 new hermetic tests in `tests/test_lira.py` (fit_gaussian, fit_gaussians_per_record, gaussian_log_pdf, compute_lira_logit, calibrate_lira_threshold, score_lira, save/load shadow_params, integration with inversion_audit).
- New doc: `docs/LIRA.md` (design, workflow, compute cost, K parameter guide, storage cost, threshold, references).

### Changed
- Version bumped: 0.4.0 → 0.5.0
- `docs/ATTACK_TAXONOMY.md` §3.4 updated from "PLANNED" to "SHIPPED"; §5 CLI mapping updated with `--lira-params` flag.

### Threat model
- LiRA scoring requires precomputed shadow parameters (4 floats per record = 16 bytes). The shadow-model training is the user's responsibility and happens OUT-OF-BAND.
- Audit-time compute is identical to the reference attack: 1 forward pass on the target model per record, plus 4 floats of Gaussian parameters.

---

## [0.4.0] — 2026-07-08 — Per-Token MIA Scoring + Attack-Class CLI Redesign

### Added
- **`score_per_token()`** in `scripts/inversion/scoring.py` — per-suffix-token NLL scoring (MUSE 2023 default). Normalizes NLL by suffix token count, removing the length bias in full-record scoring. Includes `PerTokenMIAScore` dataclass with `nll_per_token`, `nll_total`, `num_suffix_tokens`, `suffix_text`, `membership_score`, `alpha`, and `zlib_ratio`.
- **`zscore_normalize()`** helper in `scripts/inversion/scoring.py` — Z-score normalization for cross-source MIA threshold calibration. Returns all zeros for constant distributions.
- **`_extract_assistant_turn()`** helper in `scripts/inversion/scoring.py` — extracts the final assistant message content from a record (the MUSE 2023 suffix).
- **`--attack {extraction,mia,all}`** CLI flag (default `all`) — replaces the binary `--probe-carlini` / `--probe-mia` flags with an attack-class-first design. See `docs/ATTACK_TAXONOMY.md` §5 for the full mapping.
- **`--mia-method {reference,zlib,per_token,lira,all}`** CLI flag (default `reference`) — selects which MIA scoring method to use. `per_token` writes `membership_score_per_token` and related fields to the output JSONL. `all` writes both full-record and per-token columns. `lira` exits with an error message (v0.5.0+).
- 13 new hermetic tests in `tests/test_per_token_mia.py` (per-token scoring, z-score normalization, integration with full-record scoring)
- New doc: `docs/ATTACK_TAXONOMY.md` (the 3-attack taxonomy, LLM MI=TDE collapse argument, CLI mapping, references)

### Changed
- **`--probe-carlini` / `--probe-mia` / `--no-probe-*`** are now **DEPRECATED** (since v0.4.0). They still work but emit `DeprecationWarning`. They will be removed in v0.6.0. The deprecation block in `main()` maps legacy flags to `--attack` values.
- Version bumped: `0.2.0` → `0.4.0` (this is a version-skip bug fix — the code was at v0.3.1+ but `pyproject.toml` said `0.2.0`)

### Fixed
- `pyproject.toml` version now matches the actual release series (was `0.2.0`, should be `0.4.0`)
- `scripts/inversion/__init__.py` `__version__` now matches `pyproject.toml` (was `0.1.0`, now `0.4.0`)

### Threat model
- Per-token MIA scoring operates on the assistant turn only (not the full record), which is the correct unit of analysis for membership inference on generative LLMs (MUSE 2023).
- No changes to raw output handling (chmod 0600 remains).

---


### Added
- `--mia-threshold-mode {median,percentile,holdout_file}` CLI flag (default `percentile`) and `--mia-percentile` (default 5) for MIA threshold calibration. Replaces the previous median-of-scores fallback which classified 50% of records as members by construction. See `docs/MIA_THRESHOLD_CALIBRATION.md`.
- `threshold.md` artifact in each audit output directory, documenting the MIA threshold derivation
- `scripts/run_overnight_audits.sh` — resume-safe runner for 1,100-record audits (~22h at K=20, ~5.5h at K=5). See `docs/AUDIT_RUNNER.md`.
- 25 new hermetic tests in `tests/test_probe_token_budget.py` (probe adaptive cap, percentile helper, MIA threshold mode CLI)
- 3 new docs: `docs/PROBE_TOKEN_BUDGET.md`, `docs/MIA_THRESHOLD_CALIBRATION.md`, `docs/AUDIT_RUNNER.md`

### Changed
- `max_new_tokens` for Carlini probe is now adaptive: `min(256, max(64, 2*suffix_token_count))` per `docs/PROBE_TOKEN_BUDGET.md` (Carlini 2021, MUSE 2023, DecodingTrust §C.2 all use 256). Previous hard-coded 64 was smaller than the median suffix length of our audit data (75-106 tokens), suppressing real matches.
- `--probe-count` is now per-source (was a global cap that only probed atomic-red-team by alphabetical ordering). Fixes `8d34ba9`.

### Fixed
- Probe truncation: `max_new_tokens=64` could not produce exact matches on suffixes >64 tokens
- MIA threshold calibration: median-of-scores classified 75/150 = 50% of records as members by construction
- Per-source probing: all 3 sources now get probed equally

### Threat model
- Raw audit outputs (`inversion_results.jsonl`) stay workspace-internal (chmod 0600)
- Aggregate metrics (`summary.json`, `exportable_summary.json`, `threshold.md`) are safe to share
- BSD-3-Clause (Metasploit) and DRL-1.1 (Sigma) record contents must never be exported; only their per-source statistics

---

## [0.2.0] — 2026-07-06 — Inversion Audit Harness

### Added
- Inversion-attack audit harness (`scripts/inversion_audit.py`) with Carlini prefix-completion extraction and MIA loss+zlib scoring
- Helper modules: `scripts/inversion/{provenance,model_loader,probe,scoring,reporting}.py`
- Per-record license carry-through, restricted-source denylist (RTA, infection_monkey, BPL), and three-tier output classification (INTERNAL_RAW/INTERNAL_SUMMARY/EXPORTABLE_SUMMARY)
- Hermetic test suite (`tests/test_inversion_audit.py`) with 30+ tests, all mocked (no GPU required)
- `[inversion]` optional dependency group in pyproject.toml (transformers, torch, nltk, llama-cpp-python)

### Changed
- No data changes; backward compatible with v0.1.0 datasets

---

## [0.1.0] — 2026-07-06 — Initial Release (Split from AttackLM)

### Added
- Initial release of attacklm-dataset as a standalone package
- 24,652 training pairs across 16 active security sources
- 18 source directories (2 reserved for future: azure-pyrit, cyberark-fuzzyai)
- Per-record provenance with license, source URI, and attribution fields
- CLI with `init`, `balance`, `evolve`, `audit`, and `package` commands
- All extractor scripts for building from upstream sources
- Data tooling: bucket loader, balance engine, pair evolution, replay mixer

### Changed
- Split from AttackLM v0.10.1 — dataset is now an independent package
- `init_pipeline.py`: Updated `DEFAULT_DATASET_URL` to point to `attacklm-dataset` releases