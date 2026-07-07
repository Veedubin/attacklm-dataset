## [0.3.1] — 2026-07-07 — Inversion Audit Fixes + Runner

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