## [Unreleased] — 2026-07-08 — MIA Track 2: per-token + LiRA + audit screen

This is a single combined entry for the work on the
`wip/inversion-audit-track2-2026-07-08` branch. It is NOT yet a tagged
release; the version will be bumped to **v0.4.0** when this branch is
merged and tagged. The per-token MIA scoring and the LiRA shadow-model
MIA are the two major new capabilities. The Audit TUI screen and the
tooltips retrofit are shipped in the `AttackLM/attacklm-gui` repo as
**v0.11.0** (separate package, same release cycle).

### Added (in `attacklm-dataset`)
- **`score_per_token()`** in `scripts/inversion/scoring.py` — per-suffix-token NLL scoring (MUSE 2023 default). Normalizes NLL by suffix token count, removing the length bias in full-record scoring. Returns a `PerTokenMIAScore` dataclass with `nll_per_token`, `nll_total`, `num_suffix_tokens`, `suffix_text`, `membership_score`, `alpha`, and `zlib_ratio`.
- **`zscore_normalize()`** helper in `scripts/inversion/scoring.py` — Z-score normalization for cross-source MIA threshold calibration. Returns all zeros for constant distributions.
- **`_extract_assistant_turn()`** helper in `scripts/inversion/scoring.py` — extracts the final assistant message content from a record (the MUSE 2023 suffix).
- **`--attack {extraction,mia,all}`** CLI flag (default `all`) — replaces the binary `--probe-carlini` / `--probe-mia` flags with an attack-class-first design. See `docs/ATTACK_TAXONYMARKDOWN.md` §5 for the full mapping.
- **`--mia-method {reference,zlib,per_token,lira,all}`** CLI flag (default `reference`) — selects which MIA scoring method to use. `per_token` writes `membership_score_per_token` and related fields to the output JSONL. `all` writes both full-record and per-token columns. `lira` now works (was exiting with an error in the v0.4.0 draft; fixed in this branch).
- **LiRA scoring module** at `scripts/inversion/lira.py` — `LiRAScore`, `GaussianParams`, `fit_gaussian`, `fit_gaussians_per_record`, `gaussian_log_pdf`, `compute_lira_logit`, `score_lira`, `calibrate_lira_threshold`, `save_shadow_params`, `load_shadow_params`. LiRA is "10× more powerful at low FPR" than the reference attack per Carlini 2022 §4.4.
- **`shadow_train.py`** CLI scaffold at `scripts/inversion/shadow_train.py` — reads precomputed shadow loss files and produces `shadow_params.json` for use with `--mia-method lira`.
- **`--lira-k`** flag (default 16) — Number of shadow models for LiRA. K=1 = reference-model MIA, K=4 = cheap LiRA, K=16 = gold standard.
- **`--lira-params`** flag — Path to shadow_params.json (output of `inversion.shadow_train`). Required when `--mia-method lira` is used.
- **`--mia-threshold-mode lrt`** — Natural 0.0 threshold for LiRA (positive logit = member). No `holdout_file` needed.
- 13 new hermetic tests in `tests/test_per_token_mia.py` (per-token scoring, z-score normalization, integration with full-record scoring).
- 21 new hermetic tests in `tests/test_lira.py` (fit_gaussian, fit_gaussians_per_record, gaussian_log_pdf, compute_lira_logit, calibrate_lira_threshold, score_lira, save/load shadow_params, integration with inversion_audit).
- New doc: `docs/ATTACK_TAXONOMY.md` (~255 lines, the 3-attack taxonomy, the LLM MI=TDE collapse argument, CLI mapping, references).
- New doc: `docs/LIRA.md` (~155 lines, design, workflow, compute cost, K parameter guide, storage cost, threshold, references).

### Changed
- **`--probe-carlini` / `--probe-mia` / `--no-probe-*`** are now **DEPRECATED** (since this release). They still work but emit `DeprecationWarning`. They will be removed in a future major version. The deprecation block in `main()` maps legacy flags to `--attack` values.
- `pyproject.toml` version fixed: was `0.2.0`, will become `0.4.0` at release time. The actual code is at v0.3.1+ but `pyproject.toml` was stale (HANDOFF.md 2026-07-06 flagged this).
- `scripts/inversion/__init__.py` `__version__` will become `0.4.0` at release time (was `0.3.1`).
- `src/attacklm_dataset/__init__.py` `__version__` will become `0.4.0` at release time (was `0.3.1`).
- `docs/ATTACK_TAXONOMY.md` §3.4 (LiRA) is in the doc as SHIPPED (was PLANNED in the v0.4.0 draft).
- The CLI exit code is unchanged. Raw audit outputs (`inversion_results.jsonl`) stay workspace-internal (chmod 0600). Aggregate metrics remain safe to share.

### Threat model
- Per-token MIA scoring operates on the assistant turn only (not the full record), which is the correct unit of analysis for membership inference on generative LLMs (MUSE 2023).
- LiRA scoring requires precomputed shadow parameters (4 floats per record = 16 bytes). The shadow-model training is the user's responsibility and happens OUT-OF-BAND; `docs/LIRA.md` documents the workflow.
- LiRA audit-time compute is identical to the reference attack: 1 forward pass on the target model per record, plus 4 floats of Gaussian parameters.

### Added (in `AttackLM/attacklm-gui` — same release cycle, separate package)
- **`screens/audit.py`** — new Audit screen with 2 tabs (Extraction / MIA). Each tab is a form that constructs the `attacklm audit` CLI command. TUI is a thin wrapper around the CLI (no direct `attacklm-dataset` import).
- **`widgets/tooltips.py`** — centralized tooltip text for all TUI widgets. ~30 entries. `attach_tooltip(widget, key)` helper for one-line tooltip attachment.
- **`attacklm audit` subcommand** in `src/attacklm/cli.py` — bridges the TUI/CLI to `attacklm-dataset/scripts/inversion_audit.py`. Forwards all audit flags (--attack, --mia-method, --mia-threshold-mode, --mia-percentile, --model, --dataset-root, --source-filter, --top-k, --max-new-tokens, --temperature, --max-records, --dry-run).
- **Tooltips retrofit** on existing TUI screens: 9 main menu buttons, all high-traffic train form inputs (epochs, batch_size, lora_r, lora_alpha, galore_rank, max_length, spectrum, use_qgalore, use_dora), and all command form Back buttons.
- **DEFAULT_CSS** in `app.py` for Tooltip styling (background, border, padding, max-width).
- **App.tooltip_delay = 0.5s** — hover delay before tooltips show.
- 7 new tests in `attacklm-gui/tests/test_audit.py` (Audit screen mount, widget presence, tooltip coverage, dict-key presence, no-op attachment).
- 2 new structural tests in `attacklm-gui/tests/test_gui.py::TestTooltipsRetrofit`.

### Drive-by bug fixes (not in the MIA Track 2 spec)
- `AttackLM/attacklm-gui/src/attacklm_gui/presets.py` — `FP8 (H100/Blackwell)` was creating an invalid filename (`fp8_(h100/blackwell).json`) on Linux because the slash survives the previous `lower().replace(' ', '_')` slugify. Now uses a proper regex slugify via a new `_slugify()` helper. Fixes a real bug that affected all users on first run.
- `AttackLM/attacklm-gui/src/attacklm_gui/screens/train_form.py` — `Select(value=3)` for `deepspeed_stage` was silently broken under Textual 8.x (the value is silently overwritten with `Select.NULL` when options are tuples). Dropped the explicit value; the downstream code (`values.get("deepspeed_stage", 3)`) already defaults to 3.

### Test state at session end
- `attacklm-dataset`: 106/106 tests pass (47 inversion_audit + 25 probe_token_budget + 13 per_token_mia + 21 lira). Ruff clean.
- `AttackLM/attacklm-gui`: 26/26 tests pass (7 audit + 19 gui). 1 pre-existing ruff issue in `train_live.py` (F841 unused `trend_class`) — NOT in M2 scope, left alone.

### Reference
- Per architect plan approved 2026-07-08 (deepseek-v4-pro:cloud).
- Memory: `d4657ab9-53d4-49a4-b4bd-3de7816b3868` (architectural decision), `88c25f43-5d1d-4756-804e-4b0ad6b1dc19` (MIA research), `fb19c94e-dd7b-4e02-821c-3fb32eb1abac` (session summary).
- Branch: `wip/inversion-audit-track2-2026-07-08` in both repos. Nothing pushed. Rollback is a single `git reset --hard origin/main && git branch -D wip/...` per repo.

---

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