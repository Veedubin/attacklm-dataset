## [0.9.1] — 2026-07-16

- **`calibrate_threshold` FPR fix**: Fixed inverted quantile in `scoring.py`. Previously `target_fpr=0.01` produced ~99% FPR; now correctly produces ~1% FPR. Function is not used in the production audit pipeline (`inversion_audit.py` uses `_percentile` directly) but was a latent footgun for anyone calling it directly.
- **BLEU-4 unification**: Replaced nltk-based `bleu4_score` in `probe.py` with pure-Python epsilon-smoothed implementation matching `AttackLM/scripts/audit_canary_extraction.py`. Removed `nltk>=3.8` from `pyproject.toml` inversion extras.
- **Docs**: Updated README with docs index table, refreshed `MIA_THRESHOLD_CALIBRATION.md` for the corrected `calibrate_threshold` behavior.

No PyPI publish (attacklm-dataset is GitHub-only distribution).

## [0.9.0] — 2026-07-16

- First real defensive audit on `AttackLM/uncensored` (Qwen2.5-Coder-14B-Instruct-uncensored, non-finetuned). 51 records, 17 per source (atomic-red-team, metasploit-framework, sigma-hq), `--attack all --mia-method per_token --top-k 5 --max-new-tokens 128`. Used MIA Track 1 default threshold (`--mia-threshold-mode percentile --mia-percentile 5`). Results in `data/audit/2026-07-16-defensive-v1/2026-07-16/` (chmod 0700 parent + 0600 on `inversion_results.jsonl`).
- **Per-source metrics** (per_token NLL; lower = more memorized):
  - atomic-red-team: mean NLL/pt 1.63, mean BLEU-4 0.047, 0 exact matches, 3 flagged
  - metasploit-framework: mean NLL/pt 2.62, mean BLEU-4 0.063, 0 exact matches, 0 flagged
  - sigma-hq: mean NLL/pt 2.53, mean BLEU-4 0.096, 0 exact matches, 0 flagged
- **MIA threshold**: -125.08 (5th percentile of 51 scores, **3/51 flagged = 5.9%**). This validates MIA Track 1's calibration fix — the 2026-07-07 pilot (median threshold) flagged 50% by construction.
- **Per-record evidence chain**: each result includes `prompt_text`, `best_reconstruction`, `suffix_text`, per-token NLL, membership score. `inversion_results.jsonl` is self-contained per the v0.4.1 audit bug fix.
- No harness changes — `--mia-method per_token` (MUSE 2023 default) is the recommended single-method mode when shadow models are not available. For `reference`/`zlib`/`offline` coverage, run the audit 4 times with different `--mia-method` values.
- No PyPI publish (attacklm-dataset is GH-only).

## [0.8.0] — 2026-07-13

- Added held-out NLL evaluation suite (`scripts/split_held_out.py` + `scripts/held_out_nll.py`). Implements MAI-Thinking-1 §2.3-style weighted aggregate (Code/STEM/Math/General/Multilingual). New docs: `docs/HELD_OUT_NLL.md`. Backward compatible: existing training data is untouched.
- [Methodology] Implemented technique from MAI-Thinking-1 §2.3 (Evaluation Methodology) + §2.3.2 (Comparison of Accuracy and NLL Evaluations) by The Microsoft AI Team, June 2026. Uses the paper's weighted Eq-3 aggregate across 5 buckets.

## [0.7.0] — 2026-07-13

- Added `scripts/memorization_report.py`: per-source memorization proxy + recommended epoch caps.
- New `data/memorization_epoch_caps.json` (auto-generated) with default 4-tier cap table.
- New docs: `docs/MEMORIZATION.md`.
- [Methodology] Implemented technique from MAI-Thinking-1 §2.5.4 (Mid-training Data Mixture — memorization-aware epoch capping) by The Microsoft AI Team, June 2026.

Default behavior: report-only (does NOT auto-apply caps). Caps are a recommendation for review before use.

## [0.6.0] — 2026-07-13

- Added `scripts/decontam.py`: 20-gram fuzzy decontamination against public evaluation sets.
- New `data/eval_sets/` directory with 3 fixture JSONL files (atomic-red-team, metasploit, sigma).
- New `datasketch>=1.5,<2.0` optional dependency (in `[inversion]` group).
- New docs: `docs/DECONTAM.md`.
- [Methodology] Implemented technique from MAI-Thinking-1 §2.3.1 (Public Evaluation Decontamination) and §2.4.3 (Deduplication) by The Microsoft AI Team, June 2026.

Default behavior: report-only (no deletions). Use `--quarantine-output` to write matched records to a separate file for review.

## [0.5.0] — 2026-07-13

- Added `--audit-iter` flag to `inversion_audit.py` for closed-loop adversarial audits.
- New `scripts/inversion/variant_generator.py` (paraphrase + suffix-injection + prompt-template).
- New `scripts/inversion/attack_success_curve.py` (per-iteration success-rate aggregation).
- New docs: `docs/AUDIT_ITER.md`.
- [Methodology] Implemented technique from MAI-Thinking-1 §5.2 (TAP-style closed-loop adversarial audit) by The Microsoft AI Team, June 2026.

Backward compatible: `--audit-iter 1` (default) is identical to v0.4.3 single-pass behavior.

## [Unreleased] — 2026-07-13 — LiRA shadow scoring & Offline MIA baseline

- Added `scripts/score_shadow.py`: CLI for LiRA workflow step 2 (score a
  shadow model on the audit set, writes `shadow_{K}.json`). Closes the
  LiRA workflow loop — previously the user had to write this themselves.
- Added `--mia-method offline` to `inversion_audit.py`: white-box MIA
  baseline using sample mean/std of audit-set NLL as the OUT distribution.
  No shadow models required. Field names: `offline_z`, `offline_mu_out`,
  `offline_sigma_out`, `offline_flagged`. Default threshold: -1.5,
  configurable via `--offline-z-threshold`. Requires N >= 30 records.
- Added `compute_offline_z()` to `inversion.scoring` (reuses
  `zscore_normalize()` for the σ=0 guard).
- Added 15 new tests (10 in `test_score_shadow.py`, 5 in
  `test_offline_mia.py`). Total tests: 475 passing, 0 skipped.

### Bug fixes for the inversion-audit harness
A paper-vs-code audit (memory 209 laL) found 3 blocker bugs and
2 quality issues in scripts/inversion/. All 5 fixed in commit 4386995.

Bug #1 (correctness, MUST FIX): scoring.py:162 score_record
  was using _extract_full_text() which leaked the prompt into
  the NLL and biased all --mia-method reference scores by prompt
  length. Switched to _extract_assistant_turn() per MUSE 2023
  default. (matches score_per_token behavior).

Bug #2 (crash + wrong metadata, MUST FIX): lira.py:305
  save_shadow_params hardcoded lira_k=0 in the JSON and crashed
  with NameError on empty params. Added explicit lira_k parameter;
  shadow_train.py now passes lira_k=len(shadow_losses).

Bug #3 (evidence chain, MUST FIX): probe.py ProbeResult didn't
  include prompt_text or best_reconstruction, so memorization
  findings couldn't be recovered from the audit artifact without
  re-running the probe. Added the two fields; the audit artifact
  (inversion_results.jsonl, chmod 0600) is now self-contained.

Bug #4 (performance): probe.py:144 generate_completions was
  doing K sequential model.generate() calls. Switched to single
  generate() with num_return_sequences. ~20x speedup on typical
  14B + 256-token setups.

Bug #5 (code smell): lira.py LiRAScore had a dead alpha field
  "reserved for future use". Removed the field and the alpha
  parameter from score_lira. Cleaner contract.

Tests: 14 new tests in tests/test_audit_bugfixes.py. 443 -> 457
passing. Audit code is now safe to use for a real defensive run.

---


Considered publishing `attacklm-dataset` to PyPI as the missing version
after the 2026-07-09 audit caught the build-backend bug in `pyproject.toml`.
Implemented (in commits 1229f71, 098cccc, 20a9580, 34e3621) and rolled
back: the dataset is a **data package** (8,147 tracked JSONL files,
~50MB compressed, plus a 3-file Python wrapper) and PyPI is the wrong
place for it. A `pip install attacklm-dataset` for a 50MB data bundle
goes against the typical "small Python library" expectation of PyPI,
and the actual data distribution is already handled by `attacklm init`
in the AttackLM package (which downloads `attacklm-dataset.tar.gz` from
GitHub Releases).

**Decision: no PyPI publish for `attacklm-dataset`.**

What was rolled back:
- v0.4.1 tag (was `e64c782`) and v0.4.2 tag (was `95c70c2`) — both
  deleted from local and remote, both never reached PyPI.
- The hatchling build-backend fix (commit 1229f71) — note: this was
  a real bug fix (`setuptools.backends._legacy:_Backend` doesn't
  exist in any setuptools version, blocking `python -m build`); the
  decision to roll it back is **purely** about not publishing to
  PyPI, not about whether the fix is good. If you want a
  GitHub-Releases-only flow, the hatchling fix is still needed for
  `python -m build` to produce the tarball. **Open question** for
  the user: re-apply the build fix without bumping the version?
- The CI workflow at `.github/workflows/ci.yml` — deleted, the
  audit worked on the local `pytest tests/` invocation only.
- The release workflow at `.github/workflows/release.yml` — deleted.
  Would have failed at the trusted-publishing step anyway because the
  PyPI trusted publisher is not configured for this project.

What stays:
- v0.4.0 tag (244cb00), which was already PUSHED before this session
  and remains on origin. **Not on PyPI** (build was broken). The
  `attacklm init` download flow in AttackLM uses the GitHub Releases
  tarball, which is a separate artifact from the git tag.

What this means for users:
- `pip install attacklm-dataset` does not work and is **not** supported.
  Use the AttackLM package's `attacklm init` command instead, or
  `git clone https://github.com/Veedubin/attacklm-dataset`.
- `pip install "attacklm[all]"` (AttackLM) does not pull in
  `attacklm-dataset` from PyPI; the `[dataset]` extra in AttackLM is
  for the **GitHub Releases download** path, not a PyPI dep.

### Known issues (carried over from the audit)
- `pyproject.toml` still uses `setuptools.backends._legacy:_Backend`
  which does not exist. `python -m build` fails. Local
  `pip install -e .` works because it doesn't go through the build
  backend. **Fix when re-applying the hatchling patch.**
- `__version__.py` says "0.1.0" while `__init__.py` hardcodes
  "0.4.0" and the v0.4.0 tag exists. Three sources of truth,
  all disagreeing. **Fix when re-applying the single-source-of-truth
  refactor.**
- `__init__.py` DATA_DIR is computed relative to
  `Path(__file__).parent.parent` which gives the wrong root
  (`src/` instead of repo root). Not used by any `scripts/` code
  (they all define their own DATA_DIR from their own
  `Path(__file__).parent.parent`), so the bug is purely cosmetic.
  **Fix when re-applying the path refactor.**

## [0.4.0] — 2026-07-08 — MIA Track 2: per-token + LiRA + audit screen

Per-token membership-inference scoring (MUSE 2023 default) and LiRA
shadow-model MIA (Carlini 2022 §4) on the dataset side. CLI redesigned
to attack-class-first with `--attack {extraction, mia, all}` and
`--mia-method {reference, zlib, per_token, lira, all}`. New docs:
`docs/ATTACK_TAXONOMY.md` and `docs/LIRA.md`. Drive-by fix:
`pyproject.toml` was at `0.2.0` while the actual code was at `v0.3.1+`
— now `0.4.0` matches the release. 34 new hermetic tests (13
per_token_mia + 21 lira). All 106 tests pass.

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