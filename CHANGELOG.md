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