# MITRE ATLAS (vendored distribution artifact)

- Upstream: https://github.com/mitre-atlas/atlas-data
- File: `dist/v6/ATLAS-2026.09.yaml` (format-version 6.0.0, content version 2026.09)
- License: Apache-2.0 (Copyright 2021-2026 MITRE)
- Vendored: 2026-09-22 for reproducible extraction by `scripts/extract_mitre_atlas.py`

To update: download the newer `dist/v6/ATLAS-<version>.yaml`, update
`ATLAS_VERSION` in `scripts/extract_mitre_atlas.py`, re-run the extractor,
regenerate the manifest (`scripts/rebuild_manifest.py`).
