# Security

## Reporting a vulnerability in the dataset tooling

Please email **[email protected]** (or open a private security advisory via
GitHub's Security tab). Do not file public issues for security-sensitive
reports.

## License / attribution violations

If you believe content in this dataset violates your copyright or the
upstream license terms, see `data/REMOVAL.md` for the takedown process.
The dataset carries per-record provenance (`source`, `source_uri`, `license`,
`license_uri`, `rights_contact`) so affected records can be identified and
removed.

## Inversion-audit program

This package includes an inversion-audit harness (v0.2.0) under
`scripts/inversion_audit.py`. It is intended for **defensive audits of
models the auditor owns or is authorized to test**. The output boundary
enforces workspace-internal storage of raw reconstructions; only aggregate
metrics may be exported.
