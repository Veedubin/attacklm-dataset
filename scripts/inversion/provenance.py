"""Provenance tracking and license enforcement for the inversion audit.

Three responsibilities:
1. Carry per-record license stamps from source data through to every output row.
2. Enforce the restricted-source denylist (RTA, infection_monkey, BPL) —
   the driver MUST refuse to run if --source-filter matches a restricted source.
3. Classify output rows into INTERNAL_RAW, INTERNAL_SUMMARY, or
   EXPORTABLE_SUMMARY based on the license and content type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Restricted-source denylist
# ---------------------------------------------------------------------------
# These sources are EXCLUDED from the public dataset and live only in
# archive/restricted-sources/. The audit driver MUST refuse to process
# them. Their licenses (GPL-3.0, AGPL-3.0) constrain redistribution.

RESTRICTED_SOURCES: frozenset[str] = frozenset(
    {
        "rta",  # Red Team Automation — GPL-3.0
        "infection_monkey",  # Infection Monkey — AGPL-3.0
        "bpl",  # BloodHound Payload Library — scrubbed from git
    }
)

# Known license families in the dataset. Each source in _index.json carries
# one of these (or a blank string for reserved sources).
LICENSE_FAMILIES: dict[str, str] = {
    "BSD-3-Clause": "permissive",
    "MIT": "permissive",
    "Apache-2.0": "permissive",
    "DRL-1.1": "derived-data",  # Sigma — derived-data rules apply
    "GPL-3.0": "restricted",
    "AGPL-3.0": "restricted",
    "": "unlicensed",
}


class OutputClass(Enum):
    """Three-tier output classification for audit results."""

    INTERNAL_RAW = "INTERNAL_RAW"
    INTERNAL_SUMMARY = "INTERNAL_SUMMARY"
    EXPORTABLE_SUMMARY = "EXPORTABLE_SUMMARY"


@dataclass
class RecordProvenance:
    """Per-record license provenance carried through the audit pipeline."""

    source: str
    license_id: str
    license_family: str
    license_notice: str = ""
    source_uri: str = ""

    @classmethod
    def from_record(cls, record: dict) -> "RecordProvenance":
        """Extract provenance fields from a training-record dict.

        Expected keys (all optional; missing fields are filled with defaults):
            source, license, license_notice, source_uri
        """
        source = record.get("source", "unknown")
        license_id = record.get("license", "")
        license_family = LICENSE_FAMILIES.get(license_id, "unlicensed")
        license_notice = record.get("license_notice", "")
        source_uri = record.get("source_uri", "")
        return cls(
            source=source,
            license_id=license_id,
            license_family=license_family,
            license_notice=license_notice,
            source_uri=source_uri,
        )

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "license": self.license_id,
            "license_family": self.license_family,
            "license_notice": self.license_notice,
            "source_uri": self.source_uri,
        }


def load_source_index(dataset_root: Path) -> dict:
    """Load the source _index.json from the dataset root.

    Returns the parsed JSON dict with source metadata.
    """
    index_path = dataset_root / "_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Source index not found: {index_path}")
    with open(index_path) as f:
        return json.load(f)


def validate_source_filter(source_filter: list[str], dataset_root: Path) -> list[str]:
    """Validate that --source-filter doesn't include restricted sources.

    Raises ValueError if any restricted source is requested.
    Returns the validated list of source names.
    """
    for name in source_filter:
        if name in RESTRICTED_SOURCES:
            raise ValueError(
                f"Source '{name}' is in the restricted-source denylist. "
                f"Its license constrains redistribution and it is excluded "
                f"from the public dataset. The audit driver refuses to "
                f"process it. Restricted sources: {sorted(RESTRICTED_SOURCES)}"
            )
    return source_filter


def get_available_sources(dataset_root: Path) -> list[str]:
    """Return the list of source directory names that have data on disk."""
    sources_dir = dataset_root
    if not sources_dir.exists():
        return []
    return sorted(
        d.name
        for d in sources_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )


def get_held_out_sources(dataset_root: Path) -> list[str]:
    """Return sources with n_records=0 (reserved/empty) for MIA calibration.

    These are sources that have a directory entry in _index.json but no
    actual training records. They serve as natural held-out sets for
    calibrating membership-inference thresholds.
    """
    index = load_source_index(dataset_root)
    return [
        s["name"]
        for s in index.get("sources", [])
        if s.get("n_records", -1) == 0 or s.get("status") == "reserved"
    ]


def classify_output(
    provenance: RecordProvenance,
    include_raw_text: bool = False,
) -> OutputClass:
    """Determine the output class for a record based on its license.

    Rules:
        - DRL-1.1 records: INTERNAL_RAW if raw text is included,
          EXPORTABLE_SUMMARY otherwise (derived-data rules).
        - Permissive (BSD-3, MIT, Apache-2.0): INTERNAL_RAW is OK
          but still owner-only (chmod 0600).
        - Restricted: never reaches this function (blocked by denylist).
        - Unlicensed: INTERNAL_SUMMARY only.
    """
    if provenance.license_family == "restricted":
        # Should never happen — blocked by validate_source_filter
        return OutputClass.EXPORTABLE_SUMMARY
    if provenance.license_family == "unlicensed":
        return OutputClass.INTERNAL_SUMMARY
    if provenance.license_id == "DRL-1.1":
        # Sigma derived data: raw text is workspace-internal only
        return (
            OutputClass.INTERNAL_RAW
            if include_raw_text
            else OutputClass.EXPORTABLE_SUMMARY
        )
    # Permissive licenses: raw text OK but owner-only
    return (
        OutputClass.INTERNAL_RAW if include_raw_text else OutputClass.EXPORTABLE_SUMMARY
    )


def strip_raw_text_for_export(row: dict) -> dict:
    """Remove raw reconstruction text and re-identifiable record IDs
    from an audit row to produce an EXPORTABLE_SUMMARY-safe version.

    Keeps: source, license, aggregate scores, hashes.
    Removes: prompt_text, best_reconstruction, original_text,
             any field containing >50 chars of original content.
    """
    export = {}
    for key, value in row.items():
        # Keep aggregate scores and metadata
        if key in (
            "source",
            "license",
            "license_family",
            "best_exact_match",
            "best_lcs_length",
            "best_bleu4",
            "membership_score",
            "nll",
            "zlib_ratio",
            "perplexity",
            "prompt_hash",
            "reconstruction_hash",
            "num_completions",
            "probe_type",
            "record_index",
        ):
            export[key] = value
    return export


def hash_text(text: str) -> str:
    """SHA-256 hash of text for audit-trail purposes (never store raw text)."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
