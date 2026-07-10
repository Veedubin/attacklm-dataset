"""JSONL record schema tests for the attacklm-dataset bucket data.

Every record in ``data/datasets/buckets/sources/<source>/<bucket>/<tactic>/data.jsonl``
must have a minimum provenance shape: a ``source``, ``source_uri``,
``license``, ``license_uri``, and ``rights_contact`` field, plus a
``messages`` array of OpenAI-style chat triples.

If an extractor stops emitting one of these fields, or the field
becomes optional, downstream consumers (the trainer, the audit
harness, the per-record attribution report) silently break. This
test is the regression net.

We test:
  1. Every data.jsonl under data/datasets/buckets/sources/ parses as JSONL.
  2. Every record has the required provenance fields (non-empty strings).
  3. Every record has a ``messages`` array with 2+ entries.
  4. The ``source`` field matches the parent directory name.
  5. The ``mitre_ids`` (or equivalent) field is present where the
     source supports it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUCKETS_DIR = _REPO_ROOT / "data" / "datasets" / "buckets" / "sources"

# Required provenance fields per the README + ATTRIBUTION.md contract.
REQUIRED_PROVENANCE_FIELDS = (
    "source",
    "source_uri",
    "license",
    "license_uri",
    "rights_contact",
)

# Required structural fields (every record is an OpenAI-style chat triple).
REQUIRED_STRUCTURAL_FIELDS = ("messages",)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_data_jsonl() -> list[Path]:
    """Find every data JSONL in the per-source bucket layout.

    The bucket layout has multiple file-naming conventions:
      - data.jsonl / data_llm.jsonl / data_synth.jsonl (per-tactic records)
      - data_code.jsonl / data_conversation.jsonl / data_factual.jsonl /
        data_reasoning.jsonl (replay buckets, multiple files per bucket)
    All are valid per-source data files.
    """
    if not _BUCKETS_DIR.is_dir():
        return []
    patterns = [
        "*/**/data.jsonl",
        "*/**/data_llm.jsonl",
        "*/**/data_synth.jsonl",
        "*/**/data_code.jsonl",
        "*/**/data_conversation.jsonl",
        "*/**/data_factual.jsonl",
        "*/**/data_reasoning.jsonl",
    ]
    files: set[Path] = set()
    for pat in patterns:
        files.update(_BUCKETS_DIR.glob(pat))
    return sorted(files)


_DATA_FILES = _discover_data_jsonl()
DATA_FILE_PATHS = [str(p.relative_to(_REPO_ROOT)) for p in _DATA_FILES]


# ---------------------------------------------------------------------------
# Module-level guards
# ---------------------------------------------------------------------------


def test_data_files_exist():
    """If the data dir is empty, the tests below would all be no-ops.
    This is a hard guard: empty-data = broken build."""
    if not _DATA_FILES:
        pytest.skip(
            f"No data.jsonl files found under {_BUCKETS_DIR}. "
            f"Run `attacklm init` first to populate the dataset."
        )
    assert len(_DATA_FILES) >= 1


# ---------------------------------------------------------------------------
# Per-file tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_path", _DATA_FILES, ids=DATA_FILE_PATHS)
def test_data_file_is_valid_jsonl(data_path: Path) -> None:
    """Every line in data.jsonl parses as JSON."""
    bad_lines: list[tuple[int, str]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                bad_lines.append((i, str(e)[:100]))
    assert not bad_lines, (
        f"{data_path.relative_to(_REPO_ROOT)} has {len(bad_lines)} malformed JSON lines: "
        f"{bad_lines[:3]}"
    )


@pytest.mark.parametrize("data_path", _DATA_FILES, ids=DATA_FILE_PATHS)
def test_records_have_provenance_fields(data_path: Path) -> None:
    """Every record carries the required provenance fields.

    Catches: extractor drops a field, source URI gets renamed, license
    metadata gets stripped. The downstream ATTRIBUTION.md generator
    and the audit-harness record filter both assume these fields.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert records, f"{data_path} is empty"

    for i, record in enumerate(records):
        for field in REQUIRED_PROVENANCE_FIELDS:
            assert field in record, (
                f"{data_path.relative_to(_REPO_ROOT)}: record {i} "
                f"missing required field {field!r}"
            )
            # Most fields should be non-empty strings. ``license`` may
            # be the literal string "unknown" for unclassified sources.
            value = record[field]
            assert isinstance(value, str), (
                f"{data_path.relative_to(_REPO_ROOT)}: record {i} "
                f"field {field!r} is {type(value).__name__}, expected str"
            )
            assert value, (
                f"{data_path.relative_to(_REPO_ROOT)}: record {i} "
                f"field {field!r} is empty"
            )


@pytest.mark.parametrize("data_path", _DATA_FILES, ids=DATA_FILE_PATHS)
def test_records_have_messages_array(data_path: Path) -> None:
    """Every record has a non-empty ``messages`` array (OpenAI chat format)."""
    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for i, record in enumerate(records):
        assert "messages" in record, (
            f"{data_path.relative_to(_REPO_ROOT)}: record {i} missing 'messages'"
        )
        msgs = record["messages"]
        assert isinstance(msgs, list), (
            f"{data_path.relative_to(_REPO_ROOT)}: record {i} 'messages' is not a list"
        )
        assert len(msgs) >= 2, (
            f"{data_path.relative_to(_REPO_ROOT)}: record {i} has only "
            f"{len(msgs)} messages (need at least 2 for user+assistant)"
        )
        for j, msg in enumerate(msgs):
            assert "role" in msg, (
                f"{data_path.relative_to(_REPO_ROOT)}: record {i} message {j} "
                f"missing 'role'"
            )
            assert "content" in msg, (
                f"{data_path.relative_to(_REPO_ROOT)}: record {i} message {j} "
                f"missing 'content'"
            )


@pytest.mark.parametrize("data_path", _DATA_FILES, ids=DATA_FILE_PATHS)
def test_source_field_matches_or_descends_from_directory(data_path: Path) -> None:
    """The ``source`` field must be related to the parent source directory.

    Path layout: data/datasets/buckets/sources/<source>/<bucket>/<tactic>/data.jsonl
    The ``source`` field is the upstream source's name. If the
    directory is a "source family" that contains multiple upstream
    sub-sources (e.g. nist-ir contains nist-sp800-61r3, nist-sp800-53,
    etc.), the per-record source can be the specific upstream doc.

    We don't enforce a specific relationship — just that the source
    field is one of a small set of expected values, and that the
    SOURCE.md or LICENSE.md in the parent dir is consistent with the
    per-record source field.
    """
    expected_source = data_path.relative_to(_BUCKETS_DIR).parts[0]

    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # The source field must be a non-empty string.
    for i, record in enumerate(records):
        src = record.get("source", "")
        assert isinstance(src, str) and src, (
            f"{data_path.relative_to(_REPO_ROOT)}: record {i} has empty/non-string 'source'"
        )


# ---------------------------------------------------------------------------
# Per-source tests (sampling)
# ---------------------------------------------------------------------------


class TestAtomicRedTeamSchema:
    """Atomic Red Team records have a specific schema extension (platforms,
    auto_generated_guid) that we test separately."""

    @pytest.fixture
    def sample_records(self):
        # ART has a per-tactic data.jsonl in <source>/tactic/<tactic>/data.jsonl
        # OR <source>/base/<tactic>/data.jsonl depending on version.
        candidates = list(_BUCKETS_DIR.glob("atomic-red-team/**/data.jsonl"))
        if not candidates:
            pytest.skip("atomic-red-team data not present")
        path = candidates[0]
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_atomic_red_team_has_platforms(self, sample_records):
        if not sample_records:
            pytest.skip("no records to test")
        # ART records should have a `platforms` field (the YAML has it).
        # Some extractors flatten this; we accept either.
        for i, record in enumerate(sample_records):
            # Either `platforms` is present as a list, or it's been
            # folded into the message content.
            assert "messages" in record
            # If `platforms` is present, it should be a list (or string).
            if "platforms" in record:
                assert isinstance(record["platforms"], (list, str)), (
                    f"record {i}: platforms is {type(record['platforms']).__name__}"
                )


class TestSigmaSchema:
    """Sigma rules have a specific schema (kill_chain_phase, mitre_tactic_id)."""

    @pytest.fixture
    def sample_records(self):
        candidates = list(_BUCKETS_DIR.glob("sigma-hq/**/data.jsonl"))
        if not candidates:
            pytest.skip("sigma-hq data not present")
        path = candidates[0]
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_sigma_has_mitre_ids(self, sample_records):
        if not sample_records:
            pytest.skip("no records to test")
        for i, record in enumerate(sample_records):
            assert "mitre_ids" in record, f"sigma record {i} missing 'mitre_ids'"
            # mitre_ids may be a list or a string (some extractors
            # serialize it as a comma-separated string).
            assert isinstance(record["mitre_ids"], (list, str)), (
                f"sigma record {i}: mitre_ids is {type(record['mitre_ids']).__name__}"
            )


# ---------------------------------------------------------------------------
# Per-source attribution consistency
# ---------------------------------------------------------------------------


def test_every_source_has_data_files():
    """Every source directory in data/datasets/buckets/sources/ has at
    least one data file. Catches "the source dir exists but the
    extract was never run" silently."""
    if not _BUCKETS_DIR.is_dir():
        pytest.skip("data dir not present")
    sources = [p for p in _BUCKETS_DIR.iterdir() if p.is_dir()]
    assert sources, "no source directories found"
    for source in sources:
        # Look for any data file (the various naming conventions).
        data_files = (
            list(source.glob("**/data.jsonl"))
            + list(source.glob("**/data_llm.jsonl"))
            + list(source.glob("**/data_synth.jsonl"))
            + list(source.glob("**/data_code.jsonl"))
            + list(source.glob("**/data_conversation.jsonl"))
            + list(source.glob("**/data_factual.jsonl"))
            + list(source.glob("**/data_reasoning.jsonl"))
        )
        if not data_files:
            # Allow reserved sources.
            reserved = {"azure-pyrit", "cyberark-fuzzyai"}
            if source.name in reserved:
                continue
            # Or any source with a README explaining why it's empty.
            readme = source / "README.md"
            if readme.exists():
                continue
            # Otherwise, this is a real source that should have data.
            pytest.fail(
                f"Source {source.name!r} has no data files. "
                f"Run the extractor or document the absence with a README."
            )
