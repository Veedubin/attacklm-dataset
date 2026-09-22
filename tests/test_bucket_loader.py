"""Tests for bucket_loader.resolve_dataset_spec().

Tests verify that:
  - defensive/ buckets are reachable via resolve_dataset_spec
  - "all" now includes defensive/ (composition change as of 2026-09-22)
  - "all-offensive" is frozen at the pre-2026-09-22 meaning of "all"
  - backward compat is preserved for models trained before this date
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/lib to path so we can import bucket_loader
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "lib"))

from bucket_loader import resolve_dataset_spec


class TestDefensiveBuckets:
    """Test defensive/ bucket resolution."""

    def test_defensive_category_resolves(self):
        """resolve_dataset_spec("defensive/") returns 3 defensive buckets."""
        result = resolve_dataset_spec("defensive/")
        assert len(result) == 3, f"Expected 3 defensive buckets, got {len(result)}"
        paths = {b["path"] for b in result}
        expected = {
            "defensive/detection_engineering",
            "defensive/threat_hunting",
            "defensive/incident_response",
        }
        assert paths == expected, f"Expected {expected}, got {paths}"

    def test_defensive_subcategory_resolves(self):
        """resolve_dataset_spec("defensive/detection_engineering/") returns 1 bucket."""
        result = resolve_dataset_spec("defensive/detection_engineering/")
        assert len(result) == 1, f"Expected 1 bucket, got {len(result)}"
        assert result[0]["path"] == "defensive/detection_engineering"

    def test_defensive_threat_hunting_resolves(self):
        """resolve_dataset_spec("defensive/threat_hunting/") returns 1 bucket."""
        result = resolve_dataset_spec("defensive/threat_hunting/")
        assert len(result) == 1, f"Expected 1 bucket, got {len(result)}"
        assert result[0]["path"] == "defensive/threat_hunting"

    def test_defensive_incident_response_resolves(self):
        """resolve_dataset_spec("defensive/incident_response/") returns 1 bucket."""
        result = resolve_dataset_spec("defensive/incident_response/")
        assert len(result) == 1, f"Expected 1 bucket, got {len(result)}"
        assert result[0]["path"] == "defensive/incident_response"


class TestAllAlias:
    """Test "all" alias includes defensive buckets."""

    def test_all_includes_defensive(self):
        """resolve_dataset_spec("all") includes defensive/ buckets."""
        result = resolve_dataset_spec("all")
        paths = {b["path"] for b in result}

        # Check that at least one defensive bucket is present
        defensive_buckets = {p for p in paths if p.startswith("defensive/")}
        assert len(defensive_buckets) == 3, (
            f"Expected 3 defensive buckets in 'all', got {len(defensive_buckets)}: {defensive_buckets}"
        )

        # Expected defensive buckets
        expected_defensive = {
            "defensive/detection_engineering",
            "defensive/threat_hunting",
            "defensive/incident_response",
        }
        assert defensive_buckets == expected_defensive, (
            f"Expected {expected_defensive}, got {defensive_buckets}"
        )

    def test_all_contains_37_buckets(self):
        """resolve_dataset_spec("all") returns 37 buckets (including defensive).

        The count changed from 34 to 37 on 2026-09-22 when defensive/ was added.
        """
        result = resolve_dataset_spec("all")
        assert len(result) == 37, (
            f"Expected 37 buckets in 'all' (after adding defensive/), got {len(result)}"
        )


class TestAllOffensiveAlias:
    """Test "all-offensive" frozen alias preserves pre-2026-09-22 behavior."""

    def test_all_offensive_excludes_defensive(self):
        """resolve_dataset_spec("all-offensive") does NOT include any defensive bucket."""
        result = resolve_dataset_spec("all-offensive")
        paths = {b["path"] for b in result}

        defensive_buckets = {p for p in paths if p.startswith("defensive/")}
        assert len(defensive_buckets) == 0, (
            f"'all-offensive' should not include defensive buckets, but got: {defensive_buckets}"
        )

    def test_all_offensive_contains_34_buckets(self):
        """resolve_dataset_spec("all-offensive") returns 34 buckets (pre-2026-09-22 "all").

        This is the frozen baseline for backward compat.
        """
        result = resolve_dataset_spec("all-offensive")
        assert len(result) == 34, (
            f"Expected 34 buckets in 'all-offensive', got {len(result)}"
        )

    def test_all_offensive_excludes_defensive_detection_engineering(self):
        """'all-offensive' does not include defensive/detection_engineering."""
        result = resolve_dataset_spec("all-offensive")
        paths = {b["path"] for b in result}
        assert "defensive/detection_engineering" not in paths

    def test_all_offensive_excludes_defensive_threat_hunting(self):
        """'all-offensive' does not include defensive/threat_hunting."""
        result = resolve_dataset_spec("all-offensive")
        paths = {b["path"] for b in result}
        assert "defensive/threat_hunting" not in paths

    def test_all_offensive_excludes_defensive_incident_response(self):
        """'all-offensive' does not include defensive/incident_response."""
        result = resolve_dataset_spec("all-offensive")
        paths = {b["path"] for b in result}
        assert "defensive/incident_response" not in paths


class TestAllVsAllOffensive:
    """Test that "all" and "all-offensive" differ only by defensive buckets."""

    def test_all_contains_all_offensive_plus_defensive(self):
        """"all" = "all-offensive" + 3 defensive buckets."""
        all_result = resolve_dataset_spec("all")
        all_offensive_result = resolve_dataset_spec("all-offensive")

        # Convert to sets of paths for comparison
        all_paths = {b["path"] for b in all_result}
        all_offensive_paths = {b["path"] for b in all_offensive_result}

        # All-offensive paths should be a subset of all paths
        assert all_offensive_paths.issubset(all_paths), (
            "all-offensive should be a subset of all"
        )

        # The difference should be exactly the 3 defensive buckets
        difference = all_paths - all_offensive_paths
        assert len(difference) == 3, (
            f"Expected exactly 3 defensive buckets as the difference, got {len(difference)}"
        )

        expected_defensive = {
            "defensive/detection_engineering",
            "defensive/threat_hunting",
            "defensive/incident_response",
        }
        assert difference == expected_defensive, (
            f"Expected {expected_defensive} as the difference, got {difference}"
        )


class TestBackwardCompat:
    """Test backward compat for models trained before defensive/ was added."""

    def test_all_offensive_is_stable(self):
        """'all-offensive' should be usable for reproducing pre-2026-09-22 training runs.

        It must never gain new categories or buckets beyond the frozen set.
        It should return exactly 34 buckets and exclude "defensive" category.
        """
        result = resolve_dataset_spec("all-offensive")

        # Count categories
        categories = {}
        for b in result:
            cat = b.get("category")
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1

        # All-offensive should NOT contain defensive
        assert "defensive" not in categories, (
            "all-offensive must not contain defensive category"
        )

        # All-offensive must return exactly 34 buckets (the pre-2026-09-22 count)
        assert len(result) == 34, (
            f"all-offensive must return exactly 34 buckets for reproducibility, got {len(result)}"
        )
