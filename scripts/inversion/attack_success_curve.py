"""Attack success curve aggregation for closed-loop audit.
Implements MAI-Thinking-1 §5.2 success-rate tracking.
See docs/AUDIT_ITER.md for the full spec.
"""

# Aggregates per-iteration results into the attack-success curve,
# per MAI-Thinking-1 §5.2. See docs/AUDIT_ITER.md.

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


def compute_success_curve(per_iter_results: list[dict]) -> dict:
    """Compute the attack-success curve from per-iteration results.

    Each per_iter_result is a dict like:
        {"iteration": 0, "probed_count": 200, "fooling_count": 8,
         "fooling_record_ids": ["rec-0", ...], "attack": "mia"}

    Returns a curve dict:
        {"0": {"mia": {"probed": 200, "fooling": 8, "success_rate": 0.04}}, ...}
    """
    if not per_iter_results:
        return {}

    curve: dict[str, dict[str, dict[str, Any]]] = {}

    for result in per_iter_results:
        iter_key = str(result.get("iteration", 0))
        attack_type = result.get("attack", "unknown")
        probed = result.get("probed_count", 0)
        fooling = result.get("fooling_count", 0)

        # Skip entries with zero probed (avoid division by zero)
        if probed == 0:
            if iter_key not in curve:
                curve[iter_key] = {}
            curve[iter_key][attack_type] = {
                "probed": 0,
                "fooling": 0,
                "success_rate": 0.0,
            }
            continue

        success_rate = fooling / probed

        if iter_key not in curve:
            curve[iter_key] = {}

        curve[iter_key][attack_type] = {
            "probed": probed,
            "fooling": fooling,
            "success_rate": round(success_rate, 10),
        }

    return curve


def write_curve(
    curve: dict,
    output_path: Path,
    metadata: dict | None = None,
    fooling_records_by_iter: dict[str, list[str]] | None = None,
) -> None:
    """Write the attack-success curve JSON to disk.

    Output schema:
    {
      "schema_version": "1.0",
      "audit_date": "YYYY-MM-DD",
      "metadata": {<user-supplied>},
      "curve": {<the curve dict>},
      "fooling_records_by_iter": {"0": ["<id>", ...], "1": [...]}
    }
    """
    output_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_date": date.today().isoformat(),
    }

    if metadata is not None:
        output_data["metadata"] = metadata

    output_data["curve"] = curve

    if fooling_records_by_iter is not None:
        output_data["fooling_records_by_iter"] = fooling_records_by_iter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2) + "\n")
