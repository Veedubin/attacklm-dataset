"""attacklm-dataset — MITRE ATT&CK-grounded security fine-tuning dataset."""

from pathlib import Path

__version__ = "0.5.0"

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "data" / "datasets" / "buckets" / "sources"
