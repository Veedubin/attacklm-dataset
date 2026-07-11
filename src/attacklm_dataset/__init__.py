"""attacklm-dataset — MITRE ATT&CK-grounded security fine-tuning dataset."""

from pathlib import Path

from .__version__ import __version__

PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "data" / "datasets" / "buckets" / "sources"
