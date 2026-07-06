"""attacklm-dataset CLI — dataset management commands."""

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _run_python_script(script_name: str, argv: list[str]) -> int:
    """Run a Python script from the scripts directory."""
    script_path = _SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"Error: script not found: {script_path}", file=sys.stderr)
        return 1
    result = subprocess.run(
        [sys.executable, str(script_path)] + argv,
        cwd=Path.cwd(),
    )
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attacklm-dataset",
        description="AttackLM Dataset — MITRE ATT&CK-grounded security fine-tuning data",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # init
    init_p = sub.add_parser("init", help="Initialize the dataset")
    init_p.add_argument("--yes", action="store_true", help="Skip confirmation prompts")
    init_p.add_argument(
        "--from-source", action="store_true", help="Build from upstream git repos"
    )
    init_p.add_argument("--dataset-url", type=str, help="Override download URL")
    init_p.add_argument(
        "--extract-only", action="store_true", help="Run data extractors only"
    )
    init_p.add_argument(
        "--buckets-only", action="store_true", help="Organize data into buckets only"
    )
    init_p.add_argument(
        "--attribute-only",
        action="store_true",
        help="Add source/license attribution only",
    )
    init_p.add_argument(
        "--clone-only", action="store_true", help="Clone upstream repos only"
    )
    init_p.add_argument("argv", nargs=argparse.REMAINDER, help="Additional arguments")

    # balance
    balance_p = sub.add_parser("balance", help="Build a balanced training subset")
    balance_p.add_argument(
        "--profile", type=str, help="Hardware profile (e.g., 7b-16gb)"
    )
    balance_p.add_argument(
        "--preset", type=str, help="Training preset (e.g., red-team)"
    )
    balance_p.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    balance_p.add_argument(
        "argv", nargs=argparse.REMAINDER, help="Additional arguments"
    )

    # evolve
    evolve_p = sub.add_parser("evolve", help="Evolve training pairs")
    evolve_p.add_argument(
        "--strategy", type=str, default="all", help="Evolution strategy"
    )
    evolve_p.add_argument("--source", type=str, help="Source to evolve")
    evolve_p.add_argument(
        "--count", type=int, default=500, help="Number of pairs to evolve"
    )
    evolve_p.add_argument("argv", nargs=argparse.REMAINDER, help="Additional arguments")

    # audit
    audit_p = sub.add_parser("audit", help="Audit the dataset")
    audit_p.add_argument("argv", nargs=argparse.REMAINDER, help="Additional arguments")

    # package
    package_p = sub.add_parser("package", help="Package dataset for distribution")
    package_p.add_argument(
        "argv", nargs=argparse.REMAINDER, help="Additional arguments"
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init":
        return _run_python_script("init_pipeline.py", args.argv)
    elif args.command == "balance":
        return _run_python_script("balance_buckets.py", args.argv)
    elif args.command == "evolve":
        return _run_python_script("evolve_pairs.py", args.argv)
    elif args.command == "audit":
        return _run_python_script("audit_dataset.py", args.argv)
    elif args.command == "package":
        return _run_python_script("package_dataset.py", args.argv)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
