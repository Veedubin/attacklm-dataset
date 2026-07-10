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


def _args_to_argv(args: argparse.Namespace) -> list[str]:
    """Serialize the parsed argparse namespace into a CLI argv list.

    Walks the known subcommand fields and emits the ones that were set
    on the command line. This is the bridge between the wrapper CLI's
    strict argparse and the underlying script's argparse (which has
    its own --yes / --from-source / etc. flags that the wrapper
    accepts but does not re-implement).
    """
    # Map of attr name → CLI flag (long form, no abbreviation).
    flag_map = {
        "yes": "--yes",
        "from_source": "--from-source",
        "dataset_url": "--dataset-url",
        "extract_only": "--extract-only",
        "buckets_only": "--buckets-only",
        "attribute_only": "--attribute-only",
        "clone_only": "--clone-only",
        "profile": "--profile",
        "preset": "--preset",
        "strategy": "--strategy",
        "source": "--source",
        "count": "--count",
    }
    out: list[str] = []
    for attr, flag in flag_map.items():
        if not hasattr(args, attr):
            continue
        value = getattr(args, attr)
        if value is None:
            continue
        # Boolean flags: only emit if True.
        if isinstance(value, bool):
            if value:
                out.append(flag)
        else:
            out.extend([flag, str(value)])
    # Trailing positional argv (REMAINDER) goes last, untouched.
    if getattr(args, "argv", None):
        out.extend(args.argv)
    return out


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    script_map = {
        "init": "init_pipeline.py",
        "balance": "balance_buckets.py",
        "evolve": "evolve_pairs.py",
        "audit": "audit_dataset.py",
        "package": "package_dataset.py",
    }
    script = script_map.get(args.command)
    if script is None:
        parser.print_help()
        return 0
    return _run_python_script(script, _args_to_argv(args))


if __name__ == "__main__":
    sys.exit(main())
