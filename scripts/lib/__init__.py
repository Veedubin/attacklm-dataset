"""scripts.lib — Shared library code for the AttackLM-dataset scripts.

These are imported by the various extractor/maintenance scripts in
``scripts/`` via::

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent / "lib"))
    from bucket_loader import list_buckets

Moved out of the top-level ``scripts/`` directory so that the
``test_scripts_smoke.py`` regression net can enforce a clean
contract: "every ``*.py`` in ``scripts/`` is a CLI entry point with
``--help``".  These library modules are not scripts; they don't
have argparse, and they shouldn't be expected to.
"""
