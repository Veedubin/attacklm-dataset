#!/usr/bin/env bash
# Overnight inversion-audit runner.
# Splits 1,100 records across 4 runs, each ~3-6 hours.
# Resume-safe: re-running skips runs whose summary.json already exists.
# Usage: ./scripts/run_overnight_audits.sh [--dry-run]
#
# Plan doc: docs/AUDIT_RUNNER.md
# Prerequisites: commit a93c970 or later (probe + MIA Track 1 defaults)

set -euo pipefail

# Config — paths are absolute; edit here if the workspace moves.
PYTHON="/home/jcharles/Projects/reverse_engineering/AttackLM/.venv/bin/python"
MODEL="/home/jcharles/Projects/reverse_engineering/AttackLM/uncensored"
DATASET_ROOT="/home/jcharles/Projects/reverse_engineering/attacklm-dataset/data/datasets/buckets/sources"
AUDIT_ROOT="/home/jcharles/Projects/reverse_engineering/attacklm-dataset/data/audit"
LOG_ROOT="/tmp/audit-overnight-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$LOG_ROOT"

# Per-run definitions: date|probe_count|extra_source_args
# - probe_count is the per-source limit (--probe-count)
# - extra_source_args must include --source-filter to ensure disjoint sources
RUNS=(
  "2026-07-08-a|300|--source-filter atomic-red-team"
  "2026-07-08-b|300|--source-filter atomic-red-team"
  "2026-07-08-c|300|--source-filter metasploit-framework"
  "2026-07-08-d|200|--source-filter sigma-hq"
)

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "DRY RUN — would execute:"
  for run in "${RUNS[@]}"; do
    IFS='|' read -r date count extra <<< "$run"
    echo "  $date  $count records  $extra"
  done
  exit 0
fi

# Pre-flight: confirm we're in the right directory
if [[ ! -f "scripts/inversion_audit.py" ]]; then
  echo "ERROR: must be run from attacklm-dataset/ root (no scripts/inversion_audit.py found)"
  exit 1
fi

# Pre-flight: confirm working venv
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" 2>/dev/null; then
  echo "ERROR: $PYTHON does not have working CUDA torch. Use AttackLM/.venv/bin/python"
  exit 1
fi

# Pre-flight: GPU is free
GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [[ "$GPU_MEM_USED" -gt 1500 ]]; then
  echo "WARNING: GPU is using ${GPU_MEM_USED}MB. Audit needs ~14GB free."
  read -p "Continue? (y/N) " -n 1 -r
  echo
  [[ "$REPLY" =~ ^[Yy]$ ]] || exit 1
fi

echo "Pre-flight passed. Starting runs at $(date)"

for run in "${RUNS[@]}"; do
  IFS='|' read -r date count extra <<< "$run"
  out_dir="$AUDIT_ROOT/$date"

  if [[ -d "$out_dir" && -f "$out_dir/summary.json" ]]; then
    echo "[SKIP] $date — summary.json already exists at $out_dir"
    continue
  fi

  echo ""
  echo "[START] $date  $count records  $extra"
  echo "  log: $LOG_ROOT/$date.log"
  echo "  out: $out_dir"

  # Clean partial run if present (no summary.json means it didn't complete)
  if [[ -d "$out_dir" ]]; then
    find "$out_dir" -type f -delete
    rmdir "$out_dir" 2>/dev/null || true
  fi
  mkdir -p "$out_dir"
  chmod 0700 "$out_dir"

  # setsid + nohup-style detachment: the audit runs in a new session
  # so killing this script doesn't take the audit down.
  setsid "$PYTHON" scripts/inversion_audit.py \
    --model "$MODEL" \
    --dataset-root "$DATASET_ROOT" \
    $extra \
    --probe-count "$count" \
    --top-k 20 \
    --mia-threshold-mode percentile \
    --mia-percentile 5 \
    --audit-output-root "$AUDIT_ROOT" \
    --date "$date" \
    > "$LOG_ROOT/$date.log" 2>&1 < /dev/null

  echo "[DONE]  $date — log: $LOG_ROOT/$date.log"
  tail -20 "$LOG_ROOT/$date.log"
  echo "---"
done

echo ""
echo "ALL RUNS COMPLETE at $(date)"
echo "Per-run logs: $LOG_ROOT/"
echo "Per-run output: $AUDIT_ROOT/{2026-07-08-a,2026-07-08-b,2026-07-08-c,2026-07-08-d}/"
