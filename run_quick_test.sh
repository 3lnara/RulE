#!/usr/bin/env bash
# run_quick_test.sh — run RulE on the family dataset with family_quick_test config
# Usage: ./run_quick_test.sh [--skip-pretrain] [--pretrain-checkpoint PATH]
#
# Run from the project root directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config/family_quick_test.json"
SRC_DIR="${SCRIPT_DIR}/src"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-pretrain)
      EXTRA_ARGS+=(--skip_pretrain)
      shift ;;
    --pretrain-checkpoint)
      EXTRA_ARGS+=(--pretrain_checkpoint "$2")
      shift 2 ;;
    *)
      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "Config    : ${CONFIG}"
echo "Output    : $(python3 -c "import json; d=json.load(open('${CONFIG}')); print(d['save_path'])" 2>/dev/null || echo "(see config)")"
echo "Started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "----------------------------------------"

cd "${SRC_DIR}"
python3 main.py \
  --init_checkpoint_config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"

echo "----------------------------------------"
echo "Finished  : $(date '+%Y-%m-%d %H:%M:%S')"
