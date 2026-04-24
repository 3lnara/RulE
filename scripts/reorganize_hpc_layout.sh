#!/usr/bin/env bash
# Create slurm/ layout and experiment-keyed output folders.
# Usage:
#   ./scripts/reorganize_hpc_layout.sh --experiment-key wn18rr_apples
#   ./scripts/reorganize_hpc_layout.sh --experiment-key wn18rr_apples --dry-run

set -euo pipefail

EXPERIMENT_KEY=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-key)
      EXPERIMENT_KEY="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${EXPERIMENT_KEY}" ]]; then
  echo "Missing --experiment-key" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_DIR="${ROOT_DIR}/slurm"
OUT_BASE="${ROOT_DIR}/outputs/experiments/${EXPERIMENT_KEY}"

run_cmd() {
  if ${DRY_RUN}; then
    echo "[DRY-RUN] $*"
  else
    eval "$@"
  fi
}

echo "Project root: ${ROOT_DIR}"
echo "Experiment key: ${EXPERIMENT_KEY}"
echo "Dry run: ${DRY_RUN}"

run_cmd "mkdir -p \"${SLURM_DIR}\""
run_cmd "mkdir -p \"${OUT_BASE}\"/{original_fixed_alpha,gat_fixed_alpha,evaluations,manifests,summaries,slurm_logs}"

for f in adaptive_beta_only.slurm gat_grounding_only.slurm rule_comparison_gpu.slurm rule_original_wn18rr.slurm; do
  src="${ROOT_DIR}/${f}"
  dst="${SLURM_DIR}/${f}"
  if [[ -f "${src}" ]]; then
    if [[ -d "${ROOT_DIR}/.git" ]]; then
      run_cmd "git -C \"${ROOT_DIR}\" mv \"${src}\" \"${dst}\""
    else
      run_cmd "mv \"${src}\" \"${dst}\""
    fi
  fi
done

echo "Layout ready."
echo "Experiment output root: ${OUT_BASE}"
