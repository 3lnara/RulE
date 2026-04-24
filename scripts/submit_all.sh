#!/usr/bin/env bash
# Submit full apples-to-apples experiment pipeline:
# original_fixed -> gat_fixed -> unified eval (fixed + adaptive) -> summary
#
# Usage:
#   ./scripts/submit_all.sh \
#     --experiment-key wn18rr_apples \
#     --config ~/RulE/config/wn18rr_hpc.json \
#     --seeds 800,801,802 \
#     --alpha 1.0 \
#     --adaptive-epochs 5 \
#     --adaptive-lr 0.01

set -euo pipefail

EXPERIMENT_KEY=""
CONFIG=""
SEEDS="800,801,802"
PROJECT_DIR="${PROJECT_DIR:-$HOME/RulE}"
ALPHA="1.0"
ADAPTIVE_EPOCHS="5"
ADAPTIVE_LR="0.01"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-key) EXPERIMENT_KEY="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --project-dir) PROJECT_DIR="${2:-}"; shift 2 ;;
    --alpha) ALPHA="${2:-}"; shift 2 ;;
    --adaptive-epochs) ADAPTIVE_EPOCHS="${2:-}"; shift 2 ;;
    --adaptive-lr) ADAPTIVE_LR="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${EXPERIMENT_KEY}" ]]; then
  echo "Missing --experiment-key" >&2
  exit 1
fi

if [[ -z "${CONFIG}" ]]; then
  echo "Missing --config" >&2
  exit 1
fi

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
if [[ ${#SEED_ARR[@]} -eq 0 ]]; then
  echo "No seeds provided" >&2
  exit 1
fi
ARRAY_RANGE="0-$((${#SEED_ARR[@]} - 1))"

BASE_OUT="${PROJECT_DIR}/outputs/experiments/${EXPERIMENT_KEY}"
SLURM_LOG_DIR="${BASE_OUT}/slurm_logs"
mkdir -p "${SLURM_LOG_DIR}" "${BASE_OUT}/manifests" "${BASE_OUT}/summaries"

echo "Project dir: ${PROJECT_DIR}"
echo "Config: ${CONFIG}"
echo "Experiment key: ${EXPERIMENT_KEY}"
echo "Seeds: ${SEEDS}"
echo "Array range: ${ARRAY_RANGE}"

COMMON_EXPORTS="ALL,PROJECT_DIR=${PROJECT_DIR},CONFIG=${CONFIG},EXPERIMENT_KEY=${EXPERIMENT_KEY},SEEDS=${SEEDS},ALPHA=${ALPHA},ADAPTIVE_EPOCHS=${ADAPTIVE_EPOCHS},ADAPTIVE_LR=${ADAPTIVE_LR}"

J1=$(sbatch --parsable \
  --array="${ARRAY_RANGE}" \
  --export="${COMMON_EXPORTS}" \
  --output="${SLURM_LOG_DIR}/%x_%A_%a.out" \
  --error="${SLURM_LOG_DIR}/%x_%A_%a.err" \
  "${PROJECT_DIR}/slurm/00_train_original_array.slurm")
echo "Submitted original training array: ${J1}"

J2=$(sbatch --parsable \
  --dependency="afterok:${J1}" \
  --array="${ARRAY_RANGE}" \
  --export="${COMMON_EXPORTS}" \
  --output="${SLURM_LOG_DIR}/%x_%A_%a.out" \
  --error="${SLURM_LOG_DIR}/%x_%A_%a.err" \
  "${PROJECT_DIR}/slurm/01_train_gat_array.slurm")
echo "Submitted GAT training array: ${J2}"

J3=$(sbatch --parsable \
  --dependency="afterok:${J2}" \
  --array="${ARRAY_RANGE}" \
  --export="${COMMON_EXPORTS}" \
  --output="${SLURM_LOG_DIR}/%x_%A_%a.out" \
  --error="${SLURM_LOG_DIR}/%x_%A_%a.err" \
  "${PROJECT_DIR}/slurm/02_eval_adaptive_array.slurm")
echo "Submitted unified evaluator array: ${J3}"

J4=$(sbatch --parsable \
  --dependency="afterok:${J3}" \
  --export="${COMMON_EXPORTS}" \
  --output="${SLURM_LOG_DIR}/%x_%j.out" \
  --error="${SLURM_LOG_DIR}/%x_%j.err" \
  "${PROJECT_DIR}/slurm/03_summarize_results.slurm")
echo "Submitted summary job: ${J4}"

echo ""
echo "Pipeline submitted."
echo "Final summary target: ${BASE_OUT}/summaries/unified_summary.json"
