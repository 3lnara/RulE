#!/usr/bin/env bash
#SBATCH --job-name=rule_eval_adapt
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/RulE}"
CONFIG="${CONFIG:-$PROJECT_DIR/config/wn18rr_hpc.json}"
EXPERIMENT_KEY="${EXPERIMENT_KEY:-wn18rr_apples}"
SEEDS="${SEEDS:-800,801,802}"
ALPHA="${ALPHA:-1.0}"
ADAPTIVE_EPOCHS="${ADAPTIVE_EPOCHS:-5}"
ADAPTIVE_LR="${ADAPTIVE_LR:-0.01}"

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
SEED="${SEED_LIST[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "${SEED}" ]]; then
  echo "No seed mapped for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

BASE_OUT="${PROJECT_DIR}/outputs/experiments/${EXPERIMENT_KEY}"
ORIG_CKPT="${BASE_OUT}/original_fixed_alpha/seed_${SEED}/grounding.pt"
GAT_CKPT="${BASE_OUT}/gat_fixed_alpha/seed_${SEED}/grounding.pt"
EVAL_DIR="${BASE_OUT}/evaluations/seed_${SEED}"
MANIFEST="${BASE_OUT}/manifests/runs.jsonl"

if [[ ! -f "${ORIG_CKPT}" ]]; then
  echo "Missing original grounding checkpoint: ${ORIG_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${GAT_CKPT}" ]]; then
  echo "Missing GAT grounding checkpoint: ${GAT_CKPT}" >&2
  exit 1
fi

mkdir -p "${EVAL_DIR}" "${BASE_OUT}/manifests"

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "${PROJECT_DIR}/rule_env/bin/activate"

ORIG_JSON="${EVAL_DIR}/original_eval.json"
GAT_JSON="${EVAL_DIR}/gat_eval.json"

python3 -u "${PROJECT_DIR}/scripts/evaluate_unified.py" \
  --project_root "${PROJECT_DIR}" \
  --backend original \
  --config "${CONFIG}" \
  --checkpoint "${ORIG_CKPT}" \
  --output_json "${ORIG_JSON}" \
  --alpha "${ALPHA}" \
  --seed "${SEED}" \
  --train_adaptive \
  --adaptive_epochs "${ADAPTIVE_EPOCHS}" \
  --adaptive_lr "${ADAPTIVE_LR}"

python3 -u "${PROJECT_DIR}/scripts/evaluate_unified.py" \
  --project_root "${PROJECT_DIR}" \
  --backend gat \
  --config "${CONFIG}" \
  --checkpoint "${GAT_CKPT}" \
  --output_json "${GAT_JSON}" \
  --alpha "${ALPHA}" \
  --seed "${SEED}" \
  --train_adaptive \
  --adaptive_epochs "${ADAPTIVE_EPOCHS}" \
  --adaptive_lr "${ADAPTIVE_LR}"

GIT_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
CONFIG_HASH="$(shasum -a 256 "${CONFIG}" | awk '{print $1}')"

python3 - <<PY
import json, time
row = {
  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
  "job_id": "${SLURM_JOB_ID}",
  "array_task_id": int("${SLURM_ARRAY_TASK_ID}"),
  "stage": "evaluate_unified",
  "variant": "all_four",
  "seed": int("${SEED}"),
  "git_commit": "${GIT_COMMIT}",
  "config_path": "${CONFIG}",
  "config_hash": "${CONFIG_HASH}",
  "original_eval_json": "${ORIG_JSON}",
  "gat_eval_json": "${GAT_JSON}",
  "status": "ok"
}
with open("${MANIFEST}", "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\\n")
PY
