#!/usr/bin/env bash
#SBATCH --job-name=rule_gat_train
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/RulE}"
CONFIG="${CONFIG:-$PROJECT_DIR/config/wn18rr_hpc.json}"
EXPERIMENT_KEY="${EXPERIMENT_KEY:-wn18rr_apples}"
SEEDS="${SEEDS:-800,801,802}"

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
SEED="${SEED_LIST[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "${SEED}" ]]; then
  echo "No seed mapped for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

BASE_OUT="${PROJECT_DIR}/outputs/experiments/${EXPERIMENT_KEY}"
PRETRAIN_CKPT="${BASE_OUT}/original_fixed_alpha/seed_${SEED}/checkpoint"
PERSIST_DIR="${BASE_OUT}/gat_fixed_alpha/seed_${SEED}"
MANIFEST="${BASE_OUT}/manifests/runs.jsonl"

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "Missing pretrain checkpoint: ${PRETRAIN_CKPT}" >&2
  exit 1
fi

mkdir -p "${PERSIST_DIR}" "${BASE_OUT}/manifests"

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "${PROJECT_DIR}/rule_env/bin/activate"

SCRATCH_ROOT="${TMPDIR:-/tmp}/${USER}/rule_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_OUT="${SCRATCH_ROOT}/output"
TMP_CFG="${SCRATCH_ROOT}/config_seed.json"
mkdir -p "${RUN_OUT}"

python3 - <<PY
import json
cfg = json.load(open("${CONFIG}", "r"))
cfg["seed"] = int("${SEED}")
cfg["save_path"] = "${RUN_OUT}"
with open("${TMP_CFG}", "w") as f:
    json.dump(cfg, f, indent=2)
PY

TRAIN_RC=0
python3 -u "${PROJECT_DIR}/src/main.py" \
  --init_checkpoint_config "${TMP_CFG}" \
  --skip_pretrain \
  --pretrain_checkpoint "${PRETRAIN_CKPT}" \
  --save_path "${RUN_OUT}" || TRAIN_RC=$?

rsync -a "${RUN_OUT}/" "${PERSIST_DIR}/"

GIT_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
CONFIG_HASH="$(shasum -a 256 "${CONFIG}" | awk '{print $1}')"

python3 - <<PY
import json, time
row = {
  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
  "job_id": "${SLURM_JOB_ID}",
  "array_task_id": int("${SLURM_ARRAY_TASK_ID}"),
  "stage": "train_gat",
  "variant": "gat_fixed_alpha",
  "seed": int("${SEED}"),
  "git_commit": "${GIT_COMMIT}",
  "config_path": "${CONFIG}",
  "config_hash": "${CONFIG_HASH}",
  "checkpoint_path": "${PERSIST_DIR}/checkpoint",
  "grounding_path": "${PERSIST_DIR}/grounding.pt",
  "pretrain_checkpoint": "${PRETRAIN_CKPT}",
  "status": "ok" if ${TRAIN_RC} == 0 else "failed"
}
with open("${MANIFEST}", "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\\n")
PY

exit "${TRAIN_RC}"
