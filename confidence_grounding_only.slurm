#!/usr/bin/env bash
#SBATCH --job-name=conf_grounding
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=24:00:00
#SBATCH --constraint=v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl

# ── GPU info ───────────────────────────────────────────────────────────────────
echo "=== GPU info ==="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<not set>}"
echo "================"

# ── Python environment ─────────────────────────────────────────────────────────
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

if [ -n "${SLURM_JOB_GPUS}" ]; then
    export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
fi

VENV="$HOME/RulE/rule_env"
if [ -f "${VENV}/bin/activate" ]; then
    echo "Activating virtualenv: ${VENV}"
    source "${VENV}/bin/activate"
else
    echo "ERROR: Virtualenv not found at ${VENV}."
    exit 1
fi

echo "Python  : $(which python3)"
echo "Version : $(python3 --version)"
echo "PyTorch : $(python3 -c 'import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available())')"
echo ""

# ── Configuration ──────────────────────────────────────────────────────────────
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/family_comparison_hpc/checkpoint"
SAVE_PATH="$HOME/RulE/outputs/confidence_grounding_g32"

echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Save path              : ${SAVE_PATH}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    exit 1
fi

mkdir -p "${SAVE_PATH}"

# ── Run confidence + counts grounding (no GAT) ────────────────────────────────
TRAIN_START=$(date +%s)

cd "$HOME/RulE/src"
python3 -u main.py \
    --init_checkpoint_config "$HOME/RulE/config/family_comparison_hpc.json" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --g_batch_size 32 \
    --save_path "${SAVE_PATH}"
    # Note: --use_gat is NOT passed → RulEConfidence (counts × rule_confidence) is used

TRAIN_RC=$?
TRAIN_END=$(date +%s)
TRAIN_WALL=$(( TRAIN_END - TRAIN_START ))

echo ""
echo "Training exit code : ${TRAIN_RC}"
printf "Training wall time : %dh%02dm%02ds\n" \
    $(( TRAIN_WALL / 3600 )) $(( (TRAIN_WALL % 3600) / 60 )) $(( TRAIN_WALL % 60 ))

if [ "${TRAIN_RC}" -ne 0 ]; then
    echo "ERROR: training failed."
    exit "${TRAIN_RC}"
fi

echo ""
echo "=== Per-iteration metrics ==="
python3 "$HOME/RulE/scripts/parse_gat_metrics.py" \
    --log  "${SAVE_PATH}/run.log" \
    --out  "${SAVE_PATH}/metrics_table.txt"

echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
