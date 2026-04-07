#!/usr/bin/env bash
#SBATCH --job-name=gat_grounding_only
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl

# ── GPU info ───────────────────────────────────────────────────────────────────
# echo "=== GPU info ==="
# nvidia-smi
# echo "================"

# ── Python environment ─────────────────────────────────────────────────────────
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

VENV="$HOME/RulE/rule_env"
if [ -f "${VENV}/bin/activate" ]; then
    echo "Activating virtualenv: ${VENV}"
    source "${VENV}/bin/activate"
else
    echo "ERROR: Virtualenv not found at ${VENV}."
    echo "Run setup_env_hpc.sh on the login node first: bash ~/RulE/setup_env_hpc.sh"
    exit 1
fi

echo "Python  : $(which python3)"
echo "Version : $(python3 --version)"
echo "PyTorch : $(python3 -c 'import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available())')"
echo ""

# ── Configuration ──────────────────────────────────────────────────────────────
# Path to an existing pre-trained checkpoint from a previous run.
# Update this path to point to the actual checkpoint produced by the full
# comparison run (e.g. outputs/family_comparison_hpc/checkpoint).
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/family_comparison_hpc/checkpoint"

# Output directory for this GAT grounding run
SAVE_PATH="$HOME/RulE/outputs/gat_grounding_g32"

# nvidia-smi dmon log (framebuffer memory, sampled every 10 s)
# GPU_LOG="${SAVE_PATH}/gpu_memory.log"

echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Save path              : ${SAVE_PATH}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    echo "Run the full comparison job first or update PRETRAIN_CHECKPOINT in this script."
    exit 1
fi

mkdir -p "${SAVE_PATH}"

# ── Start GPU memory monitor in background ─────────────────────────────────────
# -s m  : memory metrics only (fb = framebuffer used MiB, bar1 = BAR1 used MiB)
# -d 10 : sample every 10 seconds
# nvidia-smi dmon -s m -d 10 > "${GPU_LOG}" &
# GPU_MON_PID=$!
# echo "GPU memory monitor PID : ${GPU_MON_PID}  (log: ${GPU_LOG})"
# echo ""

# ── Run GAT grounding only ─────────────────────────────────────────────────────
TRAIN_START=$(date +%s)

cd "$HOME/RulE/src"
python3 -u main.py \
    --init_checkpoint_config "$HOME/RulE/config/family_comparison_hpc.json" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --g_batch_size 32 \
    --save_path "${SAVE_PATH}"

TRAIN_RC=$?
TRAIN_END=$(date +%s)
TRAIN_WALL=$(( TRAIN_END - TRAIN_START ))

# Stop GPU monitor
kill "${GPU_MON_PID}" 2>/dev/null
wait "${GPU_MON_PID}" 2>/dev/null

echo ""
echo "Training exit code : ${TRAIN_RC}"
printf "Training wall time : %dh%02dm%02ds\n" \
    $(( TRAIN_WALL / 3600 )) $(( (TRAIN_WALL % 3600) / 60 )) $(( TRAIN_WALL % 60 ))

if [ "${TRAIN_RC}" -ne 0 ]; then
    echo "ERROR: training failed — skipping metrics parsing."
    echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
    exit "${TRAIN_RC}"
fi

# ── Parse and tabulate per-iteration metrics ───────────────────────────────────
echo ""
echo "=== Per-iteration metrics ==="
python3 "$HOME/RulE/scripts/parse_gat_metrics.py" \
    --log  "${SAVE_PATH}/run.log" \
    --gpu_log "${GPU_LOG}" \
    --out  "${SAVE_PATH}/metrics_table.txt"

echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
