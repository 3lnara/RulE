#!/usr/bin/env bash
#SBATCH --job-name=adaptive_beta_only_wn18rr_densnorm_perrel
#SBATCH --partition=mcs.gpu.q
#SBATCH --constraint=v100
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── GPU info ───────────────────────────────────────────────────────────────────
echo "=== GPU info ==="
nvidia-smi
echo "================"

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
# The original MLP run saves two checkpoints we reuse here:
#   checkpoint    — best pre-trained KGE + rule embeddings (from PreTrainer)
#   grounding.pt  — best MLP grounding weights (from GroundTrainer)
# Both are produced by rule_comparison_gpu.slurm (or a standalone original run).
ORIG_OUTPUT="$HOME/RulE/outputs/wn18rr_original"
PRETRAIN_CHECKPOINT="${ORIG_OUTPUT}/checkpoint"
GROUNDING_CHECKPOINT="${ORIG_OUTPUT}/grounding.pt"
BETA_CHECKPOINT="${ORIG_OUTPUT}/beta.pt"

# Config file — same one used for the original run
CONFIG="$HOME/RulE/config/wn18rr_hpc.json"

# Output directory for this adaptive-beta run
SAVE_PATH="$HOME/RulE/outputs/adaptive_beta_only_wn18rr_densnorm_perrel"
GPU_LOG="${SAVE_PATH}/gpu_memory.log"

# Adaptive-beta hyper-parameters
BETA_EPOCHS=10
BETA_LR=0.001
BETA_DENSITY_LR=0.001
FIXED_ALPHA=1.0   # baseline alpha for comparison table
SEED = 42


echo "Pre-train checkpoint   : ${PRETRAIN_CHECKPOINT}"
echo "Grounding checkpoint   : ${GROUNDING_CHECKPOINT}"
echo "Config                 : ${CONFIG}"
echo "Save path              : ${SAVE_PATH}"
echo "Beta epochs / lr       : ${BETA_EPOCHS} / ${BETA_LR} / ${BETA_DENSITY_LR}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── Validate checkpoints ───────────────────────────────────────────────────────
missing=0
if [ ! -f "${PRETRAIN_CHECKPOINT}" ] && [ ! -d "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-training checkpoint not found at ${PRETRAIN_CHECKPOINT}"
    missing=1
fi
if [ ! -f "${GROUNDING_CHECKPOINT}" ]; then
    echo "ERROR: Grounding checkpoint not found at ${GROUNDING_CHECKPOINT}"
    echo "       Run rule_comparison_gpu.slurm first to produce grounding.pt."
    missing=1
fi
if [ "${missing}" -ne 0 ]; then
    exit 1
fi

mkdir -p "${SAVE_PATH}"

# ── Start GPU memory monitor in background ─────────────────────────────────────
nvidia-smi dmon -s m -d 10 > "${GPU_LOG}" &
GPU_MON_PID=$!
echo "GPU memory monitor PID : ${GPU_MON_PID}  (log: ${GPU_LOG})"
echo ""

COMMON_ARGS=(
    --config             "${CONFIG}"
    --checkpoint         "${GROUNDING_CHECKPOINT}"
    --epochs             "${BETA_EPOCHS}"
    --lr                 "${BETA_LR}"
    --density_lr         "${BETA_DENSITY_LR}"
    --alpha              "${FIXED_ALPHA}"
    --seed               "${SEED}"
)
run_variant() {
    local name="$1"
    shift
    local variant_dir="${ABLATION_ROOT}/${name}"
    mkdir -p "${variant_dir}"

    echo "────────────────────────────────────────────────────────────────"
    echo "  Variant : ${name}"
    echo "  Args    : $*"
    echo "  Dir     : ${variant_dir}"
    echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "────────────────────────────────────────────────────────────────"

    local t0
    t0=$(date +%s)

    python3 -u "$HOME/RulE/scripts/train_beta_grounding_chunked.py" \
        "${COMMON_ARGS[@]}" \
        --beta_checkpoint "${variant_dir}/beta.pt" \
        "$@" \
        2>&1 | tee "${variant_dir}/training.log"

    VARIANT_RC=${PIPESTATUS[0]}
    local t1
    t1=$(date +%s)
    VARIANT_WALL=$(( t1 - t0 ))

    printf "  Finished : $(date '+%Y-%m-%d %H:%M:%S')  |  wall %dh%02dm%02ds  |  rc=%d\n" \
        $(( VARIANT_WALL / 3600 )) $(( (VARIANT_WALL % 3600) / 60 )) $(( VARIANT_WALL % 60 )) \
        "${VARIANT_RC}"
    echo ""
}

# ── Run adaptive beta training ─────────────────────────────────────────────────
# train_beta_grounding.py imports from the src/ package (data, model, utils),
# so it must be run with src/ on the Python path.
TRAIN_START=$(date +%s)

cd "$HOME/RulE/RulE_original/src"
python3 -u "$HOME/RulE/scripts/train_beta_grounding_chunked.py" \
    --config  "${CONFIG}" \
    --checkpoint "${GROUNDING_CHECKPOINT}" \
    --epochs  "${BETA_EPOCHS}" \
    --lr      "${BETA_LR}" \
    --density_lr "${BETA_DENSITY_LR}" \
    --alpha   "${FIXED_ALPHA}" \
    --normalize_scores \
    --standardize_feature \
    --feature density \
    --seed 42 \
    --beta_checkpoint "${BETA_CHECKPOINT}" \
    --neg_sampling mixed \
    --num_negatives 100 \
    2>&1 | tee "${SAVE_PATH}/beta_training.log"

TRAIN_RC=${PIPESTATUS[0]}
TRAIN_END=$(date +%s)
TRAIN_WALL=$(( TRAIN_END - TRAIN_START ))

# Stop GPU monitor
kill "${GPU_MON_PID}" 2>/dev/null
wait "${GPU_MON_PID}" 2>/dev/null

# ── Timing summary ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Adaptive beta training complete"
echo "=================================================="
echo "  Exit code  : ${TRAIN_RC}"
printf "  Wall time  : %dh%02dm%02ds\n" \
    $(( TRAIN_WALL / 3600 )) $(( (TRAIN_WALL % 3600) / 60 )) $(( TRAIN_WALL % 60 ))

if [ -f "${GPU_LOG}" ]; then
    # Extract peak GPU framebuffer memory from dmon log (fb column)
    PEAK_MEM=$(awk '!/^#/ && NF>=2 {val=$2+0; if(val>max) max=val} END{print max+0}' "${GPU_LOG}")
    echo "  Peak GPU mem (MiB) : ${PEAK_MEM}"
fi

echo "  Full log   : ${SAVE_PATH}/beta_training.log"
echo "=================================================="
echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"

exit "${TRAIN_RC}"
