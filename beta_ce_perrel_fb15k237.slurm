#!/usr/bin/env bash
#SBATCH --job-name=beta_ce_perrel_fb15k237_topruleconf
#SBATCH --partition=mcs.gpu.q
#SBATCH --constraint=v100
#SBATCH --time=28:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─────────────────────────────────────────────────────────────────────────────
# Adaptive top_rule_confidence mixing-weight (beta) training — FB15k-237,
# multi-seed.
#
# Feature: top_rule_confidence — max RulE plausibility (sigmoid of rule
# embedding distance) among rules that fired for a query; 0 if none fired.
# High confidence = the best-matching rule has a strong embedding alignment.
# Requires eval_compute_rule_confidence() in the trainer (automatically called
# after eval_compute_rule_weight via the hasattr guard).
#
# Multi-seed job array (--array=0-2) runs seeds 42, 43, 44 in parallel,
# each in its own output directory. Fixed config (ls=0.2) matched to
# FB15k-237's grounding-stage smoothing.
#
# Key settings:
#   --feature top_rule_confidence : per-query adaptive feature.
#   --standardize_feature         : z-scores on the beta-train split.
#   --loss ce                     : filtered softmax cross-entropy.
#   --density_lr 0.01             : LR for the single beta_density scalar.
#   FIXED_ALPHA=3.0               : FB15k-237's published operating point.
#
# Both stages run: [3] trains per-relation intercept beta[r], then [6] trains
# the adaptive beta_density slope on top_rule_confidence.
# ─────────────────────────────────────────────────────────────────────────────

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

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
ORIG_OUTPUT="$HOME/RulE/outputs/fb15k237_original"
PRETRAIN_CHECKPOINT="${ORIG_OUTPUT}/checkpoint"
GROUNDING_CHECKPOINT="${ORIG_OUTPUT}/grounding.pt"

CONFIG="$HOME/RulE/config/fb15k237_hpc.json"

SAVE_PATH="$HOME/RulE/outputs/beta_ce_perrel_fb15k237_topruleconf/seed${SEED}"
GPU_LOG="${SAVE_PATH}/gpu_memory.log"

BETA_EPOCHS=6      # plateau typically reached by ep 6-7 on FB15k-237
BETA_LR=0.001
DENSITY_LR=0.01
FIXED_ALPHA=3.0    # FB15k-237's published operating point (config: alpha=3.0)
ABLATION_ROOT="${SAVE_PATH}"

echo "Array task / seed      : ${SLURM_ARRAY_TASK_ID} / ${SEED}"
echo "Pre-train checkpoint   : ${PRETRAIN_CHECKPOINT}"
echo "Grounding checkpoint   : ${GROUNDING_CHECKPOINT}"
echo "Config                 : ${CONFIG}"
echo "Save path              : ${SAVE_PATH}"
echo "Feature                : top_rule_confidence (standardized, per-query adaptive)"
echo "Beta epochs / lr       : ${BETA_EPOCHS} / ${BETA_LR} (intercept), ${DENSITY_LR} (density)"
echo "Fixed alpha (baseline) : ${FIXED_ALPHA}"
echo "Loss                   : cross-entropy"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── Validate checkpoints ───────────────────────────────────────────────────────
missing=0
if [ ! -f "${PRETRAIN_CHECKPOINT}" ] && [ ! -d "${PRETRAIN_CHECKPOINT}" ]; then
    echo "WARN: Pre-training checkpoint not found at ${PRETRAIN_CHECKPOINT}"
    echo "      (beta training only needs grounding.pt, continuing)"
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

# ── Shared arguments ───────────────────────────────────────────────────────────
COMMON_ARGS=(
    --config             "${CONFIG}"
    --checkpoint         "${GROUNDING_CHECKPOINT}"
    --epochs             "${BETA_EPOCHS}"
    --lr                 "${BETA_LR}"
    --density_lr         "${DENSITY_LR}"
    --alpha              "${FIXED_ALPHA}"
    --seed               "${SEED}"
    --loss               ce
    --feature            top_rule_confidence
    --standardize_feature
)

# ── run_variant <name> [extra args…] ──────────────────────────────────────────
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
    local wall=$(( t1 - t0 ))

    printf "  Finished : $(date '+%Y-%m-%d %H:%M:%S')  |  wall %dh%02dm%02ds  |  rc=%d\n" \
        $(( wall / 3600 )) $(( (wall % 3600) / 60 )) $(( wall % 60 )) \
        "${VARIANT_RC}"
    echo ""
}

# ── Run variants ───────────────────────────────────────────────────────────────
cd "$HOME/RulE/RulE_original/src"
TRAIN_START=$(date +%s)

# Fixed config: label smoothing matched to FB15k-237's grounding objective (0.2).
run_variant "toprule_ls0.2" \
    --label_smoothing 0.2

TRAIN_END=$(date +%s)
TRAIN_WALL=$(( TRAIN_END - TRAIN_START ))

# Stop GPU monitor
kill "${GPU_MON_PID}" 2>/dev/null
wait "${GPU_MON_PID}" 2>/dev/null

# ── Timing summary ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  All variants complete  (seed=${SEED})"
echo "=================================================="
printf "  Total wall time : %dh%02dm%02ds\n" \
    $(( TRAIN_WALL / 3600 )) $(( (TRAIN_WALL % 3600) / 60 )) $(( TRAIN_WALL % 60 ))

if [ -f "${GPU_LOG}" ]; then
    PEAK_MEM=$(awk '!/^#/ && NF>=2 {val=$2+0; if(val>max) max=val} END{print max+0}' "${GPU_LOG}")
    echo "  Peak GPU mem (MiB) : ${PEAK_MEM}"
fi

echo "  Logs under : ${ABLATION_ROOT}/"
echo "=================================================="
echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"

exit "${VARIANT_RC}"
