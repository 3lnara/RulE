#!/usr/bin/env bash
#SBATCH --job-name=beta_ce_perrel_wn18rr_densnorm
#SBATCH --partition=mcs.gpu.q
#SBATCH --constraint=v100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─────────────────────────────────────────────────────────────────────────────
# Per-relation mixing-weight (beta) training with a CROSS-ENTROPY loss.
#
# This is the cross-entropy counterpart to adaptive_beta_only_wn18rr.slurm.
# Two deliberate differences from that script:
#   1. --loss ce       : trains beta[r] with filtered softmax cross-entropy
#                        (-log softmax(score)[true tail] over the full filtered
#                        candidate set) instead of the pairwise margin/hinge.
#                        This is the listwise objective RulE's GroundTrainer used
#                        to fit the rule scorer and matches the filtered-ranking
#                        eval protocol. Because CE consumes EVERY candidate, the
#                        negative-sampling knobs (--neg_sampling/--num_negatives/
#                        --mixed_hard_frac) are irrelevant and omitted.
#   2. --skip_adaptive : trains/evaluates the per-relation intercept beta[r]
#                        ONLY -- stages [5]/[6]/[7] (the query-adaptive
#                        beta_density head) are removed entirely.
#
# Ablation here therefore varies the two things that still matter under CE:
#   --label_smoothing  (eps; 0.0 = hard target, 0.3 = the grounding default)
#   --beta_l2          (shrink beta[r] toward 0 -> sigmoid 0.5, neutral routing)
# ─────────────────────────────────────────────────────────────────────────────

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
# Reuse the grounding checkpoint produced by the original MLP run -- the same
# rule + KGE heads the margin runs trained beta on top of.
ORIG_OUTPUT="$HOME/RulE/outputs/wn18rr_original"
PRETRAIN_CHECKPOINT="${ORIG_OUTPUT}/checkpoint"
GROUNDING_CHECKPOINT="${ORIG_OUTPUT}/grounding.pt"

# Config file — same one used for the original run
CONFIG="$HOME/RulE/config/wn18rr_hpc.json"

# Output directory for this cross-entropy per-relation run
SAVE_PATH="$HOME/RulE/outputs/beta_ce_perrel_wn18rr_nonorm"
GPU_LOG="${SAVE_PATH}/gpu_memory.log"

# Beta hyper-parameters (stage 3 only; adaptive stage is skipped)
BETA_EPOCHS=10
BETA_LR=0.001
FIXED_ALPHA=1.0   # baseline alpha for the comparison table
SEED=42

# Root directory for per-variant subdirectories
ABLATION_ROOT="${SAVE_PATH}"

echo "Pre-train checkpoint   : ${PRETRAIN_CHECKPOINT}"
echo "Grounding checkpoint   : ${GROUNDING_CHECKPOINT}"
echo "Config                 : ${CONFIG}"
echo "Save path              : ${SAVE_PATH}"
echo "Beta epochs / lr       : ${BETA_EPOCHS} / ${BETA_LR}"
echo "Loss                   : cross-entropy (per-relation only, adaptive disabled)"
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

# ── Shared arguments for every variant ────────────────────────────────────────
# --loss ce + --skip_adaptive are the defining flags of this script. The score
# normalization / feature standardization are kept identical to the margin runs
# so the only changed factor versus those logs is the loss function.
COMMON_ARGS=(
    --config             "${CONFIG}"
    --checkpoint         "${GROUNDING_CHECKPOINT}"
    --epochs             "${BETA_EPOCHS}"
    --lr                 "${BETA_LR}"
    --alpha              "${FIXED_ALPHA}"
    --seed               "${SEED}"
    --loss               ce
    --skip_adaptive
    --standardize_feature
    --feature            density
)

# ── run_variant <name> [extra args…] ──────────────────────────────────────────
# Runs train_beta_grounding_chunked.py with COMMON_ARGS + any extra args.
# Each variant gets its own subdirectory under ABLATION_ROOT and its own
# beta.pt / training.log so results don't overwrite each other.
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

# Plain CE, hard target -- the direct margin -> cross-entropy swap.
# run_variant "ce_ls0_l2_0" \
#     --label_smoothing 0.0 \
#     --beta_l2 0.0
# # CE + label smoothing 0.3 (matches the grounding-stage 'smoothing' default).
# run_variant "ce_ls0.3_l2_0" \
#     --label_smoothing 0.3 \
#     --beta_l2 0.0
# # CE + small L2 shrink toward neutral routing (beta -> 0.5).
# run_variant "ce_ls0_l2_0.01" \
#     --label_smoothing 0.0 \
#     --beta_l2 0.01
# # CE + label smoothing + L2.
run_variant "ce_ls0.3_l2_0.01" \
    --label_smoothing 0.3 \
    --beta_l2 0.01

TRAIN_END=$(date +%s)
TRAIN_WALL=$(( TRAIN_END - TRAIN_START ))

# Stop GPU monitor
kill "${GPU_MON_PID}" 2>/dev/null
wait "${GPU_MON_PID}" 2>/dev/null

# ── Timing summary ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  All variants complete"
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
