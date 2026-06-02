#!/usr/bin/env bash
#SBATCH --job-name=ablation_fb15k
#SBATCH --partition=mcs.gpu.q
#SBATCH --constraint=v100
#SBATCH --time=14-00:00:00
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

# ── Dataset / seed ─────────────────────────────────────────────────────────────
DATASET="fb15k"
SEED=42

# ── Checkpoints and config ─────────────────────────────────────────────────────
# Both checkpoints are produced by rule_comparison_gpu.slurm (original run):
#   checkpoint    — best KGE + rule embeddings (from PreTrainer)
#   grounding.pt  — best MLP grounding weights (from GroundTrainer)
ORIG_OUTPUT="$HOME/RulE/outputs/fb15k237_original"
PRETRAIN_CHECKPOINT="${ORIG_OUTPUT}/checkpoint"
GROUNDING_CHECKPOINT="${ORIG_OUTPUT}/grounding.pt"

CONFIG="$HOME/RulE/config/fb15k237_hpc.json"

# ── Ablation root — all 6 variant subdirs live here ───────────────────────────
ABLATION_ROOT="$HOME/RulE/outputs/ablation_${DATASET}_seed${SEED}"
GPU_LOG="${ABLATION_ROOT}/gpu_memory.log"

# ── Shared hyper-parameters ────────────────────────────────────────────────────
BETA_EPOCHS=10
BETA_LR=0.01
BETA_DENSITY_LR=0.001
FIXED_ALPHA=3.0
BETA_L2=0
NEGATIVES="mixed"
NUM_NEGATIVES=200
# FB15k237 has 14 541 entities — [32, 14541, 1000] ≈ 0.94 GB, safe on V100.
# WN18RR would OOM at 32; keep at 4 for that dataset.
EVAL_BATCH_SIZE=32

# Shared fixed-alpha per-query cache — computed once by the first variant,
# loaded (skipping step [1]) by every subsequent variant.
PQ_CACHE="${ABLATION_ROOT}/pq_fixed_cache.pt"

echo "Dataset                : ${DATASET}"
echo "Seed                   : ${SEED}"
echo "Pre-train checkpoint   : ${PRETRAIN_CHECKPOINT}"
echo "Grounding checkpoint   : ${GROUNDING_CHECKPOINT}"
echo "Config                 : ${CONFIG}"
echo "Ablation root          : ${ABLATION_ROOT}"
echo "Beta epochs / lr       : ${BETA_EPOCHS} / ${BETA_LR} / ${BETA_DENSITY_LR}"
echo "Beta L2                : ${BETA_L2}"
echo "Negatives              : ${NEGATIVES} (num=${NUM_NEGATIVES})"
echo "Eval batch size        : ${EVAL_BATCH_SIZE}"
echo "Fixed alpha            : ${FIXED_ALPHA}"
echo "pq_cache               : ${PQ_CACHE}"
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

mkdir -p "${ABLATION_ROOT}"

# ── Start GPU memory monitor in background ─────────────────────────────────────
nvidia-smi dmon -s m -d 10 > "${GPU_LOG}" &
GPU_MON_PID=$!
echo "GPU memory monitor PID : ${GPU_MON_PID}  (log: ${GPU_LOG})"
echo ""

# ── Common args shared by all variants ────────────────────────────────────────
# Stored as an array so flags with spaces (e.g. --num_negatives 100) quote correctly.
COMMON_ARGS=(
    --config             "${CONFIG}"
    --checkpoint         "${GROUNDING_CHECKPOINT}"
    --epochs             "${BETA_EPOCHS}"
    --lr                 "${BETA_LR}"
    --density_lr         "${BETA_DENSITY_LR}"
    --alpha              "${FIXED_ALPHA}"
    --seed               "${SEED}"
    --negatives          "${NEGATIVES}"
    --num_negatives      "${NUM_NEGATIVES}"
    --beta_l2            "${BETA_L2}"
    --eval_batch_size    "${EVAL_BATCH_SIZE}"
    --pq_cache           "${PQ_CACHE}"
    --skip_intermediate_eval
)

# ── run_variant NAME [EXTRA_FLAG ...] ─────────────────────────────────────────
# Runs train_beta_grounding_chunked.py for one ablation variant, writing its
# output to ${ABLATION_ROOT}/${NAME}/training.log and its beta checkpoint to
# ${ABLATION_ROOT}/${NAME}/beta.pt.
# Sets VARIANT_RC to the exit code and VARIANT_WALL to elapsed seconds.
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

# ── cd to src/ so imports resolve ─────────────────────────────────────────────
cd "$HOME/RulE/RulE_original/src"

OVERALL_RC=0

# ══════════════════════════════════════════════════════════════════════════════
#  Variants — run sequentially
# ══════════════════════════════════════════════════════════════════════════════

# fixed_alpha runs first: computes step [1] and writes pq_cache so every
# subsequent variant can load it instead of re-running the full test eval.
run_variant "fixed_alpha" \
    --baseline_only
(( OVERALL_RC |= VARIANT_RC ))

# LR sweep: all use per-relation beta + density feature, joint training.
# Negatives/num_negatives/beta_l2/eval_batch_size come from COMMON_ARGS.
run_variant "perrel_density_joint_lr01" \
    --feature density \
    --joint \
    --lr 0.01
(( OVERALL_RC |= VARIANT_RC ))

run_variant "perrel_density_joint_lr005" \
    --feature density \
    --joint \
    --lr 0.005
(( OVERALL_RC |= VARIANT_RC ))

run_variant "perrel_density_joint_lr001" \
    --feature density \
    --joint \
    --lr 0.001
(( OVERALL_RC |= VARIANT_RC ))

# ── Stop GPU monitor ───────────────────────────────────────────────────────────
kill "${GPU_MON_PID}" 2>/dev/null
wait "${GPU_MON_PID}" 2>/dev/null

if [ -f "${GPU_LOG}" ]; then
    PEAK_MEM=$(awk '!/^#/ && NF>=2 {val=$2+0; if(val>max) max=val} END{print max+0}' "${GPU_LOG}")
    echo "Peak GPU memory (MiB) : ${PEAK_MEM}"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Build SUMMARY.txt via the standalone Python summariser
# ══════════════════════════════════════════════════════════════════════════════
echo "=================================================="
echo "  ABLATION SUMMARY"
echo "=================================================="
python3 "$HOME/RulE/scripts/summarize_ablation.py" --root "${ABLATION_ROOT}"
echo "=================================================="
echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Overall exit code: ${OVERALL_RC}"

exit "${OVERALL_RC}"
