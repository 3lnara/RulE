#!/usr/bin/env bash
#SBATCH --job-name=beta_ce_perrel_umls_kgemax
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=04:00:00
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
# Adaptive mixing-weight (beta) training on UMLS with the kge_max feature.
#
# The query-adaptive stage MUST run here: kge_max is a PER-QUERY feature and it
# only enters the mix through beta = sigmoid(beta[r] + beta_density * kge_max),
# i.e. via the beta_density slope trained in stage [6]. (This is why
# --skip_adaptive is NOT set -- with it, beta_density stays 0, the feature is
# multiplied by zero, and --feature kge_max becomes a no-op.) The run therefore
# does two stages: [3] trains the per-relation intercept beta[r] (a baseline
# column), then [6] trains beta_density on kge_max on top of it.
#
# Differences from beta_ce_perrel_wn18rr.slurm:
#   1. Dataset: UMLS (small graph, much faster than WN18RR/FB15k-237).
#   2. --feature kge_max  : the adaptive head conditions on the max filtered KGE
#                           score per query rather than grounding density.
#                           kge_max is a raw RotatE score (not bounded [0,1]),
#                           so --standardize_feature is required: z-scores the
#                           feature once on the beta-train split so beta[r] is
#                           the log-odds at the average query and beta_density
#                           has a well-conditioned, dataset-invariant scale.
#   3. NO --skip_adaptive : the adaptive (per-query) stage is enabled -- it is
#                           the whole point of testing a per-query feature.
#   4. FIXED_ALPHA=2.0    : matches the WN18RR/UMLS density-norm runs for direct
#                           comparison of the mixing baseline.
#
# Ablation grid (identical to the WN18RR CE grid):
#   label_smoothing in {0.0, 0.3}  x  beta_l2 in {0.0, 0.01}  = 4 variants.
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
# Grounding checkpoint produced by the original MLP run on UMLS.
ORIG_OUTPUT="$HOME/RulE/outputs/umls"
PRETRAIN_CHECKPOINT="${ORIG_OUTPUT}/checkpoint"
GROUNDING_CHECKPOINT="${ORIG_OUTPUT}/grounding.pt"

CONFIG="$HOME/RulE/config/umls_hpc.json"

SAVE_PATH="$HOME/RulE/outputs/beta_ce_perrel_umls_kgemax_st_nonorm"
GPU_LOG="${SAVE_PATH}/gpu_memory.log"

# Beta hyper-parameters. BETA_LR drives the per-relation intercept (stage 3);
# DENSITY_LR drives the adaptive beta_density slope on kge_max (stage 6).
BETA_EPOCHS=10
BETA_LR=0.001
DENSITY_LR=0.001
FIXED_ALPHA=2.0
SEED=42

ABLATION_ROOT="${SAVE_PATH}"

echo "Pre-train checkpoint   : ${PRETRAIN_CHECKPOINT}"
echo "Grounding checkpoint   : ${GROUNDING_CHECKPOINT}"
echo "Config                 : ${CONFIG}"
echo "Save path              : ${SAVE_PATH}"
echo "Feature                : kge_max (standardized, per-query adaptive)"
echo "Beta epochs / lr       : ${BETA_EPOCHS} / ${BETA_LR} (intercept), ${DENSITY_LR} (density)"
echo "Loss                   : cross-entropy (per-relation intercept + adaptive kge_max)"
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
# --feature kge_max + --standardize_feature are the defining flags of this
# script vs the density-norm CE runs. --skip_adaptive is deliberately NOT set:
# kge_max only takes effect through the adaptive beta_density slope (stage 6).
COMMON_ARGS=(
    --config             "${CONFIG}"
    --checkpoint         "${GROUNDING_CHECKPOINT}"
    --epochs             "${BETA_EPOCHS}"
    --lr                 "${BETA_LR}"
    --density_lr         "${DENSITY_LR}"
    --alpha              "${FIXED_ALPHA}"
    --seed               "${SEED}"
    --loss               ce
    --feature            kge_max
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

# Plain CE, hard target.
run_variant "ce_ls0_l2_0" \
    --label_smoothing 0.0 \
    --beta_l2 0.0
# CE + label smoothing 0.3 (matches the grounding-stage 'smoothing' default).
run_variant "ce_ls0.2_l2_0" \
    --label_smoothing 0.2 \
    --beta_l2 0.0
# CE + small L2 shrink toward neutral routing (beta -> 0.5).
run_variant "ce_ls0_l2_0.01" \
    --label_smoothing 0.0 \
    --beta_l2 0.01
# CE + label smoothing + L2.
run_variant "ce_ls0.2_l2_0.01" \
    --label_smoothing 0.2 \
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
