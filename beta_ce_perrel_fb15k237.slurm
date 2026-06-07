#!/usr/bin/env bash
#SBATCH --job-name=beta_ce_perrel_fb15k237_norm
#SBATCH --partition=mcs.gpu.q
#SBATCH --constraint=v100
#SBATCH --time=28:00:00
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
# Per-relation mixing-weight (beta) training with CROSS-ENTROPY loss — FB15k-237.
#
# Direct FB15k-237 counterpart of beta_ce_perrel_wn18rr.slurm. Key differences:
#
#   Dataset differences:
#     - 14 541 entities, hidden_dim=1000 (WN18RR: 40 943 / 500) — GPU mem per
#       query [batch, 14541, 2000] ≈ 0.47 GB at batch=4; safe on V100.
#     - 17 535 valid / 20 466 test facts (WN18RR: 3 034 / 3 134) — ~5-6× more
#       training queries per epoch, so wall-time/variant is much higher.
#     - 131 883 mined rules vs 7 386 for WN18RR — richer rule signal.
#
#   Hyper-parameter differences vs the WN18RR script:
#     - FIXED_ALPHA = 3.0   (RulE's FB15k operating point; WN18RR used 1.0).
#                            [1] baseline therefore reflects the actual published
#                            RulE mixing rather than equal weights — the
#                            per-relation beta is compared against something
#                            meaningful.
#     - BETA_EPOCHS = 6     WN18RR valid-select MRR plateaued by epoch ~7;
#                            FB15k is ~5× slower per epoch, so 6 epochs keeps
#                            total wall-time within 24h for 3 variants.
#     - label_smoothing=0.2  FB15k's grounding stage used smoothing=0.2
#                            (config/fb15k237_hpc.json, vs 0.3 for WN18RR).
#
#   Ablation structure (3 variants, ~8h each):
#     ce_ls0.2_l2_0    -- canonical match to the FB15k grounding objective
#     ce_ls0.2_l2_0.01 -- same + light shrink toward beta=0.5 (neutral)
#     ce_ls0_l2_0      -- hard-target control: shows smoothing effect on FB15k
#
#   Fixed across all variants:
#     --loss ce, --skip_adaptive, --normalize_scores, --standardize_feature,
#     --feature density
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
# Reuse the grounding checkpoint from the original FB15k-237 run — the same
# rule + KGE heads used by feature_validity_fb15k.slurm.
ORIG_OUTPUT="$HOME/RulE/outputs/fb15k237_original"
GROUNDING_CHECKPOINT="${ORIG_OUTPUT}/grounding.pt"

# The pretrain checkpoint is used only for the guard below — the beta training
# script loads the grounding checkpoint directly.
PRETRAIN_CHECKPOINT="${ORIG_OUTPUT}/checkpoint"

CONFIG="$HOME/RulE/config/fb15k237_hpc.json"

SAVE_PATH="$HOME/RulE/outputs/beta_ce_perrel_fb15k237_norm"
GPU_LOG="${SAVE_PATH}/gpu_memory.log"

# Beta hyper-parameters
BETA_EPOCHS=6      # plateau is typically reached by ep 6-7; saves ~30% wall-time
BETA_LR=0.001
FIXED_ALPHA=3.0    # FB15k's published operating point (config: alpha=3.0)
SEED=42

ABLATION_ROOT="${SAVE_PATH}"

echo "Pre-train checkpoint   : ${PRETRAIN_CHECKPOINT}"
echo "Grounding checkpoint   : ${GROUNDING_CHECKPOINT}"
echo "Config                 : ${CONFIG}"
echo "Save path              : ${SAVE_PATH}"
echo "Beta epochs / lr       : ${BETA_EPOCHS} / ${BETA_LR}"
echo "Fixed alpha (baseline) : ${FIXED_ALPHA}"
echo "Loss                   : cross-entropy (per-relation only, adaptive disabled)"
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

# ── Shared arguments for every variant ────────────────────────────────────────
COMMON_ARGS=(
    --config             "${CONFIG}"
    --checkpoint         "${GROUNDING_CHECKPOINT}"
    --epochs             "${BETA_EPOCHS}"
    --lr                 "${BETA_LR}"
    --alpha              "${FIXED_ALPHA}"
    --seed               "${SEED}"
    --loss               ce
    --skip_adaptive
    --normalize_scores
    --standardize_feature
    --feature            density
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

# Canonical: CE with smoothing=0.2, matching FB15k's grounding-stage objective.
run_variant "ce_ls0.2_l2_0" \
    --label_smoothing 0.2 \
    --beta_l2 0.0

# Same + light L2 shrink toward neutral routing (beta -> 0.5 / equal weight).
# Was marginally best on WN18RR (MRR 0.5166 vs 0.5164).
run_variant "ce_ls0.2_l2_0.01" \
    --label_smoothing 0.2 \
    --beta_l2 0.01

# Hard-target control: shows whether smoothing helps on FB15k.
# Compare directly to ce_ls0.2_l2_0 — the only difference is label_smoothing.
run_variant "ce_ls0_l2_0" \
    --label_smoothing 0.0 \
    --beta_l2 0.0

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
