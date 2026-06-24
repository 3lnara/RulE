#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# Full 2×2 ablation: (paper_sum vs simple+no_bias) × (keep vs clamp) — UMLS
#
# Four back-to-back runs, same seed (42), same pretrain checkpoint.
# Decision metric: valid MRR (higher wins). Test/test_kge reported alongside.
#
# Output layout — one parent dir, four per-variant subfolders:
#   outputs/additive_umls_aggregation_2x2/
#     paper_sum_keep/      --paper_sum
#     paper_sum_clamp/     --paper_sum  --clamp_negative_confidence
#     simple_nobias_keep/  --simple_aggregation  --no_bias
#     simple_nobias_clamp/ --simple_aggregation  --no_bias  --clamp_negative_confidence
#
# Subfolders required: each main.py writes fixed filenames into --save_path
# (run.log, config.json, grounding.pt, g_rule_embedding.npy) and would
# overwrite if pointed at a shared flat directory.
#
# The paper_sum_clamp subfolder is the offline scorer's validation target
# (expected valid MRR 0.647622, test MRR 0.651140).
#
# This job supersedes additive_umls_paper_sum_clamp.slurm.
# ───────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=umls_2x2
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl

SEED=42

# ── Python environment ──────────────────────────────────────────────────────
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

# ── Configuration ────────────────────────────────────────────────────────────
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/umls/checkpoint"
CONFIG="$HOME/RulE/config/umls_hpc.json"
PARENT_DIR="$HOME/RulE/outputs/additive_umls_aggregation_2x2"

echo "Seed                   : ${SEED}"
echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Config                 : ${CONFIG}"
echo "Parent output dir      : ${PARENT_DIR}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    exit 1
fi

cd "$HOME/RulE/src_additive"

# ══════════════════════════════════════════════════════════════════════════════
# Variant A: paper_sum keep  (binary activation, no bias, keep negative w_R)
# ══════════════════════════════════════════════════════════════════════════════
# SAVE_PATH_A="${PARENT_DIR}/paper_sum_keep"
# mkdir -p "${SAVE_PATH_A}"

# echo "──────────────────────────────────────────────────────────────────────────"
# echo "Variant A: paper_sum keep  (seed=${SEED})"
# echo "Save path : ${SAVE_PATH_A}"
# echo "──────────────────────────────────────────────────────────────────────────"

# VA_START=$(date +%s)

# python3 -u main.py \
#     --cuda \
#     --init_checkpoint_config "${CONFIG}" \
#     --skip_pretrain \
#     --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
#     --paper_sum \
#     --num_iters 1 \
#     --seed "${SEED}" \
#     --save_path "${SAVE_PATH_A}"

# VA_RC=$?
# VA_END=$(date +%s)
# VA_WALL=$(( VA_END - VA_START ))

# echo ""
# echo "Variant A exit code : ${VA_RC}"
# printf "Variant A wall time : %dh%02dm%02ds\n" \
#     $(( VA_WALL / 3600 )) $(( (VA_WALL % 3600) / 60 )) $(( VA_WALL % 60 ))
# echo ""

# # ══════════════════════════════════════════════════════════════════════════════
# # Variant B: paper_sum clamp  (binary, no bias, w_R = max(0, w_R))
# # ══════════════════════════════════════════════════════════════════════════════
# SAVE_PATH_B="${PARENT_DIR}/paper_sum_clamp"
# mkdir -p "${SAVE_PATH_B}"

# echo "──────────────────────────────────────────────────────────────────────────"
# echo "Variant B: paper_sum clamp (w_R = max(0, w_R))  (seed=${SEED})"
# echo "Save path : ${SAVE_PATH_B}"
# echo "──────────────────────────────────────────────────────────────────────────"

# VB_START=$(date +%s)

# python3 -u main.py \
#     --cuda \
#     --init_checkpoint_config "${CONFIG}" \
#     --skip_pretrain \
#     --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
#     --paper_sum \
#     --clamp_negative_confidence \
#     --num_iters 1 \
#     --seed "${SEED}" \
#     --save_path "${SAVE_PATH_B}"

# VB_RC=$?
# VB_END=$(date +%s)
# VB_WALL=$(( VB_END - VB_START ))

# echo ""
# echo "Variant B exit code : ${VB_RC}"
# printf "Variant B wall time : %dh%02dm%02ds\n" \
#     $(( VB_WALL / 3600 )) $(( (VB_WALL % 3600) / 60 )) $(( VB_WALL % 60 ))
# echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Variant C: simple aggregation + no_bias + keep  (raw counts, keep negative w_R)
# ══════════════════════════════════════════════════════════════════════════════
SAVE_PATH_C="${PARENT_DIR}/simple_nobias_keep"
mkdir -p "${SAVE_PATH_C}"

echo "──────────────────────────────────────────────────────────────────────────"
echo "Variant C: simple_aggregation no_bias keep  (seed=${SEED})"
echo "Save path : ${SAVE_PATH_C}"
echo "──────────────────────────────────────────────────────────────────────────"

VC_START=$(date +%s)

python3 -u main.py \
    --cuda \
    --init_checkpoint_config "${CONFIG}" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --simple_aggregation \
    --no_bias \
    --num_iters 1 \
    --seed "${SEED}" \
    --save_path "${SAVE_PATH_C}"

VC_RC=$?
VC_END=$(date +%s)
VC_WALL=$(( VC_END - VC_START ))

echo ""
echo "Variant C exit code : ${VC_RC}"
printf "Variant C wall time : %dh%02dm%02ds\n" \
    $(( VC_WALL / 3600 )) $(( (VC_WALL % 3600) / 60 )) $(( VC_WALL % 60 ))
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Variant D: simple aggregation + no_bias + clamp  (raw counts, w_R = max(0, w_R))
# ══════════════════════════════════════════════════════════════════════════════
SAVE_PATH_D="${PARENT_DIR}/simple_nobias_clamp"
mkdir -p "${SAVE_PATH_D}"

echo "──────────────────────────────────────────────────────────────────────────"
echo "Variant D: simple_aggregation no_bias clamp (w_R = max(0, w_R))  (seed=${SEED})"
echo "Save path : ${SAVE_PATH_D}"
echo "──────────────────────────────────────────────────────────────────────────"

VD_START=$(date +%s)

python3 -u main.py \
    --cuda \
    --init_checkpoint_config "${CONFIG}" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --simple_aggregation \
    --no_bias \
    --clamp_negative_confidence \
    --num_iters 1 \
    --seed "${SEED}" \
    --save_path "${SAVE_PATH_D}"

VD_RC=$?
VD_END=$(date +%s)
VD_WALL=$(( VD_END - VD_START ))

echo ""
echo "Variant D exit code : ${VD_RC}"
printf "Variant D wall time : %dh%02dm%02ds\n" \
    $(( VD_WALL / 3600 )) $(( (VD_WALL % 3600) / 60 )) $(( VD_WALL % 60 ))
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════════"
echo "Seed ${SEED} summary (UMLS aggregation 2x2 ablation)"
echo "  Variant A (paper_sum keep)        : exit ${VA_RC}"
echo "  Variant B (paper_sum clamp)        : exit ${VB_RC}  <- offline scorer validation target"
echo "  Variant C (simple_nobias keep)     : exit ${VC_RC}"
echo "  Variant D (simple_nobias clamp)    : exit ${VD_RC}  <- raw_clamp cross-check"
echo "══════════════════════════════════════════════════════════════════════════"
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"

# [ "${VA_RC}" -eq 0 ] && [ "${VB_RC}" -eq 0 ] && 
[ "${VC_RC}" -eq 0 ] && [ "${VD_RC}" -eq 0 ]
