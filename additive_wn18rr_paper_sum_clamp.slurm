#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# Ablation: paper_sum keep-confidence vs clamp-negative-confidence (WN18RR)
# Two back-to-back runs, same seed, same checkpoint.
# Decision metric: valid MRR (higher wins).  Test/test_kge reported alongside.
# ───────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=wn18rr_ps_clamp
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=3:00:00
#SBATCH --constraint=v100
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
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/wn18rr_original/checkpoint"
CONFIG="$HOME/RulE/config/wn18rr_hpc.json"

echo "Seed                   : ${SEED}"
echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Config                 : ${CONFIG}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    exit 1
fi

cd "$HOME/RulE/src_additive"

# ══════════════════════════════════════════════════════════════════════════════
# Variant A: paper_sum  --  keep confidence as-is (baseline)
# ══════════════════════════════════════════════════════════════════════════════
SAVE_PATH_A="$HOME/RulE/outputs/additive_wn18rr_paper_sum_keep/seed${SEED}"
mkdir -p "${SAVE_PATH_A}"

echo "──────────────────────────────────────────────────────────────────────────"
echo "Variant A: paper_sum keep  (seed=${SEED})"
echo "Save path : ${SAVE_PATH_A}"
echo "──────────────────────────────────────────────────────────────────────────"

VA_START=$(date +%s)

python3 -u main.py \
    --cuda \
    --init_checkpoint_config "${CONFIG}" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --paper_sum \
    --num_iters 1 \
    --seed "${SEED}" \
    --save_path "${SAVE_PATH_A}"

VA_RC=$?
VA_END=$(date +%s)
VA_WALL=$(( VA_END - VA_START ))

echo ""
echo "Variant A exit code : ${VA_RC}"
printf "Variant A wall time : %dh%02dm%02ds\n" \
    $(( VA_WALL / 3600 )) $(( (VA_WALL % 3600) / 60 )) $(( VA_WALL % 60 ))
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Variant B: paper_sum  --  clamp negative confidences to zero
# ══════════════════════════════════════════════════════════════════════════════
SAVE_PATH_B="$HOME/RulE/outputs/additive_wn18rr_paper_sum_clamp/seed${SEED}"
mkdir -p "${SAVE_PATH_B}"

echo "──────────────────────────────────────────────────────────────────────────"
echo "Variant B: paper_sum clamp (w_R = max(0, w_R))  (seed=${SEED})"
echo "Save path : ${SAVE_PATH_B}"
echo "──────────────────────────────────────────────────────────────────────────"

VB_START=$(date +%s)

python3 -u main.py \
    --cuda \
    --init_checkpoint_config "${CONFIG}" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --paper_sum \
    --clamp_negative_confidence \
    --num_iters 1 \
    --seed "${SEED}" \
    --save_path "${SAVE_PATH_B}"

VB_RC=$?
VB_END=$(date +%s)
VB_WALL=$(( VB_END - VB_START ))

echo ""
echo "Variant B exit code : ${VB_RC}"
printf "Variant B wall time : %dh%02dm%02ds\n" \
    $(( VB_WALL / 3600 )) $(( (VB_WALL % 3600) / 60 )) $(( VB_WALL % 60 ))
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════════"
echo "Seed ${SEED} summary (WN18RR paper_sum clamp ablation)"
echo "  Variant A (keep)  : exit ${VA_RC}"
echo "  Variant B (clamp) : exit ${VB_RC}"
echo "══════════════════════════════════════════════════════════════════════════"
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"

[ "${VA_RC}" -eq 0 ] && [ "${VB_RC}" -eq 0 ]
