#!/usr/bin/env bash
#SBATCH --job-name=additive_baseline_umls
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl

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
# Pre-trained KGE + rule embeddings (same checkpoint used by the GAT runs).
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/umls/checkpoint"

# Output directory for this interpretable additive (Lever 1) run.
SAVE_PATH="$HOME/RulE/outputs/additive_umls_baseline_nobias"

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

# ── Run interpretable additive grounding (Lever 1: learnable per-rule weight) ───
# Simple additive aggregation replaces the MLP:  score[t] = sum_R w_R * count_R[t] + bias.
# --simple_aggregation     : use the additive sum instead of FuncToNodeSum + score_model.
# --learnable_rule_weight  : make w_R trainable (implies --simple_aggregation),
#                            warm-started from the RulE confidence.
# Ablations:
#   * frozen additive baseline : drop --learnable_rule_weight (keep --simple_aggregation).
#   * original MLP baseline    : drop both flags (== RulE_original).
TRAIN_START=$(date +%s)

cd "$HOME/RulE/src_additive"
python3 -u main.py \
    --init_checkpoint_config "$HOME/RulE/config/umls_hpc.json" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --simple_aggregation \
    --no_bias \
    --num_iters 1 \
    --save_path "${SAVE_PATH}"

TRAIN_RC=$?
TRAIN_END=$(date +%s)
TRAIN_WALL=$(( TRAIN_END - TRAIN_START ))

echo ""
echo "Training exit code : ${TRAIN_RC}"
printf "Training wall time : %dh%02dm%02ds\n" \
    $(( TRAIN_WALL / 3600 )) $(( (TRAIN_WALL % 3600) / 60 )) $(( TRAIN_WALL % 60 ))

echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
