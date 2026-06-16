#!/usr/bin/env bash
#SBATCH --job-name=additive_umls_paper_sum
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
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/umls/checkpoint"
SAVE_PATH="$HOME/RulE/outputs/additive_umls_paper_sum"

echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Save path              : ${SAVE_PATH}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    exit 1
fi

mkdir -p "${SAVE_PATH}"

# ── Run faithful paper "sum (w/o MLP)" baseline ────────────────────────────────
# score[t] = sum_R conf_R * 1[c_R[t] > 0]   (binary activation, no bias, frozen conf)
#
# --paper_sum         : binary activation, no per-entity bias, raw frozen RulE
#                       confidence; fully faithful to the paper's Eq. 6 formulation.
#                       Implies --simple_aggregation and disables --learnable_rule_weight.
# --alpha_sweep       : pick the best KGE-fusion weight on the validation split,
#                       then report test / test_kge metrics at that alpha.
# No learnable parameters beyond the pre-trained KGE/rule embeddings (which are
# frozen), so num_iters=1 is enough to run the alpha sweep; use a small budget.
TRAIN_START=$(date +%s)

cd "$HOME/RulE/src_additive"
python3 -u main.py \
    --cuda \
    --init_checkpoint_config "$HOME/RulE/config/umls_hpc.json" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --paper_sum \
    --alpha_sweep \
    --alpha_grid "0,1,2,4,6,8,10,12,16,24" \
    --num_iters_override 1 \
    --seed 42 \
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
