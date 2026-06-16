#!/usr/bin/env bash
#SBATCH --job-name=additive_umls_paper_sum_fm
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=02:00:00
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
SAVE_PATH="$HOME/RulE/outputs/additive_umls_paper_sum_fm"

echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Save path              : ${SAVE_PATH}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    exit 1
fi

mkdir -p "${SAVE_PATH}"

# ── Run faithful paper_sum + FM interactions ────────────────────────────────────
# score[t] = sum_R conf_R * 1[c_R[t] > 0]                 (faithful binary linear)
#          + sum_{R<R'} <v_R, v_R'> * 1[c_R[t]>0] * 1[c_R'[t]>0]  (learnable FM)
#          (no bias)
#
# --paper_sum         : binary activation, no per-entity bias, raw frozen RulE conf.
#                       Implies --simple_aggregation, disables --learnable_rule_weight.
# --fm_interactions   : add the learnable pairwise FM term on top of the faithful
#                       binary linear backbone. The binary basis is shared with the
#                       linear term: b_R[t] = 1[c_R[t] > 0].
# --fm_rank 32        : matches the existing additive_umls_fm_lr run for comparability.
# --fm_lr 1e-3        : dedicated higher LR for FM embeddings (start ~0, must grow).
# Any gain over the paper_sum baseline is attributable purely to rule co-fire interactions.
TRAIN_START=$(date +%s)

cd "$HOME/RulE/src_additive"
python3 -u main.py \
    --cuda \
    --init_checkpoint_config "$HOME/RulE/config/umls_hpc.json" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --paper_sum \
    --fm_interactions \
    --fm_rank 32 \
    --fm_l2 1e-5 \
    --fm_lr 1e-3 \
    --alpha_sweep \
    --alpha_grid "0,1,2,4,6,8,10,12,16,24" \
    --early_stop_patience 8 \
    --num_iters_override 60 \
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
