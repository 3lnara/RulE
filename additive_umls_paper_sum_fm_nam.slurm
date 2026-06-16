#!/usr/bin/env bash
#SBATCH --job-name=additive_umls_paper_sum_fm_nam
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
SAVE_PATH="$HOME/RulE/outputs/additive_umls_paper_sum_fm_nam"

echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Save path              : ${SAVE_PATH}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    exit 1
fi

mkdir -p "${SAVE_PATH}"

# ── Run faithful paper_sum + FM interactions + NAM residual ─────────────────────
# score[t] = sum_R conf_R * 1[c_R[t] > 0]                       (faithful binary linear)
#          + sum_{R<R'} <v_R, v_R'> * 1[c_R[t]>0] * 1[c_R'[t]>0]  (learnable FM)
#          + sum_R f_R(c_R[t]) * 1[c_R[t]>0]                     (learnable NAM residual)
#          (no bias)
#
# --paper_sum         : binary activation, no per-entity bias, raw frozen RulE conf.
#                       Implies --simple_aggregation, disables --learnable_rule_weight.
# --fm_interactions   : learnable pairwise FM term over the binary co-fire basis
#                       b_R[t] = 1[c_R[t] > 0] (shared with the linear term).
# --fm_rank 32        : matches the existing additive_umls_fm_lr run for comparability.
# --fm_lr 1e-3        : dedicated higher LR for FM embeddings (start ~0, must grow).
# --nam               : per-rule nonlinear shape-net residual f_R, zero-inited so the
#                       run starts exactly at the paper_sum+FM baseline. f_R(0)=0 via
#                       the 1[c_R[t]>0] mask, so non-firing rules contribute nothing.
# --nam_lr 1e-3       : dedicated higher LR for NAM params (start ~0, like FM).
#
# CAVEAT (faithfulness): unlike the binary FM/linear terms, NAM feeds the RAW path
# counts c_R[t] (phi = [count, log1p(count)]) into its shape-net, so it reintroduces
# count-magnitude information that --paper_sum deliberately discards. A gain over the
# paper_sum+FM run is therefore split between pairwise co-fire (FM) and per-rule
# count-based nonlinearity (NAM); it is NOT a pure co-fire-interaction ablation.
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
    --nam \
    --nam_dim 16 \
    --nam_hidden 64 \
    --nam_l2 1e-5 \
    --nam_lr 1e-3 \
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
