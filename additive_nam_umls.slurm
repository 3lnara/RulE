#!/usr/bin/env bash
#SBATCH --job-name=additive_nam_umls
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
SEED=42
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/umls/checkpoint"
SAVE_PATH="$HOME/RulE/outputs/additive_umls_nam"

echo "Seed                   : ${SEED}"
echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Save path              : ${SAVE_PATH}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    exit 1
fi

mkdir -p "${SAVE_PATH}"

# ── Run NAM + FM (headline interpretable run) ──────────────────────────────────
#
# Score decomposition (all terms additive -> clean rule attribution):
#   score[t] = bias
#            + sum_R  w_R * count_R[t]               (frozen RulE-confidence backbone)
#            + sum_R  f_R(count_R[t])                (NAM residual; f_R(0)=0 by mask)
#            + sum_{R<R'} <v_R,v_R'> * 1[c_R>0]*1[c_R'>0]  (FM pairwise)
#
# Key flags:
#   --simple_aggregation   frozen-linear backbone (w_R = RulE confidence)
#   --fm_interactions      pairwise FM term (binary co-fire basis)
#   --fm_rank 16           FM latent dimension
#   --fm_l2 1e-5           FM-only weight decay
#   --fm_lr 1e-3           FM dedicated LR (10x base; FM starts ~0 -> needs boost)
#   --nam                  NAM residual shape functions
#   --nam_dim 16           dim of per-rule conditioning embedding z_R
#   --nam_hidden 64        hidden width of the shared shape-net
#   --nam_lr 1e-3          NAM dedicated LR (same rationale as FM)
#   --nam_l2 1e-5          NAM-only weight decay
#   --alpha_sweep          pick best KGE-fusion alpha on VALID set, report test_kge
#   --early_stop_patience 8  stop if no valid MRR improvement for 8 iters
#   --num_iters_override 60  hard cap; early stopping will fire first in practice
#
# Ablation variants (toggle flags on/off):
#   FM-only:  drop  --nam --nam_dim --nam_hidden --nam_lr --nam_l2
#   NAM-only: drop  --fm_interactions --fm_rank --fm_l2 --fm_lr
#   Baseline: drop both FM and NAM blocks (keep --simple_aggregation only)

TRAIN_START=$(date +%s)

cd "$HOME/RulE/src_additive"
python3 -u main.py \
    --cuda \
    --init_checkpoint_config "$HOME/RulE/config/umls_hpc.json" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --simple_aggregation \
    --nam \
    --nam_dim 16 \
    --nam_hidden 64 \
    --nam_lr 3e-4 \
    --nam_l2 1e-5 \
    --alpha_sweep \
    --alpha_grid "0,1,2,4,6,8,10,12,16,24" \
    --early_stop_patience 8 \
    --num_iters_override 60 \
    --seed "${SEED}" \
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
