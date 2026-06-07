#!/usr/bin/env bash
#SBATCH --job-name=additive_fm_umls
#SBATCH --partition=mcs.gpu.q
#SBATCH --constraint=v100
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
# Pre-trained KGE + rule embeddings (same checkpoint used by the GAT runs).
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/umls/checkpoint"

# Output directory for this frozen-linear + FM run.
SAVE_PATH="$HOME/RulE/outputs/additive_umls_fm_lr"

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

# ── Run frozen-linear + FM interactions (headline interdependency experiment) ──
# score[t] = sum_R conf_R * count_R[t]  (frozen linear backbone)
#          + sum_{R<R'} <v_R, v_R'> * 1[c_R[t]>0] * 1[c_R'[t]>0]  (learnable FM)
#          + bias
# --simple_aggregation : frozen-linear backbone (per-rule weight frozen at RulE conf).
# --fm_interactions    : add the learnable pairwise FM term (binary co-fire basis).
# --fm_rank / --fm_l2  : FM latent dim and FM-only L2 (overfitting knob).
# --fm_lr              : dedicated (higher) LR for the FM embeddings. They start ~0 and
#                        converge slowly at the shared g_lr=1e-4, so 1e-3 (10x) lets the
#                        interaction term actually saturate within the iteration budget.
# --num_iters_override : the prior 20-iter run was still climbing at the last iteration
#                        (valid MRR monotonically rising, never plateaued) -> train longer
#                        and let best-valid model selection pick the peak.
# With the linear part frozen, any gain over the --simple_aggregation baseline is
# attributable purely to rule interactions.
TRAIN_START=$(date +%s)

cd "$HOME/RulE/src_additive"
python3 -u main.py \
    --init_checkpoint_config "$HOME/RulE/config/umls_hpc.json" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --simple_aggregation \
    --fm_interactions \
    --fm_rank 16 \
    --fm_l2 1e-5 \
    --fm_lr 1e-3 \
    --num_iters_override 60 \
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
