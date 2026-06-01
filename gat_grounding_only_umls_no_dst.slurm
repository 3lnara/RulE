#!/usr/bin/env bash
#SBATCH --job-name=gat_grounding_umls_no_dst
#SBATCH --partition=mcs.gpu.q
#SBATCH --constraint=v100
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl

# ── GPU info ───────────────────────────────────────────────────────────────────
# echo "=== GPU info ==="
# nvidia-smi
# echo "================"

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
# Variant 2-clean: drop W_dst (no additive W_rule). Confirms that W_dst was
# redundant under per-target softmax and does not hurt to remove.
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/umls/checkpoint"

# Output directory for this GAT grounding run
SAVE_PATH="$HOME/RulE/outputs/gat_umls_no_dst"

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

# ── Run GAT grounding (no_dst variant) ─────────────────────────────────────────
# Same flags as gat_grounding_only_umls_entropy.slurm except for --gat_variant.
TRAIN_START=$(date +%s)

cd "$HOME/RulE/src"
python3 -u main.py \
    --init_checkpoint_config "$HOME/RulE/config/umls_hpc.json" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --use_rule_confidence_variant_a \
    --gat_variant no_dst \
    --attn_dim 64 \
    --attn_entropy_weight 0 \
    --eval_every_batches 50 \
    --save_path "${SAVE_PATH}"
    # --freeze_conf_proj

TRAIN_RC=$?
TRAIN_END=$(date +%s)
TRAIN_WALL=$(( TRAIN_END - TRAIN_START ))

echo ""
echo "Training exit code : ${TRAIN_RC}"
printf "Training wall time : %dh%02dm%02ds\n" \
    $(( TRAIN_WALL / 3600 )) $(( (TRAIN_WALL % 3600) / 60 )) $(( TRAIN_WALL % 60 ))

if [ "${TRAIN_RC}" -ne 0 ]; then
    echo "ERROR: training failed — skipping metrics parsing."
    echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
    exit "${TRAIN_RC}"
fi

# ── Parse and tabulate per-iteration metrics ───────────────────────────────────
echo ""
echo "=== Per-iteration metrics ==="
python3 "$HOME/RulE/scripts/parse_gat_metrics.py" \
    --log  "${SAVE_PATH}/run.log" \
    --out  "${SAVE_PATH}/metrics_table.txt"

echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
