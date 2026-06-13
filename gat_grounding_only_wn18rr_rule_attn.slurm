#!/usr/bin/env bash
#SBATCH --job-name=gat_grounding_wn18rr_rule_attn
#SBATCH --partition=mcs.gpu.q
#SBATCH --constraint=v100
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
# Option A: drop W_dst AND replace the shared attn_vec with a rule-conditioned
# attn_from_rule(rule_emb). Per-rule scoring vector -> different rules see
# different per-edge attention rankings, even on the same edge set.
#
# WN18RR-specific differences vs the UMLS counterpart:
#   - hidden_dim=500 (vs 2000), so entity_dim=1000, relation_dim=500
#   - g_lr=0.005, alpha=1.0, smoothing=0.3 (from config)
#   - --eval_every_batches 0: per-batch eval disabled (WN18RR full eval is
#     expensive at 40,943 entities x ~3k valid queries; per-iter eval suffices)
#
# ── 16 GB OOM handling ──────────────────────────────────────────────────────────
# The rule_attn forward loops over EVERY rule for the query relation (up to 408
# rules on WN18RR) and each rule body is up to 5 hops, so one batch accumulates
# ~1,400 per-hop attention graphs before a single backward(). On a 16 GB V100 the
# default attn_dim=128 + g_batch_size=32 OOMs inside compute_edge_attention
# (proj = [num_edges, attn_dim], up to ~35k edges for _hypernym).
#
# Fix used here: --checkpoint_grounding gradient-checkpoints the GAT attention so
# its [num_edges, attn_dim] activations are freed after the forward and recomputed
# in backward (exact gradients, ~one extra forward per step). This keeps the full
# attn_dim=128 capacity and fits comfortably in 16 GB. g_batch_size is set to 16
# in wn18rr_gat_hpc.json for extra headroom; with checkpointing you can likely
# raise it back to 32 (check nvidia-smi). NOTE: checkpointing disables the
# attention-entropy regulariser (kept at 0 here), which is fine for this variant.
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/wn18rr_original/checkpoint"

# Output directory for this GAT grounding run
SAVE_PATH="$HOME/RulE/outputs/gat_wn18rr_rule_attn"

echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Save path              : ${SAVE_PATH}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    echo "Run rule_original_wn18rr.slurm first or update PRETRAIN_CHECKPOINT in this script."
    exit 1
fi

mkdir -p "${SAVE_PATH}"

# ── Run GAT grounding (rule_attn variant) ──────────────────────────────────────
TRAIN_START=$(date +%s)

cd "$HOME/RulE/src"
python3 -u main.py \
    --init_checkpoint_config "$HOME/RulE/config/wn18rr_gat_hpc.json" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --use_rule_confidence_variant_a \
    --gat_variant rule_attn \
    --attn_dim 128 \
    --checkpoint_grounding \
    --attn_entropy_weight 0 \
    --eval_every_batches 0 \
    --save_path "${SAVE_PATH}"

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
