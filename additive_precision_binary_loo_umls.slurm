#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# UMLS: precision*binary on the LEAVE-ONE-OUT design (additive, in-model).
#
# Apples-to-apples counterpart of additive_precision_binary_umls.slurm for
# comparing precision_binary against the learned logreg_binary aggregator. The
# only difference from that script is Step 1: precision is computed on the SAME
# edge-specific leave-one-out design the logreg fit uses
# (scripts/rule_precision_loo_train.py, which reuses rule_logreg_train.py's
# build_design_matrix) instead of the full-graph rule_precision_train.py. That
# removes the grounding-regime confound, so a precision-vs-logreg MRR gap
# reflects only the aggregation method.
#
# Two GPU steps (idempotent; each skips if its outputs already exist):
#   1. rule_precision_loo_train.py -> rule_precision.pt      (LOO precision)
#   2. src_additive/main.py --precision_binary --dump_ranks   -> ranks_{valid,test}.csv
#                                                              + valid/test MRR
#
# Compare against the logreg ranks (additive_logreg_binary_umls.slurm) OFFLINE
# by joining ranks_{valid,test}.csv on (h, r, t). The two share an identical
# grounding regime, so the join isolates precision vs learned beta_R.
# ───────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=prec_loo_umls
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
REPO="$HOME/RulE"
DATA_PATH="data/umls"                                   # relative to REPO (scripts chdir to repo root)
RULE_FILE="data/umls/mined_rules.txt"
CONFIG="${REPO}/config/umls_hpc.json"
KGE_CHECKPOINT="${REPO}/outputs/umls/checkpoint"        # KGE+rule pretrain (instantiates additive model)
PRECISION_LOO_DIR="${REPO}/outputs/additive_umls_aggregation_2x2/precision_loo"
PRECISION_FILE="${PRECISION_LOO_DIR}/rule_precision.pt"

CHUNK_SIZE="0"                                          # UMLS is small; no chunking needed

mkdir -p "${PRECISION_LOO_DIR}"

echo "Seed             : ${SEED}"
echo "Config           : ${CONFIG}"
echo "KGE checkpoint   : ${KGE_CHECKPOINT}"
echo "Precision-LOO dir: ${PRECISION_LOO_DIR}"
echo "Started          : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

for f in "${KGE_CHECKPOINT}" "${CONFIG}"; do
    if [ ! -e "${f}" ]; then
        echo "ERROR: required input not found: ${f}"
        exit 1
    fi
done

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 (GPU): per-rule LOO precision -> rule_precision.pt
# ══════════════════════════════════════════════════════════════════════════════
if [ -f "${PRECISION_FILE}" ]; then
    echo "[step 1] ${PRECISION_FILE} exists; skipping LOO precision computation."
else
    echo "[step 1] computing rule precision on the LOO design (GPU)..."
    cd "${REPO}"
    python3 -u scripts/rule_precision_loo_train.py \
        --data_path  "${DATA_PATH}" \
        --rule_file  "${RULE_FILE}" \
        --out_dir    "${PRECISION_LOO_DIR}" \
        --device     cuda \
        --chunk_size "${CHUNK_SIZE}"
    S1_RC=$?
    echo "[step 1] exit code: ${S1_RC}"
    [ "${S1_RC}" -eq 0 ] || exit "${S1_RC}"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 (GPU): additive precision*binary in-model eval + rank dump
#   Reuses the existing --precision_binary path unchanged; the LOO-derived
#   rule_precision.pt is drop-in schema-compatible with the full-graph one.
# ══════════════════════════════════════════════════════════════════════════════
if [ -f "${PRECISION_LOO_DIR}/ranks_valid.csv" ] && [ -f "${PRECISION_LOO_DIR}/ranks_test.csv" ]; then
    echo "[step 2] ranks_{valid,test}.csv exist in ${PRECISION_LOO_DIR}; skipping additive run."
else
    echo "[step 2] additive precision*binary (LOO) rank dump (GPU)..."
    cd "${REPO}/src_additive"
    python3 -u main.py \
        --cuda \
        --skip_pretrain \
        --pretrain_checkpoint "${KGE_CHECKPOINT}" \
        --init_checkpoint_config "${CONFIG}" \
        --precision_binary \
        --precision_file "${PRECISION_FILE}" \
        --dump_ranks \
        --num_iters 1 \
        --seed "${SEED}" \
        --save_path "${PRECISION_LOO_DIR}"
    S2_RC=$?
    echo "[step 2] exit code: ${S2_RC}"
    [ "${S2_RC}" -eq 0 ] || exit "${S2_RC}"
fi
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════════"
echo "UMLS precision*binary (LOO design) complete"
echo "  LOO precision : ${PRECISION_FILE}"
echo "  per-rule CSV  : ${PRECISION_LOO_DIR}/rule_precision_loo.csv"
echo "  in-model ranks: ${PRECISION_LOO_DIR}/ranks_{valid,test}.csv"
echo "  valid/test MRR: see 'Predictor: Evaluating on {valid,test}' above."
echo "  Compare vs logreg_binary OFFLINE by joining ranks on (h, r, t)."
echo "══════════════════════════════════════════════════════════════════════════"
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
