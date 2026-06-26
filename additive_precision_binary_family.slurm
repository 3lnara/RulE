#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# Family: precision*binary (additive, in-model) + MLP (original, eval-only) ranks
#
# Produces per-query rank CSVs for two methods so MLP-vs-additive rank wins can
# be compared OFFLINE later (pure pandas, no GPU). All rank production runs
# in-model on GPU so the pipeline scales unchanged to larger datasets.
#
# Three GPU steps (all idempotent; each skips if its outputs already exist):
#   1. rule_precision_train.py  -> rule_precision.pt          (train PCA precision)
#   2. src_additive/main.py --precision_binary --dump_ranks    -> ranks_{valid,test}.csv
#   3. RulE_original/main.py --eval_only --dump_ranks          -> ranks_mlp_{valid,test}.csv
#
# Validation targets (logged by each run):
#   precision*binary valid MRR ~= 0.956762   (offline precision_binary_valid.csv)
#   MLP              valid MRR ~= 0.972564 / test 0.974351   (family_comparison_hpc/run.log)
# ───────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=prec_mlp_family
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
DATA_PATH="data/family"                                  # relative to REPO (scripts chdir to repo root)
RULE_FILE="data/family/mined_rules.txt"
CONFIG="${REPO}/config/family_comparison_hpc.json"
KGE_CHECKPOINT="${REPO}/outputs/family_comparison_hpc/checkpoint"   # KGE+rule pretrain
MLP_GROUNDING="${REPO}/outputs/family_comparison_hpc/grounding.pt"  # trained original MLP grounding head
ANALYSIS_DIR="${REPO}/outputs/additive_family/analysis"
PRECISION_DIR="${REPO}/outputs/additive_family/precision_binary"
MLP_DIR="${REPO}/outputs/family_comparison_hpc/mlp_eval"
PRECISION_FILE="${ANALYSIS_DIR}/rule_precision.pt"

mkdir -p "${ANALYSIS_DIR}" "${PRECISION_DIR}" "${MLP_DIR}"

echo "Seed           : ${SEED}"
echo "Config         : ${CONFIG}"
echo "KGE checkpoint : ${KGE_CHECKPOINT}"
echo "MLP grounding  : ${MLP_GROUNDING}"
echo "Analysis dir   : ${ANALYSIS_DIR}"
echo "Precision dir  : ${PRECISION_DIR}"
echo "MLP eval dir   : ${MLP_DIR}"
echo "Started        : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

for f in "${KGE_CHECKPOINT}" "${MLP_GROUNDING}" "${CONFIG}"; do
    if [ ! -e "${f}" ]; then
        echo "ERROR: required input not found: ${f}"
        exit 1
    fi
done

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 (GPU): per-rule train PCA precision -> rule_precision.pt
# ══════════════════════════════════════════════════════════════════════════════
if [ -f "${PRECISION_FILE}" ]; then
    echo "[step 1] ${PRECISION_FILE} exists; skipping precision computation."
else
    echo "[step 1] computing rule precision (GPU)..."
    cd "${REPO}"
    python3 -u scripts/rule_precision_train.py \
        --data_path "${DATA_PATH}" \
        --rule_file "${RULE_FILE}" \
        --out_dir   "${ANALYSIS_DIR}" \
        --device    cuda \
        --no_plot
    S1_RC=$?
    echo "[step 1] exit code: ${S1_RC}"
    [ "${S1_RC}" -eq 0 ] || exit "${S1_RC}"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 (GPU): additive precision*binary in-model ranks -> ranks_{valid,test}.csv
# ══════════════════════════════════════════════════════════════════════════════
if [ -f "${PRECISION_DIR}/ranks_valid.csv" ] && [ -f "${PRECISION_DIR}/ranks_test.csv" ]; then
    echo "[step 2] ranks_{valid,test}.csv exist in ${PRECISION_DIR}; skipping additive run."
else
    echo "[step 2] additive precision*binary rank dump (GPU)..."
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
        --save_path "${PRECISION_DIR}"
    S2_RC=$?
    echo "[step 2] exit code: ${S2_RC}"
    [ "${S2_RC}" -eq 0 ] || exit "${S2_RC}"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 (GPU): original MLP eval-only ranks -> ranks_mlp_{valid,test}.csv
# ══════════════════════════════════════════════════════════════════════════════
if [ -f "${MLP_DIR}/ranks_mlp_valid.csv" ] && [ -f "${MLP_DIR}/ranks_mlp_test.csv" ]; then
    echo "[step 3] ranks_mlp_{valid,test}.csv exist in ${MLP_DIR}; skipping MLP eval."
else
    echo "[step 3] MLP eval-only rank dump (GPU)..."
    cd "${REPO}/RulE_original/src"
    python3 -u main.py \
        --cuda \
        --init_checkpoint_config "${CONFIG}" \
        --eval_only \
        --dump_ranks \
        --grounding_checkpoint "${MLP_GROUNDING}" \
        --save_path "${MLP_DIR}"
    S3_RC=$?
    echo "[step 3] exit code: ${S3_RC}"
    [ "${S3_RC}" -eq 0 ] || exit "${S3_RC}"
fi
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════════"
echo "Family precision*binary + MLP rank dump complete"
echo "  additive ranks : ${PRECISION_DIR}/ranks_{valid,test}.csv"
echo "  MLP ranks      : ${MLP_DIR}/ranks_mlp_{valid,test}.csv"
echo "  Compare later OFFLINE by joining on (h, r, t). No GPU required."
echo "══════════════════════════════════════════════════════════════════════════"
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
