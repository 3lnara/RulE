#!/usr/bin/env bash
#SBATCH --job-name=feature_validity_family
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── GPU info ───────────────────────────────────────────────────────────────────
echo "=== GPU info ==="
nvidia-smi
echo "================"

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
# Reuse the grounding checkpoint produced by the original MLP run -- the screen
# scores test queries with the same rule + KGE heads beta training sees.
ORIG_OUTPUT="$HOME/RulE/outputs/family_comparison_hpc"
GROUNDING_CHECKPOINT="${ORIG_OUTPUT}/grounding.pt"

CONFIG="$HOME/RulE/config/family_comparison_hpc.json"

SAVE_PATH="$HOME/RulE/outputs/feature_validity_family_grid"
mkdir -p "${SAVE_PATH}"

echo "Grounding checkpoint   : ${GROUNDING_CHECKPOINT}"
echo "Config                 : ${CONFIG}"
echo "Save path              : ${SAVE_PATH}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── Validate checkpoint ────────────────────────────────────────────────────────
if [ ! -f "${GROUNDING_CHECKPOINT}" ]; then
    echo "ERROR: Grounding checkpoint not found at ${GROUNDING_CHECKPOINT}"
    echo "       Run the original comparison run first to produce grounding.pt."
    exit 1
fi

# ── Run feature-validity screen ────────────────────────────────────────────────
# feature_validity_screen.py imports from RulE_original/src (data, model, utils),
# so it must run with that package on the Python path.
cd "$HOME/RulE/RulE_original/src"
python3 -u "$HOME/RulE/scripts/feature_validity_screen.py" \
    --config "${CONFIG}" \
    --checkpoint "${GROUNDING_CHECKPOINT}" \
    --split valid \
    --features "density,num_rules,kge_margin,kge_entropy,kge_max,rule_margin,rule_entropy, conf_rules, rule_kge_disagreement, top_rule_confidence, rule_concentration, cand_concentration" \
    --num_buckets 5 \
    --output "${SAVE_PATH}/feature_validity.csv" \
    --seed 42 \
    2>&1 | tee "${SAVE_PATH}/feature_validity.log"

RC=${PIPESTATUS[0]}
echo ""
echo "=================================================="
echo "  Feature-validity screen complete"
echo "=================================================="
echo "  Exit code  : ${RC}"
echo "  CSV        : ${SAVE_PATH}/feature_validity.csv"
echo "  Full log   : ${SAVE_PATH}/feature_validity.log"
echo "Job finished : $(date '+%Y-%m-%d %H:%M:%S')"

exit "${RC}"
