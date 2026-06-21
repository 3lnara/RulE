#!/usr/bin/env bash
#SBATCH --job-name=additive_lever1_family
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=2:00:00
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
# Pre-trained KGE + rule embeddings from the family comparison run.
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/family_comparison_hpc/checkpoint"

# Root output directory; each variant writes into its own subdirectory.
ABLATION_ROOT="$HOME/RulE/outputs/additive_family"

CONFIG="$HOME/RulE/config/family_comparison_hpc.json"

echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Ablation root          : ${ABLATION_ROOT}"
echo "Config                 : ${CONFIG}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    echo "Run the full comparison job first or update PRETRAIN_CHECKPOINT in this script."
    exit 1
fi

mkdir -p "${ABLATION_ROOT}"

# ── run_variant <name> [extra args…] ──────────────────────────────────────────
run_variant() {
    local name="$1"
    shift
    local variant_dir="${ABLATION_ROOT}/${name}"
    mkdir -p "${variant_dir}"

    echo "────────────────────────────────────────────────────────────────"
    echo "  Variant : ${name}"
    echo "  Args    : $*"
    echo "  Dir     : ${variant_dir}"
    echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "────────────────────────────────────────────────────────────────"

    local t0
    t0=$(date +%s)

    python3 -u main.py \
        --init_checkpoint_config "${CONFIG}" \
        --skip_pretrain \
        --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
        --num_iters 1 \
        --save_path "${variant_dir}" \
        "$@" \
        2>&1 | tee "${variant_dir}/training.log"

    local rc=${PIPESTATUS[0]}
    local t1
    t1=$(date +%s)
    local wall=$(( t1 - t0 ))

    printf "  Finished : $(date '+%Y-%m-%d %H:%M:%S')  |  wall %dh%02dm%02ds  |  rc=%d\n" \
        $(( wall / 3600 )) $(( (wall % 3600) / 60 )) $(( wall % 60 )) \
        "${rc}"
    echo ""

    LAST_RC=${rc}
}

# ── Run variants ───────────────────────────────────────────────────────────────
cd "$HOME/RulE/src_additive"
TRAIN_START=$(date +%s)

# Variant 1: simple additive sum, no bias — pure count-weighted score with no
#            per-entity popularity prior.
run_variant "simple_aggregation_nobias" \
    --simple_aggregation \
    --no_bias

# Variant 2: paper_sum — additive sum following the paper's original formulation
#            (includes bias; implies --simple_aggregation).
run_variant "paper_sum" \
    --paper_sum

TRAIN_END=$(date +%s)
TRAIN_WALL=$(( TRAIN_END - TRAIN_START ))

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  All variants complete"
echo "=================================================="
printf "  Total wall time : %dh%02dm%02ds\n" \
    $(( TRAIN_WALL / 3600 )) $(( (TRAIN_WALL % 3600) / 60 )) $(( TRAIN_WALL % 60 ))
echo "  Logs under : ${ABLATION_ROOT}/"
echo "=================================================="
echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"

exit "${LAST_RC}"
