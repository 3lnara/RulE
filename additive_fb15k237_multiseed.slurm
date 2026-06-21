#!/usr/bin/env bash
# ── Multi-seed array job ────────────────────────────────────────────────────
# Each array task = one seed, runs both variants (simple_aggregation+no_bias
# and paper_sum) sequentially within that task.
#
# Submit:  sbatch additive_fb15k237_multiseed.slurm
# Monitor: squeue -u $USER
# ───────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=additive_fb15k237_ms
#SBATCH --partition=mcs.gpu.q
#SBATCH --array=0-2                  # 3 seeds → task IDs 0, 1, 2
#SBATCH --time=3:00:00               # per-task wall time (both variants)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%A_%a.out        # %A = array job ID, %a = task index
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl

# ── Seed table ─────────────────────────────────────────────────────────────
SEEDS=(42 800 1234)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

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
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/fb15k237_original/checkpoint"
CONFIG="$HOME/RulE/config/fb15k237_hpc.json"

echo "Array task ID          : ${SLURM_ARRAY_TASK_ID}"
echo "Seed                   : ${SEED}"
echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Config                 : ${CONFIG}"
echo "Started                : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -e "${PRETRAIN_CHECKPOINT}" ]; then
    echo "ERROR: Pre-trained checkpoint not found at ${PRETRAIN_CHECKPOINT}."
    exit 1
fi

cd "$HOME/RulE/src_additive"

# ══════════════════════════════════════════════════════════════════════════════
# Variant 1: simple_aggregation + no_bias
# ══════════════════════════════════════════════════════════════════════════════
SAVE_PATH_1="$HOME/RulE/outputs/additive_fb15k237_baseline_nobias/seed${SEED}"
mkdir -p "${SAVE_PATH_1}"

echo "──────────────────────────────────────────────────────────────────────────"
echo "Variant 1: simple_aggregation + no_bias  (seed=${SEED})"
echo "Save path : ${SAVE_PATH_1}"
echo "──────────────────────────────────────────────────────────────────────────"

V1_START=$(date +%s)

python3 -u main.py \
    --init_checkpoint_config "${CONFIG}" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --simple_aggregation \
    --no_bias \
    --num_iters 1 \
    --seed "${SEED}" \
    --save_path "${SAVE_PATH_1}"

V1_RC=$?
V1_END=$(date +%s)
V1_WALL=$(( V1_END - V1_START ))

echo ""
echo "Variant 1 exit code : ${V1_RC}"
printf "Variant 1 wall time : %dh%02dm%02ds\n" \
    $(( V1_WALL / 3600 )) $(( (V1_WALL % 3600) / 60 )) $(( V1_WALL % 60 ))
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Variant 2: paper_sum
# ══════════════════════════════════════════════════════════════════════════════
SAVE_PATH_2="$HOME/RulE/outputs/additive_fb15k237_paper_sum/seed${SEED}"
mkdir -p "${SAVE_PATH_2}"

echo "──────────────────────────────────────────────────────────────────────────"
echo "Variant 2: paper_sum  (seed=${SEED})"
echo "Save path : ${SAVE_PATH_2}"
echo "──────────────────────────────────────────────────────────────────────────"

V2_START=$(date +%s)

python3 -u main.py \
    --cuda \
    --init_checkpoint_config "${CONFIG}" \
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \
    --paper_sum \
    --num_iters 1 \
    --seed "${SEED}" \
    --save_path "${SAVE_PATH_2}"

V2_RC=$?
V2_END=$(date +%s)
V2_WALL=$(( V2_END - V2_START ))

echo ""
echo "Variant 2 exit code : ${V2_RC}"
printf "Variant 2 wall time : %dh%02dm%02ds\n" \
    $(( V2_WALL / 3600 )) $(( (V2_WALL % 3600) / 60 )) $(( V2_WALL % 60 ))
echo ""

# ── Summary ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════════"
echo "Seed ${SEED} summary"
echo "  Variant 1 (simple_aggregation+no_bias) : exit ${V1_RC}"
echo "  Variant 2 (paper_sum)                  : exit ${V2_RC}"
echo "══════════════════════════════════════════════════════════════════════════"
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"

[ "${V1_RC}" -eq 0 ] && [ "${V2_RC}" -eq 0 ]
