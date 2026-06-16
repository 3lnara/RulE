#!/usr/bin/env bash
#SBATCH --job-name=rule_original_umls
#SBATCH --partition=mcs.gpu.q
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
CONFIG="$HOME/RulE/config/umls_hpc.json"
PRETRAIN_CHECKPOINT="$HOME/RulE/outputs/umls/checkpoint"
SAVE_PATH="/vast.mnt/home/20252643/RulE/outputs/umls"
mkdir -p "${SAVE_PATH}"

echo "Config     : ${CONFIG}"
echo "Pre-trained checkpoint : ${PRETRAIN_CHECKPOINT}"
echo "Save path  : ${SAVE_PATH}"
echo "Started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── Run original RulE (pre-train + MLP grounding) ─────────────────────────────
TRAIN_START=$(date +%s)

cd "$HOME/RulE/RulE_original/src"
python3 -u main.py \
    --init_checkpoint_config "${CONFIG}"\
    --skip_pretrain \
    --pretrain_checkpoint "${PRETRAIN_CHECKPOINT}" \

TRAIN_RC=$?
TRAIN_END=$(date +%s)
TRAIN_WALL=$(( TRAIN_END - TRAIN_START ))

echo ""
echo "Training exit code : ${TRAIN_RC}"
printf "Training wall time : %dh%02dm%02ds\n" \
    $(( TRAIN_WALL / 3600 )) $(( (TRAIN_WALL % 3600) / 60 )) $(( TRAIN_WALL % 60 ))

echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
