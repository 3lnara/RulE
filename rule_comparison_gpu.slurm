#!/usr/bin/env bash
#SBATCH --job-name=rule_comparison
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=24:00:00 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e.yerbolatova@student.tue.nl

# ── GPU info ───────────────────────────────────────────────────────────────────
echo "=== GPU info ==="
nvidia-smi
echo "================"

# ── Python environment ─────────────────────────────────────────────────────────
# Load the same modules used when the venv was created (setup_env_hpc.sh).
# Adjust module names if your HPC uses different versions — run `module spider Python`.
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

# ── Run comparison ─────────────────────────────────────────────────────────────
# Stage 1+2A: original RulE (MLP grounding) — pre-trains from scratch
# Stage 2B:   new RulE with GAT grounding   — reuses the checkpoint above
cd "$HOME/RulE/scripts"
python3 run_comparison.py --config ../config/family_comparison_hpc.json

echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
