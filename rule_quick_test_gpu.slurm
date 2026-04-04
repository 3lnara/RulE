#!/usr/bin/env bash
#SBATCH --job-name=rule_quick_gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Check GPU is visible ───────────────────────────────────────────────────────
echo "=== GPU info ==="
nvidia-smi
echo "================"

# ── Python environment ─────────────────────────────────────────────────────────
# Try the project virtualenv first; fall back to whatever Python 3 is on PATH.
VENV="$HOME/RulE/rule_env"
if [ -f "${VENV}/bin/activate" ]; then
    echo "Activating virtualenv: ${VENV}"
    source "${VENV}/bin/activate"
else
    echo "Virtualenv not found at ${VENV}."
    echo "Available Python modules (run 'module avail python' to see full list):"
    module avail python 2>&1 | head -20 || true
    echo ""
    echo "Loading a Python module — edit this script to pin the right one once known."
    module load Python 2>/dev/null || true
fi

echo "Python   : $(which python3)"
echo "Version  : $(python3 --version)"
echo "PyTorch  : $(python3 -c 'import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available(), "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# ── Run RulE ───────────────────────────────────────────────────────────────────
cd "$HOME/RulE/src"
python3 main.py --init_checkpoint_config ../config/family_quick_test_gpu.json

echo ""
echo "Job finished: $(date '+%Y-%m-%d %H:%M:%S')"
