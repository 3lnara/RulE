#!/usr/bin/env bash
# setup_env_hpc.sh — one-time setup: create virtualenv + install packages on TU/e HPC
# Run this interactively on the HPC login node BEFORE submitting any SLURM jobs:
#   ssh 20252643@hpc.tue.nl
#   bash ~/RulE/setup_env_hpc.sh
#
# After it completes successfully, submit jobs with:
#   sbatch ~/rule_comparison_gpu.slurm

set -euo pipefail

VENV="$HOME/RulE/rule_env"

echo "=== Setting up RulE Python environment ==="
echo "Target venv: ${VENV}"
echo ""

# ── Load modules ───────────────────────────────────────────────────────────────
# Detect available Python 3.10+ module; adjust if your HPC uses a different name.
# Run `module spider Python` on the HPC to list available versions.
echo "--- Loading modules ---"
module purge

# Try to load Python 3.10; fall back to checking what is available
if module load Python/3.10.4-GCCcore-11.3.0 2>/dev/null; then
    echo "Loaded Python/3.10.4-GCCcore-11.3.0"
elif module load Python/3.9.6-GCCcore-11.2.0 2>/dev/null; then
    echo "Loaded Python/3.9.6-GCCcore-11.2.0"
else
    echo "ERROR: Could not load a Python module. Run 'module spider Python' to find the right name."
    exit 1
fi

# Load CUDA (needed so pip can pick the right torch wheel)
if module load CUDA/12.1.1 2>/dev/null; then
    CUDA_TAG="cu121"
    echo "Loaded CUDA/12.1.1 → using torch+cu121 wheels"
elif module load CUDA/12.1.0 2>/dev/null; then
    CUDA_TAG="cu121"
    echo "Loaded CUDA/12.1.0 → using torch+cu121 wheels"
elif module load CUDA/11.8.0 2>/dev/null; then
    CUDA_TAG="cu118"
    echo "Loaded CUDA/11.8.0 → using torch+cu118 wheels"
elif module load CUDA/11.7.0 2>/dev/null; then
    CUDA_TAG="cu117"
    echo "Loaded CUDA/11.7.0 → using torch+cu117 wheels"
else
    echo "WARNING: No CUDA module loaded. Trying cuda runtime from PATH."
    # Detect from nvcc if available
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        CUDA_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"
        echo "Detected CUDA ${CUDA_VER} from nvcc → using torch+${CUDA_TAG} wheels"
    else
        echo "ERROR: Could not detect CUDA. Run 'module spider CUDA' to find the right name."
        exit 1
    fi
fi

echo ""
echo "Python  : $(which python3) — $(python3 --version)"
echo "CUDA tag: ${CUDA_TAG}"
echo ""

# ── Create virtualenv ──────────────────────────────────────────────────────────
echo "--- Creating virtualenv ---"
if [ -d "${VENV}" ]; then
    echo "Removing existing venv at ${VENV}"
    rm -rf "${VENV}"
fi
python3 -m venv "${VENV}"
source "${VENV}/bin/activate"
pip install --upgrade pip wheel
echo ""

# ── Install PyTorch (CUDA-aware) ───────────────────────────────────────────────
echo "--- Installing PyTorch ---"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
pip install torch==2.1.2+${CUDA_TAG} torchvision==0.16.2+${CUDA_TAG} torchaudio==2.1.2+${CUDA_TAG} \
    --extra-index-url "${TORCH_INDEX}"
echo ""

# ── Install torch-scatter (must match torch+CUDA version) ─────────────────────
echo "--- Installing torch-scatter ---"
pip install torch-scatter \
    -f "https://data.pyg.org/whl/torch-2.1.2+${CUDA_TAG}.html"
echo ""

# ── Install remaining dependencies ────────────────────────────────────────────
echo "--- Installing remaining dependencies ---"
pip install easydict "numpy<2" PyYAML tqdm matplotlib
echo ""

# ── Verify installation ────────────────────────────────────────────────────────
echo "--- Verifying installation ---"
python3 - <<'PYCHECK'
import torch
from torch_scatter import scatter_softmax
import easydict, yaml, numpy, tqdm, matplotlib
print(f"torch         : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"torch-scatter : OK")
print(f"easydict      : OK")
print(f"All checks passed.")
PYCHECK

echo ""
echo "=== Setup complete. Venv ready at: ${VENV} ==="
echo "You can now submit jobs: sbatch ~/rule_comparison_gpu.slurm"
