#!/bin/bash
#SBATCH --job-name=autoresearch-setup
#SBATCH --output=slurm/setup-%j.out
#SBATCH --error=slurm/setup-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=GPU
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicolaou.pavlos@ucy.ac.cy
# One-time environment setup on the SLURM cluster.
#
# Usage: sbatch slurm/setup.sh

set -euo pipefail

echo "=== autoresearch SLURM setup ==="

# Load CUDA module (adjust name to match your cluster)
if command -v module &>/dev/null; then
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || echo "Warning: could not load cuda module"
fi

# Check GPU availability
if command -v nvidia-smi &>/dev/null; then
    echo "GPU info:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "Warning: nvidia-smi not found (expected on compute nodes)"
fi

# Install Python dependencies with uv
echo ""
echo "Installing dependencies..."
if ! command -v uv &>/dev/null; then
    echo "Error: uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
uv sync

# Prepare data (load and cache GLOBEM dataset)
echo ""
echo "Preparing GLOBEM data..."
uv run prepare.py

echo ""
echo "=== Setup complete ==="
echo "Submit experiments with: bash slurm/chain.sh <num_experiments>"
