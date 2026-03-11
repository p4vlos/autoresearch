#!/usr/bin/env bash
# One-time environment setup on the SLURM cluster.
# Run this on the login node before submitting jobs.
#
# Usage: bash slurm/setup.sh

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
