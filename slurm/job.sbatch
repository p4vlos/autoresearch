#!/bin/bash
#SBATCH --job-name=autoresearch
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=GPU
#SBATCH --time=00:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicolaou.pavlos@ucy.ac.cy

set -euo pipefail

echo "=== autoresearch job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Date: $(date -Iseconds)"
echo ""

# Load CUDA module
if command -v module &>/dev/null; then
    module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
fi

# Run training
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results

uv run train.py
EXIT_CODE=$?

# Write status file for the chain wrapper
echo "${EXIT_CODE}" > "results/status-${SLURM_JOB_ID}.txt"

echo ""
echo "=== Job ${SLURM_JOB_ID} finished with exit code ${EXIT_CODE} ==="
exit ${EXIT_CODE}
