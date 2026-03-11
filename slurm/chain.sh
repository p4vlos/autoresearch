#!/usr/bin/env bash
# Submit a chain of SLURM experiments, each depending on the previous.
#
# Usage: bash slurm/chain.sh <num_experiments>
#
# Each job runs after the previous one finishes (success or failure),
# so the agent can inspect results and modify train.py between runs.

set -euo pipefail

NUM_EXPERIMENTS="${1:?Usage: bash slurm/chain.sh <num_experiments>}"

if ! [[ "${NUM_EXPERIMENTS}" =~ ^[0-9]+$ ]] || [ "${NUM_EXPERIMENTS}" -lt 1 ]; then
    echo "Error: num_experiments must be a positive integer"
    exit 1
fi

mkdir -p results

echo "Submitting chain of ${NUM_EXPERIMENTS} experiments..."

PREV_JOB_ID=""
for i in $(seq 1 "${NUM_EXPERIMENTS}"); do
    if [ -z "${PREV_JOB_ID}" ]; then
        JOB_ID=$(sbatch --parsable slurm/job.sbatch)
    else
        JOB_ID=$(sbatch --parsable --dependency=afterany:"${PREV_JOB_ID}" slurm/job.sbatch)
    fi
    echo "  Experiment ${i}/${NUM_EXPERIMENTS}: job ${JOB_ID}"
    PREV_JOB_ID="${JOB_ID}"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Latest job output will be in: results/slurm-<jobid>.out"
