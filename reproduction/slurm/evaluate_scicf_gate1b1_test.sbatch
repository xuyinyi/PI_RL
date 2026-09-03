#!/bin/bash
#SBATCH --job-name=scicf-g1b1-test
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b1-test-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b1-test-%j.err

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 DATA_ROOT FROZEN_MODEL_MANIFEST OUTPUT" >&2
    exit 2
fi

data_root="$1"
model_manifest="$2"
output="$3"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "job_id=${SLURM_JOB_ID} host=$(hostname) gate=gate1b1-sealed-test"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/gate1/evaluate_gate1b1_test.py \
    --repo-root "$repo_root" \
    --config "$repo_root/reproduction/configs/scicf-gate1b1-structure-split-v1.json" \
    --model-manifest "$model_manifest" \
    --test-report "$data_root/test/early/gate1b1-stage-report.json" \
    --test-report "$data_root/test/middle/gate1b1-stage-report.json" \
    --test-report "$data_root/test/late/gate1b1-stage-report.json" \
    --output "$output"
