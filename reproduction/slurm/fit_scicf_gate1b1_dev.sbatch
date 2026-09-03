#!/bin/bash
#SBATCH --job-name=scicf-g1b1-dev
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b1-dev-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b1-dev-%j.err

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 DATA_ROOT MODEL_ROOT" >&2
    exit 2
fi

data_root="$1"
model_root="$2"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "job_id=${SLURM_JOB_ID} host=$(hostname) gate=gate1b1-dev"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/gate1/fit_gate1b1_dev.py \
    --repo-root "$repo_root" \
    --config "$repo_root/reproduction/configs/scicf-gate1b1-structure-split-v1.json" \
    --train-report "$data_root/train/early/gate1b1-stage-report.json" \
    --train-report "$data_root/train/middle/gate1b1-stage-report.json" \
    --train-report "$data_root/train/late/gate1b1-stage-report.json" \
    --dev-report "$data_root/dev/early/gate1b1-stage-report.json" \
    --dev-report "$data_root/dev/middle/gate1b1-stage-report.json" \
    --dev-report "$data_root/dev/late/gate1b1-stage-report.json" \
    --output-root "$model_root"
