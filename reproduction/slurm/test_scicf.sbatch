#!/bin/bash
#SBATCH --job-name=scicf-tests
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-tests-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-tests-%j.err

set -euo pipefail

repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "job_id=${SLURM_JOB_ID} host=$(hostname)"
cd "$repo_root"
srun "$python_bin" -m unittest \
    reproduction.tests.test_framework_contracts \
    reproduction.tests.test_scicf_core \
    -v
