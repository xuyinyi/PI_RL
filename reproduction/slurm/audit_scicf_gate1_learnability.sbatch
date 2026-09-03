#!/bin/bash
#SBATCH --job-name=scicf-gate1b
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-gate1b-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-gate1b-%j.err

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 GATE1_ROOT AUDIT_ID" >&2
    exit 2
fi

gate1_root="$1"
audit_id="$2"
if [[ ! "$audit_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
    echo "AUDIT_ID must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
fi

repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"
output="$gate1_root/gate1b-learnability-$audit_id.json"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "job_id=${SLURM_JOB_ID} host=$(hostname) gate=gate1b-learnability audit_id=$audit_id"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/gate1/audit_learnability.py \
    --repo-root "$repo_root" \
    --config "$repo_root/reproduction/configs/scicf-gate1b-dev-v1.json" \
    --stage-report "$gate1_root/early/gate1-stage-report.json" \
    --stage-report "$gate1_root/middle/gate1-stage-report.json" \
    --stage-report "$gate1_root/late/gate1-stage-report.json" \
    --output "$output"
