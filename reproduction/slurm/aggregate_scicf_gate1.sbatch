#!/bin/bash
#SBATCH --job-name=scicf-gate1-agg
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-gate1-agg-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-gate1-agg-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 GATE1_ROOT" >&2
    exit 2
fi

gate1_root="$1"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "job_id=${SLURM_JOB_ID} host=$(hostname)"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/gate1/aggregate.py \
    --repo-root "$repo_root" \
    --scicf-config "$repo_root/reproduction/configs/scicf-gate1-run-v1.json" \
    --stage-report "$gate1_root/early/gate1-stage-report.json" \
    --stage-report "$gate1_root/middle/gate1-stage-report.json" \
    --stage-report "$gate1_root/late/gate1-stage-report.json" \
    --llm-responses "$gate1_root/llm/responses.jsonl" \
    --output "$gate1_root/gate1-decision.json"
