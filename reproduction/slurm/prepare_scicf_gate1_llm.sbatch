#!/bin/bash
#SBATCH --job-name=scicf-gate1-blind
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-gate1-blind-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-gate1-blind-%j.err

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

cd "$repo_root"
for stage in early middle late; do
    srun "$python_bin" reproduction/scicf/llm/rebuild_blinded_requests.py \
        --input "$gate1_root/$stage/llm-requests.jsonl" \
        --output "$gate1_root/$stage/llm-requests-blinded-v2.jsonl"
done
