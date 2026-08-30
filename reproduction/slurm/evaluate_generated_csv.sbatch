#!/bin/bash
#SBATCH --job-name=dapigen-evaluate
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/evaluate-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/evaluate-%j.err

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 INPUT_CSV REFERENCE_CSV OUTPUT_JSON" >&2
    exit 2
fi

repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

cd "$repo_root"
srun "$python_bin" reproduction/evaluate_generated_csv.py \
    --input-csv "$1" \
    --reference-csv "$2" \
    --output-json "$3" \
    --repo-root "$repo_root" \
    --formal
