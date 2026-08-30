#!/bin/bash
#SBATCH --job-name=scicf-adapter
#SBATCH --partition=compute
#SBATCH --nodelist=n001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-adapter-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-adapter-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 OUTPUT_JSON" >&2
    exit 2
fi

output_json="$1"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/validate_dapigen_adapter.py \
    --repo-root "$repo_root" \
    --output "$output_json"
