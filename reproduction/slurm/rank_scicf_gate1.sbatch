#!/bin/bash
#SBATCH --job-name=scicf-gate1-llm
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-gate1-llm-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-gate1-llm-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 GATE1_ROOT" >&2
    exit 2
fi

gate1_root="$1"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/scicf-llm-py38/bin/python"
model_root="/home/wch/workspaces/DAPiGen-reproduction/models/qwen2.5-7b-instruct-a09a3545"
response_root="$gate1_root/llm"
cache_root="/home/wch/workspaces/DAPiGen-reproduction/cache/scicf-llm-responses"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/llm/rank_requests.py \
    --requests "$gate1_root/early/llm-requests.jsonl" \
    --requests "$gate1_root/middle/llm-requests.jsonl" \
    --requests "$gate1_root/late/llm-requests.jsonl" \
    --output-root "$response_root" \
    --cache-root "$cache_root" \
    --model-path "$model_root" \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --max-new-tokens 384 \
    --max-retries 2 \
    --seed 2023
