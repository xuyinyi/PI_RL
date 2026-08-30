#!/bin/bash
#SBATCH --job-name=scicf-llm-setup
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-llm-setup-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-llm-setup-%j.err

set -euo pipefail

repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
legacy_python="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"
llm_env="/home/wch/workspaces/DAPiGen-reproduction/envs/scicf-llm-py38"
llm_python="$llm_env/bin/python"
model_root="/home/wch/workspaces/DAPiGen-reproduction/models/qwen2.5-7b-instruct-a09a3545"

export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export HF_HOME=/home/wch/workspaces/DAPiGen-reproduction/cache/huggingface
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -x "$llm_python" ]]; then
    "$legacy_python" -m venv --system-site-packages "$llm_env"
fi
"$llm_python" -m pip install --upgrade \
    transformers==4.46.3 \
    accelerate==1.0.1 \
    huggingface_hub==0.26.2 \
    safetensors==0.4.5

cd "$repo_root"
srun "$llm_python" reproduction/scicf/llm/download_model.py \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --output "$model_root"
