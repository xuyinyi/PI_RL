#!/bin/bash
#SBATCH --job-name=scicf-api-credential
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-api-credential-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-api-credential-%j.err

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 RAW_KEY_FILE PRIVATE_CREDENTIALS_FILE MODEL_REVISION" >&2
    exit 2
fi

raw_key_file="$1"
credentials_file="$2"
model_revision="$3"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "job_id=${SLURM_JOB_ID} host=$(hostname) action=provision-deepseek-flash-credentials"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/llm/provision_credentials.py \
    --raw-key-file "$raw_key_file" \
    --output "$credentials_file" \
    --profile deepseek-v4-flash \
    --model-revision "$model_revision"
