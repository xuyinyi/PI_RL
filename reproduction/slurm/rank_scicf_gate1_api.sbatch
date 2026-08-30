#!/bin/bash
#SBATCH --job-name=scicf-gate1-api
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-gate1-api-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-gate1-api-%j.err

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 GATE1_ROOT PRIVATE_CREDENTIALS_FILE RUN_ID" >&2
    exit 2
fi

gate1_root="$1"
credentials_file="$2"
run_id="$3"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
    echo "RUN_ID must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
fi

repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"
response_root="$gate1_root/llm-api-$run_id"
cache_root="/home/wch/workspaces/DAPiGen-reproduction/cache/scicf-llm-api-responses"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "job_id=${SLURM_JOB_ID} host=$(hostname) provider=private-api run_id=$run_id"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/llm/rank_requests_api.py \
    --repo-root "$repo_root" \
    --credentials-file "$credentials_file" \
    --requests "$gate1_root/early/llm-requests-blinded-v2.jsonl" \
    --requests "$gate1_root/middle/llm-requests-blinded-v2.jsonl" \
    --requests "$gate1_root/late/llm-requests-blinded-v2.jsonl" \
    --output-root "$response_root" \
    --cache-root "$cache_root" \
    --max-output-tokens 384 \
    --validation-retries 2 \
    --transport-retries 3 \
    --timeout-seconds 120 \
    --seed 2023 \
    --expected-request-count 24 \
    --expected-budget 4 \
    --expected-pool-size 24 \
    --expected-prompt-version scicf-dapigen-acquisition-blinded-v2 \
    --expected-candidate-presentation sha256-shuffle-v1
