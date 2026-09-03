#!/bin/bash
#SBATCH --job-name=scicf-g1b2-late-dev
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b2-late-dev-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b2-late-dev-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 OUTPUT_ROOT" >&2
    exit 2
fi

output_root="$1"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"
baseline_root="/home/wch/workspaces/DAPiGen-reproduction/runs/ppo-compat/full-20260830-v1"

export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export DAPIGEN_RAY_TMPDIR="/tmp/scicf-g1b2-late-dev-${SLURM_JOB_ID}"

echo "job_id=${SLURM_JOB_ID} split=dev stage=late host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/gate1/collect_gate1b1.py \
    --repo-root "$repo_root" \
    --framework-config "$repo_root/reproduction/configs/ppo-compat-common-v1.json" \
    --gate-config "$repo_root/reproduction/configs/scicf-gate1b2-late-dev-collection-v1.json" \
    --baseline-run-root "$baseline_root" \
    --split dev \
    --stage late \
    --output-root "$output_root"
