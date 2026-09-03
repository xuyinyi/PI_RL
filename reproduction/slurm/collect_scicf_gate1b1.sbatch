#!/bin/bash
#SBATCH --job-name=scicf-g1b1-data
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b1-data-%A_%a.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b1-data-%A_%a.err

set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "usage: $0 SPLIT OUTPUT_ROOT [FROZEN_MODEL_MANIFEST]" >&2
    exit 2
fi

split="$1"
output_root="$2"
model_manifest="${3:-}"
if [[ "$split" != "train" && "$split" != "dev" && "$split" != "test" ]]; then
    echo "SPLIT must be train, dev, or test" >&2
    exit 2
fi
if [[ "$split" == "test" && -z "$model_manifest" ]]; then
    echo "test collection requires FROZEN_MODEL_MANIFEST" >&2
    exit 2
fi
if [[ "$split" != "test" && -n "$model_manifest" ]]; then
    echo "train/dev collection must not receive FROZEN_MODEL_MANIFEST" >&2
    exit 2
fi

repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"
baseline_root="/home/wch/workspaces/DAPiGen-reproduction/runs/ppo-compat/full-20260830-v1"
stages=(early middle late)
stage="${stages[$SLURM_ARRAY_TASK_ID]}"

export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export DAPIGEN_RAY_TMPDIR="/tmp/scicf-g1b1-${split}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

echo "job_id=${SLURM_JOB_ID} split=${split} stage=${stage} host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$repo_root"
command=(
    "$python_bin" reproduction/scicf/gate1/collect_gate1b1.py
    --repo-root "$repo_root"
    --framework-config "$repo_root/reproduction/configs/ppo-compat-common-v1.json"
    --gate-config "$repo_root/reproduction/configs/scicf-gate1b1-structure-split-v1.json"
    --baseline-run-root "$baseline_root"
    --split "$split"
    --stage "$stage"
    --output-root "$output_root"
)
if [[ -n "$model_manifest" ]]; then
    command+=(--model-manifest "$model_manifest")
fi
srun "${command[@]}"
