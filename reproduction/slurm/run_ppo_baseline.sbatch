#!/bin/bash
#SBATCH --job-name=dapigen-ppo
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --gres=gpu:4

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
    echo "usage: $0 RUN_PPO_BASELINE_ARGS..." >&2
    exit 2
fi

repo_root="/home/wch/workspaces/DAPiGen-reproduction/worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export DGLBACKEND=pytorch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "arguments=$*"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

cd "$repo_root"
srun "$python_bin" reproduction/run_ppo_baseline.py "$@"
