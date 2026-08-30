#!/bin/bash
#SBATCH --job-name=dapigen-algorithm
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/algorithm-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/algorithm-%j.err

set -euo pipefail

if [[ "$#" -lt 3 ]]; then
    echo "usage: $0 CONFIG OUTPUT_ROOT RUN_ID [RUNNER_ARGS...]" >&2
    exit 2
fi

config_path="$1"
output_root="$2"
run_id="$3"
shift 3

repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export DGLBACKEND=pytorch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export DAPIGEN_RAY_TMPDIR="/tmp/dapigen-ray-${SLURM_JOB_ID}"

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "config=${config_path} output_root=${output_root} run_id=${run_id}"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

cd "$repo_root"
srun "$python_bin" reproduction/run_algorithm.py \
    --config "$config_path" \
    --output-root "$output_root" \
    --run-id "$run_id" \
    "$@"
