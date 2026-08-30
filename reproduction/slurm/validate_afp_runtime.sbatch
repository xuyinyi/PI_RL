#!/bin/bash
#SBATCH --job-name=afp-validate
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 REPO_ROOT TRAINING_ROOT OUTPUT_JSON" >&2
    exit 2
fi

repo_root="$1"
training_root="$2"
output_json="$3"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
srun "$python_bin" "$repo_root/reproduction/validate_afp_runtime.py" \
    --repo-root "$repo_root" \
    --training-root "$training_root" \
    --output "$output_json"
