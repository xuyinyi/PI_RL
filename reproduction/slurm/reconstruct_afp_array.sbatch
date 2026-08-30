#!/bin/bash
#SBATCH --job-name=afp-rebuild
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

set -euo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "usage: $0 REPO_ROOT OUTPUT_ROOT MAX_EPOCH EARLYSTOP_PATIENCE RUN_LABEL" >&2
    exit 2
fi

repo_root="$1"
output_root="$2"
max_epoch="$3"
earlystop_patience="$4"
run_label="$5"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"
properties=("transmittance(400)" "cte" "strength" "tg")
property_name="${properties[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "job_id=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID} property=${property_name}"
echo "host=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

srun "$python_bin" "$repo_root/reproduction/reconstruct_afp_assets.py" \
    --repo-root "$repo_root" \
    --output-root "$output_root" \
    --property "$property_name" \
    --max-epoch "$max_epoch" \
    --earlystopping-patience "$earlystop_patience" \
    --device cuda \
    --run-label "$run_label"
