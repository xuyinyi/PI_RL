#!/bin/bash
#SBATCH --job-name=stage0-v22-replay
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN OUTPUT_DIR" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
output_dir="$3"

mkdir -p "$output_dir"
export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
cd "$repo_root"
srun "$python_bin" reproduction/stage0/scripts/validate_replay_restore.py \
    --dapigen-root "$repo_root" \
    --transitions 1000 \
    --seed 20260904 \
    --output "$output_dir/replay_restore.json"
srun "$python_bin" reproduction/stage0/scripts/validate_ray_evaluator.py \
    --dapigen-root "$repo_root" \
    --evaluator-mode persistent \
    --device cpu \
    --duplicates 8 \
    --output "$output_dir/ray_evaluator_checkpoint.json"

echo "stage0_v22_replay_validation_complete"
