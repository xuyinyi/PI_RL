#!/bin/bash
#SBATCH --job-name=stage0-v23-models
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN OUTPUT_DIR" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
output_dir="$3"
polybert_path="$repo_root/RL_PPO/models"

mkdir -p "$output_dir"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
echo "repo_root=$repo_root"
echo "python_bin=$python_bin"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

cd "$repo_root"
srun "$python_bin" reproduction/stage0/scripts/compare_polybert.py \
    --polybert-path "$polybert_path" \
    --device cpu \
    --atol 1e-6 \
    --rtol 1e-6 \
    --output "$output_dir/polybert_parity.json"
srun "$python_bin" reproduction/stage0/scripts/compare_evaluators.py \
    --dapigen-root "$repo_root" \
    --input-csv "$repo_root/raw_data/PI.csv" \
    --sample-size 100 \
    --seed 20260903 \
    --device cuda:0 \
    --tolerance 0 \
    --output-dir "$output_dir/evaluator_parity"
srun "$python_bin" reproduction/stage0/scripts/validate_ray_evaluator.py \
    --dapigen-root "$repo_root" \
    --evaluator-mode persistent \
    --device cpu \
    --duplicates 8 \
    --output "$output_dir/ray_evaluator.json"
srun "$python_bin" reproduction/stage0/scripts/create_stage0_manifest.py \
    --dapigen-root "$repo_root" \
    --polybert-path "$polybert_path" \
    --config "$repo_root/reproduction/stage0/configs/stage0_environment.json" \
    --device cuda:0 \
    --maximum-requested-calls 100000 \
    --maximum-unique-calls 100000 \
    --output-dir "$output_dir/development_contract" \
    --run-name stage0_v23_mask_development \
    --allow-dirty

echo "stage0_v23_model_validation_complete"
