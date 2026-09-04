#!/bin/bash
#SBATCH --job-name=p4-native-ppo-formal
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-4%1

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN PYTHON_PACKAGE_ROOT RUNS_ROOT" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
python_package_root="$3"
runs_root="$4"
protocol="$repo_root/reproduction/p4/configs/native_ppo_baseline_protocol_v1.json"
seeds=(20260911 20260912 20260913 20260914 20260915)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
output_dir="$runs_root/seed-$seed"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$python_package_root:$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} array_job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} seed=$seed"
echo "repo_root=$repo_root output_dir=$output_dir"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
test -d "$python_package_root/gymnasium"
test -z "$(git -C "$repo_root" status --porcelain)"
test ! -e "$output_dir"
mkdir -p "$runs_root"

cd "$repo_root"
srun "$python_bin" -m pytest -q reproduction/p2/tests reproduction/p4/tests
srun --cpu-bind=cores "$python_bin" \
    reproduction/p4/scripts/run_native_ppo_baseline.py \
    --dapigen-root "$repo_root" \
    --protocol "$protocol" \
    --output-dir "$output_dir" \
    --mode formal \
    --seed "$seed"
srun "$python_bin" reproduction/p4/scripts/audit_native_ppo_baseline.py \
    --protocol "$protocol" \
    --report "$output_dir/run_report.json" \
    --mode formal

test -z "$(git -C "$repo_root" status --porcelain)"
echo "p4_native_ppo_formal_seed_complete seed=$seed"

