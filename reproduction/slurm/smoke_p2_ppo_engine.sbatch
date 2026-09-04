#!/bin/bash
#SBATCH --job-name=p2-native-ppo-smoke
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN PYTHON_PACKAGE_ROOT OUTPUT_DIR" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
python_package_root="$3"
output_dir="$4"
config="$repo_root/reproduction/p2/configs/ppo_engine_smoke_v1.json"
accepted_manifest="$repo_root/reproduction/results/stage0-v23-p1-373b329-20260904/models/formal_contract/environment_manifest.json"

mkdir -p "$output_dir"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
test -d "$python_package_root/gymnasium"
export PYTHONPATH="$python_package_root:$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
echo "repo_root=$repo_root"
echo "python_bin=$python_bin"
echo "python_package_root=$python_package_root"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
test -z "$(git -C "$repo_root" status --porcelain)"

cd "$repo_root"
srun "$python_bin" -c \
    'import gymnasium, torch; print("dependency_probe gymnasium=%s torch=%s" % (gymnasium.__version__, torch.__version__))'
srun "$python_bin" -m pytest -q reproduction/p2/tests
srun --cpu-bind=cores "$python_bin" \
    reproduction/p2/scripts/run_ppo_engine_smoke.py \
    --dapigen-root "$repo_root" \
    --config "$config" \
    --accepted-manifest "$accepted_manifest" \
    --output-dir "$output_dir"

test -z "$(git -C "$repo_root" status --porcelain)"
echo "p2_native_ppo_engine_smoke_complete"
