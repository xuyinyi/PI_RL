#!/bin/bash
#SBATCH --job-name=scicf-one-iter
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN PYTHON_PACKAGE_ROOT POLYBERT_PATH PRIVATE_CREDENTIALS_FILE OUTPUT_DIR" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
python_package_root="$3"
polybert_path="$4"
credentials_file="$5"
output_dir="$6"
config="$repo_root/reproduction/scicf/online/configs/single_iteration_integration_v1.json"
accepted_manifest="$repo_root/reproduction/results/stage0-v23-p1-373b329-20260904/models/formal_contract/environment_manifest.json"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$python_package_root:$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} protocol=dapigen-scicf-single-iteration-integration-smoke-v1"
echo "repo_root=$repo_root output_dir=$output_dir provider=private-api formal_training=disabled"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
test -d "$python_package_root/gymnasium"
test -d "$polybert_path"
test -f "$credentials_file"
test "$(stat -c %a "$credentials_file")" = "600"
test -z "$(git -C "$repo_root" status --porcelain)"
test ! -e "$output_dir"

cd "$repo_root"
srun "$python_bin" -m pytest -q reproduction/p2/tests reproduction/tests/test_scicf_online.py
srun --cpu-bind=cores "$python_bin" -m reproduction.scicf.online.run_single_iteration_integration \
    --dapigen-root "$repo_root" \
    --config "$config" \
    --accepted-manifest "$accepted_manifest" \
    --polybert-path "$polybert_path" \
    --credentials-file "$credentials_file" \
    --output-dir "$output_dir"

test -z "$(git -C "$repo_root" status --porcelain)"
echo "scicf_single_iteration_integration_complete"
