#!/bin/bash
#SBATCH --job-name=scicf-short-preflight
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN PYTHON_PACKAGE_ROOT POLYBERT_PATH EVALUATOR_ASSET_PATH OUTPUT_DIR" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
python_package_root="$3"
polybert_path="$4"
evaluator_asset_path="$5"
output_dir="$6"
protocol="$repo_root/reproduction/scicf/online/configs/short_horizon_multi_iteration_v1.json"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$python_package_root:$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) protocol=dapigen-scicf-short-horizon-multi-iteration-smoke-v1"
echo "scope=no-credential-full-runtime-preflight external_api=disabled credentials=disabled ppo=disabled oracle=disabled local_llm=disabled"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
test -d "$python_package_root/gymnasium"
test -d "$polybert_path"
test -d "$evaluator_asset_path"
test -f "$protocol"
test -z "$(git -C "$repo_root" status --porcelain)"
test ! -e "$output_dir"

cd "$repo_root"
srun "$python_bin" -m pytest -q \
    reproduction/tests/test_scicf_short_horizon_multi_iteration.py \
    reproduction/tests/test_scicf_soft_pair_resilience.py \
    reproduction/tests/test_scicf_single_iteration_v3.py \
    reproduction/tests/test_scicf_schema_robustness.py \
    reproduction/stage0/tests/test_embedding_contract.py
srun --cpu-bind=cores "$python_bin" -m reproduction.scicf.online.run_short_horizon_multi_iteration_preflight \
    --repo-root "$repo_root" \
    --protocol "$protocol" \
    --polybert-path "$polybert_path" \
    --evaluator-asset-path "$evaluator_asset_path" \
    --output-dir "$output_dir"

test -z "$(git -C "$repo_root" status --porcelain)"
echo "scicf_short_horizon_no_credential_preflight_complete"
