#!/bin/bash
#SBATCH --job-name=scicf-one-iter-v2
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

if [[ "$#" -ne 8 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN PYTHON_PACKAGE_ROOT POLYBERT_PATH EVALUATOR_ASSET_PATH PRIVATE_CREDENTIALS_FILE EXECUTION_AUTHORIZATION OUTPUT_DIR" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
python_package_root="$3"
polybert_path="$4"
evaluator_asset_path="$5"
credentials_file="$6"
execution_authorization="$7"
output_dir="$8"
protocol="$repo_root/reproduction/scicf/online/configs/single_iteration_integration_v2_protocol.json"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$python_package_root:$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} protocol=dapigen-scicf-single-iteration-integration-smoke-v2"
echo "scope=single-authorized-integration-v2 automatic_rerun=disabled multi_iteration=disabled"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
test -d "$python_package_root/gymnasium"
test -d "$polybert_path"
test -d "$evaluator_asset_path"
while read -r expected relative; do
    test -n "$expected"
    test -s "$polybert_path/$relative"
done < "$repo_root/reproduction/polybert-e7dce434.sha256"
test -f "$protocol"
test -f "$credentials_file"
test -f "$execution_authorization"
test "$(stat -c %a "$credentials_file")" = "600"
test "$(stat -c %a "$execution_authorization")" = "600"
test -z "$(git -C "$repo_root" status --porcelain)"
test ! -e "$output_dir"

cd "$repo_root"
srun "$python_bin" -m pytest -q \
    reproduction/p2/tests \
    reproduction/tests/test_scicf_api.py \
    reproduction/tests/test_scicf_online.py \
    reproduction/tests/test_scicf_schema_robustness.py \
    reproduction/tests/test_scicf_single_iteration_v2.py
srun --cpu-bind=cores "$python_bin" -m reproduction.scicf.online.run_single_iteration_integration_v2 \
    --dapigen-root "$repo_root" \
    --protocol "$protocol" \
    --execution-authorization "$execution_authorization" \
    --polybert-path "$polybert_path" \
    --evaluator-asset-path "$evaluator_asset_path" \
    --credentials-file "$credentials_file" \
    --output-dir "$output_dir"

test -z "$(git -C "$repo_root" status --porcelain)"
echo "scicf_single_iteration_integration_v2_complete"
