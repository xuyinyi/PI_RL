#!/bin/bash
#SBATCH --job-name=scicf-v2-preflight
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN PYTHON_PACKAGE_ROOT POLYBERT_PATH OUTPUT_DIR" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
python_package_root="$3"
polybert_path="$4"
output_dir="$5"
protocol="$repo_root/reproduction/scicf/online/configs/single_iteration_integration_v2_protocol.json"

export CUDA_VISIBLE_DEVICES=''
export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$python_package_root:$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} protocol=dapigen-scicf-single-iteration-integration-smoke-v2"
echo "scope=mock-only-preflight external_api=disabled credentials=disabled ppo=disabled oracle=disabled"
test -d "$python_package_root/gymnasium"
test -f "$protocol"
test -d "$polybert_path"
while read -r expected relative; do
    test -n "$expected"
    test -s "$polybert_path/$relative"
done < "$repo_root/reproduction/polybert-e7dce434.sha256"
test -z "$(git -C "$repo_root" status --porcelain)"
test ! -e "$output_dir"

cd "$repo_root"
srun "$python_bin" -m pytest -q \
    reproduction/tests/test_scicf_api.py \
    reproduction/tests/test_scicf_online.py \
    reproduction/tests/test_scicf_schema_robustness.py \
    reproduction/tests/test_scicf_single_iteration_v2.py \
    reproduction/stage0/tests/test_embedding_contract.py
srun --cpu-bind=cores "$python_bin" -m reproduction.scicf.online.run_single_iteration_integration_v2_preflight \
    --repo-root "$repo_root" \
    --protocol "$protocol" \
    --polybert-path "$polybert_path" \
    --output-dir "$output_dir"

test -z "$(git -C "$repo_root" status --porcelain)"
echo "scicf_single_iteration_v2_mock_preflight_complete"
