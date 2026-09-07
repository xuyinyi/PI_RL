#!/bin/bash
#SBATCH --job-name=scicf-short-multi
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

if [[ "$#" -ne 8 && "$#" -ne 9 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN PYTHON_PACKAGE_ROOT POLYBERT_PATH EVALUATOR_ASSET_PATH PRIVATE_CREDENTIALS_FILE EXECUTION_AUTHORIZATION OUTPUT_DIR [RESUME_CONTROL_CHECKPOINT]" >&2
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
resume_control_checkpoint="${9:-}"
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
echo "scope=separately-authorized-short-horizon maximum_iterations=6 automatic_resume=disabled automatic_rerun=disabled formal_training=disabled"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
test -d "$python_package_root/gymnasium"
test -d "$polybert_path"
test -d "$evaluator_asset_path"
test -f "$protocol"
test -f "$credentials_file"
test -f "$execution_authorization"
test "$(stat -c %a "$credentials_file")" = "600"
test "$(stat -c %a "$execution_authorization")" = "600"
if [[ -n "$resume_control_checkpoint" ]]; then
    test -f "$resume_control_checkpoint"
fi
test -z "$(git -C "$repo_root" status --porcelain)"
test ! -e "$output_dir"

cd "$repo_root"
srun "$python_bin" -m pytest -q \
    reproduction/p2/tests \
    reproduction/tests/test_scicf_api.py \
    reproduction/tests/test_scicf_online.py \
    reproduction/tests/test_scicf_schema_robustness.py \
    reproduction/tests/test_scicf_soft_pair_resilience.py \
    reproduction/tests/test_scicf_single_iteration_v3.py \
    reproduction/tests/test_scicf_short_horizon_multi_iteration.py

runner_args=(
    --dapigen-root "$repo_root"
    --protocol "$protocol"
    --execution-authorization "$execution_authorization"
    --polybert-path "$polybert_path"
    --evaluator-asset-path "$evaluator_asset_path"
    --credentials-file "$credentials_file"
    --output-dir "$output_dir"
)
if [[ -n "$resume_control_checkpoint" ]]; then
    runner_args+=(--resume-control-checkpoint "$resume_control_checkpoint")
fi
srun --cpu-bind=cores "$python_bin" -m reproduction.scicf.online.run_short_horizon_multi_iteration "${runner_args[@]}"

test -z "$(git -C "$repo_root" status --porcelain)"
echo "scicf_short_horizon_multi_iteration_complete"
