#!/bin/bash
#SBATCH --job-name=ppo-only-matched
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
if [[ "$#" -ne 7 && "$#" -ne 9 ]]; then
    echo 'usage: MODE REPO PYTHON PACKAGE_ROOT POLYBERT AFP OUTPUT [AUTHORIZATION PREFLIGHT_REPORT]' >&2
    exit 2
fi
mode="$1"
repo_root="$2"
python_bin="$3"
package_root="$4"
polybert_path="$5"
afp_path="$6"
output_dir="$7"
[[ "$mode" = preflight || "$mode" = real ]]
[[ "$mode" != real || "$#" = 9 ]]
[[ "$mode" != preflight || "$#" = 7 ]]
test "$(hostname)" = yanlih100n1
test -z "${SLURM_ARRAY_JOB_ID:-}"
test "${SLURM_RESTART_COUNT:-0}" = 0
test -z "$(git -C "$repo_root" status --porcelain)"
test ! -e "$output_dir"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$package_root:$repo_root"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
cd "$repo_root"
echo "job_id=$SLURM_JOB_ID mode=$mode source=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
if [[ "$mode" = preflight ]]; then
    srun "$python_bin" -m pytest -q reproduction/p2/tests reproduction/tests/test_ppo_only_control.py
fi
args=(--mode "$mode" --repo-root "$repo_root" --polybert-path "$polybert_path"
      --evaluator-asset-path "$afp_path" --output-dir "$output_dir")
if [[ "$mode" = real ]]; then
    args+=(--authorization "$8" --preflight-report "$9")
fi
srun --cpu-bind=cores "$python_bin" -m reproduction.scicf.online.run_ppo_only_control "${args[@]}"
if [[ "$mode" = real ]]; then
    srun --cpu-bind=cores "$python_bin" -m reproduction.scicf.online.analyze_ppo_only_control \
        --repo-root "$repo_root" --control-report "$output_dir/control-report.json" \
        --output-dir "$output_dir/analysis"
fi
test -z "$(git -C "$repo_root" status --porcelain)"
