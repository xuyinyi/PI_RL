#!/bin/bash
#SBATCH --job-name=scicf-softpair-preflight
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN PYTHON_PACKAGE_ROOT OUTPUT_DIR" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
python_package_root="$3"
output_dir="$4"
config="$repo_root/reproduction/scicf/online/configs/soft_pair_resilience_dev_v1.json"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$python_package_root:$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "job_id=${SLURM_JOB_ID} host=$(hostname) protocol=dapigen-scicf-soft-pair-resilience-dev-v1"
echo "scope=mock_only credentials=forbidden external_api=forbidden ppo=forbidden oracle=forbidden"
test -d "$python_package_root/gymnasium"
test -f "$config"
test -z "$(git -C "$repo_root" status --porcelain)"
test ! -e "$output_dir"

cd "$repo_root"
srun "$python_bin" -m pytest -q \
    reproduction/tests/test_scicf_online.py \
    reproduction/tests/test_scicf_soft_pair_resilience.py
srun "$python_bin" -m reproduction.scicf.online.run_soft_pair_resilience_preflight \
    --config "$config" \
    --repo-root "$repo_root" \
    --output-dir "$output_dir"

test -z "$(git -C "$repo_root" status --porcelain)"
echo "scicf_soft_pair_resilience_preflight_complete"
