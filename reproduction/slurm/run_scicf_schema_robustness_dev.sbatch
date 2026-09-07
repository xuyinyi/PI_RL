#!/bin/bash
#SBATCH --job-name=scicf-schema-dev
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN PYTHON_PACKAGE_ROOT OUTPUT_DIR" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
python_package_root="$3"
output_dir="$4"
config="$repo_root/reproduction/scicf/online/configs/schema_robustness_dev_v1.json"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$python_package_root:$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} protocol=dapigen-scicf-llm-schema-robustness-dev-v1"
echo "repo_root=$repo_root output_dir=$output_dir external_api=disabled ppo=disabled oracle=disabled"
test -d "$python_package_root/gymnasium"
test -z "$(git -C "$repo_root" status --porcelain)"
test ! -e "$output_dir"

cd "$repo_root"
srun "$python_bin" -m pytest -q \
    reproduction/tests/test_scicf_schema_robustness.py \
    reproduction/tests/test_scicf_api.py \
    reproduction/tests/test_scicf_online.py
srun "$python_bin" -m reproduction.scicf.online.run_schema_robustness_dev \
    --repo-root "$repo_root" \
    --config "$config" \
    --output-dir "$output_dir"

test -z "$(git -C "$repo_root" status --porcelain)"
echo "scicf_schema_robustness_development_complete"
