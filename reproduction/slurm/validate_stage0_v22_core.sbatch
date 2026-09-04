#!/bin/bash
#SBATCH --job-name=stage0-v22-core
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN OUTPUT_DIR [REACHABLE_STATES]" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
output_dir="$3"
reachable_states="${4:-5}"

mkdir -p "$output_dir"
export DGLBACKEND=pytorch
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
echo "repo_root=$repo_root"
echo "python_bin=$python_bin"
echo "reachable_states=$reachable_states"

cd "$repo_root"
srun "$python_bin" -m pytest -q reproduction/stage0/tests
srun "$python_bin" reproduction/stage0/scripts/validate_core_without_models.py \
    > "$output_dir/core_validation.json"
srun "$python_bin" reproduction/check_assets.py \
    --asset-mode reconstructed-afp-compatibility \
    > "$output_dir/reconstructed_asset_gate.json"
srun "$python_bin" reproduction/stage0/scripts/audit_stage0_catalogs.py \
    --dapigen-root "$repo_root" \
    --max-steps 5 \
    --reachable-states "$reachable_states" \
    --rollout-seed 20260904 \
    --output "$output_dir/catalog_mask_audit.json"

echo "stage0_v22_core_validation_complete"
