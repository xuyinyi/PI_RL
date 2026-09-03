#!/bin/bash
#SBATCH --job-name=scicf-g1b3-data-check
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b3-data-check-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b3-data-check-%j.err

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 DATA_ROOT PRIOR_DEV_MODEL_MANIFEST AUTHORIZATION_RECEIPT" >&2
    exit 2
fi

data_root="$1"
prior_dev_manifest="$2"
authorization_receipt="$3"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

cd "$repo_root"
srun "$python_bin" reproduction/scicf/gate1/validate_gate1b3_collection.py \
    --repo-root "$repo_root" \
    --config "$repo_root/reproduction/configs/scicf-gate1b3-stage-routed-development-v1.json" \
    --collection-config "$repo_root/reproduction/configs/scicf-gate1b3-unseen-dev-collection-v1.json" \
    --prior-dev-manifest "$prior_dev_manifest" \
    --authorization-receipt "$authorization_receipt" \
    --early-report "$data_root/dev/early/gate1b1-stage-report.json" \
    --middle-report "$data_root/dev/middle/gate1b1-stage-report.json" \
    --output "$data_root/collection-validation.json"
