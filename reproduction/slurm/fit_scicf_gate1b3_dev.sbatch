#!/bin/bash
#SBATCH --job-name=scicf-g1b3-dev-fit
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b3-dev-fit-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b3-dev-fit-%j.err

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
    echo "usage: $0 DATA_ROOT GATE1B1_MODEL_MANIFEST GATE1B2_MODEL_MANIFEST COLLECTION_VALIDATION EVALUATION_AUTHORIZATION_RECEIPT OUTPUT_ROOT" >&2
    exit 2
fi

data_root="$1"
gate1b1_manifest="$2"
gate1b2_manifest="$3"
collection_validation="$4"
evaluation_receipt="$5"
output_root="$6"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "job_id=${SLURM_JOB_ID} gate=gate1b3-dev test=sealed late=non-gating host=$(hostname)"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/gate1/fit_gate1b3_dev.py \
    --repo-root "$repo_root" \
    --config "$repo_root/reproduction/configs/scicf-gate1b3-stage-routed-development-v1.json" \
    --collection-config "$repo_root/reproduction/configs/scicf-gate1b3-unseen-dev-collection-v1.json" \
    --gate1b1-model-manifest "$gate1b1_manifest" \
    --gate1b2-model-manifest "$gate1b2_manifest" \
    --early-report "$data_root/dev/early/gate1b1-stage-report.json" \
    --middle-report "$data_root/dev/middle/gate1b1-stage-report.json" \
    --collection-validation "$collection_validation" \
    --evaluation-authorization-receipt "$evaluation_receipt" \
    --output-root "$output_root"
