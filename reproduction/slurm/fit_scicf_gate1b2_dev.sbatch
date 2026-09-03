#!/bin/bash
#SBATCH --job-name=scicf-g1b2-dev
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b2-dev-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b2-dev-%j.err

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 PARENT_DATA_ROOT EXPANSION_DATA_ROOT PARENT_MODEL_MANIFEST OUTPUT_ROOT" >&2
    exit 2
fi

parent_data_root="$1"
expansion_data_root="$2"
parent_model_manifest="$3"
output_root="$4"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "job_id=${SLURM_JOB_ID} host=$(hostname) gate=gate1b2-dev sealed_test=untouched"
cd "$repo_root"
srun "$python_bin" reproduction/scicf/gate1/fit_gate1b2_dev.py \
    --repo-root "$repo_root" \
    --config "$repo_root/reproduction/configs/scicf-gate1b2-development-v1.json" \
    --parent-config "$repo_root/reproduction/configs/scicf-gate1b1-structure-split-v1.json" \
    --expansion-config "$repo_root/reproduction/configs/scicf-gate1b2-late-dev-collection-v1.json" \
    --parent-model-manifest "$parent_model_manifest" \
    --train-report "$parent_data_root/train/early/gate1b1-stage-report.json" \
    --train-report "$parent_data_root/train/middle/gate1b1-stage-report.json" \
    --train-report "$parent_data_root/train/late/gate1b1-stage-report.json" \
    --dev-report "$parent_data_root/dev/early/gate1b1-stage-report.json" \
    --dev-report "$parent_data_root/dev/middle/gate1b1-stage-report.json" \
    --dev-report "$parent_data_root/dev/late/gate1b1-stage-report.json" \
    --late-expansion-report "$expansion_data_root/dev/late/gate1b1-stage-report.json" \
    --output-root "$output_root"
