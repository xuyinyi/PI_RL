#!/bin/bash
#SBATCH --job-name=scicf-g1b3-preflight
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b3-preflight-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b3-preflight-%j.err

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 GATE1B1_MODEL_MANIFEST GATE1B2_MODEL_MANIFEST" >&2
    exit 2
fi

gate1b1_manifest="$1"
gate1b2_manifest="$2"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
python_bin="/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1

cd "$repo_root"
srun "$python_bin" reproduction/scicf/gate1/preflight_gate1b3.py \
    --repo-root "$repo_root" \
    --config "$repo_root/reproduction/configs/scicf-gate1b3-stage-routed-development-v1.json" \
    --collection-config "$repo_root/reproduction/configs/scicf-gate1b3-unseen-dev-collection-v1.json" \
    --gate1b1-model-manifest "$gate1b1_manifest" \
    --gate1b2-model-manifest "$gate1b2_manifest"
