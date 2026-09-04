#!/bin/bash
#SBATCH --job-name=p4-native-ppo-aggregate
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN RUNS_ROOT OUTPUT_JSON" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
runs_root="$3"
output_json="$4"
protocol="$repo_root/reproduction/p4/configs/native_ppo_baseline_protocol_v1.json"

export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

test -z "$(git -C "$repo_root" status --porcelain)"
test ! -e "$output_json"
cd "$repo_root"
srun "$python_bin" reproduction/p4/scripts/aggregate_native_ppo_baseline.py \
    --protocol "$protocol" \
    --runs-root "$runs_root" \
    --output "$output_json"
echo "p4_native_ppo_aggregation_complete"
