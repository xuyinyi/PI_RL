#!/bin/bash
#SBATCH --job-name=p2-gymnasium-setup
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 REPO_ROOT PYTHON_BIN TARGET_DIR OUTPUT_DIR" >&2
    exit 2
fi

repo_root="$1"
python_bin="$2"
target_dir="$3"
output_dir="$4"
requirements="$repo_root/reproduction/p2/requirements-native-ppo.txt"

test ! -e "$target_dir"
mkdir -p "$target_dir" "$output_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

echo "job_id=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
echo "python_bin=$python_bin"
echo "target_dir=$target_dir"
"$python_bin" -m pip install \
    --disable-pip-version-check \
    --no-input \
    --no-deps \
    --requirement "$requirements" \
    --target "$target_dir" \
    --report "$output_dir/pip-install-report.json"
PYTHONPATH="$target_dir" "$python_bin" -c \
    'import gymnasium, farama_notifications; print("gymnasium=%s" % gymnasium.__version__)'
find "$target_dir" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$output_dir/installed-files.sha256"
echo "p2_gymnasium_setup_complete"
