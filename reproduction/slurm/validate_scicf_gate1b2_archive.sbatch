#!/bin/bash
#SBATCH --job-name=scicf-g1b2-archive
#SBATCH --partition=compute
#SBATCH --nodelist=yanlih100n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b2-archive-%j.out
#SBATCH --error=/home/wch/workspaces/DAPiGen-reproduction/logs/scicf-g1b2-archive-%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 RUN_ROOT" >&2
    exit 2
fi

run_root="$1"
repo_root="/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree"
summary="$repo_root/reproduction/results/scicf-gate1b2-development-20260903/dev-summary.json"
parent_test="/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1b1-structure-split-20260903-v1/test"

cd "$repo_root"
test -z "$(git status --porcelain)"
jq -e '
    .decision == "dev-entry-no-go" and
    .sealed_test_eligible == false and
    .test_collection_authorized == false and
    .test_collected == false and
    .test_accessed == false and
    .test_evaluated == false and
    .decision_checks.dev_entry_passed == false and
    .execution.local_code_execution == false
' "$summary" >/dev/null

dev_report="$(jq -r '.source.dev_report' "$summary")"
model_manifest="$(jq -r '.source.model_manifest' "$summary")"
late_report="$(jq -r '.source.late_expansion_report' "$summary")"
expected_dev_sha="$(jq -r '.source.dev_report_sha256' "$summary")"
expected_manifest_sha="$(jq -r '.source.model_manifest_sha256' "$summary")"
expected_late_sha="$(jq -r '.source.late_expansion_report_sha256' "$summary")"

test "$(sha256sum "$dev_report" | awk '{print $1}')" = "$expected_dev_sha"
test "$(sha256sum "$model_manifest" | awk '{print $1}')" = "$expected_manifest_sha"
test "$(sha256sum "$late_report" | awk '{print $1}')" = "$expected_late_sha"
test ! -e "$parent_test"
test -z "$(find "$run_root" -type d -name test -print -quit)"

echo "job_id=${SLURM_JOB_ID} gate1b2_archive_valid=true sealed_test_untouched=true commit=$(git rev-parse HEAD)"
