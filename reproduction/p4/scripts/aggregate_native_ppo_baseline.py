#!/usr/bin/env python
"""Aggregate all five frozen P4-A formal seeds without claim promotion."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from reproduction.p4.protocol import (
    REQUIRED_METRICS,
    evaluate_acceptance,
    file_sha256,
    load_protocol,
)


def summary(values):
    parsed = [float(value) for value in values]
    return {
        "n": len(parsed),
        "mean": statistics.mean(parsed),
        "sample_standard_deviation": statistics.stdev(parsed)
        if len(parsed) > 1
        else 0.0,
        "minimum": min(parsed),
        "maximum": max(parsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    runs_root = Path(args.runs_root).resolve()
    output_path = Path(args.output).resolve()
    protocol = load_protocol(protocol_path)
    expected_seeds = tuple(int(value) for value in protocol["formal"]["seeds"])
    reports = []
    failures = []
    for seed in expected_seeds:
        path = runs_root / ("seed-%d" % seed) / "run_report.json"
        if not path.is_file():
            failures.append("missing report for seed %d" % seed)
            continue
        report = json.loads(path.read_text())
        checks, accepted = evaluate_acceptance(report, protocol)
        if report.get("status") != "passed" or not accepted:
            failures.append("seed %d failed acceptance: %s" % (seed, checks))
        reports.append((seed, path, report))

    commits = {report.get("source", {}).get("commit") for _seed, _path, report in reports}
    protocol_hashes = {report.get("protocol_sha256") for _seed, _path, report in reports}
    if len(commits) != 1:
        failures.append("formal seeds do not share one Git commit")
    if protocol_hashes != {file_sha256(protocol_path)}:
        failures.append("formal seeds do not share the frozen protocol hash")

    curves = {}
    per_seed = []
    for seed, path, report in reports:
        evaluation_rows = []
        for evaluation in report["evaluations"]:
            metrics = evaluation["metrics"]
            row = {
                "target_requested_calls": evaluation["target_requested_calls"],
                "actual_training_requested_calls": evaluation[
                    "actual_training_requested_calls"
                ],
                **{name: metrics[name] for name in REQUIRED_METRICS},
            }
            evaluation_rows.append(row)
            curves.setdefault(str(evaluation["target_requested_calls"]), []).append(row)
        per_seed.append(
            {
                "seed": seed,
                "report": str(path),
                "report_sha256": file_sha256(path),
                "source_commit": report["source"]["commit"],
                "environment_transitions": report["environment_transitions"],
                "training_evaluator_ledger": report["training_evaluator_ledger"],
                "final_policy_sha256": report["final_policy_sha256"],
                "elapsed_seconds": report["elapsed_seconds"],
                "evaluations": evaluation_rows,
            }
        )

    aggregate_curves = {}
    if not failures:
        for target, rows in sorted(curves.items(), key=lambda item: int(item[0])):
            aggregate_curves[target] = {
                "actual_training_requested_calls": summary(
                    row["actual_training_requested_calls"] for row in rows
                ),
                **{
                    metric: summary(row[metric] for row in rows)
                    for metric in REQUIRED_METRICS
                },
            }
            if any(
                not math.isfinite(float(item["mean"]))
                for item in aggregate_curves[target].values()
            ):
                failures.append("non-finite aggregate at target %s" % target)

    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "status": "passed" if not failures else "failed",
        "classification": "P4-A native-standard baseline engineering evidence only",
        "complete_p4_gate": False,
        "algorithm_comparison_performed": False,
        "scientific_validation_performed": False,
        "expected_seeds": list(expected_seeds),
        "source_commits": sorted(value for value in commits if value),
        "failures": failures,
        "per_seed": per_seed,
        "aggregate_learning_curves": aggregate_curves,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output_path)}, sort_keys=True))
    if failures:
        raise SystemExit("P4-A aggregation failed.")


if __name__ == "__main__":
    main()

