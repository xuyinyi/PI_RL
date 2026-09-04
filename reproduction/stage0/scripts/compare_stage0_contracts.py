#!/usr/bin/env python
"""Fail unless PPO, Policy-CC and MCC-PPO manifests describe one task/budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--output", default="stage0_audit/contract_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.manifests) < 2:
        raise ValueError("At least two manifests are required.")
    records = []
    for value in args.manifests:
        path = Path(value)
        payload = json.loads(path.read_text())
        records.append(
            {
                "path": str(path.resolve()),
                "task_contract_id": payload.get("task_contract_id"),
                "budget_contract_id": payload.get("budget_contract_id"),
                "runtime_contract_id": payload.get("runtime_contract_id"),
                "environment_id": payload.get("environment", {}).get("environment_id"),
                "evaluator_version": payload.get("evaluator_version"),
                "objective_contract": payload.get("objective_contract"),
                "maximum_requested_calls": payload.get("oracle_budget", {}).get(
                    "maximum_requested_calls"
                ),
                "maximum_unique_calls": payload.get("oracle_budget", {}).get(
                    "maximum_unique_calls"
                ),
                "budget_protocol_version": payload.get("oracle_budget", {}).get(
                    "budget_protocol_version"
                ),
            }
        )
    fields = (
        "task_contract_id",
        "budget_contract_id",
        "runtime_contract_id",
        "environment_id",
        "evaluator_version",
        "objective_contract",
        "maximum_requested_calls",
        "maximum_unique_calls",
        "budget_protocol_version",
    )
    mismatches = {}
    for field in fields:
        values = [record.get(field) for record in records]
        if len(set(values)) != 1:
            mismatches[field] = values
    payload = {
        "status": "passed" if not mismatches else "failed",
        "manifest_count": len(records),
        "records": records,
        "mismatches": mismatches,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
