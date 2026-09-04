#!/usr/bin/env python
"""Require numerical parity between released and persistent QSPR evaluators."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", required=True)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="stage0_evaluator_parity")
    parser.add_argument("--tolerance", type=float, default=0.0)
    return parser.parse_args()


def load_smiles(path, column=None):
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if column is None:
            for candidate in ("smile", "smiles", "SMILES", "PI"):
                if candidate in fields:
                    column = candidate
                    break
        if column is None or column not in fields:
            raise ValueError(
                "Could not identify a SMILES column. Available fields: %s" % fields
            )
        values = [str(row[column]).strip() for row in reader if str(row[column]).strip()]
    return values, column


def as_flat_record(index, smiles, legacy, persistent):
    keys = ("transmittance", "cte", "strength", "tg", "sa_score")
    record = {
        "index": index,
        "smiles": smiles,
        "legacy_valid": legacy.valid,
        "persistent_valid": persistent.valid,
        "legacy_objective": legacy.objective,
        "persistent_objective": persistent.objective,
        "legacy_failure_reason": legacy.failure_reason,
        "persistent_failure_reason": persistent.failure_reason,
        "objective_abs_error": abs(legacy.objective - persistent.objective),
    }
    for key in keys:
        left = legacy.properties.get(key)
        right = persistent.properties.get(key)
        record["legacy_%s" % key] = left
        record["persistent_%s" % key] = right
        record["%s_abs_error" % key] = (
            None if left is None or right is None else abs(float(left) - float(right))
        )
    return record


def main():
    args = parse_args()
    root = Path(args.dapigen_root).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(repository_root))

    from RL_PPO.envs.chemistry import LegacyDAPiGenChemistryBackend
    from RL_PPO.envs.evaluator import (
        LegacyDAPiGenBenchmarkEvaluator,
        PersistentDAPiGenBenchmarkEvaluator,
    )

    input_csv = Path(args.input_csv) if args.input_csv else root / "raw_data" / "PI.csv"
    values, column = load_smiles(str(input_csv), args.smiles_column)
    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=True)
    canonical = []
    invalid_input = []
    for value in values:
        try:
            molecule = chemistry.canonicalize(value)
            if "*" not in molecule:
                canonical.append(molecule)
        except Exception as exc:
            invalid_input.append({"smiles": value, "error": str(exc)})
    canonical = sorted(set(canonical))
    if not canonical:
        raise RuntimeError("No valid complete molecules were found.")
    rng = random.Random(args.seed)
    rng.shuffle(canonical)
    panel = canonical[: min(args.sample_size, len(canonical))]

    legacy_evaluator = LegacyDAPiGenBenchmarkEvaluator()
    persistent_evaluator = PersistentDAPiGenBenchmarkEvaluator(
        str(root), device=args.device
    )
    if legacy_evaluator.objective_contract != persistent_evaluator.objective_contract:
        raise RuntimeError("Legacy and persistent objective contracts differ.")
    legacy_results = legacy_evaluator.evaluate_batch(panel)
    persistent_results = persistent_evaluator.evaluate_batch(panel)

    records = [
        as_flat_record(index, smiles, legacy, persistent)
        for index, (smiles, legacy, persistent) in enumerate(
            zip(panel, legacy_results, persistent_results)
        )
    ]
    tolerance = float(args.tolerance)
    mismatch_records = []
    metric_names = (
        "objective_abs_error",
        "transmittance_abs_error",
        "cte_abs_error",
        "strength_abs_error",
        "tg_abs_error",
        "sa_score_abs_error",
    )
    for record in records:
        # The fixed PI validation panel is expected to be evaluable. Treating
        # "both failed" as parity would let missing checkpoints pass silently.
        mismatch = bool(
            not record["legacy_valid"]
            or not record["persistent_valid"]
            or record["legacy_valid"] != record["persistent_valid"]
        )
        for metric in metric_names:
            value = record.get(metric)
            if value is not None and float(value) > tolerance:
                mismatch = True
        if mismatch:
            mismatch_records.append(record)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluator_parity.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "status": "passed" if not mismatch_records else "failed",
        "dapigen_root": str(root),
        "input_csv": str(input_csv),
        "smiles_column": column,
        "seed": args.seed,
        "requested_sample_size": args.sample_size,
        "evaluated_sample_size": len(panel),
        "invalid_input_count": len(invalid_input),
        "legacy_evaluator_version": legacy_evaluator.evaluator_version,
        "persistent_evaluator_version": persistent_evaluator.evaluator_version,
        "objective_contract": persistent_evaluator.objective_contract,
        "tolerance": tolerance,
        "mismatch_count": len(mismatch_records),
        "mismatches": mismatch_records[:20],
        "csv": str(csv_path.resolve()),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if mismatch_records:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
