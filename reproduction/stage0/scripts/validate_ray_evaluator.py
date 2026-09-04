#!/usr/bin/env python
"""Validate one global Ray evaluator ledger under concurrent worker requests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from RL_PPO.envs.ray_evaluator import create_stage0_evaluator_actor
from RL_PPO.envs.sources import ENVIRONMENT_REGRESSION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", required=True)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--evaluator-mode", choices=("persistent", "legacy"), default="persistent")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duplicates", type=int, default=8)
    parser.add_argument("--output", default="stage0_audit/ray_evaluator.json")
    return parser.parse_args()


def read_first_smiles(path: Path) -> str:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")
        column = "smile" if "smile" in reader.fieldnames else "smiles"
        if column not in reader.fieldnames:
            raise ValueError("Input CSV must contain 'smile' or 'smiles'.")
        for row in reader:
            value = str(row.get(column, "")).strip()
            if value:
                return value
    raise ValueError("Input CSV contains no molecule.")


def main() -> None:
    args = parse_args()
    if args.duplicates <= 1:
        raise ValueError("--duplicates must exceed one.")

    import ray

    root = Path(args.dapigen_root).resolve()
    input_csv = Path(args.input_csv) if args.input_csv else root / "raw_data" / "PI.csv"
    smiles = read_first_smiles(input_csv)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    actor = create_stage0_evaluator_actor(
        dapigen_root=str(root),
        evaluator_mode=args.evaluator_mode,
        device=args.device,
        maximum_requested_calls=args.duplicates + 1,
        maximum_unique_calls=1,
        allowed_sources=(ENVIRONMENT_REGRESSION,),
        cache_scope="ray_concurrency_validation",
        num_cpus=1.0,
        num_gpus=0.0,
    )
    futures = [
        actor.evaluate_batch.remote([smiles], ENVIRONMENT_REGRESSION)
        for _ in range(args.duplicates)
    ]
    outputs = ray.get(futures)
    ledger_before_restore = ray.get(actor.ledger.remote())
    checkpoint = ray.get(actor.state_dict.remote())
    ray.get(actor.reset_ledger.remote(False))
    ledger_after_restore = ray.get(actor.load_state_dict.remote(checkpoint))
    post_restore = ray.get(
        actor.evaluate_batch.remote([smiles], ENVIRONMENT_REGRESSION)
    )[0]
    ledger = ray.get(actor.ledger.remote())
    objectives = [float(batch[0].objective) for batch in outputs]
    versions = [str(batch[0].evaluator_version) for batch in outputs]

    passed = bool(
        ledger_before_restore["requested_calls"] == args.duplicates
        and ledger_before_restore["unique_calls"] == 1
        and ledger_before_restore["backend_calls"] == 1
        and ledger_before_restore["cache_hits"] == args.duplicates - 1
        and ledger_after_restore == ledger_before_restore
        and ledger["requested_calls"] == args.duplicates + 1
        and ledger["unique_calls"] == 1
        and ledger["backend_calls"] == 1
        and ledger["cache_hits"] == args.duplicates
        and ledger["objective_contract"]
        == "dapigen-paper-equation-1-weighted-average-v1"
        and float(post_restore.objective) == objectives[0]
        and len(set(objectives)) == 1
        and len(set(versions)) == 1
    )
    payload = {
        "status": "passed" if passed else "failed",
        "duplicates": int(args.duplicates),
        "molecule": smiles,
        "objectives": objectives,
        "evaluator_versions": versions,
        "post_restore_objective": float(post_restore.objective),
        "ledger_before_restore": ledger_before_restore,
        "ledger_after_restore": ledger_after_restore,
        "ledger": ledger,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
