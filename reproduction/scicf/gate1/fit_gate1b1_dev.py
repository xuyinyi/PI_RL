#!/usr/bin/env python3
"""Fit Gate 1B.1 on train structures and freeze the dev entry decision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from reproduction.framework.io import git_identity, write_json
from reproduction.scicf.gate1.gate1b1 import (
    STAGES,
    attach_predictions,
    calibrate_late_threshold,
    evaluate_early,
    evaluate_late,
    evaluate_middle,
    file_sha256,
    fit_ridge,
    flatten_rows,
    gain_feature_names,
    headroom_fraction,
    load_config,
    load_split_reports,
    mean,
    structure_keys,
    trajectory_cross_timestep_fraction,
    validity_feature_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, action="append", required=True)
    parser.add_argument("--dev-report", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _metric_entry(
    differences: Sequence[float], headrooms: Sequence[float], minimum_fraction: float
) -> Dict[str, Any]:
    difference = mean(differences)
    captured = headroom_fraction(differences, headrooms)
    return {
        "mean_difference": difference,
        "mean_oracle_headroom": mean(headrooms),
        "oracle_headroom_fraction_captured": captured,
        "minimum_required_fraction": minimum_fraction,
        "checks": {
            "mean_difference_positive": difference > 0.0,
            "minimum_headroom_fraction_captured": captured >= minimum_fraction,
        },
        "passed": difference > 0.0 and captured >= minimum_fraction,
    }


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 1B.1 dev fitting must run through Slurm")
    repo_root = args.repo_root.resolve()
    source = git_identity(repo_root)
    if source.get("dirty") is not False:
        raise RuntimeError("Gate 1B.1 dev fitting requires clean Git")
    config_path = args.config.resolve()
    config = load_config(config_path)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError("Gate 1B.1 model root already exists")
    output_root.mkdir(parents=True)

    train, train_sources = load_split_reports(args.train_report, "train", config)
    dev, dev_sources = load_split_reports(args.dev_report, "dev", config)
    for split_sources in (train_sources, dev_sources):
        for stage in STAGES:
            if split_sources[stage]["source"] != source:
                raise RuntimeError("Gate 1B.1 data source does not match fitting source")
    train_keys = structure_keys(train)
    dev_keys = structure_keys(dev)
    overlap = sorted(train_keys & dev_keys)
    train_rows = flatten_rows(train)
    validity_model_config = config["models"]["validity_filter"]
    gain_model_config = config["models"]["policy_conditioned_gain_ranker"]
    validity_model = fit_ridge(
        np.vstack([row["validity_features"] for row in train_rows]),
        np.asarray([float(row["counterfactual_valid_rate"]) for row in train_rows]),
        float(validity_model_config["alpha"]),
    )
    gain_training_rows = [
        row
        for stage in gain_model_config["training_stages"]
        for row in train[stage]
        if float(row["counterfactual_valid_rate"]) == 1.0
    ]
    if not gain_training_rows:
        raise RuntimeError("Gate 1B.1 has no fully valid gain-training row")
    gain_model = fit_ridge(
        np.vstack([row["gain_features"] for row in gain_training_rows]),
        np.asarray([float(row["gain"]) for row in gain_training_rows]),
        float(gain_model_config["alpha"]),
    )
    predicted = attach_predictions(dev, validity_model, gain_model)
    budget = int(config["candidate_pool"]["budget"])
    validity_threshold = float(validity_model_config["threshold"])
    early = evaluate_early(predicted["early"], budget)
    middle = evaluate_middle(predicted["middle"], budget, validity_threshold)
    calibration = calibrate_late_threshold(predicted["late"], validity_threshold)
    late = []
    if calibration["status"] == "complete":
        late = evaluate_late(
            predicted["late"],
            budget,
            validity_threshold,
            float(calibration["chosen"]["threshold"]),
        )

    minimum_fraction = float(
        config["dev_entry_rule"]["minimum_oracle_headroom_fraction_captured"]
    )
    early_entry = _metric_entry(
        [float(row["descriptor_minus_random"]) for row in early],
        [float(row["oracle_minus_random"]) for row in early],
        minimum_fraction,
    )
    middle_best_entry = _metric_entry(
        [float(row["best_gain_descriptor_minus_random"]) for row in middle],
        [float(row["best_gain_oracle_minus_random"]) for row in middle],
        minimum_fraction,
    )
    middle_ndcg_entry = _metric_entry(
        [float(row["ndcg_descriptor_minus_random"]) for row in middle],
        [float(row["ndcg_oracle_minus_random"]) for row in middle],
        minimum_fraction,
    )
    evaluated = {"early": early, "middle": middle, "late": late}
    cross_timestep_fraction = trajectory_cross_timestep_fraction(evaluated)
    checks = {
        "train_dev_structure_overlap_zero": len(overlap) == 0,
        "early_rescue_entry": bool(early_entry["passed"]),
        "middle_best_gain_entry": bool(middle_best_entry["passed"]),
        "middle_ndcg_entry": bool(middle_ndcg_entry["passed"]),
        "late_threshold_calibratable": calibration["status"] == "complete",
        "cross_timestep_fraction_at_least_0_25": cross_timestep_fraction >= 0.25,
    }
    passed = all(checks.values())
    model_manifest = {
        "schema_version": 1,
        "gate": config["gate"],
        "status": "frozen",
        "source": source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "feature_protocol": config["features"]["protocol"],
        "validity_feature_names": validity_feature_names(),
        "gain_feature_names": gain_feature_names(),
        "validity_model": validity_model,
        "gain_model": gain_model,
        "validity_threshold": validity_threshold,
        "late_gain_threshold": (
            float(calibration["chosen"]["threshold"])
            if calibration["status"] == "complete"
            else None
        ),
        "train_structure_keys": sorted(train_keys),
        "dev_structure_keys": sorted(dev_keys),
        "train_sources": train_sources,
        "dev_sources": dev_sources,
        "dev_entry_checks": checks,
        "dev_entry_passed": passed,
        "test_collection_authorized": passed,
        "test_evaluations_completed": 0,
        "authorization": config["authorization"],
    }
    model_path = output_root / "frozen-model-manifest.json"
    write_json(model_path, model_manifest)
    dev_report = {
        "schema_version": 1,
        "gate": config["gate"],
        "status": "complete",
        "dataset_role": "dev-entry-and-calibration",
        "source": source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "model_manifest": str(model_path),
        "model_manifest_sha256": file_sha256(model_path),
        "structure_isolation": {
            "train_keys": len(train_keys),
            "dev_keys": len(dev_keys),
            "overlap_count": len(overlap),
            "overlap": overlap,
        },
        "models": {
            "validity_training_rows": validity_model["training_rows"],
            "gain_training_rows": gain_model["training_rows"],
            "hyperparameter_search": False,
        },
        "early": {"entry": early_entry, "trajectories": early},
        "middle": {
            "best_gain_entry": middle_best_entry,
            "ndcg_entry": middle_ndcg_entry,
            "trajectories": middle,
        },
        "late": {"calibration": calibration, "trajectories": late},
        "cross_timestep_trajectory_fraction": cross_timestep_fraction,
        "decision": {
            "checks": checks,
            "dev_entry_passed": passed,
            "test_collection_authorized": passed,
            "gate1c_authorized": False,
            "pairwise_refinement_authorized": False,
            "ppo_integration_authorized": False,
        },
        "claim_boundary": (
            "This is a structure-disjoint development entry decision. Test labels "
            "have not been collected or evaluated."
        ),
    }
    write_json(output_root / "dev-entry-report.json", dev_report)
    print(
        json.dumps(
            {
                "event": "gate1b1_dev_entry_complete",
                "output_root": str(output_root),
                **dev_report["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
