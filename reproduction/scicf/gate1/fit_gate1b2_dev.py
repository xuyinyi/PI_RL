#!/usr/bin/env python3
"""Fit Gate 1B.2 on frozen train data and stop at a dev-only decision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence

import joblib
import sklearn

from reproduction.framework.io import git_identity, write_json
from reproduction.scicf.gate1.gate1b1 import (
    STAGES,
    calibrate_late_threshold,
    evaluate_early,
    evaluate_late,
    evaluate_middle,
    file_sha256,
    flatten_rows,
    headroom_fraction,
    load_split_reports,
    mean,
    raw_cross_timestep_fraction,
    structure_keys,
)
from reproduction.scicf.gate1.gate1b2 import (
    attach_predictions,
    fit_validity_filter,
    load_config,
    load_stage_report,
    model_summary,
    sha256_bytes,
    validate_parent_contracts,
    validate_parent_model_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--expansion-config", type=Path, required=True)
    parser.add_argument("--parent-model-manifest", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, action="append", required=True)
    parser.add_argument("--dev-report", type=Path, action="append", required=True)
    parser.add_argument("--late-expansion-report", type=Path, required=True)
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
        raise RuntimeError("Gate 1B.2 dev fitting must run through Slurm")
    repo_root = args.repo_root.resolve()
    source = git_identity(repo_root)
    if source.get("dirty") is not False:
        raise RuntimeError("Gate 1B.2 dev fitting requires clean Git")
    config_path = args.config.resolve()
    config = load_config(config_path)
    parent_config, expansion_config = validate_parent_contracts(
        repo_root,
        config,
        args.parent_config,
        args.expansion_config,
    )
    parent_manifest_path = args.parent_model_manifest.resolve()
    parent_manifest = validate_parent_model_manifest(parent_manifest_path, config)
    if sklearn.__version__ != config["execution"]["expected_sklearn_version"]:
        raise RuntimeError("Gate 1B.2 scikit-learn version mismatch")
    if joblib.__version__ != config["execution"]["expected_joblib_version"]:
        raise RuntimeError("Gate 1B.2 joblib version mismatch")

    train, train_sources = load_split_reports(args.train_report, "train", parent_config)
    dev, dev_sources = load_split_reports(args.dev_report, "dev", parent_config)
    expansion_rows, expansion_source = load_stage_report(
        args.late_expansion_report,
        expected_split="dev",
        expected_stage="late",
        expected_seeds=config["late_dev_expansion"]["seeds"],
        structure_config=expansion_config,
    )
    parent_data_sources = [
        split_sources[stage]["source"]
        for split_sources in (train_sources, dev_sources)
        for stage in STAGES
    ]
    parent_data_commits = {
        item.get("commit") for item in parent_data_sources if item
    }
    if (
        len(parent_data_commits) != 1
        or any(
            not item or item.get("dirty") is not False for item in parent_data_sources
        )
        or next(iter(parent_data_commits))
        != parent_manifest.get("collection_source_commit")
    ):
        raise RuntimeError("Gate 1B.2 parent data provenance mismatch")
    expansion_source_identity = expansion_source.get("source") or {}
    if (
        expansion_source_identity.get("dirty") is not False
        or expansion_source_identity.get("commit") != source.get("commit")
        or expansion_source.get("test_seal") is not None
    ):
        raise RuntimeError("Gate 1B.2 expansion provenance or sealed-test boundary mismatch")

    combined_dev = {
        "early": list(dev["early"]),
        "middle": list(dev["middle"]),
        "late": list(dev["late"]) + list(expansion_rows),
    }
    train_keys = structure_keys(train)
    dev_keys = structure_keys(combined_dev)
    overlap = sorted(train_keys & dev_keys)
    train_rows = flatten_rows(train)
    validity_model = fit_validity_filter(train_rows, config)
    gain_model = parent_manifest["gain_model"]
    predicted_original = attach_predictions(dev, validity_model, gain_model)
    predicted_expansion = attach_predictions(
        {"early": [], "middle": [], "late": expansion_rows},
        validity_model,
        gain_model,
    )
    predicted_combined_late = list(predicted_original["late"]) + list(
        predicted_expansion["late"]
    )

    budget = int(parent_config["candidate_pool"]["budget"])
    validity_threshold = float(config["models"]["validity_filter"]["threshold"])
    early = evaluate_early(predicted_original["early"], budget)
    middle = evaluate_middle(
        predicted_original["middle"], budget, validity_threshold
    )
    calibration = calibrate_late_threshold(
        predicted_combined_late, validity_threshold
    )
    late = []
    if calibration["status"] == "complete":
        late = evaluate_late(
            predicted_combined_late,
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
    original_cross_timestep_fraction = raw_cross_timestep_fraction(
        predicted_original
    )
    checks = {
        "train_combined_dev_structure_overlap_zero": len(overlap) == 0,
        "early_rescue_entry": bool(early_entry["passed"]),
        "middle_best_gain_entry": bool(middle_best_entry["passed"]),
        "middle_ndcg_entry": bool(middle_ndcg_entry["passed"]),
        "late_threshold_calibratable": calibration["status"] == "complete",
        "original_dev_cross_timestep_fraction_at_least_0_25": (
            original_cross_timestep_fraction
            >= float(
                config["dev_entry_rule"][
                    "minimum_cross_timestep_trajectory_fraction"
                ]
            )
        ),
    }
    passed = all(checks.values())

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError("Gate 1B.2 model root already exists")
    output_root.mkdir(parents=True)
    validity_model_path = output_root / "validity-filter.joblib"
    joblib.dump(validity_model, validity_model_path, compress=3)
    validity_model_sha256 = file_sha256(validity_model_path)
    gain_model_sha256 = sha256_bytes(
        json.dumps(gain_model, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    summary = model_summary(validity_model, len(train_rows))
    model_manifest = {
        "schema_version": 1,
        "gate": config["gate"],
        "status": "frozen-development",
        "source": source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "parent_config": {
            "path": str(args.parent_config.resolve()),
            "sha256": file_sha256(args.parent_config.resolve()),
        },
        "late_expansion_config": {
            "path": str(args.expansion_config.resolve()),
            "sha256": file_sha256(args.expansion_config.resolve()),
        },
        "validity_feature_names": parent_manifest["validity_feature_names"],
        "gain_feature_names": parent_manifest["gain_feature_names"],
        "validity_model": {
            **summary,
            "artifact": str(validity_model_path),
            "artifact_sha256": validity_model_sha256,
            "sklearn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        },
        "gain_model": {
            "source": "exact-parent-frozen-model-manifest",
            "parent_model_manifest": str(parent_manifest_path),
            "parent_model_manifest_sha256": file_sha256(parent_manifest_path),
            "canonical_model_sha256": gain_model_sha256,
            "family": gain_model["family"],
            "refit": False,
        },
        "validity_threshold": validity_threshold,
        "late_gain_threshold": (
            float(calibration["chosen"]["threshold"])
            if calibration["status"] == "complete"
            else None
        ),
        "train_structure_keys": sorted(train_keys),
        "combined_dev_structure_keys": sorted(dev_keys),
        "train_sources": train_sources,
        "original_dev_sources": dev_sources,
        "late_expansion_source": expansion_source,
        "dev_entry_checks": checks,
        "dev_entry_passed": passed,
        "sealed_test_eligible": passed,
        "test_collection_authorized": False,
        "test_evaluation_authorized": False,
        "test_data_accessed": False,
        "test_evaluations_completed": 0,
        "authorization": config["authorization"],
    }
    model_path = output_root / "frozen-development-model-manifest.json"
    write_json(model_path, model_manifest)
    dev_report = {
        "schema_version": 1,
        "gate": config["gate"],
        "status": "complete",
        "dataset_role": "development-entry-and-abstention-calibration",
        "source": source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "model_manifest": str(model_path),
        "model_manifest_sha256": file_sha256(model_path),
        "sealed_test": {
            "accessed": False,
            "collected": False,
            "evaluated": False,
            "eligible": passed,
            "collection_authorized": False,
            "separate_user_authorization_required": True,
        },
        "structure_isolation": {
            "train_keys": len(train_keys),
            "original_dev_keys": len(structure_keys(dev)),
            "combined_dev_keys": len(dev_keys),
            "overlap_count": len(overlap),
            "overlap": overlap,
        },
        "models": {
            "validity_filter": summary,
            "gain_ranker": {
                "source": "exact-parent-frozen-model-manifest",
                "canonical_model_sha256": gain_model_sha256,
                "refit": False,
            },
        },
        "early": {"entry": early_entry, "trajectories": early},
        "middle": {
            "best_gain_entry": middle_best_entry,
            "ndcg_entry": middle_ndcg_entry,
            "trajectories": middle,
        },
        "late": {
            "original_trajectories": len(dev["late"])
            // int(parent_config["candidate_pool"]["size"]),
            "expanded_trajectories": len(expansion_rows) // int(
                parent_config["candidate_pool"]["size"]
            ),
            "combined_trajectories": len(predicted_combined_late)
            // int(parent_config["candidate_pool"]["size"]),
            "calibration": calibration,
            "trajectories": late,
            "selection_count_range": [0, budget],
        },
        "original_dev_cross_timestep_trajectory_fraction": original_cross_timestep_fraction,
        "decision": {
            "checks": checks,
            "dev_entry_passed": passed,
            "sealed_test_eligible": passed,
            "test_collection_authorized": False,
            "test_evaluation_authorized": False,
            "gate1c_authorized": False,
            "pairwise_refinement_authorized": False,
            "ppo_integration_authorized": False,
        },
        "claim_boundary": (
            "This is a structure-disjoint development decision. The sealed test "
            "was not collected, read, or evaluated; a dev pass requires separate "
            "user authorization before any one-shot test access."
        ),
    }
    write_json(output_root / "dev-entry-report.json", dev_report)
    print(
        json.dumps(
            {
                "event": "gate1b2_dev_entry_complete",
                "output_root": str(output_root),
                **dev_report["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
