#!/usr/bin/env python3
"""Evaluate the frozen stage-routed Gate 1B.3 candidate on fresh dev only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from reproduction.framework.io import git_identity, write_json
from reproduction.scicf.gate1.gate1b1 import (
    evaluate_early,
    evaluate_middle,
    group_rows,
    headroom_fraction,
    mean,
)
from reproduction.scicf.gate1.gate1b2 import load_stage_report
from reproduction.scicf.gate1.gate1b3 import (
    attach_stage_routed_predictions,
    file_sha256,
    load_config,
    load_early_validity_model,
    validate_collection_config,
    validate_model_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--collection-config", type=Path, required=True)
    parser.add_argument("--gate1b1-model-manifest", type=Path, required=True)
    parser.add_argument("--gate1b2-model-manifest", type=Path, required=True)
    parser.add_argument("--early-report", type=Path, required=True)
    parser.add_argument("--middle-report", type=Path, required=True)
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
        "passed": difference > 0.0 and captured >= minimum_fraction,
    }


def _fresh_cross_timestep_fraction(
    rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]]
) -> float:
    trajectories = [
        current
        for stage in ("early", "middle")
        for current in group_rows(rows_by_stage[stage]).values()
    ]
    if not trajectories:
        raise ValueError("Gate 1B.3 fresh dev has no trajectories")
    return sum(
        len({int(row["timestep"]) for row in current}) > 1
        for current in trajectories
    ) / float(len(trajectories))


def _validate_exclusion_seal(
    source: Mapping[str, Any], expected_sha: str, expected_key_count: int
) -> None:
    seal = source.get("structure_exclusion_seal") or {}
    if (
        seal.get("manifest_sha256") != expected_sha
        or int(seal.get("excluded_key_count", -1)) != expected_key_count
    ):
        raise RuntimeError("Gate 1B.3 report does not carry the frozen exclusion seal")


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 1B.3 dev evaluation must run through Slurm")
    repo_root = args.repo_root.resolve()
    source = git_identity(repo_root)
    if source.get("dirty") is not False:
        raise RuntimeError("Gate 1B.3 dev evaluation requires clean Git")
    config_path = args.config.resolve()
    config = load_config(config_path)
    if config["authorization"].get("fresh_dev_evaluation_authorized") is not True:
        raise RuntimeError(
            "Gate 1B.3 fresh-dev evaluation requires separate authorization"
        )
    collection_config = validate_collection_config(
        repo_root, config, args.collection_config
    )
    b1_manifest, b2_manifest = validate_model_manifests(
        args.gate1b1_model_manifest,
        args.gate1b2_model_manifest,
        config,
    )
    seeds = config["fresh_development_confirmation"]["seeds"]
    early_rows, early_source = load_stage_report(
        args.early_report,
        expected_split="dev",
        expected_stage="early",
        expected_seeds=seeds,
        structure_config=collection_config,
    )
    middle_rows, middle_source = load_stage_report(
        args.middle_report,
        expected_split="dev",
        expected_stage="middle",
        expected_seeds=seeds,
        structure_config=collection_config,
    )
    sources = (early_source, middle_source)
    source_commits = {
        item.get("source", {}).get("commit") for item in sources
    }
    if (
        source_commits != {source.get("commit")}
        or any(item.get("source", {}).get("dirty") is not False for item in sources)
        or any(item.get("test_seal") is not None for item in sources)
    ):
        raise RuntimeError("Gate 1B.3 fresh dev provenance violates the test seal")
    receipt_seals = [item.get("authorization_receipt_seal") or {} for item in sources]
    receipt_hashes = {item.get("receipt_sha256") for item in receipt_seals}
    if (
        len(receipt_hashes) != 1
        or None in receipt_hashes
        or any(item.get("action") != "fresh-dev-collection" for item in receipt_seals)
    ):
        raise RuntimeError("Gate 1B.3 fresh dev lacks one collection authorization seal")

    prior_dev_keys = {str(key) for key in b2_manifest["combined_dev_structure_keys"]}
    train_keys = {str(key) for key in b2_manifest["train_structure_keys"]}
    expected_exclusion_sha = config["fresh_development_confirmation"][
        "prior_dev_structure_manifest_sha256"
    ]
    for item in sources:
        _validate_exclusion_seal(
            item, expected_exclusion_sha, len(prior_dev_keys)
        )
    fresh_rows = list(early_rows) + list(middle_rows)
    fresh_keys = {str(row["structure_key"]) for row in fresh_rows}
    train_overlap = sorted(fresh_keys & train_keys)
    prior_dev_overlap = sorted(fresh_keys & prior_dev_keys)

    early_model = load_early_validity_model(b2_manifest, config)
    predicted = attach_stage_routed_predictions(
        early_rows, middle_rows, early_model, b1_manifest
    )
    budget = int(collection_config["candidate_pool"]["budget"])
    threshold = float(config["models"]["validity_threshold"])
    early = evaluate_early(predicted["early"], budget)
    middle = evaluate_middle(predicted["middle"], budget, threshold)
    minimum_fraction = float(
        config["entry_rule"]["minimum_oracle_headroom_fraction_captured"]
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
    cross_timestep_fraction = _fresh_cross_timestep_fraction(predicted)
    checks = {
        "fresh_dev_train_structure_overlap_zero": len(train_overlap) == 0,
        "fresh_dev_prior_dev_structure_overlap_zero": len(prior_dev_overlap) == 0,
        "early_rescue_entry": bool(early_entry["passed"]),
        "middle_best_gain_entry": bool(middle_best_entry["passed"]),
        "middle_ndcg_entry": bool(middle_ndcg_entry["passed"]),
        "fresh_cross_timestep_fraction_at_least_0_25": (
            cross_timestep_fraction
            >= float(config["entry_rule"]["minimum_cross_timestep_trajectory_fraction"])
        ),
    }
    passed = all(checks.values())

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError("Gate 1B.3 output root already exists")
    output_root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "gate": config["gate"],
        "status": "fresh-development-confirmed" if passed else "fresh-development-no-go",
        "source": source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "gate1b1_model_manifest": {
            "path": str(args.gate1b1_model_manifest.resolve()),
            "sha256": file_sha256(args.gate1b1_model_manifest.resolve()),
        },
        "gate1b2_model_manifest": {
            "path": str(args.gate1b2_model_manifest.resolve()),
            "sha256": file_sha256(args.gate1b2_model_manifest.resolve()),
        },
        "fresh_sources": {"early": early_source, "middle": middle_source},
        "fresh_structure_keys": sorted(fresh_keys),
        "entry_checks": checks,
        "dev_entry_passed": passed,
        "sealed_test_eligible": passed,
        "test_collection_authorized": False,
        "test_evaluation_authorized": False,
        "test_data_accessed": False,
        "test_evaluations_completed": 0,
        "late_saturation_diagnostic": config["late_saturation_diagnostic"],
        "authorization": config["authorization"],
    }
    manifest_path = output_root / "frozen-stage-routed-model-manifest.json"
    write_json(manifest_path, manifest)
    report = {
        "schema_version": 1,
        "gate": config["gate"],
        "status": "complete",
        "source": source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "model_manifest": str(manifest_path),
        "model_manifest_sha256": file_sha256(manifest_path),
        "structure_isolation": {
            "fresh_keys": len(fresh_keys),
            "train_overlap_count": len(train_overlap),
            "prior_dev_overlap_count": len(prior_dev_overlap),
        },
        "early": {"entry": early_entry, "trajectories": early},
        "middle": {
            "best_gain_entry": middle_best_entry,
            "ndcg_entry": middle_ndcg_entry,
            "trajectories": middle,
        },
        "late": {
            **config["late_saturation_diagnostic"],
            "gating": False,
            "selected_count": 0,
        },
        "fresh_cross_timestep_trajectory_fraction": cross_timestep_fraction,
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
            "Gate 1B.3 is a fresh-seed, prior-dev-structure-excluded development "
            "confirmation. The sealed test is not read or evaluated, and late "
            "saturation is diagnostic rather than a gate."
        ),
    }
    write_json(output_root / "dev-confirmation-report.json", report)
    print(
        json.dumps(
            {
                "event": "gate1b3_fresh_dev_complete",
                "output_root": str(output_root),
                **report["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
