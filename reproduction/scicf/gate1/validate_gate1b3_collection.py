#!/usr/bin/env python3
"""Validate Gate 1B.3 collection metadata without evaluating its labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from reproduction.framework.io import git_identity, write_json
from reproduction.scicf.gate1.gate1b1 import file_sha256
from reproduction.scicf.gate1.gate1b3 import (
    load_config,
    validate_collection_authorization,
    validate_collection_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--collection-config", type=Path, required=True)
    parser.add_argument("--prior-dev-manifest", type=Path, required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--early-report", type=Path, required=True)
    parser.add_argument("--middle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 1B.3 collection validation must run through Slurm")
    repo_root = args.repo_root.resolve()
    source = git_identity(repo_root)
    if source.get("dirty") is not False:
        raise RuntimeError("Gate 1B.3 collection validation requires clean Git")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Gate 1B.3 collection validation already exists")
    config = load_config(args.config)
    collection_config = validate_collection_config(
        repo_root, config, args.collection_config
    )
    _, receipt_seal = validate_collection_authorization(
        args.authorization_receipt, collection_config
    )
    prior_path = args.prior_dev_manifest.resolve()
    prior_sha = file_sha256(prior_path)
    expected_prior_sha = collection_config["structure_exclusion"]["manifest_sha256"]
    if prior_sha != expected_prior_sha:
        raise RuntimeError("Gate 1B.3 prior-dev manifest checksum drift")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_dev_keys = {str(key) for key in prior["combined_dev_structure_keys"]}
    train_keys = {str(key) for key in prior["train_structure_keys"]}
    expected_seeds = tuple(int(seed) for seed in collection_config["seeds"]["dev"])
    expected_pool_size = int(collection_config["candidate_pool"]["size"])

    reports = {}
    collection_commits = set()
    all_fresh_keys = set()
    oracle_counts = {"factual": 0, "counterfactual": 0, "evaluation": 0, "total": 0}
    for stage, unresolved in (
        ("early", args.early_report),
        ("middle", args.middle_report),
    ):
        path = unresolved.resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "complete"
            or report.get("split") != "dev"
            or report.get("stage") != stage
            or tuple(int(seed) for seed in report.get("seeds", ())) != expected_seeds
            or len(report.get("trajectories", ())) != len(expected_seeds)
            or report.get("test_seal") is not None
        ):
            raise RuntimeError("Gate 1B.3 {} collection report mismatch".format(stage))
        identity = report.get("source") or {}
        if identity.get("dirty") is not False or not identity.get("commit"):
            raise RuntimeError("Gate 1B.3 collection source is not clean")
        collection_commits.add(identity["commit"])
        exclusion_seal = report.get("structure_exclusion_seal") or {}
        if (
            exclusion_seal.get("manifest_sha256") != expected_prior_sha
            or int(exclusion_seal.get("excluded_key_count", -1))
            != len(prior_dev_keys)
        ):
            raise RuntimeError("Gate 1B.3 structure exclusion seal mismatch")
        observed_receipt = report.get("authorization_receipt_seal") or {}
        if (
            observed_receipt.get("receipt_sha256")
            != receipt_seal["receipt_sha256"]
            or observed_receipt.get("action") != "fresh-dev-collection"
        ):
            raise RuntimeError("Gate 1B.3 collection authorization seal mismatch")
        stage_keys = set()
        for trajectory in report["trajectories"]:
            candidates = trajectory["pool"]["candidates"]
            if len(candidates) != expected_pool_size:
                raise RuntimeError("Gate 1B.3 candidate pool size mismatch")
            observed_timesteps = {
                int(value) for value in trajectory["cross_timestep"]["observed_timesteps"]
            }
            selected_timesteps = {
                int(value)
                for value in trajectory["cross_timestep"][
                    "selected_candidates_by_timestep"
                ]
            }
            if observed_timesteps != selected_timesteps:
                raise RuntimeError("Gate 1B.3 pool missed an observed timestep")
            stage_keys.update(str(item["structure_key"]) for item in candidates)
        if stage_keys & prior_dev_keys or stage_keys & train_keys:
            raise RuntimeError("Gate 1B.3 fresh collection has structure leakage")
        all_fresh_keys.update(stage_keys)
        counts = report["oracle_counts"]
        for key in oracle_counts:
            oracle_counts[key] += int(counts[key])
        reports[stage] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "trajectory_count": len(report["trajectories"]),
            "unique_structure_keys": len(stage_keys),
            "oracle_counts": counts,
            "source": identity,
        }
    if len(collection_commits) != 1:
        raise RuntimeError("Gate 1B.3 collection stages have different source commits")

    result = {
        "schema_version": 1,
        "gate": config["gate"],
        "status": "collection-complete-valid",
        "dataset_role": "fresh-development collection only",
        "source": source,
        "collection_source_commit": next(iter(collection_commits)),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "reports": reports,
        "trajectory_count": len(expected_seeds) * 2,
        "unique_fresh_structure_keys": len(all_fresh_keys),
        "prior_dev_structure_overlap_count": len(all_fresh_keys & prior_dev_keys),
        "train_structure_overlap_count": len(all_fresh_keys & train_keys),
        "authorization_receipt": receipt_seal,
        "oracle_counts": oracle_counts,
        "labels_evaluated": False,
        "fresh_dev_evaluation_authorized": False,
        "test_data_accessed": False,
        "test_collection_authorized": False,
        "test_evaluation_authorized": False,
        "ppo_integration_authorized": False,
    }
    write_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
