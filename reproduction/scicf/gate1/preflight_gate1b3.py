#!/usr/bin/env python3
"""Read-only preflight for the frozen Gate 1B.3 implementation candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from reproduction.framework.io import git_identity
from reproduction.scicf.gate1.gate1b3 import (
    feature_contract,
    load_config,
    load_early_validity_model,
    load_excluded_structure_keys,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 1B.3 preflight must run through Slurm")
    repo_root = args.repo_root.resolve()
    source = git_identity(repo_root)
    if source.get("dirty") is not False:
        raise RuntimeError("Gate 1B.3 preflight requires clean Git")
    config = load_config(args.config)
    collection = validate_collection_config(
        repo_root, config, args.collection_config
    )
    b1_manifest, b2_manifest = validate_model_manifests(
        args.gate1b1_model_manifest,
        args.gate1b2_model_manifest,
        config,
    )
    excluded, exclusion_seal = load_excluded_structure_keys(
        args.gate1b2_model_manifest, collection
    )
    early_model = load_early_validity_model(b2_manifest, config)
    features = feature_contract()
    if int(early_model.n_features_in_) != int(features["validity_feature_count"]):
        raise RuntimeError("Gate 1B.3 early model feature count mismatch")
    if len(b1_manifest["validity_model"]["coefficients"]) != int(
        features["validity_feature_count"]
    ):
        raise RuntimeError("Gate 1B.3 middle validity feature count mismatch")
    if len(b1_manifest["gain_model"]["coefficients"]) != int(
        features["gain_feature_count"]
    ):
        raise RuntimeError("Gate 1B.3 middle gain feature count mismatch")
    print(
        json.dumps(
            {
                "event": "gate1b3_preflight_complete",
                "source": source,
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "excluded_prior_dev_structure_keys": len(excluded),
                "exclusion_seal": exclusion_seal,
                "early_model_class": type(early_model).__name__,
                "feature_contract": features,
                "fresh_dev_collection_authorized": config["authorization"][
                    "fresh_dev_collection_authorized"
                ],
                "test_collection_authorized": config["authorization"][
                    "test_collection_authorized"
                ],
                "test_evaluation_authorized": config["authorization"][
                    "test_evaluation_authorized"
                ],
                "ppo_integration_authorized": config["authorization"][
                    "ppo_integration_authorized"
                ],
                "oracle_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
