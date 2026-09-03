"""Contracts for sealed-test SciCF Gate 1B.2 development evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from reproduction.scicf.gate1.gate1b1 import (
    STAGES,
    candidate_features,
    file_sha256,
    gain_feature_names,
    predict_ridge,
    scientific_object_valid,
    structure_key,
    structure_split,
    validity_feature_names,
)


GATE_VERSION = "scicf-gate1b2-nonlinear-validity-development-v1"
VALIDITY_FAMILY = "sklearn.ensemble.RandomForestRegressor"
FIXED_RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 512,
    "criterion": "squared_error",
    "max_depth": 8,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 71021,
    "n_jobs": 1,
}


def load_config(path: Path) -> Mapping[str, Any]:
    config = json.loads(path.resolve().read_text(encoding="utf-8"))
    if config.get("gate") != GATE_VERSION:
        raise ValueError("Gate 1B.2 configuration identity mismatch")
    validity = config["models"]["validity_filter"]
    if validity.get("family") != VALIDITY_FAMILY:
        raise ValueError("Gate 1B.2 validity family is not frozen")
    if validity.get("fixed_parameters") != FIXED_RANDOM_FOREST_PARAMETERS:
        raise ValueError("Gate 1B.2 random-forest parameters are not frozen")
    if config["models"].get("hyperparameter_search") is not False:
        raise ValueError("Gate 1B.2 forbids hyperparameter search")
    if config["models"].get("model_family_search") is not False:
        raise ValueError("Gate 1B.2 forbids model-family search")
    authorization = config["authorization"]
    if (
        authorization.get("test_collection_authorized") is not False
        or authorization.get("test_evaluation_authorized") is not False
        or authorization.get("separate_user_authorization_required_for_test")
        is not True
    ):
        raise ValueError("Gate 1B.2 sealed-test boundary is not fail-closed")
    return config


def validate_parent_contracts(
    repo_root: Path,
    config: Mapping[str, Any],
    parent_config_path: Path,
    expansion_config_path: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    from reproduction.scicf.gate1.gate1b1 import load_config as load_parent_config

    parent_config_path = parent_config_path.resolve()
    expansion_config_path = expansion_config_path.resolve()
    if file_sha256(parent_config_path) != config["parent"]["config_sha256"]:
        raise RuntimeError("Gate 1B.1 parent config checksum drift")
    if (
        file_sha256(expansion_config_path)
        != config["late_dev_expansion"]["collection_config_sha256"]
    ):
        raise RuntimeError("Gate 1B.2 late-dev collection config checksum drift")
    expected_parent = (repo_root / config["parent"]["config_path"]).resolve()
    expected_expansion = (
        repo_root / config["late_dev_expansion"]["collection_config_path"]
    ).resolve()
    if parent_config_path != expected_parent or expansion_config_path != expected_expansion:
        raise RuntimeError("Gate 1B.2 config paths do not match the frozen contract")
    parent = load_parent_config(parent_config_path)
    expansion = load_parent_config(expansion_config_path)
    if expansion.get("extension_role") != "Gate 1B.2 late-dev calibration expansion only":
        raise RuntimeError("Gate 1B.2 expansion role mismatch")
    if expansion["execution"].get("allowed_split") != "dev" or expansion[
        "execution"
    ].get("allowed_stage") != "late":
        raise RuntimeError("Gate 1B.2 expansion is not restricted to late dev")
    for key in ("structure_split", "checkpoints", "candidate_pool", "features"):
        if expansion[key] != parent[key]:
            raise RuntimeError("Gate 1B.2 expansion changed parent {}".format(key))
    verification_keys = (
        "paired_replicates",
        "common_random_numbers",
        "confidence_rule",
        "sign_consistency_fraction",
        "numerical_tolerance",
        "max_steps",
    )
    if any(
        expansion["verification"][key] != parent["verification"][key]
        for key in verification_keys
    ):
        raise RuntimeError("Gate 1B.2 expansion changed the verification protocol")
    expected_seeds = tuple(int(seed) for seed in config["late_dev_expansion"]["seeds"])
    observed_seeds = tuple(int(seed) for seed in expansion["seeds"]["dev"])
    if observed_seeds != expected_seeds:
        raise RuntimeError("Gate 1B.2 late-dev seed freeze mismatch")
    parent_seeds = {
        int(seed)
        for split in ("train", "dev", "test")
        for seed in parent["seeds"][split]
    }
    if parent_seeds & set(expected_seeds):
        raise RuntimeError("Gate 1B.2 late-dev seeds overlap the parent seeds")
    return parent, expansion


def load_stage_report(
    path: Path,
    expected_split: str,
    expected_stage: str,
    expected_seeds: Sequence[int],
    structure_config: Mapping[str, Any],
) -> Tuple[list, Dict[str, Any]]:
    resolved = path.resolve()
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete"
        or report.get("split") != expected_split
        or report.get("stage") != expected_stage
    ):
        raise ValueError("invalid Gate 1B.2 expansion report")
    observed_seeds = tuple(int(seed) for seed in report["seeds"])
    if observed_seeds != tuple(int(seed) for seed in expected_seeds):
        raise ValueError("Gate 1B.2 expansion seed manifest mismatch")
    expected_pool = int(structure_config["candidate_pool"]["size"])
    trajectories = report["trajectories"]
    if len(trajectories) != len(observed_seeds):
        raise ValueError("Gate 1B.2 expansion trajectory count mismatch")
    rows = []
    for item in trajectories:
        trajectory = item["trajectory"]
        candidates = item["pool"]["candidates"]
        if len(candidates) != expected_pool:
            raise ValueError("Gate 1B.2 expansion candidate pool size mismatch")
        gains = item["verified_gains"]
        verifications = {
            str(entry["intervention"]["intervention_id"]): entry
            for entry in item["verifications"]
        }
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            verification = verifications[candidate_id]
            outcomes = verification["paired_outcomes"]
            counterfactual_valid_rate = sum(
                scientific_object_valid(outcome.get("counterfactual_terminal_object"))
                for outcome in outcomes
            ) / float(len(outcomes))
            validity_features, gain_features = candidate_features(trajectory, candidate)
            component = str(candidate["intervention"]["component"])
            alternative = str(candidate["intervention"]["alternative_structure"])
            key = structure_key(component, alternative)
            if key != candidate["structure_key"]:
                raise ValueError("stored structure key does not match expansion candidate")
            if structure_split(component, alternative, structure_config) != expected_split:
                raise ValueError("expansion candidate is assigned to the wrong structure split")
            gain = float(gains[candidate_id])
            if not math.isclose(
                gain,
                float(verification["mean_delta"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Gate 1B.2 verified gain identity mismatch")
            rows.append(
                {
                    "split": expected_split,
                    "stage": expected_stage,
                    "trajectory_id": str(trajectory["trajectory_id"]),
                    "candidate_id": candidate_id,
                    "structure_key": key,
                    "timestep": int(candidate["intervention"]["timestep"]),
                    "terminal_valid": scientific_object_valid(
                        trajectory.get("terminal_scientific_object")
                    ),
                    "counterfactual_valid_rate": counterfactual_valid_rate,
                    "gain": gain,
                    "validity_features": validity_features,
                    "gain_features": gain_features,
                }
            )
    if expected_stage == "early" and any(bool(row["terminal_valid"]) for row in rows):
        raise ValueError("Gate 1B.2 early report is not all-invalid")
    if expected_stage in {"middle", "late"} and any(
        not bool(row["terminal_valid"]) for row in rows
    ):
        raise ValueError("Gate 1B.2 {} report is not all-valid".format(expected_stage))
    source = {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "slurm_job_id": report.get("slurm_job_id"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "source": report.get("source"),
        "test_seal": report.get("test_seal"),
        "structure_exclusion_seal": report.get("structure_exclusion_seal"),
        "authorization_receipt_seal": report.get("authorization_receipt_seal"),
    }
    return rows, source


def fit_validity_filter(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    parameters = config["models"]["validity_filter"]["fixed_parameters"]
    if parameters != FIXED_RANDOM_FOREST_PARAMETERS:
        raise RuntimeError("Gate 1B.2 attempted to alter fixed model parameters")
    features = np.vstack([row["validity_features"] for row in rows])
    targets = np.asarray(
        [float(row["counterfactual_valid_rate"]) for row in rows], dtype=float
    )
    model = RandomForestRegressor(**parameters)
    model.fit(features, targets)
    return model


def attach_predictions(
    rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]],
    validity_model: Any,
    gain_model: Mapping[str, Any],
) -> Dict[str, list]:
    result = {}
    for stage in STAGES:
        rows = rows_by_stage[stage]
        if not rows:
            result[stage] = []
            continue
        validity = validity_model.predict(
            np.vstack([row["validity_features"] for row in rows])
        )
        gain = predict_ridge(
            gain_model, np.vstack([row["gain_features"] for row in rows])
        )
        result[stage] = [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if not key.endswith("features")
                },
                "predicted_validity": float(validity[index]),
                "predicted_gain": float(gain[index]),
            }
            for index, row in enumerate(rows)
        ]
    return result


def validate_parent_model_manifest(
    path: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    resolved = path.resolve()
    if file_sha256(resolved) != config["parent"]["frozen_model_manifest_sha256"]:
        raise RuntimeError("Gate 1B.1 frozen model manifest checksum drift")
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        manifest.get("gate") != config["parent"]["gate"]
        or manifest.get("status") != "frozen"
        or manifest.get("gain_model", {}).get("family") != "ridge"
        or manifest.get("gain_feature_names") != list(gain_feature_names())
        or int(manifest.get("test_evaluations_completed", -1)) != 0
    ):
        raise RuntimeError("Gate 1B.1 frozen gain model contract mismatch")
    return manifest


def model_summary(model: Any, training_rows: int) -> Dict[str, Any]:
    return {
        "family": VALIDITY_FAMILY,
        "fixed_parameters": dict(FIXED_RANDOM_FOREST_PARAMETERS),
        "training_rows": int(training_rows),
        "feature_count": len(validity_feature_names()),
        "feature_importances": [
            {"feature": name, "importance": float(importance)}
            for name, importance in sorted(
                zip(validity_feature_names(), model.feature_importances_),
                key=lambda item: (-float(item[1]), item[0]),
            )
        ],
        "hyperparameter_search": False,
        "model_family_search": False,
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
