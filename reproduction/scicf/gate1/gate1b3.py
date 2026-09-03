"""Stage-routed Gate 1B.3 contracts with a sealed-test boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from reproduction.scicf.gate1.gate1b1 import (
    file_sha256,
    gain_feature_names,
    predict_ridge,
    rank_rows,
    validity_feature_names,
)


GATE_VERSION = "scicf-gate1b3-stage-routed-unseen-development-v1"
COLLECTION_ROLE = (
    "Gate 1B.3 unseen-structure early-middle dev confirmation only"
)
STAGE_ROUTER = {
    "early": "nonlinear-validity",
    "middle": "linear-validity-then-policy-conditioned-gain",
    "late": "abstain-while-uncalibrated",
}


def load_config(path: Path) -> Mapping[str, Any]:
    config = json.loads(path.resolve().read_text(encoding="utf-8"))
    if config.get("gate") != GATE_VERSION:
        raise ValueError("Gate 1B.3 configuration identity mismatch")
    models = config["models"]
    if (
        models.get("hyperparameter_search") is not False
        or models.get("model_family_search") is not False
        or models["early_validity_filter"].get("refit") is not False
        or models["middle_validity_filter"].get("refit") is not False
        or models["middle_gain_ranker"].get("refit") is not False
    ):
        raise ValueError("Gate 1B.3 model identities must remain frozen")
    if config["entry_rule"].get("late_is_gating") is not False:
        raise ValueError("Gate 1B.3 late saturation diagnostic must be non-gating")
    if config["stage_router"].get("late_default_selection_count") != 0:
        raise ValueError("Gate 1B.3 uncalibrated late stage must abstain")
    authorization = config["authorization"]
    closed = (
        "fresh_dev_collection_authorized",
        "fresh_dev_evaluation_authorized",
        "test_collection_authorized",
        "test_evaluation_authorized",
        "gate1c_authorized",
        "pairwise_refinement_authorized",
        "ppo_integration_authorized",
    )
    if any(authorization.get(key) is not False for key in closed):
        raise ValueError("Gate 1B.3 implementation config must fail closed")
    return config


def validate_collection_config(
    repo_root: Path,
    config: Mapping[str, Any],
    collection_config_path: Path,
) -> Mapping[str, Any]:
    from reproduction.scicf.gate1.gate1b1 import load_config as load_parent_config

    path = collection_config_path.resolve()
    expected_path = (
        repo_root
        / config["fresh_development_confirmation"]["collection_config_path"]
    ).resolve()
    if path != expected_path:
        raise RuntimeError("Gate 1B.3 collection config path mismatch")
    if (
        file_sha256(path)
        != config["fresh_development_confirmation"]["collection_config_sha256"]
    ):
        raise RuntimeError("Gate 1B.3 collection config checksum drift")
    collection = load_parent_config(path)
    if collection.get("extension_role") != COLLECTION_ROLE:
        raise RuntimeError("Gate 1B.3 collection role mismatch")
    execution = collection["execution"]
    if execution.get("allowed_split") != "dev" or tuple(
        execution.get("allowed_stages", ())
    ) != ("early", "middle"):
        raise RuntimeError("Gate 1B.3 collection is not restricted to dev early/middle")
    confirmation = config["fresh_development_confirmation"]
    if tuple(collection["seeds"]["dev"]) != tuple(confirmation["seeds"]):
        raise RuntimeError("Gate 1B.3 fresh dev seed freeze mismatch")
    exclusion = collection.get("structure_exclusion", {})
    if (
        exclusion.get("required") is not True
        or exclusion.get("manifest_sha256")
        != confirmation["prior_dev_structure_manifest_sha256"]
    ):
        raise RuntimeError("Gate 1B.3 prior-dev structure exclusion is not frozen")
    return collection


def load_excluded_structure_keys(
    manifest_path: Path, collection_config: Mapping[str, Any]
) -> Tuple[set, Dict[str, Any]]:
    exclusion = collection_config.get("structure_exclusion")
    if not exclusion or exclusion.get("required") is not True:
        raise RuntimeError("structure exclusion is not required by this config")
    resolved = manifest_path.resolve()
    observed_sha = file_sha256(resolved)
    if observed_sha != exclusion["manifest_sha256"]:
        raise RuntimeError("prior development structure manifest checksum drift")
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    if manifest.get("gate") != exclusion["manifest_gate"]:
        raise RuntimeError("prior development structure manifest gate mismatch")
    if (
        manifest.get("test_data_accessed") is not False
        or int(manifest.get("test_evaluations_completed", -1)) != 0
    ):
        raise RuntimeError("prior development manifest does not preserve the test seal")
    manifest_key = str(exclusion["manifest_key"])
    keys = {str(value) for value in manifest.get(manifest_key, ())}
    if not keys:
        raise RuntimeError("prior development structure exclusion set is empty")
    return keys, {
        "manifest": str(resolved),
        "manifest_sha256": observed_sha,
        "manifest_key": manifest_key,
        "excluded_key_count": len(keys),
    }


def validate_model_manifests(
    gate1b1_manifest_path: Path,
    gate1b2_manifest_path: Path,
    config: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    b1_path = gate1b1_manifest_path.resolve()
    b2_path = gate1b2_manifest_path.resolve()
    expected_b1_sha = config["models"]["middle_gain_ranker"][
        "source_manifest_sha256"
    ]
    if file_sha256(b1_path) != expected_b1_sha:
        raise RuntimeError("Gate 1B.1 routed model manifest checksum drift")
    if (
        config["models"]["middle_validity_filter"]["source_manifest_sha256"]
        != expected_b1_sha
    ):
        raise RuntimeError("Gate 1B.3 middle models do not share one frozen source")
    if (
        file_sha256(b2_path)
        != config["models"]["early_validity_filter"]["source_manifest_sha256"]
    ):
        raise RuntimeError("Gate 1B.2 routed model manifest checksum drift")
    b1 = json.loads(b1_path.read_text(encoding="utf-8"))
    b2 = json.loads(b2_path.read_text(encoding="utf-8"))
    if (
        b1.get("gate") != config["models"]["middle_gain_ranker"]["source_gate"]
        or b1.get("validity_model", {}).get("family") != "ridge"
        or b1.get("gain_model", {}).get("family") != "ridge"
        or int(b1.get("test_evaluations_completed", -1)) != 0
    ):
        raise RuntimeError("Gate 1B.1 routed models violate their frozen contract")
    if (
        b2.get("gate") != config["models"]["early_validity_filter"]["source_gate"]
        or b2.get("validity_model", {}).get("family")
        != "sklearn.ensemble.RandomForestRegressor"
        or b2.get("test_data_accessed") is not False
        or int(b2.get("test_evaluations_completed", -1)) != 0
    ):
        raise RuntimeError("Gate 1B.2 routed model violates its frozen contract")
    return b1, b2


def load_early_validity_model(
    gate1b2_manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> Any:
    import joblib

    artifact = Path(gate1b2_manifest["validity_model"]["artifact"]).resolve()
    expected_sha = config["models"]["early_validity_filter"]["artifact_sha256"]
    if file_sha256(artifact) != expected_sha:
        raise RuntimeError("Gate 1B.2 validity model artifact checksum drift")
    return joblib.load(artifact)


def attach_stage_routed_predictions(
    early_rows: Sequence[Mapping[str, Any]],
    middle_rows: Sequence[Mapping[str, Any]],
    early_validity_model: Any,
    gate1b1_manifest: Mapping[str, Any],
) -> Dict[str, list]:
    early_values = early_validity_model.predict(
        np.vstack([row["validity_features"] for row in early_rows])
    )
    early = [
        {
            **{
                key: value
                for key, value in row.items()
                if not key.endswith("features")
            },
            "predicted_validity": float(early_values[index]),
            "routed_validity_model": STAGE_ROUTER["early"],
        }
        for index, row in enumerate(early_rows)
    ]
    middle_validity = predict_ridge(
        gate1b1_manifest["validity_model"],
        np.vstack([row["validity_features"] for row in middle_rows]),
    )
    middle_gain = predict_ridge(
        gate1b1_manifest["gain_model"],
        np.vstack([row["gain_features"] for row in middle_rows]),
    )
    middle = [
        {
            **{
                key: value
                for key, value in row.items()
                if not key.endswith("features")
            },
            "predicted_validity": float(middle_validity[index]),
            "predicted_gain": float(middle_gain[index]),
            "routed_validity_model": "linear-validity",
            "routed_gain_model": "policy-conditioned-gain",
        }
        for index, row in enumerate(middle_rows)
    ]
    return {"early": early, "middle": middle, "late": []}


def select_routed_rows(
    stage: str,
    rows: Sequence[Mapping[str, Any]],
    budget: int,
    validity_threshold: float,
) -> Sequence[Mapping[str, Any]]:
    if budget < 1:
        raise ValueError("Gate 1B.3 budget must be positive")
    if stage == "late":
        return []
    if len(rows) < budget:
        raise ValueError("Gate 1B.3 budget exceeds available candidates")
    if stage == "early":
        return rank_rows(
            rows, lambda row: (-float(row["predicted_validity"]),)
        )[:budget]
    if stage == "middle":
        return rank_rows(
            rows,
            lambda row: (
                -int(float(row["predicted_validity"]) >= validity_threshold),
                -float(row["predicted_gain"]),
                -float(row["predicted_validity"]),
            ),
        )[:budget]
    raise ValueError("Gate 1B.3 stage must be early, middle, or late")


def feature_contract() -> Dict[str, Any]:
    return {
        "validity_feature_count": len(validity_feature_names()),
        "gain_feature_count": len(gain_feature_names()),
        "policy_score_in_validity": "policy_probability" in validity_feature_names(),
        "policy_score_in_gain": "policy_probability" in gain_feature_names(),
    }
