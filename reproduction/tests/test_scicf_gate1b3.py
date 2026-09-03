from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduction.scicf.gate1.gate1b1 import (
    gain_feature_names,
    validity_feature_names,
)
from reproduction.scicf.gate1.gate1b3 import (
    STAGE_ROUTER,
    attach_stage_routed_predictions,
    feature_contract,
    load_config,
    load_excluded_structure_keys,
    select_routed_rows,
    validate_collection_authorization,
    validate_collection_config,
)


class _EarlyModel:
    def predict(self, features):
        return np.asarray([0.8 - 0.1 * index for index in range(len(features))])


def _ridge(feature_count, intercept):
    return {
        "family": "ridge",
        "mean": [0.0] * feature_count,
        "scale": [1.0] * feature_count,
        "intercept": float(intercept),
        "coefficients": [0.0] * feature_count,
    }


class SciCFGate1B3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.config_path = (
            cls.repo_root
            / "reproduction/configs/scicf-gate1b3-stage-routed-development-v1.json"
        )
        cls.collection_path = (
            cls.repo_root
            / "reproduction/configs/scicf-gate1b3-unseen-dev-collection-v1.json"
        )

    def test_stage_router_is_frozen_and_fail_closed(self):
        config = load_config(self.config_path)
        self.assertEqual(STAGE_ROUTER["early"], "nonlinear-validity")
        self.assertEqual(
            STAGE_ROUTER["middle"],
            "linear-validity-then-policy-conditioned-gain",
        )
        self.assertEqual(STAGE_ROUTER["late"], "abstain-while-uncalibrated")
        self.assertFalse(config["entry_rule"]["late_is_gating"])
        self.assertFalse(config["authorization"]["fresh_dev_collection_authorized"])
        self.assertFalse(config["authorization"]["test_collection_authorized"])
        self.assertFalse(config["authorization"]["ppo_integration_authorized"])

    def test_collection_is_dev_early_middle_only_and_checksum_frozen(self):
        config = load_config(self.config_path)
        collection = validate_collection_config(
            self.repo_root, config, self.collection_path
        )
        self.assertEqual(collection["execution"]["allowed_split"], "dev")
        self.assertEqual(
            collection["execution"]["allowed_stages"], ["early", "middle"]
        )
        self.assertTrue(collection["structure_exclusion"]["required"])

    def test_collection_authorization_is_narrow_and_versioned(self):
        collection = json.loads(self.collection_path.read_text(encoding="utf-8"))
        receipt_path = (
            self.repo_root
            / "reproduction/results/scicf-gate1b3-implementation-20260903"
            / "fresh-dev-collection-authorization-v1.json"
        )
        receipt, seal = validate_collection_authorization(
            receipt_path, collection
        )
        self.assertTrue(
            receipt["authorization"]["fresh_dev_collection_authorized"]
        )
        self.assertFalse(
            receipt["authorization"]["fresh_dev_evaluation_authorized"]
        )
        self.assertFalse(receipt["authorization"]["test_collection_authorized"])
        self.assertEqual(seal["action"], "fresh-dev-collection")

    def test_structure_exclusion_requires_exact_sealed_manifest(self):
        payload = {
            "gate": "scicf-gate1b2-nonlinear-validity-development-v1",
            "combined_dev_structure_keys": ["diamine|N", "dianhydride|C"],
            "test_data_accessed": False,
            "test_evaluations_completed": 0,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_bytes(encoded)
            collection = json.loads(self.collection_path.read_text(encoding="utf-8"))
            collection = copy.deepcopy(collection)
            collection["structure_exclusion"]["manifest_sha256"] = hashlib.sha256(
                encoded
            ).hexdigest()
            keys, seal = load_excluded_structure_keys(path, collection)
        self.assertEqual(keys, {"diamine|N", "dianhydride|C"})
        self.assertEqual(seal["excluded_key_count"], 2)

    def test_routed_predictions_use_different_validity_models_by_stage(self):
        early_row = {
            "trajectory_id": "early-a",
            "candidate_id": "early-candidate",
            "timestep": 0,
            "counterfactual_valid_rate": 1.0,
            "gain": 0.0,
            "validity_features": np.zeros(len(validity_feature_names())),
            "gain_features": np.zeros(len(gain_feature_names())),
        }
        middle_row = {
            **early_row,
            "trajectory_id": "middle-a",
            "candidate_id": "middle-candidate",
            "gain": 0.2,
        }
        b1_manifest = {
            "validity_model": _ridge(len(validity_feature_names()), 0.6),
            "gain_model": _ridge(len(gain_feature_names()), 0.25),
        }
        predicted = attach_stage_routed_predictions(
            [early_row], [middle_row], _EarlyModel(), b1_manifest
        )
        self.assertEqual(predicted["early"][0]["predicted_validity"], 0.8)
        self.assertEqual(predicted["middle"][0]["predicted_validity"], 0.6)
        self.assertEqual(predicted["middle"][0]["predicted_gain"], 0.25)

    def test_late_defaults_to_abstention(self):
        selected = select_routed_rows(
            "late",
            [{"candidate_id": "unused"}],
            budget=4,
            validity_threshold=0.5,
        )
        self.assertEqual(selected, [])

    def test_feature_contract_keeps_policy_out_of_validity(self):
        contract = feature_contract()
        self.assertFalse(contract["policy_score_in_validity"])
        self.assertTrue(contract["policy_score_in_gain"])


if __name__ == "__main__":
    unittest.main()
