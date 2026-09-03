from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from reproduction.scicf.gate1.gate1b1 import (
    evaluate_late,
    gain_feature_names,
    validity_feature_names,
)
from reproduction.scicf.gate1.gate1b2 import (
    FIXED_RANDOM_FOREST_PARAMETERS,
    attach_predictions,
    load_config,
    validate_parent_contracts,
)


class _FixedValidityModel:
    def predict(self, features):
        return np.asarray([0.9 for _ in range(len(features))], dtype=float)


class SciCFGate1B2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.config_path = (
            cls.repo_root
            / "reproduction/configs/scicf-gate1b2-development-v1.json"
        )
        cls.parent_path = (
            cls.repo_root
            / "reproduction/configs/scicf-gate1b1-structure-split-v1.json"
        )
        cls.expansion_path = (
            cls.repo_root
            / "reproduction/configs/scicf-gate1b2-late-dev-collection-v1.json"
        )

    def test_config_freezes_one_nonlinear_model_without_search(self):
        config = load_config(self.config_path)
        self.assertEqual(
            config["models"]["validity_filter"]["fixed_parameters"],
            FIXED_RANDOM_FOREST_PARAMETERS,
        )
        self.assertFalse(config["models"]["hyperparameter_search"])
        self.assertFalse(config["models"]["model_family_search"])

    def test_parent_and_expansion_contracts_are_frozen(self):
        config = load_config(self.config_path)
        parent, expansion = validate_parent_contracts(
            self.repo_root, config, self.parent_path, self.expansion_path
        )
        self.assertEqual(parent["structure_split"], expansion["structure_split"])
        self.assertEqual(expansion["execution"]["allowed_split"], "dev")
        self.assertEqual(expansion["execution"]["allowed_stage"], "late")

    def test_expansion_seeds_do_not_overlap_parent(self):
        parent = json.loads(self.parent_path.read_text(encoding="utf-8"))
        expansion = json.loads(self.expansion_path.read_text(encoding="utf-8"))
        parent_seeds = {
            int(seed)
            for split in ("train", "dev", "test")
            for seed in parent["seeds"][split]
        }
        self.assertFalse(parent_seeds & set(expansion["seeds"]["dev"]))

    def test_predictions_use_nonlinear_validity_and_frozen_ridge_gain(self):
        row = {
            "trajectory_id": "late-a",
            "candidate_id": "candidate-a",
            "timestep": 0,
            "counterfactual_valid_rate": 1.0,
            "gain": 0.2,
            "validity_features": np.zeros(len(validity_feature_names())),
            "gain_features": np.zeros(len(gain_feature_names())),
        }
        gain_model = {
            "family": "ridge",
            "mean": [0.0] * len(gain_feature_names()),
            "scale": [1.0] * len(gain_feature_names()),
            "intercept": 0.25,
            "coefficients": [0.0] * len(gain_feature_names()),
        }
        predicted = attach_predictions(
            {"early": [], "middle": [], "late": [row]},
            _FixedValidityModel(),
            gain_model,
        )
        self.assertEqual(predicted["early"], [])
        self.assertEqual(predicted["late"][0]["predicted_validity"], 0.9)
        self.assertEqual(predicted["late"][0]["predicted_gain"], 0.25)

    def test_late_selection_can_abstain_or_select_up_to_four(self):
        rows = [
            {
                "trajectory_id": "negative",
                "candidate_id": "n{}".format(index),
                "timestep": 0,
                "predicted_validity": 1.0,
                "predicted_gain": -0.2,
                "gain": -0.1,
            }
            for index in range(6)
        ]
        rows.extend(
            {
                "trajectory_id": "positive",
                "candidate_id": "p{}".format(index),
                "timestep": 0,
                "predicted_validity": 1.0,
                "predicted_gain": 0.4 - index * 0.01,
                "gain": 0.2,
            }
            for index in range(6)
        )
        evaluated = evaluate_late(
            rows, budget=4, validity_threshold=0.5, gain_threshold=0.0
        )
        by_id = {item["trajectory_id"]: item for item in evaluated}
        self.assertTrue(by_id["negative"]["abstained"])
        self.assertEqual(by_id["negative"]["selected_count"], 0)
        self.assertEqual(by_id["positive"]["selected_count"], 4)

    def test_sealed_test_is_not_authorized_by_development_config(self):
        config = load_config(self.config_path)
        authorization = config["authorization"]
        self.assertFalse(authorization["test_collection_authorized"])
        self.assertFalse(authorization["test_evaluation_authorized"])
        self.assertTrue(
            authorization["separate_user_authorization_required_for_test"]
        )


if __name__ == "__main__":
    unittest.main()
