from __future__ import annotations

import unittest

from reproduction.scicf.acquisition.base import AcquisitionCandidate
from reproduction.scicf.core.records import Intervention
from reproduction.scicf.gate1.gate1b1 import (
    build_cross_timestep_pool,
    calibrate_late_threshold,
    common_feature_names,
    gain_feature_names,
    raw_cross_timestep_fraction,
    structure_key,
    structure_split,
    validity_feature_names,
)


class SciCFGate1B1Tests(unittest.TestCase):
    def _config(self):
        return {
            "structure_split": {
                "salt": "test-salt",
                "train_upper_exclusive": 60,
                "dev_upper_exclusive": 80,
                "modulus": 100,
            }
        }

    def _candidate(self, timestep, alternative, score):
        intervention = Intervention(
            intervention_id="trajectory:t{:03d}:dianhydride:0000-{:04d}".format(
                timestep, alternative
            ),
            trajectory_id="trajectory",
            timestep=timestep,
            component="dianhydride",
            factual_action=(0, 0),
            alternative_action=(alternative, 0),
            factual_component_value=0,
            alternative_component_value=alternative,
            alternative_structure="C" * (alternative + 1),
            metadata={"factual_structure": "C"},
        )
        return AcquisitionCandidate(
            intervention=intervention,
            policy_score=float(score),
            structural_score=float(score),
            heuristic_score=float(score),
        )

    def test_structure_split_is_deterministic_and_exclusive(self):
        config = self._config()
        first = structure_split("diamine", "c1ccccc1N", config)
        second = structure_split("diamine", "Nc1ccccc1", config)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "dev", "test"})
        self.assertEqual(
            structure_key("diamine", "c1ccccc1N"),
            structure_key("diamine", "Nc1ccccc1"),
        )

    def test_cross_timestep_pool_covers_every_observed_timestep(self):
        candidates = []
        for timestep in (0, 1):
            for offset in range(1, 8):
                candidates.append(
                    self._candidate(timestep, timestep * 10 + offset, offset)
                )
        pool = build_cross_timestep_pool(
            candidates,
            {"policy_near": 2, "random_legal": 2, "structural": 2},
            requested_size=6,
            seed=17,
        )
        self.assertEqual(len(pool.candidates), 6)
        self.assertEqual(
            {candidate.intervention.timestep for candidate in pool.candidates}, {0, 1}
        )

    def test_feature_contract_keeps_policy_only_in_gain_ranker(self):
        self.assertEqual(len(common_feature_names()), 87)
        self.assertEqual(validity_feature_names(), common_feature_names())
        self.assertEqual(len(gain_feature_names()), 89)
        self.assertNotIn("policy_probability", validity_feature_names())
        self.assertIn("policy_probability", gain_feature_names())

    def test_late_threshold_calibration_prefers_balanced_separation(self):
        rows = [
            {"trajectory_id": "negative-a", "predicted_validity": 1.0, "predicted_gain": -0.2, "gain": -0.1},
            {"trajectory_id": "negative-b", "predicted_validity": 1.0, "predicted_gain": -0.1, "gain": 0.0},
            {"trajectory_id": "positive-a", "predicted_validity": 1.0, "predicted_gain": 0.2, "gain": 0.1},
            {"trajectory_id": "positive-b", "predicted_validity": 1.0, "predicted_gain": 0.3, "gain": 0.2},
        ]
        calibration = calibrate_late_threshold(rows, validity_threshold=0.5)
        self.assertEqual(calibration["status"], "complete")
        self.assertEqual(calibration["chosen"]["balanced_accuracy"], 1.0)
        self.assertGreaterEqual(calibration["chosen"]["threshold"], -0.1)
        self.assertLess(calibration["chosen"]["threshold"], 0.2)

    def test_raw_cross_timestep_fraction_keeps_uncalibrated_late_in_denominator(self):
        rows = {
            "early": [
                {"trajectory_id": "early-a", "timestep": 0},
                {"trajectory_id": "early-a", "timestep": 1},
            ],
            "middle": [{"trajectory_id": "middle-a", "timestep": 0}],
            "late": [{"trajectory_id": "late-a", "timestep": 0}],
        }
        self.assertAlmostEqual(raw_cross_timestep_fraction(rows), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
