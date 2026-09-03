from __future__ import annotations

import math
import unittest

import numpy as np

from reproduction.scicf.gate1.audit_learnability import (
    binary_auc,
    build_loto_operators,
    decide_gate1b,
    feature_names,
    molecule_descriptors,
    predict_with_operators,
    spearman_rho,
)


class SciCFLearnabilityTests(unittest.TestCase):
    def test_descriptor_protocol_is_fixed_and_deterministic(self):
        first = molecule_descriptors("[1*]C(=O)C[8*]")
        second = molecule_descriptors("[1*]C(=O)C[8*]")
        self.assertEqual(len(feature_names()), 81)
        self.assertEqual(first.shape, second.shape)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(np.isfinite(first)))

    def test_loto_ridge_excludes_the_held_trajectory(self):
        features = np.asarray(
            [[0.0, 1.0], [0.1, 1.0], [1.0, 0.0], [1.1, 0.0], [2.0, 1.0], [2.1, 1.0]]
        )
        groups = ["a", "a", "b", "b", "c", "c"]
        targets = np.asarray([0.0, 0.1, 1.0, 0.9, 2.0, 2.1])
        operators, receipts = build_loto_operators(features, groups, alpha=1.0)
        predictions = predict_with_operators(operators, targets)
        self.assertEqual(predictions.shape, targets.shape)
        self.assertTrue(np.all(np.isfinite(predictions)))
        for receipt in receipts:
            self.assertNotIn(receipt["held_group"], receipt["train_groups"])
            self.assertEqual(receipt["test_candidates"], 2)

    def test_rank_metrics_handle_ties(self):
        self.assertEqual(binary_auc([False, True], [0.0, 1.0]), 1.0)
        self.assertEqual(binary_auc([False, True], [1.0, 0.0]), 0.0)
        self.assertEqual(binary_auc([False, True], [1.0, 1.0]), 0.5)
        self.assertAlmostEqual(spearman_rho([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]), 1.0)

    def test_late_diagnostic_cannot_change_gate_decision(self):
        passed = {"passed": True}
        failed = {"passed": False}
        decision = decide_gate1b(passed, passed, passed)
        self.assertTrue(decision["gate1b_passed"])
        self.assertTrue(decision["gate1c_eligible"])
        self.assertFalse(decision["llm_gate1c_authorized"])
        decision = decide_gate1b(passed, failed, passed)
        self.assertFalse(decision["gate1b_passed"])
        self.assertFalse(decision["gate1c_eligible"])


if __name__ == "__main__":
    unittest.main()
