from __future__ import annotations

import unittest

from reproduction.scicf.gate1.audit_headroom import (
    expected_random_best_gain,
    expected_random_ndcg,
    terminal_valid,
)


class SciCFHeadroomTests(unittest.TestCase):
    def test_expected_random_best_gain_is_exact(self):
        observed = expected_random_best_gain([1.0, 2.0, 3.0], budget=2)
        self.assertAlmostEqual(observed, 8.0 / 3.0)

    def test_expected_random_ndcg_is_one_for_equal_positive_gains(self):
        observed = expected_random_ndcg([0.5, 0.5, 0.5, 0.5], budget=2)
        self.assertAlmostEqual(observed, 1.0)

    def test_expected_random_ndcg_is_zero_without_positive_gain(self):
        observed = expected_random_ndcg([-0.1, 0.0, -0.3], budget=2)
        self.assertEqual(observed, 0.0)

    def test_terminal_validity_does_not_treat_missing_as_valid(self):
        self.assertFalse(terminal_valid({"terminal_scientific_object": None}))
        self.assertFalse(terminal_valid({"terminal_scientific_object": "None"}))
        self.assertTrue(terminal_valid({"terminal_scientific_object": "polymer"}))


if __name__ == "__main__":
    unittest.main()
