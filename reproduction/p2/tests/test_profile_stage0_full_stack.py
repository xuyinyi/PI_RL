import json
import unittest
from pathlib import Path

from reproduction.p2.scripts.profile_stage0_full_stack import (
    classify_profile,
    ledger_delta,
    load_profile_config,
)


ROOT = Path(__file__).resolve().parents[3]


def _phase(backend_calls, unique_calls, cache_hits):
    return {
        "transition_count": 100,
        "successful_terminal_count": 10,
        "termination_reasons": {"success": 10},
        "no_product_count": 0,
        "trajectory_digest": "trajectory",
        "observation_digest": "observation",
        "terminal_evaluations_digest": "evaluations",
        "ledger_delta": {
            "requested_calls": 20,
            "unique_calls": unique_calls,
            "backend_calls": backend_calls,
            "cache_hits": cache_hits,
            "invalid_results": 0,
            "requested_by_source": {
                "environment_regression": 10,
                "evaluation": 10,
            },
            "unique_by_source": {},
            "backend_by_source": {},
        },
    }


class FullStackProfileTests(unittest.TestCase):
    def test_frozen_full_stack_profile_config(self):
        config = load_profile_config(
            ROOT / "reproduction/p2/configs/stage0_full_stack_profile_v1.json"
        )
        self.assertEqual(config["transition_count"], 100)
        self.assertEqual(config["minimum_successful_terminals"], 10)
        self.assertEqual(config["expected_observation_dimension"], 1246)
        self.assertTrue(config["duplicate_each_success_for_cache_measurement"])

    def test_ledger_delta_preserves_global_and_source_counts(self):
        before = {
            "requested_calls": 2,
            "unique_calls": 1,
            "backend_calls": 1,
            "cache_hits": 1,
            "invalid_results": 0,
            "requested_by_source": {"a": 2},
            "unique_by_source": {"a": 1},
            "backend_by_source": {"a": 1},
        }
        after = {
            "requested_calls": 5,
            "unique_calls": 2,
            "backend_calls": 2,
            "cache_hits": 3,
            "invalid_results": 0,
            "requested_by_source": {"a": 3, "b": 2},
            "unique_by_source": {"a": 1, "b": 1},
            "backend_by_source": {"a": 1, "b": 1},
        }
        self.assertEqual(
            ledger_delta(before, after),
            {
                "requested_calls": 3,
                "unique_calls": 1,
                "backend_calls": 1,
                "cache_hits": 2,
                "invalid_results": 0,
                "requested_by_source": {"a": 1, "b": 2},
                "unique_by_source": {"b": 1},
                "backend_by_source": {"b": 1},
            },
        )

    def test_profile_classification_passes_without_performance_claim(self):
        config = json.loads(
            (
                ROOT
                / "reproduction/p2/configs/stage0_full_stack_profile_v1.json"
            ).read_text()
        )
        result = classify_profile(
            _phase(backend_calls=8, unique_calls=8, cache_hits=12),
            _phase(backend_calls=0, unique_calls=0, cache_hits=20),
            {
                "profile_git_dirty": False,
                "stage0_source_changed_from_accepted": False,
            },
            {"status": "passed"},
            True,
            config,
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            result["performance_admission"],
            "not_defined_characterization_only",
        )
        self.assertFalse(result["parallel_scaling_profiled"])

    def test_profile_classification_fails_on_cache_or_determinism_break(self):
        config = json.loads(
            (
                ROOT
                / "reproduction/p2/configs/stage0_full_stack_profile_v1.json"
            ).read_text()
        )
        replay = _phase(backend_calls=1, unique_calls=0, cache_hits=19)
        replay["trajectory_digest"] = "changed"
        result = classify_profile(
            _phase(backend_calls=8, unique_calls=8, cache_hits=12),
            replay,
            {
                "profile_git_dirty": False,
                "stage0_source_changed_from_accepted": False,
            },
            {"status": "passed"},
            True,
            config,
        )
        self.assertEqual(result["status"], "failed_functional_gate")
        self.assertIn("cache_replay_reached_backend", result["functional_failures"])
        self.assertIn(
            "nondeterministic_trajectory_digest", result["functional_failures"]
        )


if __name__ == "__main__":
    unittest.main()
