import copy
import json
import tempfile
import unittest
from pathlib import Path

from reproduction.p4.protocol import (
    REQUIRED_METRICS,
    evaluate_acceptance,
    load_protocol,
    resolved_run,
)


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = (
    ROOT / "reproduction/p4/configs/native_ppo_baseline_protocol_v1.json"
)


def passing_report(protocol):
    seed, spec = resolved_run(protocol, "preflight")
    metrics = {
        "samples": spec["evaluation"]["samples"],
        "valid_samples": 20,
        **{name: 0.5 for name in REQUIRED_METRICS},
    }
    evaluations = []
    for target in spec["checkpoint_requested_calls"]:
        evaluations.append(
            {
                "target_requested_calls": target,
                "metrics": metrics,
                "evaluator_ledger": {
                    "requested_by_source": {"evaluation": 20},
                    "invalid_results": 0,
                },
            }
        )
    iterations = []
    for _index in range(4):
        iterations.append(
            {
                "transition_count": protocol["ppo"]["rollout_steps"],
                "credit_evaluator_delta": {"requested_calls": 0},
                "actor_advantages_sha256": "gae",
                "gae_sha256": "gae",
                "update_metrics": {"actor_loss": 0.1, "value_loss": 0.2},
            }
        )
    return {
        "mode": "preflight",
        "seed": seed,
        "source": {"dirty": False},
        "slurm": {"job_id": "1"},
        "host": "yanlih100n1",
        "accepted_binding": {
            "stage0_source_changed_from_accepted": False,
            "accepted_git_commit": protocol["accepted_binding"]["accepted_p1_commit"],
        },
        "stack_specification": {
            "environment_id": protocol["accepted_binding"]["environment_id"],
            "config": protocol["task"]["configuration"],
            "observation_dimension": protocol["accepted_binding"][
                "observation_dimension"
            ],
        },
        "training_evaluator_ledger": {
            "requested_calls": 400,
            "requested_by_source": {"ppo/on_policy": 400},
            "invalid_results": 0,
            "evaluator_version": protocol["accepted_binding"]["evaluator_version"],
            "objective_contract": protocol["accepted_binding"]["objective_contract"],
        },
        "iterations": iterations,
        "evaluations": evaluations,
        "initial_policy_sha256": "before",
        "final_policy_sha256": "after",
        "final_policy_version": len(iterations),
        "checkpoint_roundtrip": {
            "policy": True,
            "version": True,
            "ledger": True,
        },
        "sealed_test_accessed": False,
        "external_api_invoked": False,
    }


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_validates_and_separates_seeds(self):
        protocol = load_protocol(PROTOCOL_PATH)
        preflight_seed, preflight = resolved_run(protocol, "preflight")
        self.assertEqual(preflight_seed, 20260910)
        self.assertEqual(preflight["maximum_training_requested_calls"], 512)
        for seed in protocol["formal"]["seeds"]:
            observed, formal = resolved_run(protocol, "formal", seed)
            self.assertEqual(observed, seed)
            self.assertEqual(formal["maximum_training_requested_calls"], 10000)
        with self.assertRaisesRegex(ValueError, "absent"):
            resolved_run(protocol, "formal", 1)

    def test_acceptance_is_fail_closed(self):
        protocol = load_protocol(PROTOCOL_PATH)
        report = passing_report(protocol)
        checks, accepted = evaluate_acceptance(report, protocol)
        self.assertTrue(accepted)
        self.assertTrue(all(checks.values()))
        contaminated = copy.deepcopy(report)
        contaminated["sealed_test_accessed"] = True
        checks, accepted = evaluate_acceptance(contaminated, protocol)
        self.assertFalse(accepted)
        self.assertFalse(checks["sealed_test_not_accessed"])

    def test_protocol_rejects_in_place_retuning(self):
        payload = json.loads(PROTOCOL_PATH.read_text())
        payload["ppo"]["learning_rate"] = 1e-3
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            changed.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "PPO configuration changed"):
                load_protocol(changed)


if __name__ == "__main__":
    unittest.main()
