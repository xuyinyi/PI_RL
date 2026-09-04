import unittest

import numpy as np

from reproduction.p2.contracts import (
    PPO,
    ContractViolation,
    CreditRequest,
    FrozenPolicyHandle,
    RolloutBatch,
)
from reproduction.p2.gae import GAECreditEstimator, compute_gae


class GAEProviderTests(unittest.TestCase):
    def test_terminal_and_truncation_boundaries(self):
        advantages, returns = compute_gae(
            rewards=[0.0, 1.0, 0.0, 0.0],
            values=[0.5, 0.25, 1.0, 2.0],
            next_values=[0.25, 0.0, 2.0, 3.0],
            terminated=[False, True, False, False],
            truncated=[False, False, False, True],
            episode_ids=[0, 0, 1, 1],
            gamma=0.9,
            gae_lambda=0.8,
        )
        np.testing.assert_allclose(advantages, [0.265, 0.75, 1.304, 0.7])
        np.testing.assert_allclose(returns, [0.765, 1.0, 2.304, 2.7])

    def test_gae_never_crosses_episode_identifier(self):
        advantages, _ = compute_gae(
            rewards=[0.0, 1.0],
            values=[0.0, 0.0],
            next_values=[0.0, 0.0],
            terminated=[False, True],
            truncated=[False, False],
            episode_ids=[0, 1],
            gamma=1.0,
            gae_lambda=1.0,
        )
        np.testing.assert_array_equal(advantages, [0.0, 1.0])

    def test_ppo_provider_returns_exact_gae_without_queries(self):
        frozen = FrozenPolicyHandle(2, "policy", "env", "task")
        rollout = RolloutBatch(
            batch_id="batch",
            environment_id="env",
            task_contract_id="task",
            budget_contract_id="budget",
            frozen_policy=frozen,
            transition_ids=("t0", "t1"),
            transitions=(object(), object()),
            gae_advantages=np.asarray([0.25, -0.5]),
            critic_returns=np.asarray([1.0, 0.0]),
        )
        provider = GAECreditEstimator()
        estimate = provider.estimate(CreditRequest(PPO, rollout, 0, 0, 17))
        np.testing.assert_array_equal(estimate.actor_advantages, rollout.gae_advantages)
        self.assertEqual(estimate.evaluator_delta.requested_calls, 0)
        self.assertIsNone(estimate.pending_labels)
        self.assertEqual(provider.state_dict()["model_version"], 0)
        with self.assertRaisesRegex(ContractViolation, "checkpoint identity"):
            provider.load_state_dict({"schema_version": 1})

    def test_rejects_misaligned_inputs_and_double_terminal_flags(self):
        common = dict(
            rewards=[0.0],
            values=[0.0],
            next_values=[0.0],
            terminated=[False],
            truncated=[False],
            episode_ids=[0],
            gamma=1.0,
            gae_lambda=0.95,
        )
        with self.assertRaisesRegex(ValueError, "align"):
            compute_gae(**dict(common, values=[0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "both"):
            compute_gae(
                **dict(common, terminated=[True], truncated=[True])
            )


if __name__ == "__main__":
    unittest.main()
