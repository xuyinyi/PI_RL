import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from reproduction.p2.contracts import (
    CREDIT_ESTIMATOR_CONTRACT_ID,
    EVALUATION,
    MCC_PPO,
    METHOD_EVALUATOR_SOURCES,
    POLICY_CC,
    POLICY_CC_FACTUAL,
    PPO,
    PPO_ENGINE_CONTRACT_ID,
    ContractViolation,
    CreditEstimate,
    CreditRequest,
    EvaluatorLedgerDelta,
    FrozenPolicyHandle,
    MethodRunContract,
    PPOUpdateReceipt,
    PendingLabelBatch,
    RolloutBatch,
    validate_credit_estimate,
    validate_method_contracts,
    validate_pending_label_commit,
    validate_update_receipt,
)


ROOT = Path(__file__).resolve().parents[3]


def _rollout():
    frozen = FrozenPolicyHandle(
        policy_version=4,
        state_sha256="policy-sha",
        environment_id="env",
        task_contract_id="task",
    )
    return RolloutBatch(
        batch_id="batch",
        environment_id="env",
        task_contract_id="task",
        budget_contract_id="budget",
        frozen_policy=frozen,
        transition_ids=("t0", "t1"),
        transitions=(object(), object()),
        gae_advantages=np.asarray([1.0, -1.0]),
        critic_returns=np.asarray([0.4, 0.2]),
    )


def _method_contract(method):
    return MethodRunContract(
        method=method,
        environment_id="env",
        task_contract_id="task",
        budget_contract_id="budget",
        evaluator_version="evaluator",
        objective_contract="objective",
        ppo_engine_contract_id=PPO_ENGINE_CONTRACT_ID,
        ppo_hyperparameters_sha256="ppo-config",
        credit_estimator_contract_id=CREDIT_ESTIMATOR_CONTRACT_ID,
        allowed_evaluator_sources=tuple(METHOD_EVALUATOR_SOURCES[method]),
    )


class P2ContractTests(unittest.TestCase):
    def test_frozen_test_matrix_has_unique_complete_ids(self):
        payload = json.loads(
            (
                ROOT
                / "reproduction/p2/configs/p2_contract_test_matrix_v1.json"
            ).read_text()
        )
        ids = [row["id"] for row in payload["tests"]]
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            payload["contract_ids"],
            {
                "credit_estimator": CREDIT_ESTIMATOR_CONTRACT_ID,
                "ppo_engine": PPO_ENGINE_CONTRACT_ID,
            },
        )
        self.assertEqual(
            ids,
            [
                "P2-C01",
                "P2-C02",
                "P2-C03",
                "P2-C04",
                "P2-C05",
                "P2-C06",
                "P2-C07",
                "P2-I01",
                "P2-I02",
                "P2-I03",
                "P2-I04",
                "P2-I05",
                "P2-I06",
            ],
        )
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            {row["level"] for row in payload["tests"]},
            {
                "contract_validator",
                "integrated_pending",
                "integrated_engine_passed",
                "integrated_ppo_passed",
            },
        )

    def test_three_methods_share_everything_outside_credit_seam(self):
        contracts = [
            _method_contract(PPO),
            _method_contract(POLICY_CC),
            _method_contract(MCC_PPO),
        ]
        validate_method_contracts(contracts)
        with self.assertRaisesRegex(ContractViolation, "outside the credit seam"):
            validate_method_contracts(
                [
                    contracts[0],
                    replace(contracts[1], task_contract_id="changed"),
                    contracts[2],
                ]
            )

    def test_ppo_credit_is_exact_gae_and_never_queries_or_stages_labels(self):
        rollout = _rollout()
        request = CreditRequest(PPO, rollout, 0, 0, 11)
        estimate = CreditEstimate(
            method=PPO,
            source_batch_id=rollout.batch_id,
            source_policy_version=4,
            actor_advantages=rollout.gae_advantages,
            credit_model_version_before=0,
            credit_model_version_after=0,
        )
        validate_credit_estimate(request, estimate)
        with self.assertRaisesRegex(ContractViolation, "must equal engine GAE"):
            validate_credit_estimate(
                request,
                replace(estimate, actor_advantages=np.asarray([1.0, -0.5])),
            )

    def test_credit_estimate_fails_on_model_leak_budget_or_source_violation(self):
        rollout = _rollout()
        request = CreditRequest(POLICY_CC, rollout, 3, 2, 12)
        ledger = EvaluatorLedgerDelta(
            requested_calls=2,
            unique_calls=2,
            backend_calls=2,
            cache_hits=0,
            requested_by_source={POLICY_CC_FACTUAL: 2},
        )
        estimate = CreditEstimate(
            method=POLICY_CC,
            source_batch_id="batch",
            source_policy_version=4,
            actor_advantages=np.asarray([0.2, 0.3]),
            credit_model_version_before=3,
            credit_model_version_after=3,
            evaluator_delta=ledger,
        )
        validate_credit_estimate(request, estimate)
        with self.assertRaisesRegex(ContractViolation, "changed before"):
            validate_credit_estimate(
                request, replace(estimate, credit_model_version_after=4)
            )
        with self.assertRaisesRegex(ContractViolation, "exceeded"):
            validate_credit_estimate(
                replace(request, reserved_query_requested_calls=1), estimate
            )
        wrong_source = replace(
            estimate,
            evaluator_delta=replace(ledger, requested_by_source={EVALUATION: 2}),
        )
        with self.assertRaisesRegex(
            ContractViolation, "unauthorized evaluator source"
        ):
            validate_credit_estimate(request, wrong_source)

    def test_update_receipt_binds_actor_credit_and_critic_environment_returns(self):
        rollout = _rollout()
        estimate = CreditEstimate(
            method=POLICY_CC,
            source_batch_id="batch",
            source_policy_version=4,
            actor_advantages=np.asarray([0.2, 0.3]),
            credit_model_version_before=3,
            credit_model_version_after=3,
        )
        receipt = PPOUpdateReceipt(
            engine_contract_id=PPO_ENGINE_CONTRACT_ID,
            source_batch_id="batch",
            policy_version_before=4,
            policy_version_after=5,
            actor_advantages_sha256=estimate.actor_advantages_sha256,
            critic_returns_sha256=rollout.critic_returns_sha256,
            optimizer_step_completed=True,
        )
        validate_update_receipt(rollout, estimate, receipt)
        with self.assertRaisesRegex(ContractViolation, "Critic targets"):
            validate_update_receipt(
                rollout,
                estimate,
                replace(receipt, critic_returns_sha256="changed"),
            )

    def test_pending_labels_commit_only_after_policy_advance(self):
        pending = PendingLabelBatch("labels", "batch", 4, 2, "payload")
        receipt = PPOUpdateReceipt(
            engine_contract_id=PPO_ENGINE_CONTRACT_ID,
            source_batch_id="batch",
            policy_version_before=4,
            policy_version_after=5,
            actor_advantages_sha256="actor",
            critic_returns_sha256="critic",
            optimizer_step_completed=True,
        )
        validate_pending_label_commit(pending, receipt)
        with self.assertRaisesRegex(ContractViolation, "optimizer completion"):
            validate_pending_label_commit(
                pending, replace(receipt, optimizer_step_completed=False)
            )
        with self.assertRaisesRegex(ContractViolation, "source iteration"):
            validate_pending_label_commit(
                pending, replace(receipt, policy_version_after=4)
            )


if __name__ == "__main__":
    unittest.main()
