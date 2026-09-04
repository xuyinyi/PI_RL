"""Environment-return GAE and the evaluator-free PPO credit provider."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from reproduction.p2.contracts import (
    CREDIT_ESTIMATOR_CONTRACT_ID,
    PPO,
    ContractViolation,
    CreditEstimate,
    CreditRequest,
    EvaluatorLedgerDelta,
    PPOUpdateReceipt,
    PendingLabelBatch,
    validate_credit_estimate,
)


GAE_PROVIDER_STATE_SCHEMA_VERSION = 1


def _one_dimensional_finite(values: Any, name: str, dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("%s must have non-empty shape [T]." % name)
    if not bool(np.isfinite(array).all()):
        raise ValueError("%s must contain only finite values." % name)
    return array


def compute_gae(
    *,
    rewards: Any,
    values: Any,
    next_values: Any,
    terminated: Any,
    truncated: Any,
    episode_ids: Any,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE without propagating recurrences across episode boundaries.

    A true terminal state has zero bootstrap value. A time-limit truncation may
    bootstrap from ``next_values`` but still cuts the backwards GAE recurrence.
    The returned critic target is always ``advantage + old value`` and therefore
    remains derived solely from the real environment reward path.
    """

    rewards_array = _one_dimensional_finite(rewards, "rewards", np.float64)
    values_array = _one_dimensional_finite(values, "values", np.float64)
    next_values_array = _one_dimensional_finite(
        next_values, "next_values", np.float64
    )
    terminated_array = np.asarray(terminated, dtype=bool)
    truncated_array = np.asarray(truncated, dtype=bool)
    episode_array = np.asarray(episode_ids)
    expected_shape = rewards_array.shape
    for name, array in (
        ("values", values_array),
        ("next_values", next_values_array),
        ("terminated", terminated_array),
        ("truncated", truncated_array),
        ("episode_ids", episode_array),
    ):
        if array.shape != expected_shape:
            raise ValueError("%s must align with rewards." % name)
    if not np.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    if not np.isfinite(float(gae_lambda)) or not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("gae_lambda must lie in [0, 1].")
    if bool(np.logical_and(terminated_array, truncated_array).any()):
        raise ValueError("A transition cannot be both terminated and truncated.")

    bootstrap = np.logical_not(terminated_array).astype(np.float64)
    deltas = (
        rewards_array
        + float(gamma) * bootstrap * next_values_array
        - values_array
    )
    advantages = np.zeros_like(deltas)
    running = 0.0
    for index in range(deltas.size - 1, -1, -1):
        boundary = bool(terminated_array[index] or truncated_array[index])
        same_episode_next = bool(
            index + 1 < deltas.size
            and episode_array[index + 1] == episode_array[index]
        )
        continue_recurrence = (not boundary) and same_episode_next
        running = float(deltas[index]) + (
            float(gamma) * float(gae_lambda) * running
            if continue_recurrence
            else 0.0
        )
        advantages[index] = running
    returns = advantages + values_array
    return advantages.astype(np.float64), returns.astype(np.float64)


class GAECreditEstimator:
    """PPO credit provider: return the common engine's GAE exactly.

    It has no evaluator, replay buffer or trainable state. This deliberately
    makes a PPO iteration incapable of issuing a counterfactual evaluator call
    or staging labels through the credit seam.
    """

    method = PPO
    contract_id = CREDIT_ESTIMATOR_CONTRACT_ID
    model_version = 0

    def estimate(self, request: CreditRequest) -> CreditEstimate:
        if request.method != PPO:
            raise ContractViolation("GAE credit is valid only for PPO.")
        estimate = CreditEstimate(
            method=PPO,
            source_batch_id=request.rollout.batch_id,
            source_policy_version=request.rollout.frozen_policy.policy_version,
            actor_advantages=request.rollout.gae_advantages,
            credit_model_version_before=self.model_version,
            credit_model_version_after=self.model_version,
            evaluator_delta=EvaluatorLedgerDelta(),
            pending_labels=None,
            same_iteration_labels_committed=False,
            diagnostics={
                "provider": "environment_return_gae",
                "evaluator_free": True,
            },
        )
        validate_credit_estimate(request, estimate)
        return estimate

    def commit_after_update(
        self, pending: PendingLabelBatch, receipt: PPOUpdateReceipt
    ) -> Mapping[str, Any]:
        del pending, receipt
        raise ContractViolation("PPO GAE credit cannot commit pending labels.")

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": GAE_PROVIDER_STATE_SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "method": self.method,
            "model_version": self.model_version,
            "evaluator_free": True,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        if dict(state) != expected:
            raise ContractViolation("GAE credit checkpoint identity mismatch.")
