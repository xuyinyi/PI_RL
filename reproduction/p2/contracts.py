"""Executable P2 interface contracts shared by PPO, Policy-CC and MCC-PPO.

This module freezes the seam between one PPO engine and method-specific credit
estimators.  It deliberately contains no optimizer or chemistry implementation.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from RL_PPO.envs.sources import (
    EVALUATION,
    MCC_PPO_COUNTERFACTUAL,
    MCC_PPO_FACTUAL,
    MCC_PPO_ON_POLICY,
    POLICY_CC_COUNTERFACTUAL,
    POLICY_CC_FACTUAL,
    POLICY_CC_ON_POLICY,
    PPO_ON_POLICY,
)


P2_CONTRACT_SCHEMA_VERSION = 1
PPO_ENGINE_CONTRACT_ID = "dapigen-p2-single-ppo-engine-v1"
CREDIT_ESTIMATOR_CONTRACT_ID = "dapigen-p2-credit-estimator-v1"

PPO = "ppo"
POLICY_CC = "policy_cc"
MCC_PPO = "mcc_ppo"
METHODS = (PPO, POLICY_CC, MCC_PPO)
SCICF_PPO = "scicf_ppo"
SCICF_PPO_ON_POLICY = "scicf_ppo/on_policy"
SCICF_PPO_FACTUAL = "scicf_ppo/factual"
SCICF_PPO_COUNTERFACTUAL = "scicf_ppo/counterfactual"
EXPERIMENTAL_METHODS = (SCICF_PPO,)
SUPPORTED_METHODS = METHODS + EXPERIMENTAL_METHODS

METHOD_EVALUATOR_SOURCES = {
    PPO: frozenset((PPO_ON_POLICY, EVALUATION)),
    POLICY_CC: frozenset(
        (
            POLICY_CC_ON_POLICY,
            POLICY_CC_FACTUAL,
            POLICY_CC_COUNTERFACTUAL,
            EVALUATION,
        )
    ),
    MCC_PPO: frozenset(
        (
            MCC_PPO_ON_POLICY,
            MCC_PPO_FACTUAL,
            MCC_PPO_COUNTERFACTUAL,
            EVALUATION,
        )
    ),
    SCICF_PPO: frozenset(
        (
            SCICF_PPO_ON_POLICY,
            SCICF_PPO_FACTUAL,
            SCICF_PPO_COUNTERFACTUAL,
            EVALUATION,
        )
    ),
}

METHOD_QUERY_SOURCES = {
    PPO: frozenset(),
    POLICY_CC: frozenset((POLICY_CC_FACTUAL, POLICY_CC_COUNTERFACTUAL)),
    MCC_PPO: frozenset((MCC_PPO_FACTUAL, MCC_PPO_COUNTERFACTUAL)),
    SCICF_PPO: frozenset((SCICF_PPO_FACTUAL, SCICF_PPO_COUNTERFACTUAL)),
}

METHOD_ON_POLICY_SOURCE = {
    PPO: PPO_ON_POLICY,
    POLICY_CC: POLICY_CC_ON_POLICY,
    MCC_PPO: MCC_PPO_ON_POLICY,
    SCICF_PPO: SCICF_PPO_ON_POLICY,
}


class ContractViolation(ValueError):
    """Raised before an update when a P2 invariant is violated."""


def canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractViolation("%s must be a non-empty identifier." % name)
    return value


def _validate_method(method: str) -> str:
    if not isinstance(method, str):
        raise ContractViolation("P2 method must be a string identifier.")
    if method not in SUPPORTED_METHODS:
        raise ContractViolation("Unsupported P2 method: %s" % method)
    return method


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ContractViolation("%s must be a non-negative integer." % name)
    parsed = int(value)
    if parsed < 0:
        raise ContractViolation("%s must be a non-negative integer." % name)
    return parsed


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if not bool(np.isfinite(array).all()):
        raise ContractViolation("%s must contain only finite values." % name)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FrozenPolicyHandle:
    policy_version: int
    state_sha256: str
    environment_id: str
    task_contract_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_version",
            _nonnegative_integer(self.policy_version, "policy_version"),
        )
        for name in ("state_sha256", "environment_id", "task_contract_id"):
            object.__setattr__(
                self, name, _validate_identifier(getattr(self, name), name)
            )


@dataclass(frozen=True)
class RolloutBatch:
    """Method-independent data owned by the single PPO engine.

    ``transitions`` must contain the accepted Stage-0 transition records.  The
    engine owns GAE and critic returns; a credit estimator can only replace the
    actor-advantage vector returned in :class:`CreditEstimate`.
    """

    batch_id: str
    environment_id: str
    task_contract_id: str
    budget_contract_id: str
    frozen_policy: FrozenPolicyHandle
    transition_ids: Tuple[str, ...]
    transitions: Tuple[Any, ...]
    gae_advantages: np.ndarray
    critic_returns: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "batch_id",
            "environment_id",
            "task_contract_id",
            "budget_contract_id",
        ):
            object.__setattr__(
                self, name, _validate_identifier(getattr(self, name), name)
            )
        if self.environment_id != self.frozen_policy.environment_id:
            raise ContractViolation("Rollout and frozen policy environment differ.")
        if self.task_contract_id != self.frozen_policy.task_contract_id:
            raise ContractViolation("Rollout and frozen policy task contract differ.")
        transition_ids = tuple(str(value) for value in self.transition_ids)
        transitions = tuple(self.transitions)
        if not transition_ids or len(transition_ids) != len(set(transition_ids)):
            raise ContractViolation("transition_ids must be non-empty and unique.")
        if len(transitions) != len(transition_ids):
            raise ContractViolation("transitions and transition_ids must align.")
        gae = _finite_vector(self.gae_advantages, "gae_advantages")
        returns = _finite_vector(self.critic_returns, "critic_returns")
        if gae.size != len(transition_ids) or returns.size != len(transition_ids):
            raise ContractViolation("Rollout tensors must align with transitions.")
        object.__setattr__(self, "transition_ids", transition_ids)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "gae_advantages", gae)
        object.__setattr__(self, "critic_returns", returns)

    @property
    def gae_sha256(self) -> str:
        return array_sha256(self.gae_advantages)

    @property
    def critic_returns_sha256(self) -> str:
        return array_sha256(self.critic_returns)


@dataclass(frozen=True)
class CreditRequest:
    method: str
    rollout: RolloutBatch
    credit_model_version: int
    reserved_query_requested_calls: int
    query_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _validate_method(self.method))
        for name in (
            "credit_model_version",
            "reserved_query_requested_calls",
            "query_seed",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, _nonnegative_integer(value, name))
        if self.method == PPO and self.reserved_query_requested_calls != 0:
            raise ContractViolation("PPO cannot reserve a counterfactual query budget.")


@dataclass(frozen=True)
class EvaluatorLedgerDelta:
    requested_calls: int = 0
    unique_calls: int = 0
    backend_calls: int = 0
    cache_hits: int = 0
    requested_by_source: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed = {}
        for name in ("requested_calls", "unique_calls", "backend_calls", "cache_hits"):
            value = getattr(self, name)
            object.__setattr__(self, name, _nonnegative_integer(value, name))
        for source, count in self.requested_by_source.items():
            parsed[_validate_identifier(source, "evaluator source")] = (
                _nonnegative_integer(count, "evaluator source count")
            )
        object.__setattr__(self, "requested_by_source", parsed)
        if sum(parsed.values()) != self.requested_calls:
            raise ContractViolation("Source-level requested calls do not match total.")
        if self.backend_calls > self.requested_calls:
            raise ContractViolation("backend_calls cannot exceed requested_calls.")
        if self.cache_hits != self.requested_calls - self.backend_calls:
            raise ContractViolation("cache_hits must equal requested minus backend calls.")
        if self.unique_calls > self.backend_calls:
            raise ContractViolation("unique_calls cannot exceed backend_calls.")


@dataclass(frozen=True)
class PendingLabelBatch:
    label_batch_id: str
    source_batch_id: str
    source_policy_version: int
    label_count: int
    payload_sha256: str

    def __post_init__(self) -> None:
        for name in ("label_batch_id", "source_batch_id", "payload_sha256"):
            object.__setattr__(
                self, name, _validate_identifier(getattr(self, name), name)
            )
        for name in ("source_policy_version", "label_count"):
            value = getattr(self, name)
            object.__setattr__(self, name, _nonnegative_integer(value, name))
        if self.label_count == 0:
            raise ContractViolation("A pending-label batch cannot be empty.")


@dataclass(frozen=True)
class CreditEstimate:
    method: str
    source_batch_id: str
    source_policy_version: int
    actor_advantages: np.ndarray
    credit_model_version_before: int
    credit_model_version_after: int
    evaluator_delta: EvaluatorLedgerDelta = field(default_factory=EvaluatorLedgerDelta)
    pending_labels: Optional[PendingLabelBatch] = None
    same_iteration_labels_committed: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _validate_method(self.method))
        object.__setattr__(
            self,
            "source_batch_id",
            _validate_identifier(self.source_batch_id, "source_batch_id"),
        )
        for name in (
            "source_policy_version",
            "credit_model_version_before",
            "credit_model_version_after",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, _nonnegative_integer(value, name))
        object.__setattr__(
            self,
            "actor_advantages",
            _finite_vector(self.actor_advantages, "actor_advantages"),
        )
        if not isinstance(self.same_iteration_labels_committed, bool):
            raise ContractViolation("same_iteration_labels_committed must be Boolean.")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def actor_advantages_sha256(self) -> str:
        return array_sha256(self.actor_advantages)


@dataclass(frozen=True)
class PPOUpdateReceipt:
    engine_contract_id: str
    source_batch_id: str
    policy_version_before: int
    policy_version_after: int
    actor_advantages_sha256: str
    critic_returns_sha256: str
    optimizer_step_completed: bool

    def __post_init__(self) -> None:
        for name in (
            "engine_contract_id",
            "source_batch_id",
            "actor_advantages_sha256",
            "critic_returns_sha256",
        ):
            object.__setattr__(
                self, name, _validate_identifier(getattr(self, name), name)
            )
        for name in ("policy_version_before", "policy_version_after"):
            object.__setattr__(
                self, name, _nonnegative_integer(getattr(self, name), name)
            )
        if not isinstance(self.optimizer_step_completed, bool):
            raise ContractViolation("optimizer_step_completed must be Boolean.")


@dataclass(frozen=True)
class MethodRunContract:
    method: str
    environment_id: str
    task_contract_id: str
    budget_contract_id: str
    evaluator_version: str
    objective_contract: str
    ppo_engine_contract_id: str
    ppo_hyperparameters_sha256: str
    credit_estimator_contract_id: str
    allowed_evaluator_sources: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _validate_method(self.method))
        for name in (
            "environment_id",
            "task_contract_id",
            "budget_contract_id",
            "evaluator_version",
            "objective_contract",
            "ppo_engine_contract_id",
            "ppo_hyperparameters_sha256",
            "credit_estimator_contract_id",
        ):
            object.__setattr__(
                self, name, _validate_identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "allowed_evaluator_sources",
            tuple(sorted(str(value) for value in self.allowed_evaluator_sources)),
        )
        if len(self.allowed_evaluator_sources) != len(
            set(self.allowed_evaluator_sources)
        ):
            raise ContractViolation("Evaluator-source allowlist contains duplicates.")


class CreditEstimator(Protocol):
    method: str
    contract_id: str

    def estimate(self, request: CreditRequest) -> CreditEstimate:
        ...

    def commit_after_update(
        self, pending: PendingLabelBatch, receipt: PPOUpdateReceipt
    ) -> Mapping[str, Any]:
        ...

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


class PPOEngine(Protocol):
    engine_contract_id: str

    def freeze_policy(self) -> FrozenPolicyHandle:
        ...

    def collect_rollout(self, frozen_policy: FrozenPolicyHandle) -> RolloutBatch:
        ...

    def update(
        self, rollout: RolloutBatch, credit: CreditEstimate
    ) -> PPOUpdateReceipt:
        ...

    def state_dict(self) -> Mapping[str, Any]:
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


def validate_credit_estimate(
    request: CreditRequest, estimate: CreditEstimate
) -> None:
    rollout = request.rollout
    if estimate.method != request.method:
        raise ContractViolation("Credit method does not match its request.")
    if estimate.source_batch_id != rollout.batch_id:
        raise ContractViolation("Credit estimate is bound to another rollout batch.")
    if estimate.source_policy_version != rollout.frozen_policy.policy_version:
        raise ContractViolation("Credit estimate used another policy version.")
    if estimate.actor_advantages.size != rollout.gae_advantages.size:
        raise ContractViolation("Actor advantages do not align with the rollout.")
    if estimate.credit_model_version_before != request.credit_model_version:
        raise ContractViolation("Credit model version differs from the frozen request.")
    if estimate.credit_model_version_after != request.credit_model_version:
        raise ContractViolation("Credit model changed before the actor update.")
    if estimate.same_iteration_labels_committed:
        raise ContractViolation("New labels were committed before the actor update.")
    if estimate.evaluator_delta.requested_calls > request.reserved_query_requested_calls:
        raise ContractViolation("Counterfactual queries exceeded the reserved budget.")
    observed_sources = frozenset(estimate.evaluator_delta.requested_by_source)
    if not observed_sources.issubset(METHOD_QUERY_SOURCES[request.method]):
        raise ContractViolation("Credit estimator used an unauthorized evaluator source.")
    if request.method == PPO:
        if not np.array_equal(estimate.actor_advantages, rollout.gae_advantages):
            raise ContractViolation("PPO actor advantages must equal engine GAE.")
        if estimate.evaluator_delta.requested_calls != 0:
            raise ContractViolation("PPO credit cannot query the terminal evaluator.")
        if estimate.pending_labels is not None:
            raise ContractViolation("PPO credit cannot stage counterfactual labels.")
    if estimate.pending_labels is not None:
        if estimate.pending_labels.source_batch_id != rollout.batch_id:
            raise ContractViolation("Pending labels belong to another rollout batch.")
        if (
            estimate.pending_labels.source_policy_version
            != rollout.frozen_policy.policy_version
        ):
            raise ContractViolation("Pending labels used another policy version.")


def validate_update_receipt(
    rollout: RolloutBatch,
    estimate: CreditEstimate,
    receipt: PPOUpdateReceipt,
) -> None:
    if receipt.engine_contract_id != PPO_ENGINE_CONTRACT_ID:
        raise ContractViolation("Update used a different PPO engine contract.")
    if receipt.source_batch_id != rollout.batch_id:
        raise ContractViolation("Update receipt belongs to another rollout batch.")
    expected_before = rollout.frozen_policy.policy_version
    if receipt.policy_version_before != expected_before:
        raise ContractViolation("Update did not start from the frozen policy version.")
    if receipt.policy_version_after != expected_before + 1:
        raise ContractViolation("A completed update must advance policy_version by one.")
    if not receipt.optimizer_step_completed:
        raise ContractViolation("Pending labels cannot commit before optimizer completion.")
    if receipt.actor_advantages_sha256 != estimate.actor_advantages_sha256:
        raise ContractViolation("Actor update did not use the validated credit vector.")
    if receipt.critic_returns_sha256 != rollout.critic_returns_sha256:
        raise ContractViolation("Critic targets differ from environment returns.")


def validate_pending_label_commit(
    pending: PendingLabelBatch, receipt: PPOUpdateReceipt
) -> None:
    if pending.source_batch_id != receipt.source_batch_id:
        raise ContractViolation("Pending labels and PPO receipt refer to different batches.")
    if pending.source_policy_version != receipt.policy_version_before:
        raise ContractViolation("Pending labels and PPO receipt use different policies.")
    if not receipt.optimizer_step_completed:
        raise ContractViolation("Pending labels cannot commit before optimizer completion.")
    if receipt.policy_version_after <= pending.source_policy_version:
        raise ContractViolation("Pending labels cannot commit in their source iteration.")


def validate_method_contracts(contracts: Sequence[MethodRunContract]) -> None:
    by_method = {item.method: item for item in contracts}
    if set(by_method) != set(METHODS) or len(contracts) != len(METHODS):
        raise ContractViolation("Exactly one contract is required for every P2 method.")
    shared_names = (
        "environment_id",
        "task_contract_id",
        "budget_contract_id",
        "evaluator_version",
        "objective_contract",
        "ppo_engine_contract_id",
        "ppo_hyperparameters_sha256",
    )
    reference = by_method[PPO]
    for method in METHODS:
        item = by_method[method]
        for name in shared_names:
            if getattr(item, name) != getattr(reference, name):
                raise ContractViolation(
                    "Method contracts differ outside the credit seam: %s" % name
                )
        if item.ppo_engine_contract_id != PPO_ENGINE_CONTRACT_ID:
            raise ContractViolation("Method uses an unrecognized PPO engine contract.")
        if item.credit_estimator_contract_id != CREDIT_ESTIMATOR_CONTRACT_ID:
            raise ContractViolation("Method uses an unrecognized credit contract.")
        if frozenset(item.allowed_evaluator_sources) != METHOD_EVALUATOR_SOURCES[method]:
            raise ContractViolation("Method evaluator-source allowlist is incorrect.")
