"""Typed records for the bounded online SciCF-PPO architecture smoke."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ONLINE_PROTOCOL_ID = "dapigen-scicf-online-architecture-smoke-v1"
ACQUISITION_VERIFIER_PROTOCOL_ID = "dapigen-scicf-acquisition-verifier-dev-v1"
PAIRWISE_STABILITY_PROTOCOL_ID = "dapigen-scicf-pairwise-stability-dev-v1"
SINGLE_ITERATION_INTEGRATION_PROTOCOL_ID = (
    "dapigen-scicf-single-iteration-integration-smoke-v1"
)
SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID = (
    "dapigen-scicf-single-iteration-integration-smoke-v2"
)
SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID = (
    "dapigen-scicf-single-iteration-integration-smoke-v3"
)
SCHEMA_ROBUSTNESS_PROTOCOL_ID = "dapigen-scicf-llm-schema-robustness-dev-v1"
SOFT_PAIR_RESILIENCE_PROTOCOL_ID = "dapigen-scicf-soft-pair-resilience-dev-v1"
ONLINE_PROMPT_VERSION = "scicf-dapigen-online-acquisition-blinded-v1"
ONLINE_PRESENTATION_PROTOCOL = "opaque-id-sha256-shuffle-v1"


def _finite_array(value: Any, name: str, dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1).copy()
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise ValueError("%s must be a non-empty finite vector" % name)
    array.setflags(write=False)
    return array


def _action(value: Sequence[int], name: str) -> Tuple[int, int]:
    if len(value) != 2:
        raise ValueError("%s must contain two action ids" % name)
    result = (int(value[0]), int(value[1]))
    if result[0] < 0 or result[1] < 0:
        raise ValueError("%s cannot contain negative action ids" % name)
    return result


@dataclass(frozen=True)
class OnlineCandidate:
    candidate_id: str
    transition_id: str
    episode_id: int
    timestep: int
    component: str
    factual_action: Tuple[int, int]
    alternative_action: Tuple[int, int]
    state_snapshot: Mapping[str, Any]
    observation: np.ndarray
    dianhydride_mask: np.ndarray
    diamine_mask: np.ndarray
    state_context: Mapping[str, Any]
    factual_block: Mapping[str, Any]
    alternative_block: Mapping[str, Any]
    behavior_policy_probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.startswith("cf-"):
            raise ValueError("candidate_id must be an opaque cf- identifier")
        if not isinstance(self.transition_id, str) or not self.transition_id:
            raise ValueError("transition_id must be non-empty")
        if int(self.episode_id) < 0 or int(self.timestep) < 0:
            raise ValueError("episode_id and timestep must be non-negative")
        if self.component not in {"dianhydride", "diamine"}:
            raise ValueError("unsupported action component")
        factual = _action(self.factual_action, "factual_action")
        alternative = _action(self.alternative_action, "alternative_action")
        changed = [index for index in (0, 1) if factual[index] != alternative[index]]
        expected = 0 if self.component == "dianhydride" else 1
        if changed != [expected]:
            raise ValueError("an online candidate must change exactly its named component")
        observation = _finite_array(self.observation, "observation", np.float32)
        d_mask = np.asarray(self.dianhydride_mask, dtype=bool).reshape(-1).copy()
        a_mask = np.asarray(self.diamine_mask, dtype=bool).reshape(-1).copy()
        if not bool(d_mask.any()) or not bool(a_mask.any()):
            raise ValueError("candidate masks must retain a legal action")
        if not d_mask[alternative[0]] or not a_mask[alternative[1]]:
            raise ValueError("alternative action is masked")
        d_mask.setflags(write=False)
        a_mask.setflags(write=False)
        probability = float(self.behavior_policy_probability)
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError("behavior policy probability must lie in [0,1]")
        object.__setattr__(self, "factual_action", factual)
        object.__setattr__(self, "alternative_action", alternative)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "dianhydride_mask", d_mask)
        object.__setattr__(self, "diamine_mask", a_mask)
        object.__setattr__(self, "state_snapshot", dict(self.state_snapshot))
        object.__setattr__(self, "state_context", dict(self.state_context))
        object.__setattr__(self, "factual_block", dict(self.factual_block))
        object.__setattr__(self, "alternative_block", dict(self.alternative_block))
        object.__setattr__(self, "behavior_policy_probability", probability)

    def public_prompt_row(self) -> Dict[str, Any]:
        """Return chemistry context while excluding policy and reward scores."""

        return {
            "intervention_id": self.candidate_id,
            "timestep": int(self.timestep),
            "component": self.component,
            "pre_action_state": dict(self.state_context),
            "factual_component_action": dict(self.factual_block),
            "alternative_component_action": dict(self.alternative_block),
        }

    def audit_row(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "transition_id": self.transition_id,
            "episode_id": int(self.episode_id),
            "timestep": int(self.timestep),
            "component": self.component,
            "factual_action": list(self.factual_action),
            "alternative_action": list(self.alternative_action),
            "state_id": self.state_snapshot.get("state_id"),
            "behavior_policy_probability": float(self.behavior_policy_probability),
            "factual_block": dict(self.factual_block),
            "alternative_block": dict(self.alternative_block),
        }


@dataclass(frozen=True)
class OnlineVerification:
    candidate_id: str
    paired_seeds: Tuple[int, ...]
    factual_returns: Tuple[float, ...]
    counterfactual_returns: Tuple[float, ...]
    deltas: Tuple[float, ...]
    accepted: bool
    preferred: Optional[str]
    rejection_reason: Optional[str]
    factual_terminal_records: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    counterfactual_terminal_records: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        size = len(self.paired_seeds)
        if size < 1:
            raise ValueError("verification requires at least one paired replicate")
        if not (
            len(self.factual_returns)
            == len(self.counterfactual_returns)
            == len(self.deltas)
            == len(self.factual_terminal_records)
            == len(self.counterfactual_terminal_records)
            == size
        ):
            raise ValueError("verification replicate fields must align")
        for factual, counterfactual, delta in zip(
            self.factual_returns, self.counterfactual_returns, self.deltas
        ):
            if not all(math.isfinite(float(value)) for value in (factual, counterfactual, delta)):
                raise ValueError("verification returns must be finite")
            if not math.isclose(
                float(counterfactual) - float(factual),
                float(delta),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError("verification delta does not match paired returns")
        mean_delta = self.mean_delta
        expected = "counterfactual" if mean_delta > 0.0 else "factual"
        if self.accepted:
            if mean_delta == 0.0 or self.preferred != expected or self.rejection_reason is not None:
                raise ValueError("accepted verification direction is inconsistent")
        elif self.preferred is not None or not self.rejection_reason:
            raise ValueError("rejected verification must record a reason and no preference")

    @property
    def mean_delta(self) -> float:
        return float(sum(self.deltas) / len(self.deltas))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "paired_seeds": [int(value) for value in self.paired_seeds],
            "factual_returns": [float(value) for value in self.factual_returns],
            "counterfactual_returns": [
                float(value) for value in self.counterfactual_returns
            ],
            "deltas": [float(value) for value in self.deltas],
            "mean_delta": self.mean_delta,
            "accepted": bool(self.accepted),
            "preferred": self.preferred,
            "rejection_reason": self.rejection_reason,
            "factual_terminal_records": [
                dict(value) for value in self.factual_terminal_records
            ],
            "counterfactual_terminal_records": [
                dict(value) for value in self.counterfactual_terminal_records
            ],
        }


@dataclass(frozen=True)
class PairwiseRefinementConfig:
    learning_rate: float = 1e-5
    maximum_gradient_norm: float = 0.5
    target_kl: float = 0.01
    maximum_weight: float = 2.0
    delta_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "learning_rate",
            "maximum_gradient_norm",
            "target_kl",
            "maximum_weight",
            "delta_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be finite and positive" % name)


@dataclass(frozen=True)
class SchemaRecoveryConfig:
    """Hard bounds for one online response plus at most one semantic repair."""

    maximum_schema_attempts: int = 2
    transport_retries_per_attempt: int = 1
    max_output_tokens: int = 512
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_schema_attempts, bool)
            or not isinstance(self.maximum_schema_attempts, int)
            or self.maximum_schema_attempts not in {1, 2}
        ):
            raise ValueError("maximum_schema_attempts must be 1 or 2")
        if (
            isinstance(self.transport_retries_per_attempt, bool)
            or not isinstance(self.transport_retries_per_attempt, int)
            or not 0 <= self.transport_retries_per_attempt <= 1
        ):
            raise ValueError("transport_retries_per_attempt must be 0 or 1")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
        ):
            raise ValueError("max_output_tokens must be positive")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0.0
        ):
            raise ValueError("timeout_seconds must be finite and positive")

    @property
    def maximum_semantic_repairs(self) -> int:
        return int(self.maximum_schema_attempts) - 1

    @property
    def maximum_http_transmissions(self) -> int:
        return int(self.maximum_schema_attempts) * (
            int(self.transport_retries_per_attempt) + 1
        )
