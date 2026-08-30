"""Typed, JSON-serializable records used by the SciCF pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


ActionTuple = Tuple[int, int]
VALID_COMPONENTS = {"dianhydride", "diamine"}


def _require_finite(value: float, locator: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError("{} must be finite".format(locator))


def _require_action(value: ActionTuple, locator: str) -> None:
    if len(value) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise ValueError("{} must contain two non-negative action indices".format(locator))


@dataclass(frozen=True)
class TrajectoryStep:
    """One factual decision with both PPO and domain-readable context."""

    timestep: int
    observation: Tuple[float, ...]
    factual_action: ActionTuple
    pre_state: Mapping[str, Any]
    post_state: Mapping[str, Any]
    terminated: bool
    environment_reward: float
    snapshot_id: str
    oracle_outputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")
        if not self.snapshot_id:
            raise ValueError("snapshot_id is required")
        _require_action(self.factual_action, "factual_action")
        _require_finite(self.environment_reward, "environment_reward")


@dataclass(frozen=True)
class TrajectoryRecord:
    """A complete factual scientific-design episode."""

    trajectory_id: str
    policy_version: str
    seed: int
    steps: Tuple[TrajectoryStep, ...]
    episode_return: float
    terminal_scientific_object: Optional[str]
    terminal_properties: Mapping[str, Any]
    environment_transitions: int
    atomic_oracle_calls: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.policy_version:
            raise ValueError("trajectory_id and policy_version are required")
        if self.schema_version != 1:
            raise ValueError("unsupported trajectory schema_version")
        if self.environment_transitions != len(self.steps):
            raise ValueError("environment_transitions must equal recorded step count")
        if self.atomic_oracle_calls < 0:
            raise ValueError("atomic_oracle_calls must be non-negative")
        _require_finite(self.episode_return, "episode_return")


@dataclass(frozen=True)
class Intervention:
    """One legal, action-component-local counterfactual intervention."""

    intervention_id: str
    trajectory_id: str
    timestep: int
    component: str
    factual_action: ActionTuple
    alternative_action: ActionTuple
    factual_component_value: int
    alternative_component_value: int
    alternative_structure: str
    source_memberships: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.intervention_id or not self.trajectory_id:
            raise ValueError("intervention_id and trajectory_id are required")
        if self.component not in VALID_COMPONENTS:
            raise ValueError("unsupported action component: {}".format(self.component))
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")
        _require_action(self.factual_action, "factual_action")
        _require_action(self.alternative_action, "alternative_action")
        changed = [
            index
            for index, values in enumerate(zip(self.factual_action, self.alternative_action))
            if values[0] != values[1]
        ]
        expected = 0 if self.component == "dianhydride" else 1
        if changed != [expected]:
            raise ValueError("an intervention must change exactly its declared component")
        if self.factual_component_value == self.alternative_component_value:
            raise ValueError("alternative component must differ from factual component")


@dataclass(frozen=True)
class PairedOutcome:
    """Raw result of one matched factual/counterfactual replicate."""

    replicate: int
    continuation_seed: int
    factual_return: float
    counterfactual_return: float
    factual_oracle_calls: int
    counterfactual_oracle_calls: int
    factual_terminal_object: Optional[str] = None
    counterfactual_terminal_object: Optional[str] = None

    def __post_init__(self) -> None:
        if self.replicate < 0 or self.factual_oracle_calls < 0 or self.counterfactual_oracle_calls < 0:
            raise ValueError("replicate and oracle counts must be non-negative")
        _require_finite(self.factual_return, "factual_return")
        _require_finite(self.counterfactual_return, "counterfactual_return")

    @property
    def delta(self) -> float:
        return self.counterfactual_return - self.factual_return


@dataclass(frozen=True)
class VerificationResult:
    """Auditable scientific-oracle decision for one intervention."""

    verification_id: str
    intervention: Intervention
    policy_version: str
    paired_outcomes: Tuple[PairedOutcome, ...]
    mean_delta: float
    uncertainty: float
    confidence_rule: str
    accepted_for_learning: bool
    rejection_reason: Optional[str]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.verification_id or not self.policy_version:
            raise ValueError("verification_id and policy_version are required")
        if not self.paired_outcomes:
            raise ValueError("at least one paired outcome is required")
        _require_finite(self.mean_delta, "mean_delta")
        _require_finite(self.uncertainty, "uncertainty")
        observed_mean = sum(outcome.delta for outcome in self.paired_outcomes) / len(
            self.paired_outcomes
        )
        if not math.isclose(observed_mean, self.mean_delta, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("mean_delta does not match paired outcomes")
        if not self.accepted_for_learning and not self.rejection_reason:
            raise ValueError("rejected verification requires a reason")

    @property
    def atomic_oracle_calls(self) -> int:
        return sum(
            outcome.factual_oracle_calls + outcome.counterfactual_oracle_calls
            for outcome in self.paired_outcomes
        )


@dataclass(frozen=True)
class VerifiedPair:
    """A pairwise target created only from an accepted verification."""

    pair_id: str
    verification_id: str
    trajectory_id: str
    timestep: int
    component: str
    observation: Tuple[float, ...]
    factual_component_value: int
    alternative_component_value: int
    preferred: str
    verified_delta: float
    weight: float
    source_policy_version: str
    created_iteration: int
    expires_after_iteration: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.component not in VALID_COMPONENTS:
            raise ValueError("unsupported action component")
        if self.preferred not in {"factual", "counterfactual"}:
            raise ValueError("preferred must be factual or counterfactual")
        if self.factual_component_value == self.alternative_component_value:
            raise ValueError("pair actions must differ")
        if self.weight <= 0:
            raise ValueError("pair weight must be positive")
        if self.expires_after_iteration < self.created_iteration:
            raise ValueError("pair expiry cannot precede creation")
        _require_finite(self.verified_delta, "verified_delta")
        _require_finite(self.weight, "weight")
        expected = "counterfactual" if self.verified_delta > 0 else "factual"
        if self.verified_delta == 0 or self.preferred != expected:
            raise ValueError("pair direction must follow the non-zero verified delta")


def record_to_dict(record: Any) -> Dict[str, Any]:
    """Convert a SciCF dataclass record to a plain nested mapping."""

    return asdict(record)
