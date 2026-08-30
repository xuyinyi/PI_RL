"""Stable scientific-domain contract for SciCF acquisition and verification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from reproduction.scicf.core.records import ActionTuple, Intervention


PolicyCallable = Callable[[Any, np.random.RandomState], ActionTuple]


@dataclass(frozen=True)
class ContinuationOutcome:
    terminal_return: float
    terminal_scientific_object: Optional[str]
    terminal_properties: Mapping[str, Any]
    actions: Tuple[ActionTuple, ...]
    environment_transitions: int
    atomic_oracle_calls: int
    policy_version: str
    continuation_seed: int


class ScientificDomainAdapter(ABC):
    """Domain operations required by the domain-independent SciCF core."""

    @abstractmethod
    def serialize_state(self) -> Mapping[str, Any]:
        """Return domain-readable state without exposing raw PPO vectors alone."""

    @abstractmethod
    def capture_snapshot(self, observation: Any) -> Any:
        """Capture all scientific and stochastic state before an action."""

    @abstractmethod
    def restore_snapshot(self, snapshot: Any) -> Any:
        """Restore a captured pre-action state and return its PPO observation."""

    @abstractmethod
    def enumerate_interventions(
        self,
        trajectory_id: str,
        timestep: int,
        factual_action: ActionTuple,
    ) -> Sequence[Intervention]:
        """Enumerate legal interventions that each change one action component."""

    @abstractmethod
    def apply_intervention(
        self, factual_action: ActionTuple, intervention: Intervention
    ) -> ActionTuple:
        """Return an intervened action while preserving the unrelated component."""

    @abstractmethod
    def continue_from_snapshot(
        self,
        snapshot: Any,
        first_action: ActionTuple,
        continuation_policy: PolicyCallable,
        policy_version: str,
        continuation_seed: int,
        oracle_scope: str,
        max_steps: int,
    ) -> ContinuationOutcome:
        """Run one auditable continuation from a restored state."""
