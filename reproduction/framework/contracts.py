"""Stable interfaces shared by all DAPiGen optimization algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type


class ContractError(RuntimeError):
    """Raised when an algorithm violates the comparison contract."""


@dataclass(frozen=True)
class AlgorithmContext:
    """Runtime inputs supplied to an algorithm without changing task semantics."""

    repo_root: Path
    output_root: Path
    env_class: Type[Any]
    env_config: Mapping[str, Any]
    legacy_algorithm_config: Mapping[str, Any]
    seed: int
    resources: Mapping[str, Any]
    slurm_cpus: int


class AlgorithmAdapter(ABC):
    """Interface that makes different optimizers comparable in one runner.

    Implementations own policy construction, updates, native logging, and
    checkpoints. They must not mutate ``context.env_config`` or the reward
    backend.
    """

    @property
    @abstractmethod
    def identity(self) -> Mapping[str, Any]:
        """Return a JSON-serializable algorithm name/version identity."""

    @abstractmethod
    def initialize(
        self, context: AlgorithmContext, parameters: Mapping[str, Any]
    ) -> None:
        """Allocate the algorithm and its workers."""

    @abstractmethod
    def train_step(self, target_environment_steps: Optional[int] = None) -> Dict[str, Any]:
        """Perform one update and return normalized cumulative metrics.

        The result must contain an integer ``environment_steps`` value that is
        strictly increasing across calls. When the runner uses strict budgets,
        the adapter must stop exactly at ``target_environment_steps``.
        """

    @abstractmethod
    def act(self, observation: Any, explore: bool = False) -> Any:
        """Return one environment action."""

    @abstractmethod
    def save(self, checkpoint_dir: Path) -> str:
        """Save a restorable checkpoint and return its locator."""

    @abstractmethod
    def restore(self, checkpoint_path: str) -> None:
        """Restore a previously saved checkpoint."""

    @abstractmethod
    def effective_config(self) -> Mapping[str, Any]:
        """Return the effective, JSON-serializable algorithm configuration."""

    @abstractmethod
    def close(self) -> None:
        """Release workers and other runtime resources."""
