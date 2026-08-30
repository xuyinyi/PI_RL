"""Common acquisition records that guarantee matched candidate pools."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from reproduction.scicf.core.records import Intervention


@dataclass(frozen=True)
class AcquisitionCandidate:
    intervention: Intervention
    policy_score: Optional[float] = None
    structural_score: Optional[float] = None
    heuristic_score: Optional[float] = None

    @property
    def candidate_id(self) -> str:
        return self.intervention.intervention_id


@dataclass(frozen=True)
class CandidatePool:
    candidates: Tuple[AcquisitionCandidate, ...]
    requested_size: int
    source_selected_counts: Mapping[str, int]
    source_shortfalls: Mapping[str, int]
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifiers = self.candidate_ids
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate pool contains duplicate intervention IDs")
        if len(identifiers) > self.requested_size:
            raise ValueError("candidate pool exceeds requested size")

    @property
    def candidate_ids(self) -> Tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def pool_id(self) -> str:
        joined = "\n".join(sorted(self.candidate_ids)).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()

    @property
    def shortfall(self) -> int:
        return self.requested_size - len(self.candidates)


@dataclass(frozen=True)
class AcquisitionResult:
    strategy: str
    strategy_version: str
    pool_id: str
    candidate_ids: Tuple[str, ...]
    selected_ids: Tuple[str, ...]
    budget: int
    ranking_scores: Mapping[str, float]
    seed: int

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("acquisition budget must be positive")
        if len(self.selected_ids) > self.budget:
            raise ValueError("selected interventions exceed acquisition budget")
        if len(self.selected_ids) != len(set(self.selected_ids)):
            raise ValueError("acquisition result contains duplicate IDs")
        if not set(self.selected_ids).issubset(self.candidate_ids):
            raise ValueError("acquisition selected an ID outside the fixed pool")


class AcquisitionStrategy(ABC):
    name = "abstract"
    version = "v1"

    @abstractmethod
    def select(self, pool: CandidatePool, budget: int, seed: int) -> AcquisitionResult:
        """Rank and select from exactly the supplied candidate pool."""

    def _validate_budget(self, pool: CandidatePool, budget: int) -> None:
        if budget < 1:
            raise ValueError("budget must be positive")
        if budget > len(pool.candidates):
            raise ValueError("budget exceeds available fixed-pool candidates")
