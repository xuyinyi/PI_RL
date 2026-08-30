"""Matched-budget non-LLM acquisition baselines for offline Gate 1."""

from __future__ import annotations

import random
from typing import Callable, Dict, List

from reproduction.scicf.acquisition.base import (
    AcquisitionCandidate,
    AcquisitionResult,
    AcquisitionStrategy,
    CandidatePool,
)


def _ranked_result(
    strategy: AcquisitionStrategy,
    pool: CandidatePool,
    ranked: List[AcquisitionCandidate],
    budget: int,
    seed: int,
    score: Callable[[AcquisitionCandidate], float],
) -> AcquisitionResult:
    return AcquisitionResult(
        strategy=strategy.name,
        strategy_version=strategy.version,
        pool_id=pool.pool_id,
        candidate_ids=pool.candidate_ids,
        selected_ids=tuple(candidate.candidate_id for candidate in ranked[:budget]),
        budget=budget,
        ranking_scores={candidate.candidate_id: float(score(candidate)) for candidate in ranked},
        seed=seed,
    )


class RandomAcquisition(AcquisitionStrategy):
    name = "random"
    version = "v1"

    def select(self, pool: CandidatePool, budget: int, seed: int) -> AcquisitionResult:
        self._validate_budget(pool, budget)
        ranked = list(pool.candidates)
        random.Random(seed).shuffle(ranked)
        scores: Dict[str, float] = {
            candidate.candidate_id: float(len(ranked) - index)
            for index, candidate in enumerate(ranked)
        }
        return AcquisitionResult(
            strategy=self.name,
            strategy_version=self.version,
            pool_id=pool.pool_id,
            candidate_ids=pool.candidate_ids,
            selected_ids=tuple(candidate.candidate_id for candidate in ranked[:budget]),
            budget=budget,
            ranking_scores=scores,
            seed=seed,
        )


class PolicyProbabilityAcquisition(AcquisitionStrategy):
    name = "policy_probability"
    version = "v1"

    def select(self, pool: CandidatePool, budget: int, seed: int) -> AcquisitionResult:
        self._validate_budget(pool, budget)
        if any(candidate.policy_score is None for candidate in pool.candidates):
            raise ValueError("policy acquisition requires scores for every candidate")
        ranked = sorted(
            pool.candidates,
            key=lambda candidate: (float(candidate.policy_score), candidate.candidate_id),
            reverse=True,
        )
        return _ranked_result(
            self,
            pool,
            ranked,
            budget,
            seed,
            lambda candidate: float(candidate.policy_score),
        )


class HeuristicAcquisition(AcquisitionStrategy):
    name = "chemistry_heuristic"
    version = "v1"

    def select(self, pool: CandidatePool, budget: int, seed: int) -> AcquisitionResult:
        self._validate_budget(pool, budget)
        if any(candidate.heuristic_score is None for candidate in pool.candidates):
            raise ValueError("heuristic acquisition requires scores for every candidate")
        ranked = sorted(
            pool.candidates,
            key=lambda candidate: (float(candidate.heuristic_score), candidate.candidate_id),
            reverse=True,
        )
        return _ranked_result(
            self,
            pool,
            ranked,
            budget,
            seed,
            lambda candidate: float(candidate.heuristic_score),
        )
