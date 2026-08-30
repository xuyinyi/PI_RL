"""Deterministic fixed candidate-pool construction with source quotas."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Sequence

from reproduction.scicf.acquisition.base import AcquisitionCandidate, CandidatePool


SOURCE_ORDER = ("policy_near", "random_legal", "structural")


def _merge_candidates(
    groups: Iterable[Sequence[AcquisitionCandidate]],
) -> Dict[str, AcquisitionCandidate]:
    merged: Dict[str, AcquisitionCandidate] = {}
    for group in groups:
        for candidate in group:
            previous = merged.get(candidate.candidate_id)
            if previous is None:
                merged[candidate.candidate_id] = candidate
                continue
            if previous.intervention != candidate.intervention:
                raise ValueError(
                    "candidate ID maps to inconsistent interventions: {}".format(
                        candidate.candidate_id
                    )
                )
            merged[candidate.candidate_id] = AcquisitionCandidate(
                intervention=candidate.intervention,
                policy_score=(
                    candidate.policy_score
                    if candidate.policy_score is not None
                    else previous.policy_score
                ),
                structural_score=(
                    candidate.structural_score
                    if candidate.structural_score is not None
                    else previous.structural_score
                ),
                heuristic_score=(
                    candidate.heuristic_score
                    if candidate.heuristic_score is not None
                    else previous.heuristic_score
                ),
            )
    return merged


class CandidatePoolBuilder:
    """Build one pool that every acquisition strategy must share."""

    version = "fixed-source-quota-v1"

    def build(
        self,
        source_candidates: Mapping[str, Sequence[AcquisitionCandidate]],
        source_quotas: Mapping[str, int],
        requested_size: int,
        seed: int,
    ) -> CandidatePool:
        if requested_size < 1:
            raise ValueError("requested_size must be positive")
        if set(source_quotas) != set(SOURCE_ORDER):
            raise ValueError("source quotas must define {}".format(SOURCE_ORDER))
        if sum(source_quotas.values()) != requested_size:
            raise ValueError("source quotas must sum to requested_size")
        if any(value < 0 for value in source_quotas.values()):
            raise ValueError("source quotas must be non-negative")

        all_candidates = _merge_candidates(source_candidates.values())
        normalized: Dict[str, List[AcquisitionCandidate]] = {}
        for source in SOURCE_ORDER:
            candidates = [
                all_candidates[candidate.candidate_id]
                for candidate in source_candidates.get(source, ())
            ]
            if source == "policy_near":
                candidates.sort(
                    key=lambda item: (
                        item.policy_score is not None,
                        item.policy_score if item.policy_score is not None else float("-inf"),
                        item.candidate_id,
                    ),
                    reverse=True,
                )
            elif source == "structural":
                candidates.sort(
                    key=lambda item: (
                        item.structural_score is not None,
                        item.structural_score
                        if item.structural_score is not None
                        else float("-inf"),
                        item.candidate_id,
                    ),
                    reverse=True,
                )
            else:
                random.Random(seed).shuffle(candidates)
            normalized[source] = candidates

        selected: List[AcquisitionCandidate] = []
        selected_ids = set()
        counts: Dict[str, int] = {source: 0 for source in SOURCE_ORDER}
        shortfalls: Dict[str, int] = {source: 0 for source in SOURCE_ORDER}
        for source in SOURCE_ORDER:
            quota = int(source_quotas[source])
            for candidate in normalized[source]:
                if candidate.candidate_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.candidate_id)
                counts[source] += 1
                if counts[source] == quota:
                    break
            shortfalls[source] = quota - counts[source]

        if len(selected) < requested_size:
            remainder = [
                candidate
                for identifier, candidate in sorted(all_candidates.items())
                if identifier not in selected_ids
            ]
            random.Random(seed + 1).shuffle(remainder)
            for candidate in remainder[: requested_size - len(selected)]:
                selected.append(candidate)
                selected_ids.add(candidate.candidate_id)

        return CandidatePool(
            candidates=tuple(selected),
            requested_size=requested_size,
            source_selected_counts=counts,
            source_shortfalls=shortfalls,
        )
