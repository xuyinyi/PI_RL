"""Deterministic, gain-blind prompt generation for LLM acquisition."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, Mapping, Sequence

from reproduction.scicf.acquisition.base import CandidatePool
from reproduction.scicf.core.records import TrajectoryRecord


PROMPT_VERSION = "scicf-dapigen-acquisition-blinded-v2"
PRESENTATION_PROTOCOL = "sha256-shuffle-v1"


def blind_candidate_rows(
    request_id: str, pool_id: str, candidate_rows: Sequence[Mapping[str, Any]]
) -> list:
    """Deterministically blind source-group and policy-near presentation order."""

    identity = "{}|{}|{}".format(request_id, pool_id, PRESENTATION_PROTOCOL)
    seed = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)
    blinded = [dict(row) for row in candidate_rows]
    random.Random(seed).shuffle(blinded)
    return blinded


def acquisition_system_prompt(budget: int) -> str:
    return (
        "You rank proposed scientific experiments. You do not provide reward, "
        "value, advantage, or truth. Rank only intervention IDs explicitly "
        "listed by the user. Return exactly {} unique IDs in best-first order "
        "inside one JSON object and no markdown."
    ).format(budget)


def build_acquisition_request(
    request_id: str,
    trajectory: TrajectoryRecord,
    decision_timestep: int,
    pool: CandidatePool,
    budget: int,
) -> Dict[str, Any]:
    if decision_timestep < 0 or decision_timestep >= len(trajectory.steps):
        raise ValueError("decision_timestep is outside the trajectory")
    if budget < 1 or budget > len(pool.candidates):
        raise ValueError("invalid acquisition budget")
    step = trajectory.steps[decision_timestep]
    candidate_rows = []
    for candidate in pool.candidates:
        intervention = candidate.intervention
        candidate_rows.append(
            {
                "intervention_id": intervention.intervention_id,
                "component": intervention.component,
                "factual_component_index": intervention.factual_component_value,
                "alternative_component_index": intervention.alternative_component_value,
                "alternative_building_block_smiles": intervention.alternative_structure,
            }
        )
    candidate_rows = blind_candidate_rows(request_id, pool.pool_id, candidate_rows)
    scientific_context = {
        "objective": (
            "Select atomic polymer-building-block interventions most worth evaluating "
            "for improvement in the frozen DAPiGen terminal multi-property reward."
        ),
        "budget": budget,
        "decision_timestep": decision_timestep,
        "pre_action_scientific_state": step.pre_state,
        "factual_action": list(step.factual_action),
        "factual_terminal_outcome": {
            "polymer": trajectory.terminal_scientific_object,
            "properties": trajectory.terminal_properties,
            "episode_return": trajectory.episode_return,
        },
        "candidate_interventions": candidate_rows,
        "required_output": {
            "ranked_intervention_ids": (
                "exactly {} candidate ID strings in best-first order".format(budget)
            ),
            "reasoning": "brief scientific rationale; advisory only",
            "confidence": "optional number in [0,1]; advisory only",
        },
    }
    messages = [
        {
            "role": "system",
            "content": acquisition_system_prompt(budget),
        },
        {
            "role": "user",
            "content": json.dumps(
                scientific_context, sort_keys=True, separators=(",", ":")
            ),
        },
    ]
    prompt_bytes = json.dumps(
        messages, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "request_id": request_id,
        "prompt_version": PROMPT_VERSION,
        "pool_id": pool.pool_id,
        "candidate_ids": list(pool.candidate_ids),
        "budget": budget,
        "candidate_presentation": PRESENTATION_PROTOCOL,
        "presented_candidate_ids": [
            row["intervention_id"] for row in candidate_rows
        ],
        "messages": messages,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "verified_gain_exposed": False,
    }
