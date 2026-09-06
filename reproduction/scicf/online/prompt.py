"""Blinded, reward-free prompt and response schema for online acquisition."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, Mapping, Sequence

from .contracts import (
    ONLINE_PRESENTATION_PROTOCOL,
    ONLINE_PROMPT_VERSION,
    OnlineCandidate,
)


def _blinded_rows(
    request_id: str, candidates: Sequence[OnlineCandidate]
) -> Sequence[Mapping[str, Any]]:
    rows = [candidate.public_prompt_row() for candidate in candidates]
    seed = int(
        hashlib.sha256(
            (request_id + "|" + ONLINE_PRESENTATION_PROTOCOL).encode("utf-8")
        ).hexdigest()[:16],
        16,
    )
    random.Random(seed).shuffle(rows)
    return rows


def build_online_acquisition_request(
    *,
    request_id: str,
    trajectory_context: Mapping[str, Any],
    candidates: Sequence[OnlineCandidate],
    maximum_budget: int,
) -> Dict[str, Any]:
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be non-empty")
    if maximum_budget < 1 or maximum_budget > len(candidates):
        raise ValueError("maximum_budget is outside the candidate pool")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate ids must be unique")
    rows = list(_blinded_rows(request_id, candidates))
    scientific_context = {
        "role_boundary": (
            "You choose which legal counterfactual experiments are worth spending "
            "scientific-Oracle budget on. You do not predict or provide reward truth."
        ),
        "polymer_design_task": {
            "material": "polyimide assembled from dianhydride and diamine fragments",
            "action_semantics": (
                "Each candidate changes exactly one legal building-block action at one "
                "recorded timestep; later actions will be sampled from one frozen PPO policy."
            ),
            "objective": (
                "Maximize the DAPiGen terminal score: optical transmittance/100 multiplied "
                "by the average of 1 and four bounded desirabilities for low absolute CTE, "
                "high tensile strength, high glass-transition temperature, and low synthetic "
                "accessibility score. The terminal scientific Oracle, not you, computes it."
            ),
            "desirability_anchors": {
                "absolute_cte": "better when closer to 0; zero contribution at or above 80",
                "tensile_strength": "better when higher; saturates above 500",
                "glass_transition_temperature": "better when higher; saturates above 600",
                "synthetic_accessibility": "better when lower; best below 2 and zero at or above 5",
                "transmittance": "multiplicative; better when higher",
            },
            "mechanistic_considerations": [
                "rigidity, aromaticity, symmetry and planarity may improve thermal or mechanical behavior",
                "bulky, non-coplanar or fluorinated motifs may alter packing and optical behavior",
                "excessive size, attachment incompatibility or difficult synthesis can invalidate a trajectory",
                "trade-offs across all five objectives matter more than optimizing one property alone",
            ],
        },
        "observed_trajectory_without_reward": dict(trajectory_context),
        "maximum_oracle_budget": int(maximum_budget),
        "candidate_interventions": rows,
        "selection_rule": (
            "Return between 1 and the maximum budget candidates in best-first order. "
            "Use abstain=true with an empty list only if none is scientifically worth "
            "evaluating. Do not invent IDs."
        ),
        "required_output": {
            "ranked_intervention_ids": "JSON list containing zero to maximum_oracle_budget unique listed IDs",
            "abstain": "JSON boolean; true requires an empty ranked_intervention_ids list",
            "reasoning": "brief scientific rationale; advisory only",
            "confidence": "optional number in [0,1]; advisory only",
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Act as a cautious scientific counterfactual acquisition selector. "
                "Use only the supplied chemistry context. Return one JSON object and no markdown."
            ),
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
        "prompt_version": ONLINE_PROMPT_VERSION,
        "candidate_presentation": ONLINE_PRESENTATION_PROTOCOL,
        "candidate_ids": candidate_ids,
        "presented_candidate_ids": [row["intervention_id"] for row in rows],
        "maximum_budget": int(maximum_budget),
        "messages": messages,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "api_key_exposed": False,
        "candidate_reward_truth_exposed": False,
        "factual_reward_truth_exposed": False,
        "policy_score_exposed": False,
    }


def validate_online_ranked_response(
    value: Any, candidate_ids: Sequence[str], maximum_budget: int
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    allowed = {"ranked_intervention_ids", "abstain", "reasoning", "confidence"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError("LLM response has unknown fields: %s" % unknown)
    ranked = value.get("ranked_intervention_ids")
    abstain = value.get("abstain")
    if not isinstance(ranked, list):
        raise ValueError("ranked_intervention_ids must be a JSON list")
    if not isinstance(abstain, bool):
        raise ValueError("abstain must be a JSON boolean")
    if any(not isinstance(identifier, str) for identifier in ranked):
        raise ValueError("ranked intervention ids must be strings")
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranked intervention ids must be unique")
    invented = sorted(set(ranked) - set(candidate_ids))
    if invented:
        raise ValueError("LLM invented intervention ids: %s" % invented)
    if abstain and ranked:
        raise ValueError("abstention requires an empty ranking")
    if not abstain and not (1 <= len(ranked) <= int(maximum_budget)):
        raise ValueError("a non-abstaining ranking must contain 1..budget ids")
    if len(ranked) > int(maximum_budget):
        raise ValueError("LLM response exceeds the acquisition budget")
    reasoning = value.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise ValueError("reasoning must be a string")
    confidence = value.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must lie in [0,1]")
    return {
        "schema_version": "scicf-online-ranked-interventions-v1",
        "ranked_intervention_ids": list(ranked),
        "selected_intervention_ids": list(ranked),
        "abstain": bool(abstain),
        "reasoning": reasoning,
        "confidence": None if confidence is None else float(confidence),
        "advisory_only": True,
    }
