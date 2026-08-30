"""Fail-closed validation for LLM-ranked intervention identifiers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


SCHEMA_VERSION = "scicf-ranked-interventions-v1"


def validate_ranked_response(
    value: Any, candidate_ids: Sequence[str], budget: int
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    allowed_fields = {"ranked_intervention_ids", "reasoning", "confidence"}
    unknown_fields = sorted(set(value) - allowed_fields)
    if unknown_fields:
        raise ValueError("LLM response has unknown fields: {}".format(unknown_fields))
    ranked = value.get("ranked_intervention_ids")
    if not isinstance(ranked, list) or len(ranked) < budget:
        raise ValueError("LLM response must rank at least budget interventions")
    if any(not isinstance(identifier, str) for identifier in ranked):
        raise ValueError("ranked intervention IDs must be strings")
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranked intervention IDs must be unique")
    invented = sorted(set(ranked) - set(candidate_ids))
    if invented:
        raise ValueError("LLM invented intervention IDs: {}".format(invented))
    reasoning = value.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise ValueError("reasoning must be a string when supplied")
    confidence = value.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric when supplied")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
    return {
        "schema_version": SCHEMA_VERSION,
        "ranked_intervention_ids": ranked,
        "selected_intervention_ids": ranked[:budget],
        "reasoning": reasoning,
        "confidence": float(confidence) if confidence is not None else None,
        "advisory_only": True,
    }
