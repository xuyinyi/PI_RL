"""Pre-declared offline acquisition metrics for SciCF Gate 1."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


METRIC_DEFINITIONS = {
    "hit_rate_at_b": "fraction of selected interventions with verified delta > 0",
    "best_gain_at_b": "largest verified delta among selected interventions",
    "regret_at_b": "global best verified delta minus selected best verified delta",
    "ndcg_at_b": "NDCG using max(delta, 0) as relevance",
}


def _dcg(relevances: Sequence[float]) -> float:
    return sum(
        (2.0 ** relevance - 1.0) / math.log2(index + 2.0)
        for index, relevance in enumerate(relevances)
    )


def acquisition_metrics(
    verified_gains: Mapping[str, float],
    ranked_selected_ids: Sequence[str],
    budget: int,
) -> Dict[str, float]:
    if budget < 1:
        raise ValueError("budget must be positive")
    if not verified_gains:
        raise ValueError("verified gain table cannot be empty")
    if len(ranked_selected_ids) < budget:
        raise ValueError("ranked selections do not fill the requested budget")
    selected_ids = tuple(ranked_selected_ids[:budget])
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("ranked selections contain duplicate IDs")
    unknown = sorted(set(selected_ids) - set(verified_gains))
    if unknown:
        raise ValueError("selected IDs are missing verified gains: {}".format(unknown))

    selected_gains = [float(verified_gains[identifier]) for identifier in selected_ids]
    if any(not math.isfinite(gain) for gain in selected_gains):
        raise ValueError("verified gains must be finite")
    global_best = max(float(gain) for gain in verified_gains.values())
    selected_best = max(selected_gains)
    relevances = [max(gain, 0.0) for gain in selected_gains]
    ideal_relevances = sorted(
        (max(float(gain), 0.0) for gain in verified_gains.values()), reverse=True
    )[:budget]
    ideal_dcg = _dcg(ideal_relevances)
    return {
        "hit_rate_at_b": sum(gain > 0.0 for gain in selected_gains) / float(budget),
        "best_gain_at_b": selected_best,
        "regret_at_b": global_best - selected_best,
        "ndcg_at_b": _dcg(relevances) / ideal_dcg if ideal_dcg > 0.0 else 0.0,
    }
