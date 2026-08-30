"""Paired scientific-oracle verification with common random numbers."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from reproduction.scicf.core.domain import PolicyCallable, ScientificDomainAdapter
from reproduction.scicf.core.records import (
    Intervention,
    PairedOutcome,
    VerificationResult,
)


@dataclass(frozen=True)
class VerificationConfig:
    paired_replicates: int
    confidence_rule: str
    max_steps: int
    numerical_tolerance: float = 1e-12
    sign_consistency_fraction: float = 1.0
    interval_z: float = 1.96

    def __post_init__(self) -> None:
        if self.paired_replicates < 1 or self.max_steps < 1:
            raise ValueError("paired_replicates and max_steps must be positive")
        if self.confidence_rule not in {"sign-consistency", "interval-excludes-zero"}:
            raise ValueError("unsupported confidence rule")
        if not 0.5 < self.sign_consistency_fraction <= 1.0:
            raise ValueError("sign_consistency_fraction must be in (0.5, 1]")
        if self.numerical_tolerance < 0.0 or self.interval_z <= 0.0:
            raise ValueError("invalid numerical tolerance or interval_z")


class PairedCounterfactualVerifier:
    """Turn matched continuations into an auditable oracle-derived preference."""

    version = "paired-crn-v1"

    def __init__(
        self, domain: ScientificDomainAdapter, config: VerificationConfig
    ) -> None:
        self.domain = domain
        self.config = config

    def verify(
        self,
        intervention: Intervention,
        snapshot: Any,
        continuation_policy: PolicyCallable,
        policy_version: str,
        continuation_seeds: Sequence[int],
    ) -> VerificationResult:
        if len(continuation_seeds) != self.config.paired_replicates:
            raise ValueError("continuation seed count must equal paired_replicates")
        if len(set(continuation_seeds)) != len(continuation_seeds):
            raise ValueError("continuation seeds must be unique within a verification")

        alternative_action = self.domain.apply_intervention(
            intervention.factual_action, intervention
        )
        outcomes = []
        for replicate, seed in enumerate(continuation_seeds):
            factual = self.domain.continue_from_snapshot(
                snapshot=snapshot,
                first_action=intervention.factual_action,
                continuation_policy=continuation_policy,
                policy_version=policy_version,
                continuation_seed=int(seed),
                oracle_scope="factual",
                max_steps=self.config.max_steps,
            )
            counterfactual = self.domain.continue_from_snapshot(
                snapshot=snapshot,
                first_action=alternative_action,
                continuation_policy=continuation_policy,
                policy_version=policy_version,
                continuation_seed=int(seed),
                oracle_scope="counterfactual",
                max_steps=self.config.max_steps,
            )
            outcomes.append(
                PairedOutcome(
                    replicate=replicate,
                    continuation_seed=int(seed),
                    factual_return=factual.terminal_return,
                    counterfactual_return=counterfactual.terminal_return,
                    factual_oracle_calls=factual.atomic_oracle_calls,
                    counterfactual_oracle_calls=counterfactual.atomic_oracle_calls,
                    factual_terminal_object=factual.terminal_scientific_object,
                    counterfactual_terminal_object=counterfactual.terminal_scientific_object,
                )
            )

        deltas = [outcome.delta for outcome in outcomes]
        mean_delta = statistics.mean(deltas)
        standard_error = (
            statistics.stdev(deltas) / math.sqrt(len(deltas))
            if len(deltas) > 1
            else 0.0
        )
        accepted, rejection_reason = self._confidence_decision(
            deltas, mean_delta, standard_error
        )
        identity = {
            "intervention_id": intervention.intervention_id,
            "policy_version": policy_version,
            "seeds": [int(seed) for seed in continuation_seeds],
            "deltas": deltas,
        }
        verification_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return VerificationResult(
            verification_id=verification_id,
            intervention=intervention,
            policy_version=policy_version,
            paired_outcomes=tuple(outcomes),
            mean_delta=mean_delta,
            uncertainty=standard_error,
            confidence_rule=self.config.confidence_rule,
            accepted_for_learning=accepted,
            rejection_reason=rejection_reason,
        )

    def _confidence_decision(
        self, deltas: Sequence[float], mean_delta: float, standard_error: float
    ) -> tuple:
        tolerance = self.config.numerical_tolerance
        if abs(mean_delta) <= tolerance:
            return False, "mean effect is numerically indistinguishable from zero"
        if self.config.confidence_rule == "sign-consistency":
            preferred_sign = 1 if mean_delta > 0.0 else -1
            consistent = sum(
                1
                for delta in deltas
                if (delta > tolerance and preferred_sign == 1)
                or (delta < -tolerance and preferred_sign == -1)
            )
            fraction = consistent / float(len(deltas))
            if fraction < self.config.sign_consistency_fraction:
                return False, "paired effects fail the configured sign-consistency rule"
            return True, None

        half_width = self.config.interval_z * standard_error
        lower = mean_delta - half_width
        upper = mean_delta + half_width
        if lower <= 0.0 <= upper:
            return False, "effect interval includes zero"
        return True, None
