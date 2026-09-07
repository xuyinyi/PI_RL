"""K=5 soft Oracle preference aggregation and weighted refinement.

This module is additive.  The frozen integration-v2 strict-pair implementation
remains unchanged and continues to reproduce its archived decision exactly.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from reproduction.p2.contracts import ContractViolation

from .contracts import OnlineCandidate, OnlineVerification, PairwiseRefinementConfig
from .pipeline import _joint_kl, _masked_log_probabilities


@dataclass(frozen=True)
class SoftPairAggregationConfig:
    """Frozen empirical-confidence mapping for five matched continuations."""

    replicates: int = 5
    practical_delta_tolerance: float = 0.005
    beta_prior_alpha: float = 0.5
    magnitude_scale: float = 0.05
    maximum_training_weight: float = 2.0
    minimum_training_weight: float = 0.05
    minimum_effective_pair_mass: float = 0.5
    minimum_weighted_candidate_count: int = 2
    maximum_single_pair_mass_fraction: float = 0.75

    def __post_init__(self) -> None:
        if isinstance(self.replicates, bool) or int(self.replicates) != 5:
            raise ValueError("soft pair aggregation requires exactly K=5")
        for name in (
            "practical_delta_tolerance",
            "beta_prior_alpha",
            "magnitude_scale",
            "maximum_training_weight",
            "minimum_training_weight",
            "minimum_effective_pair_mass",
            "maximum_single_pair_mass_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be finite and positive" % name)
        if self.minimum_training_weight > self.maximum_training_weight:
            raise ValueError("minimum training weight exceeds maximum")
        if not 0.0 < self.maximum_single_pair_mass_fraction <= 1.0:
            raise ValueError("maximum single-pair fraction must lie in (0,1]")
        if (
            isinstance(self.minimum_weighted_candidate_count, bool)
            or int(self.minimum_weighted_candidate_count) < 1
        ):
            raise ValueError("minimum weighted candidate count must be positive")


@dataclass(frozen=True)
class SoftPairEvidence:
    """One verified candidate plus its soft empirical preference evidence."""

    verification: OnlineVerification
    positive_count: int
    negative_count: int
    tie_count: int
    posterior_positive_probability: float
    preferred: Optional[str]
    sign_confidence: float
    nonzero_fraction: float
    median_delta: float
    magnitude_confidence: float
    training_weight: float
    status: str

    def __post_init__(self) -> None:
        counts = (self.positive_count, self.negative_count, self.tie_count)
        if any(isinstance(value, bool) or int(value) < 0 for value in counts):
            raise ValueError("soft-pair counts must be non-negative integers")
        if sum(int(value) for value in counts) != len(self.verification.deltas):
            raise ValueError("soft-pair sign counts do not match replicates")
        for name in (
            "posterior_positive_probability",
            "sign_confidence",
            "nonzero_fraction",
            "magnitude_confidence",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("%s must lie in [0,1]" % name)
        if not math.isfinite(float(self.median_delta)):
            raise ValueError("median delta must be finite")
        if not math.isfinite(float(self.training_weight)) or self.training_weight < 0.0:
            raise ValueError("training weight must be finite and non-negative")
        if self.training_weight > 0.0:
            if self.preferred not in {"factual", "counterfactual"}:
                raise ValueError("weighted soft pair requires a preference")
            if self.status != "weighted_preference":
                raise ValueError("weighted soft pair status mismatch")
        elif self.preferred is not None:
            raise ValueError("zero-weight soft pair must abstain")

    @property
    def candidate_id(self) -> str:
        return self.verification.candidate_id

    @property
    def deltas(self) -> Tuple[float, ...]:
        return self.verification.deltas

    def to_dict(self) -> Dict[str, Any]:
        payload = self.verification.to_dict()
        payload["strict_all_nonzero_same_sign"] = bool(
            self.verification.accepted
        )
        payload["soft_aggregation"] = {
            "positive_count": int(self.positive_count),
            "negative_count": int(self.negative_count),
            "tie_count": int(self.tie_count),
            "posterior_positive_probability": float(
                self.posterior_positive_probability
            ),
            "preferred": self.preferred,
            "sign_confidence": float(self.sign_confidence),
            "nonzero_fraction": float(self.nonzero_fraction),
            "median_delta": float(self.median_delta),
            "magnitude_confidence": float(self.magnitude_confidence),
            "training_weight": float(self.training_weight),
            "status": self.status,
        }
        return payload


def aggregate_soft_verification(
    verification: OnlineVerification,
    config: SoftPairAggregationConfig,
) -> SoftPairEvidence:
    """Convert five matched deltas into a direction and continuous weight."""

    if len(verification.deltas) != int(config.replicates):
        raise ValueError("soft aggregation received a non-K=5 verification")
    tolerance = float(config.practical_delta_tolerance)
    deltas = np.asarray(verification.deltas, dtype=np.float64)
    positive_count = int(np.sum(deltas > tolerance))
    negative_count = int(np.sum(deltas < -tolerance))
    tie_count = int(deltas.size - positive_count - negative_count)
    nonzero_count = positive_count + negative_count
    alpha = float(config.beta_prior_alpha)
    posterior_positive = float(
        (positive_count + alpha) / (nonzero_count + 2.0 * alpha)
    )
    sign_confidence = float(abs(2.0 * posterior_positive - 1.0))
    nonzero_fraction = float(nonzero_count / deltas.size)
    median_delta = float(np.median(deltas))
    magnitude_confidence = float(
        min(abs(median_delta) / float(config.magnitude_scale), 1.0)
    )

    if positive_count > negative_count:
        preferred = "counterfactual"
    elif negative_count > positive_count:
        preferred = "factual"
    else:
        preferred = None

    raw_weight = float(
        float(config.maximum_training_weight)
        * sign_confidence
        * nonzero_fraction
        * magnitude_confidence
    )
    if nonzero_count == 0:
        status = "abstained_all_ties"
        training_weight = 0.0
        preferred = None
    elif preferred is None:
        status = "abstained_direction_tie"
        training_weight = 0.0
    elif raw_weight < float(config.minimum_training_weight):
        status = "abstained_below_minimum_weight"
        training_weight = 0.0
        preferred = None
    else:
        status = "weighted_preference"
        training_weight = min(raw_weight, float(config.maximum_training_weight))

    return SoftPairEvidence(
        verification=verification,
        positive_count=positive_count,
        negative_count=negative_count,
        tie_count=tie_count,
        posterior_positive_probability=posterior_positive,
        preferred=preferred,
        sign_confidence=sign_confidence,
        nonzero_fraction=nonzero_fraction,
        median_delta=median_delta,
        magnitude_confidence=magnitude_confidence,
        training_weight=float(training_weight),
        status=status,
    )


def aggregate_soft_verifications(
    verifications: Sequence[OnlineVerification],
    config: SoftPairAggregationConfig,
) -> Tuple[SoftPairEvidence, ...]:
    return tuple(
        aggregate_soft_verification(item, config) for item in verifications
    )


def soft_pair_gate(
    evidence: Sequence[SoftPairEvidence],
    config: SoftPairAggregationConfig,
) -> Mapping[str, Any]:
    """Decide whether the optional auxiliary update has enough effective mass."""

    identifiers = [item.candidate_id for item in evidence]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("soft-pair evidence contains duplicate candidates")
    weighted = [item for item in evidence if item.training_weight > 0.0]
    total_weight = float(sum(item.training_weight for item in weighted))
    effective_mass = float(total_weight / config.maximum_training_weight)
    maximum_fraction = float(
        max((item.training_weight for item in weighted), default=0.0)
        / total_weight
        if total_weight > 0.0
        else 0.0
    )
    failures = []
    if len(weighted) < int(config.minimum_weighted_candidate_count):
        failures.append("insufficient_weighted_candidate_count")
    if effective_mass < float(config.minimum_effective_pair_mass):
        failures.append("insufficient_effective_pair_mass")
    if maximum_fraction > float(config.maximum_single_pair_mass_fraction):
        failures.append("single_pair_mass_dominates")
    return {
        "eligible_for_optional_update": not failures,
        "weighted_candidate_count": len(weighted),
        "total_training_weight": total_weight,
        "effective_pair_mass": effective_mass,
        "maximum_single_pair_mass_fraction": maximum_fraction,
        "failures": failures,
        "continue_primary_training": True,
        "config": asdict(config),
    }


def refine_soft_pairs(
    *,
    engine,
    candidates_by_id: Mapping[str, OnlineCandidate],
    evidence: Sequence[SoftPairEvidence],
    aggregation_config: SoftPairAggregationConfig,
    refinement_config: PairwiseRefinementConfig,
) -> Mapping[str, Any]:
    """Apply at most one weighted auxiliary step without blocking primary PPO."""

    if not math.isclose(
        float(refinement_config.maximum_weight),
        float(aggregation_config.maximum_training_weight),
    ):
        raise ContractViolation("soft-pair maximum-weight contracts differ")
    if not math.isclose(
        float(refinement_config.delta_tolerance),
        float(aggregation_config.practical_delta_tolerance),
    ):
        raise ContractViolation("soft-pair delta-tolerance contracts differ")
    gate = soft_pair_gate(evidence, aggregation_config)
    policy_version_before = int(engine.policy_version)
    policy_sha256_before = engine.policy_state_sha256
    if not gate["eligible_for_optional_update"]:
        return {
            "status": "skipped_insufficient_soft_pair_mass",
            "optimizer_steps": 0,
            "policy_version_before": policy_version_before,
            "policy_version_after": policy_version_before,
            "policy_sha256_before": policy_sha256_before,
            "policy_sha256_after": policy_sha256_before,
            "continue_primary_training": True,
            "method_observed": "ppo_only_degraded",
            "llm_confidence_used_for_weight": False,
            "oracle_empirical_confidence_used_for_weight": True,
            "counterfactual_actions_in_ppo_clipping": False,
            "soft_pair_gate": gate,
        }

    weighted = [item for item in evidence if item.training_weight > 0.0]
    candidates = [candidates_by_id[item.candidate_id] for item in weighted]
    model_before = copy.deepcopy(engine.model).to(engine.device).eval()
    model_state_before = copy.deepcopy(engine.model.state_dict())
    optimizer_state_before = copy.deepcopy(engine.optimizer.state_dict())
    original_learning_rates = [
        float(group["lr"]) for group in engine.optimizer.param_groups
    ]
    losses = []
    for item, candidate in zip(weighted, candidates):
        d_log, a_log = _masked_log_probabilities(
            engine.model, candidate, engine.device
        )
        index = 0 if candidate.component == "dianhydride" else 1
        component_log = d_log if index == 0 else a_log
        factual_id = candidate.factual_action[index]
        alternative_id = candidate.alternative_action[index]
        margin = component_log[alternative_id] - component_log[factual_id]
        direction = 1.0 if item.preferred == "counterfactual" else -1.0
        losses.append(
            float(item.training_weight)
            * torch.nn.functional.softplus(-direction * margin)
        )
    loss = torch.stack(losses).sum() / float(
        sum(item.training_weight for item in weighted)
    )
    for group in engine.optimizer.param_groups:
        group["lr"] = float(refinement_config.learning_rate)
    engine.model.train()
    engine.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        engine.model.parameters(), float(refinement_config.maximum_gradient_norm)
    )
    gradient_norm_value = float(gradient_norm)
    if not math.isfinite(gradient_norm_value):
        engine.model.load_state_dict(model_state_before, strict=True)
        engine.optimizer.load_state_dict(optimizer_state_before)
        raise ContractViolation("soft pairwise refinement produced a non-finite gradient")
    engine.optimizer.step()
    for group, learning_rate in zip(
        engine.optimizer.param_groups, original_learning_rates
    ):
        group["lr"] = learning_rate
    mean_kl, maximum_kl = _joint_kl(
        model_before, engine.model, candidates, engine.device
    )
    rolled_back = (
        not math.isfinite(mean_kl)
        or mean_kl > float(refinement_config.target_kl)
    )
    if rolled_back:
        engine.model.load_state_dict(model_state_before, strict=True)
        engine.optimizer.load_state_dict(optimizer_state_before)
    else:
        engine.policy_version += 1
    return {
        "status": "rolled_back_for_kl" if rolled_back else "applied",
        "optimizer_steps": 0 if rolled_back else 1,
        "attempted_optimizer_steps": 1,
        "policy_version_before": policy_version_before,
        "policy_version_after": int(engine.policy_version),
        "policy_sha256_before": policy_sha256_before,
        "policy_sha256_after": engine.policy_state_sha256,
        "pairwise_loss": float(loss.item()),
        "gradient_norm": gradient_norm_value,
        "mean_joint_kl": float(mean_kl),
        "maximum_joint_kl": float(maximum_kl),
        "continue_primary_training": True,
        "method_observed": (
            "ppo_plus_scicf_soft_pair" if not rolled_back else "ppo_only_degraded"
        ),
        "llm_confidence_used_for_weight": False,
        "oracle_empirical_confidence_used_for_weight": True,
        "counterfactual_actions_in_ppo_clipping": False,
        "weights": [float(item.training_weight) for item in weighted],
        "candidate_ids": [item.candidate_id for item in weighted],
        "soft_pair_gate": gate,
        "refinement_config": asdict(refinement_config),
    }
