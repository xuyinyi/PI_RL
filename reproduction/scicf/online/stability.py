"""Read-only diagnostics for conservative SciCF pairwise policy updates."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from .contracts import OnlineCandidate, OnlineVerification


def _model_outputs(model, candidate: OnlineCandidate, device):
    observation = torch.as_tensor(
        np.asarray(candidate.observation, dtype=np.float32).copy(),
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)
    d_mask = torch.as_tensor(
        np.asarray(candidate.dianhydride_mask, dtype=bool).copy(),
        dtype=torch.bool,
        device=device,
    ).reshape(1, -1)
    a_mask = torch.as_tensor(
        np.asarray(candidate.diamine_mask, dtype=bool).copy(),
        dtype=torch.bool,
        device=device,
    ).reshape(1, -1)
    d_logits, a_logits, value = model(observation)
    d_minimum = torch.finfo(d_logits.dtype).min
    a_minimum = torch.finfo(a_logits.dtype).min
    return (
        torch.log_softmax(d_logits.masked_fill(~d_mask, d_minimum), dim=-1)[0],
        torch.log_softmax(a_logits.masked_fill(~a_mask, a_minimum), dim=-1)[0],
        value.reshape(-1)[0],
    )


def pairwise_preference_metrics(
    *,
    model,
    candidates_by_id: Mapping[str, OnlineCandidate],
    verifications: Sequence[OnlineVerification],
    device,
) -> Mapping[str, Any]:
    """Measure signed log-probability margins on accepted Oracle pairs."""

    if not verifications:
        raise ValueError("pairwise metrics require at least one verification")
    margins = []
    with torch.no_grad():
        for verification in verifications:
            if not verification.accepted:
                raise ValueError("pairwise metrics accept only stable verified pairs")
            candidate = candidates_by_id[verification.candidate_id]
            d_log, a_log, _value = _model_outputs(model, candidate, device)
            component_log = d_log if candidate.component == "dianhydride" else a_log
            index = 0 if candidate.component == "dianhydride" else 1
            factual = int(candidate.factual_action[index])
            alternative = int(candidate.alternative_action[index])
            raw_margin = component_log[alternative] - component_log[factual]
            direction = 1.0 if verification.mean_delta > 0.0 else -1.0
            margins.append(float((direction * raw_margin).item()))
    if any(not math.isfinite(value) for value in margins):
        raise ValueError("pairwise margins must be finite")
    array = np.asarray(margins, dtype=np.float64)
    losses = np.logaddexp(0.0, -array)
    return {
        "pair_count": int(array.size),
        "preference_accuracy": float(np.mean(array > 0.0)),
        "tie_rate": float(np.mean(array == 0.0)),
        "mean_signed_margin": float(np.mean(array)),
        "minimum_signed_margin": float(np.min(array)),
        "maximum_signed_margin": float(np.max(array)),
        "mean_unweighted_pairwise_loss": float(np.mean(losses)),
    }


def factorized_policy_drift(
    *,
    before_model,
    after_model,
    candidates: Sequence[OnlineCandidate],
    device,
) -> Mapping[str, float]:
    """Measure factor-wise KL and value drift on a fixed candidate support."""

    if not candidates:
        raise ValueError("policy drift requires a non-empty candidate support")
    d_values = []
    a_values = []
    target_values = []
    non_target_values = []
    value_drift = []
    with torch.no_grad():
        for candidate in candidates:
            before_d, before_a, before_value = _model_outputs(
                before_model, candidate, device
            )
            after_d, after_a, after_value = _model_outputs(
                after_model, candidate, device
            )
            d_kl = float(
                torch.sum(torch.exp(before_d) * (before_d - after_d)).item()
            )
            a_kl = float(
                torch.sum(torch.exp(before_a) * (before_a - after_a)).item()
            )
            # Floating-point summation can produce a tiny negative KL.
            d_kl = max(0.0, d_kl)
            a_kl = max(0.0, a_kl)
            d_values.append(d_kl)
            a_values.append(a_kl)
            if candidate.component == "dianhydride":
                target_values.append(d_kl)
                non_target_values.append(a_kl)
            else:
                target_values.append(a_kl)
                non_target_values.append(d_kl)
            value_drift.append(float(torch.abs(after_value - before_value).item()))
    joint = np.asarray(d_values, dtype=np.float64) + np.asarray(
        a_values, dtype=np.float64
    )
    target = np.asarray(target_values, dtype=np.float64)
    non_target = np.asarray(non_target_values, dtype=np.float64)
    values = np.asarray(value_drift, dtype=np.float64)
    return {
        "mean_joint_kl": float(np.mean(joint)),
        "maximum_joint_kl": float(np.max(joint)),
        "mean_dianhydride_kl": float(np.mean(d_values)),
        "maximum_dianhydride_kl": float(np.max(d_values)),
        "mean_diamine_kl": float(np.mean(a_values)),
        "maximum_diamine_kl": float(np.max(a_values)),
        "mean_target_factor_kl": float(np.mean(target)),
        "maximum_target_factor_kl": float(np.max(target)),
        "mean_non_target_factor_kl": float(np.mean(non_target)),
        "maximum_non_target_factor_kl": float(np.max(non_target)),
        "mean_absolute_value_drift": float(np.mean(values)),
        "maximum_absolute_value_drift": float(np.max(values)),
    }
