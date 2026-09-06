"""Candidate, verification and conservative refinement stages for online smoke."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from RL_PPO.envs.rng import derive_seed, uniform01
from RL_PPO.envs.types import DAPiGenState
from reproduction.p2.contracts import (
    CREDIT_ESTIMATOR_CONTRACT_ID,
    SCICF_PPO,
    SCICF_PPO_COUNTERFACTUAL,
    SCICF_PPO_FACTUAL,
    ContractViolation,
    CreditEstimate,
    CreditRequest,
    EvaluatorLedgerDelta,
    PendingLabelBatch,
    PPOUpdateReceipt,
    validate_credit_estimate,
)
from reproduction.p2.engine import torch_state_sha256

from .contracts import OnlineCandidate, OnlineVerification, PairwiseRefinementConfig


class SciCFGAECreditEstimator:
    """Keep the standard PPO update equal to GAE before pairwise refinement."""

    method = SCICF_PPO
    contract_id = CREDIT_ESTIMATOR_CONTRACT_ID
    model_version = 0

    def estimate(self, request: CreditRequest) -> CreditEstimate:
        if request.method != self.method or request.reserved_query_requested_calls != 0:
            raise ContractViolation("SciCF PPO phase must use evaluator-free GAE")
        estimate = CreditEstimate(
            method=self.method,
            source_batch_id=request.rollout.batch_id,
            source_policy_version=request.rollout.frozen_policy.policy_version,
            actor_advantages=request.rollout.gae_advantages,
            credit_model_version_before=self.model_version,
            credit_model_version_after=self.model_version,
            evaluator_delta=EvaluatorLedgerDelta(),
            pending_labels=None,
            same_iteration_labels_committed=False,
            diagnostics={
                "provider": "environment_return_gae_before_scicf_pairwise",
                "evaluator_free": True,
                "counterfactual_actions_in_ppo_clipping": False,
            },
        )
        validate_credit_estimate(request, estimate)
        return estimate

    def commit_after_update(
        self, pending: PendingLabelBatch, receipt: PPOUpdateReceipt
    ) -> Mapping[str, Any]:
        del pending, receipt
        raise ContractViolation("SciCF GAE phase cannot commit pairwise labels")

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "contract_id": self.contract_id,
            "method": self.method,
            "model_version": self.model_version,
            "evaluator_free": True,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if dict(state) != self.state_dict():
            raise ContractViolation("SciCF GAE provider checkpoint identity mismatch")


class FrozenPolicySampler:
    """Read-only sampler used by both matched continuation branches."""

    def __init__(self, model, device: str) -> None:
        self.device = torch.device(device)
        self.model = copy.deepcopy(model).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.state_sha256 = torch_state_sha256(self.model.state_dict())

    @staticmethod
    def _masked(logits, mask):
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def probabilities(
        self, observation: Any, dianhydride_mask: Any, diamine_mask: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        vector = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).copy(),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        d_mask = torch.as_tensor(
            np.asarray(dianhydride_mask, dtype=bool).copy(),
            dtype=torch.bool,
            device=self.device,
        ).reshape(1, -1)
        a_mask = torch.as_tensor(
            np.asarray(diamine_mask, dtype=bool).copy(),
            dtype=torch.bool,
            device=self.device,
        ).reshape(1, -1)
        with torch.no_grad():
            d_logits, a_logits, _value = self.model(vector)
            d = torch.softmax(self._masked(d_logits, d_mask), dim=-1)[0]
            a = torch.softmax(self._masked(a_logits, a_mask), dim=-1)[0]
        return d.cpu().numpy(), a.cpu().numpy()

    @staticmethod
    def _inverse_cdf(probabilities: np.ndarray, uniform: float) -> int:
        cumulative = np.cumsum(np.asarray(probabilities, dtype=np.float64))
        return min(int(np.searchsorted(cumulative, float(uniform), side="right")), len(cumulative) - 1)

    def matched_action(
        self,
        observation: Any,
        dianhydride_mask: Any,
        diamine_mask: Any,
        pair_seed: int,
        continuation_index: int,
    ) -> Tuple[int, int]:
        d, a = self.probabilities(observation, dianhydride_mask, diamine_mask)
        return (
            self._inverse_cdf(
                d, uniform01(pair_seed, "policy", continuation_index, "dianhydride")
            ),
            self._inverse_cdf(
                a, uniform01(pair_seed, "policy", continuation_index, "diamine")
            ),
        )


def _block_row(core, component: str, action_id: int) -> Dict[str, Any]:
    if component == "dianhydride":
        noop_id = int(core.dianhydride_noop_id)
        metadata = core.dianhydride_metadata
    else:
        noop_id = int(core.diamine_noop_id)
        metadata = core.diamine_metadata
    if int(action_id) == noop_id:
        return {
            "action_id": int(action_id),
            "operation": "NOOP",
            "canonical_smiles": None,
            "attachment_labels": [],
            "attachment_count": 0,
            "atom_count": 0,
            "is_complete_for_side": True,
        }
    item = metadata[int(action_id)]
    return {
        "action_id": int(action_id),
        "operation": "select_building_block",
        "canonical_smiles": item.canonical_smiles,
        "attachment_labels": sorted(int(value) for value in item.attachment_labels),
        "attachment_count": int(item.attachment_count),
        "atom_count": int(item.atom_count),
        "is_complete_for_side": bool(item.is_complete_for_side),
    }


def _candidate_identity(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return "cf-" + digest[:16]


def build_online_candidate_pool(
    *,
    rollout,
    core,
    behavior_policy: FrozenPolicySampler,
    pool_size: int,
    seed: int,
) -> Tuple[Sequence[OnlineCandidate], Mapping[str, Any]]:
    """Build one deterministic cross-timestep pool from a successful episode."""

    episodes = defaultdict(list)
    for transition in rollout.transitions:
        episodes[int(transition.episode_id)].append(transition)
    selected = None
    complete_fallback = None
    for episode_id in sorted(episodes):
        rows = sorted(episodes[episode_id], key=lambda item: int(item.timestep))
        complete = bool(rows and rows[0].timestep == 0 and (rows[-1].terminated or rows[-1].truncated))
        successful = bool(rows and "terminal_evaluation" in dict(rows[-1].info))
        cross_timestep = len({int(item.timestep) for item in rows}) >= 2
        if complete and cross_timestep and complete_fallback is None:
            complete_fallback = rows
        if complete and cross_timestep and successful:
            selected = rows
            break
    if selected is None:
        selected = complete_fallback
    if selected is None:
        raise RuntimeError("online smoke found no complete cross-timestep factual episode")

    groups = defaultdict(list)
    for transition in selected:
        if transition.state_snapshot_before is None:
            raise RuntimeError("rollout transition lacks an exact pre-action state")
        state = DAPiGenState.from_dict(transition.state_snapshot_before)
        d_probabilities, a_probabilities = behavior_policy.probabilities(
            transition.observation,
            transition.dianhydride_mask,
            transition.diamine_mask,
        )
        factual = (int(transition.dianhydride_action), int(transition.diamine_action))
        state_context = {
            "timestep": int(transition.timestep),
            "remaining_steps": int(state.remaining_steps),
            "dianhydride_partial_smiles": state.dianhydride_smiles,
            "diamine_partial_smiles": state.diamine_smiles,
            "dianhydride_complete": bool(state.dianhydride_complete),
            "diamine_complete": bool(state.diamine_complete),
            "dianhydride_growth_steps": int(state.dianhydride_growth_steps),
            "diamine_growth_steps": int(state.diamine_growth_steps),
        }
        components = (
            ("dianhydride", 0, transition.dianhydride_mask, d_probabilities),
            ("diamine", 1, transition.diamine_mask, a_probabilities),
        )
        for component, index, mask, probabilities in components:
            for alternative_id in np.flatnonzero(mask):
                alternative_id = int(alternative_id)
                if alternative_id == factual[index]:
                    continue
                alternative = list(factual)
                alternative[index] = alternative_id
                identity = {
                    "protocol": "scicf-online-opaque-candidate-v1",
                    "seed": int(seed),
                    "transition_id": transition.transition_id,
                    "component": component,
                    "factual": factual[index],
                    "alternative": alternative_id,
                }
                candidate = OnlineCandidate(
                    candidate_id=_candidate_identity(identity),
                    transition_id=transition.transition_id,
                    episode_id=int(transition.episode_id),
                    timestep=int(transition.timestep),
                    component=component,
                    factual_action=factual,
                    alternative_action=tuple(alternative),
                    state_snapshot=transition.state_snapshot_before,
                    observation=transition.observation,
                    dianhydride_mask=transition.dianhydride_mask,
                    diamine_mask=transition.diamine_mask,
                    state_context=state_context,
                    factual_block=_block_row(core, component, factual[index]),
                    alternative_block=_block_row(core, component, alternative_id),
                    behavior_policy_probability=float(probabilities[alternative_id]),
                )
                groups[(int(transition.timestep), component)].append(candidate)
    for key, values in groups.items():
        values.sort(
            key=lambda candidate: hashlib.sha256(
                (str(seed) + "|" + candidate.candidate_id).encode("utf-8")
            ).hexdigest()
        )
    ordered_keys = sorted(groups, key=lambda item: (item[0], item[1]))
    pool = []
    cursor = 0
    while len(pool) < int(pool_size) and any(groups[key] for key in ordered_keys):
        key = ordered_keys[cursor % len(ordered_keys)]
        cursor += 1
        if groups[key]:
            pool.append(groups[key].pop(0))
    if len(pool) != int(pool_size):
        raise RuntimeError("candidate pool cannot satisfy the declared size")
    timesteps = sorted({candidate.timestep for candidate in pool})
    if len(timesteps) < 2:
        raise RuntimeError("online candidate pool lacks cross-timestep coverage")
    trajectory_context = {
        "episode_id": int(selected[0].episode_id),
        "step_count": len(selected),
        "candidate_timesteps": timesteps,
        "states_and_factual_actions": [
            {
                "timestep": int(item.timestep),
                "pre_action_state": {
                    key: value
                    for key, value in dict(item.state_snapshot_before).items()
                    if key
                    in {
                        "dianhydride_smiles",
                        "diamine_smiles",
                        "dianhydride_complete",
                        "diamine_complete",
                        "dianhydride_growth_steps",
                        "diamine_growth_steps",
                        "step_index",
                        "max_steps",
                    }
                },
                "factual_action": [
                    int(item.dianhydride_action),
                    int(item.diamine_action),
                ],
            }
            for item in selected
        ],
        "terminal_reward_or_properties_included": False,
        "episode_selection_rule": "first-cross-timestep-successful-episode-else-first-cross-timestep-complete-episode",
    }
    return tuple(pool), trajectory_context


def _terminal_record(evaluated, actions: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    evaluation = None
    if evaluated.evaluation is not None:
        evaluation = evaluated.evaluation.to_dict()
    return {
        "return": float(evaluated.reward),
        "termination_reason": evaluated.state.termination_reason,
        "terminal_smiles": evaluated.core.terminal_smiles,
        "terminal_evaluation": evaluation,
        "actions": [list(value) for value in actions],
        "environment_transitions": len(actions),
    }


def _run_branch(
    *,
    environment,
    policy: FrozenPolicySampler,
    candidate: OnlineCandidate,
    first_action: Tuple[int, int],
    pair_seed: int,
    source: str,
) -> Dict[str, Any]:
    state = DAPiGenState.from_dict(candidate.state_snapshot)
    actions = [tuple(first_action)]
    transition_seed = derive_seed(pair_seed, "chemistry", int(state.step_index))
    evaluated = environment.transition_from(
        state, first_action, transition_seed=transition_seed, source=source
    )
    continuation_index = 0
    while not evaluated.state.done:
        action = policy.matched_action(
            evaluated.observation,
            evaluated.action_mask.dianhydride,
            evaluated.action_mask.diamine,
            pair_seed,
            continuation_index,
        )
        continuation_index += 1
        actions.append(action)
        transition_seed = derive_seed(
            pair_seed, "chemistry", int(evaluated.state.step_index)
        )
        evaluated = environment.transition_from(
            evaluated.state,
            action,
            transition_seed=transition_seed,
            source=source,
        )
        if len(actions) > int(state.remaining_steps):
            raise RuntimeError("matched continuation exceeded the state horizon")
    return _terminal_record(evaluated, actions)


def verify_selected_candidates(
    *,
    environment,
    policy: FrozenPolicySampler,
    candidates_by_id: Mapping[str, OnlineCandidate],
    selected_ids: Sequence[str],
    replicates: int,
    seed: int,
    delta_tolerance: float,
) -> Sequence[OnlineVerification]:
    if int(replicates) < 1:
        raise ValueError("replicates must be positive")
    results = []
    for candidate_id in selected_ids:
        candidate = candidates_by_id[candidate_id]
        paired_seeds = []
        factual_returns = []
        counterfactual_returns = []
        deltas = []
        factual_records = []
        counterfactual_records = []
        for replicate in range(int(replicates)):
            pair_seed = derive_seed(seed, "pair", candidate_id, replicate)
            factual = _run_branch(
                environment=environment,
                policy=policy,
                candidate=candidate,
                first_action=candidate.factual_action,
                pair_seed=pair_seed,
                source=SCICF_PPO_FACTUAL,
            )
            counterfactual = _run_branch(
                environment=environment,
                policy=policy,
                candidate=candidate,
                first_action=candidate.alternative_action,
                pair_seed=pair_seed,
                source=SCICF_PPO_COUNTERFACTUAL,
            )
            factual_return = float(factual["return"])
            counterfactual_return = float(counterfactual["return"])
            paired_seeds.append(pair_seed)
            factual_returns.append(factual_return)
            counterfactual_returns.append(counterfactual_return)
            deltas.append(counterfactual_return - factual_return)
            factual_records.append(factual)
            counterfactual_records.append(counterfactual)
        positive = all(delta > float(delta_tolerance) for delta in deltas)
        negative = all(delta < -float(delta_tolerance) for delta in deltas)
        accepted = bool(positive or negative)
        mean_delta = float(sum(deltas) / len(deltas))
        results.append(
            OnlineVerification(
                candidate_id=candidate_id,
                paired_seeds=tuple(paired_seeds),
                factual_returns=tuple(factual_returns),
                counterfactual_returns=tuple(counterfactual_returns),
                deltas=tuple(deltas),
                accepted=accepted,
                preferred=(
                    "counterfactual"
                    if accepted and mean_delta > 0.0
                    else "factual" if accepted else None
                ),
                rejection_reason=None if accepted else "zero_or_sign_inconsistent_delta",
                factual_terminal_records=tuple(factual_records),
                counterfactual_terminal_records=tuple(counterfactual_records),
            )
        )
    return tuple(results)


def _masked_log_probabilities(model, candidate: OnlineCandidate, device):
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
    d_logits, a_logits, _value = model(observation)
    minimum = torch.finfo(d_logits.dtype).min
    d_log = torch.log_softmax(d_logits.masked_fill(~d_mask, minimum), dim=-1)[0]
    a_log = torch.log_softmax(a_logits.masked_fill(~a_mask, minimum), dim=-1)[0]
    return d_log, a_log


def _joint_kl(pre_model, post_model, candidates, device) -> Tuple[float, float]:
    values = []
    with torch.no_grad():
        for candidate in candidates:
            pre_d, pre_a = _masked_log_probabilities(pre_model, candidate, device)
            post_d, post_a = _masked_log_probabilities(post_model, candidate, device)
            d_kl = torch.sum(torch.exp(pre_d) * (pre_d - post_d))
            a_kl = torch.sum(torch.exp(pre_a) * (pre_a - post_a))
            values.append(float((d_kl + a_kl).item()))
    return float(np.mean(values)), float(np.max(values))


def refine_verified_pairs(
    *,
    engine,
    candidates_by_id: Mapping[str, OnlineCandidate],
    verifications: Sequence[OnlineVerification],
    config: PairwiseRefinementConfig,
) -> Mapping[str, Any]:
    accepted = [item for item in verifications if item.accepted]
    policy_version_before = int(engine.policy_version)
    policy_sha256_before = engine.policy_state_sha256
    if not accepted:
        return {
            "status": "skipped_no_verified_pairs",
            "optimizer_steps": 0,
            "policy_version_before": policy_version_before,
            "policy_version_after": policy_version_before,
            "policy_sha256_before": policy_sha256_before,
            "policy_sha256_after": policy_sha256_before,
            "llm_confidence_used_for_weight": False,
            "counterfactual_actions_in_ppo_clipping": False,
        }

    candidates = [candidates_by_id[item.candidate_id] for item in accepted]
    magnitudes = np.asarray([abs(item.mean_delta) for item in accepted], dtype=np.float64)
    robust_scale = max(float(np.median(magnitudes)), float(config.delta_tolerance))
    weights = np.clip(magnitudes / robust_scale, 0.0, float(config.maximum_weight))
    model_before = copy.deepcopy(engine.model).to(engine.device).eval()
    model_state_before = copy.deepcopy(engine.model.state_dict())
    optimizer_state_before = copy.deepcopy(engine.optimizer.state_dict())
    original_learning_rates = [float(group["lr"]) for group in engine.optimizer.param_groups]
    losses = []
    for verification, candidate, weight in zip(accepted, candidates, weights):
        d_log, a_log = _masked_log_probabilities(engine.model, candidate, engine.device)
        index = 0 if candidate.component == "dianhydride" else 1
        component_log = d_log if index == 0 else a_log
        factual_id = candidate.factual_action[index]
        alternative_id = candidate.alternative_action[index]
        z = component_log[alternative_id] - component_log[factual_id]
        direction = 1.0 if verification.mean_delta > 0.0 else -1.0
        losses.append(float(weight) * torch.nn.functional.softplus(-direction * z))
    loss = torch.stack(losses).mean()
    for group in engine.optimizer.param_groups:
        group["lr"] = float(config.learning_rate)
    engine.model.train()
    engine.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        engine.model.parameters(), float(config.maximum_gradient_norm)
    )
    if not bool(torch.isfinite(gradient_norm).item()):
        engine.model.load_state_dict(model_state_before, strict=True)
        engine.optimizer.load_state_dict(optimizer_state_before)
        raise ContractViolation("pairwise refinement produced a non-finite gradient")
    engine.optimizer.step()
    for group, learning_rate in zip(engine.optimizer.param_groups, original_learning_rates):
        group["lr"] = learning_rate
    mean_kl, maximum_kl = _joint_kl(
        model_before, engine.model, candidates, engine.device
    )
    rolled_back = not math.isfinite(mean_kl) or mean_kl > float(config.target_kl)
    if rolled_back:
        engine.model.load_state_dict(model_state_before, strict=True)
        engine.optimizer.load_state_dict(optimizer_state_before)
    else:
        engine.policy_version += 1
    policy_sha256_after = engine.policy_state_sha256
    return {
        "status": "rolled_back_for_kl" if rolled_back else "applied",
        "optimizer_steps": 0 if rolled_back else 1,
        "attempted_optimizer_steps": 1,
        "pair_count": len(accepted),
        "pairwise_loss": float(loss.item()),
        "gradient_norm": float(gradient_norm.item()),
        "robust_scale": robust_scale,
        "weights": [float(value) for value in weights],
        "target_kl": float(config.target_kl),
        "mean_joint_kl": mean_kl,
        "maximum_joint_kl": maximum_kl,
        "policy_version_before": policy_version_before,
        "policy_version_after": int(engine.policy_version),
        "policy_sha256_before": policy_sha256_before,
        "policy_sha256_after": policy_sha256_after,
        "llm_confidence_used_for_weight": False,
        "oracle_delta_used_for_weight": True,
        "action_component_local_loss": True,
        "counterfactual_actions_in_ppo_clipping": False,
        "config": asdict(config),
    }
