import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from RL_PPO.envs.chemistry import BlockMetadata
from reproduction.p2.contracts import (
    CREDIT_ESTIMATOR_CONTRACT_ID,
    METHODS,
    SCICF_PPO,
    CreditRequest,
    FrozenPolicyHandle,
    RolloutBatch,
)
from reproduction.p2.engine import FactorizedActorCritic, torch_state_sha256
from reproduction.scicf.online.contracts import (
    OnlineCandidate,
    OnlineVerification,
    PairwiseRefinementConfig,
)
from reproduction.scicf.online.pipeline import (
    SciCFGAECreditEstimator,
    build_online_candidate_pool,
    eligible_online_episode_ids,
    refine_verified_pairs,
)
from reproduction.scicf.acquisition.metrics import acquisition_metrics_with_abstention
from reproduction.scicf.online.prompt import (
    build_online_acquisition_request,
    validate_online_ranked_response,
)
from reproduction.scicf.online.stability import (
    factorized_policy_drift,
    pairwise_preference_metrics,
)


def _candidate(component="dianhydride", candidate_id="cf-0123456789abcdef"):
    factual = (0, 0)
    alternative = (1, 0) if component == "dianhydride" else (0, 1)
    return OnlineCandidate(
        candidate_id=candidate_id,
        transition_id="transition-0",
        episode_id=0,
        timestep=0,
        component=component,
        factual_action=factual,
        alternative_action=alternative,
        state_snapshot={
            "schema_version": 3,
            "dianhydride_smiles": "[*]C",
            "diamine_smiles": "[*]N",
            "environment_id": "env",
            "dianhydride_complete": False,
            "diamine_complete": False,
            "dianhydride_growth_steps": 0,
            "diamine_growth_steps": 0,
            "step_index": 0,
            "max_steps": 3,
            "terminated": False,
            "truncated": False,
            "termination_reason": None,
        },
        observation=np.asarray([0.1, 0.2, 0.3]),
        dianhydride_mask=np.asarray([True, True, True]),
        diamine_mask=np.asarray([True, True, True]),
        state_context={"dianhydride_partial_smiles": "[*]C"},
        factual_block={"action_id": 0, "canonical_smiles": "C"},
        alternative_block={"action_id": 1, "canonical_smiles": "CC"},
        behavior_policy_probability=0.25,
    )


def test_scicf_is_experimental_without_changing_frozen_three_method_set():
    assert METHODS == ("ppo", "policy_cc", "mcc_ppo")
    assert SCICF_PPO not in METHODS


def test_online_prompt_is_blinded_reward_free_and_policy_score_free():
    candidates = tuple(
        _candidate(candidate_id="cf-%016x" % index) for index in range(1, 5)
    )
    request = build_online_acquisition_request(
        request_id="request",
        trajectory_context={
            "episode_id": 0,
            "states_and_factual_actions": [],
            "terminal_reward_or_properties_included": False,
        },
        candidates=candidates,
        maximum_budget=2,
    )
    user_payload = json.loads(request["messages"][1]["content"])
    serialized = json.dumps(user_payload, sort_keys=True)
    assert "behavior_policy_probability" not in serialized
    assert "terminal_evaluation" not in serialized
    assert "episode_return" not in serialized
    assert request["candidate_reward_truth_exposed"] is False
    assert request["factual_reward_truth_exposed"] is False
    assert request["policy_score_exposed"] is False
    assert set(request["presented_candidate_ids"]) == set(request["candidate_ids"])


def test_online_response_supports_partial_budget_and_explicit_abstention():
    candidate_ids = ["cf-a", "cf-b", "cf-c"]
    partial = validate_online_ranked_response(
        {
            "ranked_intervention_ids": ["cf-b"],
            "abstain": False,
            "reasoning": "one is sufficient",
        },
        candidate_ids,
        2,
    )
    assert partial["selected_intervention_ids"] == ["cf-b"]
    abstain = validate_online_ranked_response(
        {"ranked_intervention_ids": [], "abstain": True}, candidate_ids, 2
    )
    assert abstain["abstain"] is True
    with pytest.raises(ValueError, match="abstention"):
        validate_online_ranked_response(
            {"ranked_intervention_ids": ["cf-a"], "abstain": True},
            candidate_ids,
            2,
        )


def test_scicf_ppo_phase_returns_exact_gae_without_querying():
    frozen = FrozenPolicyHandle(0, "policy", "env", "task")
    rollout = RolloutBatch(
        batch_id="batch",
        environment_id="env",
        task_contract_id="task",
        budget_contract_id="budget",
        frozen_policy=frozen,
        transition_ids=("t0",),
        transitions=(object(),),
        gae_advantages=np.asarray([0.4]),
        critic_returns=np.asarray([0.7]),
    )
    request = CreditRequest(SCICF_PPO, rollout, 0, 0, 17)
    estimator = SciCFGAECreditEstimator()
    estimate = estimator.estimate(request)
    np.testing.assert_array_equal(estimate.actor_advantages, rollout.gae_advantages)
    assert estimate.evaluator_delta.requested_calls == 0
    assert estimate.pending_labels is None
    assert estimator.contract_id == CREDIT_ESTIMATOR_CONTRACT_ID


def _metadata(action_id):
    return BlockMetadata(
        action_id=action_id,
        raw_smiles="[*]C%d" % action_id,
        canonical_smiles="[*]C%d" % action_id,
        has_attachment=True,
        attachment_labels=frozenset({1}),
        attachment_count=1,
        atom_count=2,
        is_complete_for_side=False,
    )


def _state(step):
    return {
        "schema_version": 3,
        "dianhydride_smiles": "[*]C",
        "diamine_smiles": "[*]N",
        "environment_id": "env",
        "dianhydride_complete": False,
        "diamine_complete": False,
        "dianhydride_growth_steps": step,
        "diamine_growth_steps": step,
        "step_index": step,
        "max_steps": 3,
        "terminated": False,
        "truncated": False,
        "termination_reason": None,
    }


class _UniformPolicy:
    def probabilities(self, observation, dianhydride_mask, diamine_mask):
        del observation
        d = np.asarray(dianhydride_mask, dtype=float)
        a = np.asarray(diamine_mask, dtype=float)
        return d / d.sum(), a / a.sum()


def test_candidate_pool_is_opaque_legal_and_cross_timestep():
    transitions = []
    for step in (0, 1):
        transitions.append(
            SimpleNamespace(
                transition_id="t%d" % step,
                episode_id=0,
                timestep=step,
                state_snapshot_before=_state(step),
                observation=np.asarray([step, 0.0, 1.0]),
                dianhydride_mask=np.asarray([True, True, True, True, True]),
                diamine_mask=np.asarray([True, True, True, True, True]),
                dianhydride_action=0,
                diamine_action=0,
                terminated=step == 1,
                truncated=False,
                info={"terminal_evaluation": {"objective": 0.5}} if step == 1 else {},
            )
        )
    core = SimpleNamespace(
        dianhydride_noop_id=4,
        diamine_noop_id=4,
        dianhydride_metadata=tuple(_metadata(index) for index in range(4)),
        diamine_metadata=tuple(_metadata(index) for index in range(4)),
    )
    pool, context = build_online_candidate_pool(
        rollout=SimpleNamespace(transitions=tuple(transitions)),
        core=core,
        behavior_policy=_UniformPolicy(),
        pool_size=8,
        seed=12,
    )
    assert len(pool) == 8
    assert len({item.candidate_id for item in pool}) == 8
    assert {item.timestep for item in pool} == {0, 1}
    assert context["terminal_reward_or_properties_included"] is False
    for item in pool:
        assert item.candidate_id.startswith("cf-")
        changed = sum(
            int(left != right)
            for left, right in zip(item.factual_action, item.alternative_action)
        )
        assert changed == 1


def test_episode_selection_can_be_outcome_blind_and_explicit():
    transitions = []
    for episode, successful in ((0, False), (1, True)):
        for step in (0, 1):
            transitions.append(
                SimpleNamespace(
                    transition_id="e%d-t%d" % (episode, step),
                    episode_id=episode,
                    timestep=step,
                    state_snapshot_before=_state(step),
                    observation=np.asarray([step, 0.0, 1.0]),
                    dianhydride_mask=np.asarray([True, True, True, True, True]),
                    diamine_mask=np.asarray([True, True, True, True, True]),
                    dianhydride_action=0,
                    diamine_action=0,
                    terminated=step == 1,
                    truncated=False,
                    info=(
                        {"terminal_evaluation": {"objective": 0.5}}
                        if step == 1 and successful
                        else {}
                    ),
                )
            )
    rollout = SimpleNamespace(transitions=tuple(transitions))
    assert eligible_online_episode_ids(rollout) == (1, 0)
    assert eligible_online_episode_ids(rollout, prefer_successful=False) == (0, 1)
    core = SimpleNamespace(
        dianhydride_noop_id=4,
        diamine_noop_id=4,
        dianhydride_metadata=tuple(_metadata(index) for index in range(4)),
        diamine_metadata=tuple(_metadata(index) for index in range(4)),
    )
    _pool, context = build_online_candidate_pool(
        rollout=rollout,
        core=core,
        behavior_policy=_UniformPolicy(),
        pool_size=8,
        seed=12,
        episode_id=0,
    )
    assert context["episode_id"] == 0
    assert context["episode_selection_rule"] == (
        "explicit-predeclared-episode-id-without-outcome-filtering"
    )


def test_acquisition_metrics_preserve_abstention_without_padding():
    gains = {"a": 0.8, "b": 0.2, "c": -0.4, "d": -0.8}
    abstained = acquisition_metrics_with_abstention(gains, (), 2)
    assert abstained["selected_count"] == 0
    assert abstained["budget_utilization"] == 0.0
    assert abstained["effective_best_gain_at_b"] == 0.0
    assert abstained["regret_at_b"] == 0.8
    partial = acquisition_metrics_with_abstention(gains, ("b",), 2)
    assert partial["selected_count"] == 1
    assert partial["hit_rate_at_b"] == 0.5
    assert partial["effective_best_gain_at_b"] == 0.2


class _Engine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = FactorizedActorCritic(3, 3, 3, (8,))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.policy_version = 1

    @property
    def policy_state_sha256(self):
        return torch_state_sha256(self.model.state_dict())


def test_verified_pair_refinement_is_separate_and_changes_policy_once():
    torch.manual_seed(5)
    engine = _Engine()
    candidate = _candidate()
    verification = OnlineVerification(
        candidate_id=candidate.candidate_id,
        paired_seeds=(11,),
        factual_returns=(0.2,),
        counterfactual_returns=(0.8,),
        deltas=(0.6,),
        accepted=True,
        preferred="counterfactual",
        rejection_reason=None,
        factual_terminal_records=({},),
        counterfactual_terminal_records=({},),
    )
    before = engine.policy_state_sha256
    receipt = refine_verified_pairs(
        engine=engine,
        candidates_by_id={candidate.candidate_id: candidate},
        verifications=(verification,),
        config=PairwiseRefinementConfig(target_kl=1.0),
    )
    assert receipt["status"] == "applied"
    assert receipt["optimizer_steps"] == 1
    assert receipt["counterfactual_actions_in_ppo_clipping"] is False
    assert receipt["llm_confidence_used_for_weight"] is False
    assert engine.policy_version == 2
    assert engine.policy_state_sha256 != before


def test_pairwise_stability_metrics_are_finite_and_identity_drift_is_zero():
    torch.manual_seed(9)
    model = FactorizedActorCritic(3, 3, 3, (8,))
    candidate = _candidate()
    verification = OnlineVerification(
        candidate_id=candidate.candidate_id,
        paired_seeds=(11, 12),
        factual_returns=(0.2, 0.3),
        counterfactual_returns=(0.8, 0.7),
        deltas=(0.6, 0.4),
        accepted=True,
        preferred="counterfactual",
        rejection_reason=None,
        factual_terminal_records=({}, {}),
        counterfactual_terminal_records=({}, {}),
    )
    metrics = pairwise_preference_metrics(
        model=model,
        candidates_by_id={candidate.candidate_id: candidate},
        verifications=(verification,),
        device=torch.device("cpu"),
    )
    assert metrics["pair_count"] == 1
    assert np.isfinite(metrics["mean_signed_margin"])
    drift = factorized_policy_drift(
        before_model=model,
        after_model=copy.deepcopy(model),
        candidates=(candidate,),
        device=torch.device("cpu"),
    )
    assert drift["maximum_joint_kl"] == 0.0
    assert drift["maximum_non_target_factor_kl"] == 0.0
    assert drift["maximum_absolute_value_drift"] == 0.0
