import copy
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reproduction.p2.contracts import ContractViolation
from reproduction.p2.engine import FactorizedActorCritic, torch_state_sha256
from reproduction.scicf.online.contracts import (
    OnlineCandidate,
    OnlineVerification,
    PairwiseRefinementConfig,
)
from reproduction.scicf.online.resilience import (
    LLMAuxiliaryCircuitBreaker,
    LLMAuxiliaryResilienceConfig,
    RecoverableLLMError,
    finalize_optional_auxiliary,
    run_optional_llm_acquisition,
)
from reproduction.scicf.online.soft_pair import (
    SoftPairAggregationConfig,
    aggregate_soft_verification,
    aggregate_soft_verifications,
    refine_soft_pairs,
    soft_pair_gate,
)


def _verification(candidate_id, deltas):
    tolerance = 0.005
    strict_positive = all(value > tolerance for value in deltas)
    strict_negative = all(value < -tolerance for value in deltas)
    strict = strict_positive or strict_negative
    factual = tuple(0.5 for _ in deltas)
    counterfactual = tuple(0.5 + value for value in deltas)
    mean_delta = sum(deltas) / len(deltas)
    return OnlineVerification(
        candidate_id=candidate_id,
        paired_seeds=tuple(range(10, 10 + len(deltas))),
        factual_returns=factual,
        counterfactual_returns=counterfactual,
        deltas=tuple(deltas),
        accepted=strict,
        preferred=(
            "counterfactual"
            if strict and mean_delta > 0.0
            else "factual" if strict else None
        ),
        rejection_reason=None if strict else "strict_rule_rejected",
        factual_terminal_records=tuple({} for _ in deltas),
        counterfactual_terminal_records=tuple({} for _ in deltas),
    )


def _candidate(candidate_id, component="dianhydride"):
    factual = (0, 0)
    alternative = (1, 0) if component == "dianhydride" else (0, 1)
    return OnlineCandidate(
        candidate_id=candidate_id,
        transition_id="transition-" + candidate_id,
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


class _Engine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = FactorizedActorCritic(3, 3, 3, (8,))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.policy_version = 1

    @property
    def policy_state_sha256(self):
        return torch_state_sha256(self.model.state_dict())


def test_soft_pair_config_requires_exactly_five_replicates():
    assert SoftPairAggregationConfig().replicates == 5
    with pytest.raises(ValueError, match="K=5"):
        SoftPairAggregationConfig(replicates=3)


def test_soft_weights_distinguish_five_four_and_three_vote_support():
    config = SoftPairAggregationConfig()
    records = aggregate_soft_verifications(
        (
            _verification("cf-0000000000000001", (0.10, 0.08, 0.06, 0.04, 0.02)),
            _verification("cf-0000000000000002", (0.08, 0.06, 0.04, 0.02, -0.01)),
            _verification("cf-0000000000000003", (0.03, 0.02, 0.01, -0.01, -0.02)),
        ),
        config,
    )
    assert [item.preferred for item in records] == [
        "counterfactual",
        "counterfactual",
        "counterfactual",
    ]
    assert records[0].training_weight > records[1].training_weight
    assert records[1].training_weight > records[2].training_weight > 0.0
    assert records[0].sign_confidence > records[1].sign_confidence
    assert records[1].sign_confidence > records[2].sign_confidence


def test_zeros_reduce_mass_without_forcing_rejection_and_ties_abstain():
    config = SoftPairAggregationConfig()
    positive_with_ties = aggregate_soft_verification(
        _verification("cf-0000000000000004", (0.04, 0.03, 0.02, 0.0, 0.0)),
        config,
    )
    direction_tie = aggregate_soft_verification(
        _verification("cf-0000000000000005", (0.04, 0.03, -0.02, -0.01, 0.0)),
        config,
    )
    all_ties = aggregate_soft_verification(
        _verification("cf-0000000000000006", (0.0, 0.0, 0.0, 0.0, 0.0)),
        config,
    )
    assert positive_with_ties.preferred == "counterfactual"
    assert positive_with_ties.training_weight > 0.0
    assert positive_with_ties.nonzero_fraction == pytest.approx(0.6)
    assert direction_tie.status == "abstained_direction_tie"
    assert direction_tie.training_weight == 0.0
    assert all_ties.status == "abstained_all_ties"
    assert all_ties.training_weight == 0.0


def test_soft_gate_uses_effective_mass_and_prevents_single_pair_dominance():
    config = SoftPairAggregationConfig()
    evidence = aggregate_soft_verifications(
        (
            _verification("cf-0000000000000001", (0.10, 0.08, 0.06, 0.04, 0.02)),
            _verification("cf-0000000000000002", (-0.08, -0.06, -0.04, -0.02, 0.01)),
        ),
        config,
    )
    gate = soft_pair_gate(evidence, config)
    assert gate["eligible_for_optional_update"] is True
    assert gate["weighted_candidate_count"] == 2
    assert gate["continue_primary_training"] is True
    dominated = soft_pair_gate((evidence[0],), config)
    assert dominated["eligible_for_optional_update"] is False
    assert "insufficient_weighted_candidate_count" in dominated["failures"]
    assert "single_pair_mass_dominates" in dominated["failures"]


def test_soft_refinement_applies_once_and_never_uses_llm_confidence():
    torch.manual_seed(17)
    config = SoftPairAggregationConfig()
    verifications = (
        _verification("cf-0000000000000001", (0.10, 0.08, 0.06, 0.04, 0.02)),
        _verification("cf-0000000000000002", (-0.08, -0.06, -0.04, -0.02, 0.01)),
    )
    evidence = aggregate_soft_verifications(verifications, config)
    candidates = {
        "cf-0000000000000001": _candidate("cf-0000000000000001"),
        "cf-0000000000000002": _candidate(
            "cf-0000000000000002", component="diamine"
        ),
    }
    engine = _Engine()
    value_before = copy.deepcopy(engine.model.value_head.state_dict())
    before = engine.policy_state_sha256
    receipt = refine_soft_pairs(
        engine=engine,
        candidates_by_id=candidates,
        evidence=evidence,
        aggregation_config=config,
        refinement_config=PairwiseRefinementConfig(
            target_kl=1.0, delta_tolerance=0.005
        ),
    )
    assert receipt["status"] == "applied"
    assert receipt["optimizer_steps"] == 1
    assert receipt["method_observed"] == "ppo_plus_scicf_soft_pair"
    assert receipt["llm_confidence_used_for_weight"] is False
    assert receipt["oracle_empirical_confidence_used_for_weight"] is True
    assert receipt["counterfactual_actions_in_ppo_clipping"] is False
    assert engine.policy_version == 2
    assert engine.policy_state_sha256 != before
    for name, value in value_before.items():
        assert torch.equal(value, engine.model.value_head.state_dict()[name])


def test_no_soft_pair_mass_skips_auxiliary_but_keeps_primary_training_alive():
    config = SoftPairAggregationConfig()
    evidence = aggregate_soft_verifications(
        (
            _verification("cf-0000000000000001", (0.0, 0.0, 0.0, 0.0, 0.0)),
            _verification("cf-0000000000000002", (0.0, 0.0, 0.0, 0.0, 0.0)),
        ),
        config,
    )
    engine = _Engine()
    before = engine.policy_state_sha256
    receipt = refine_soft_pairs(
        engine=engine,
        candidates_by_id={},
        evidence=evidence,
        aggregation_config=config,
        refinement_config=PairwiseRefinementConfig(delta_tolerance=0.005),
    )
    assert receipt["status"] == "skipped_insufficient_soft_pair_mass"
    assert receipt["optimizer_steps"] == 0
    assert receipt["continue_primary_training"] is True
    assert receipt["method_observed"] == "ppo_only_degraded"
    assert engine.policy_state_sha256 == before


def test_recoverable_llm_failures_open_circuit_without_stopping_ppo(tmp_path):
    config = LLMAuxiliaryResilienceConfig(
        maximum_consecutive_llm_failures=2,
        circuit_cooldown_iterations=3,
    )
    breaker = LLMAuxiliaryCircuitBreaker(config)
    checkpoint_path = tmp_path / "ppo-checkpoint.pt"
    checkpoint_path.write_bytes(b"committed-primary-ppo")
    checkpoint = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    first = run_optional_llm_acquisition(
        iteration=1,
        primary_ppo_checkpoint_path=checkpoint_path,
        primary_ppo_checkpoint_sha256=checkpoint,
        breaker=breaker,
        acquire=lambda: (_ for _ in ()).throw(
            RecoverableLLMError("provider_timeout")
        ),
    )
    second = run_optional_llm_acquisition(
        iteration=2,
        primary_ppo_checkpoint_path=checkpoint_path,
        primary_ppo_checkpoint_sha256=checkpoint,
        breaker=breaker,
        acquire=lambda: (_ for _ in ()).throw(
            RecoverableLLMError("schema_exhausted")
        ),
    )
    third = run_optional_llm_acquisition(
        iteration=3,
        primary_ppo_checkpoint_path=checkpoint_path,
        primary_ppo_checkpoint_sha256=checkpoint,
        breaker=breaker,
        acquire=lambda: (_ for _ in ()).throw(AssertionError("must not call")),
    )
    assert first["continue_training"] is True
    assert second["continue_training"] is True
    assert second["circuit_breaker"]["open_until_iteration"] == 6
    assert third["auxiliary_status"] == "skipped_circuit_open"
    assert third["method_observed"] == "ppo_only_degraded"

    recovered = run_optional_llm_acquisition(
        iteration=6,
        primary_ppo_checkpoint_path=checkpoint_path,
        primary_ppo_checkpoint_sha256=checkpoint,
        breaker=breaker,
        acquire=lambda: {"status": "validated", "payload": {"ids": ["cf-a"]}},
    )
    assert recovered["auxiliary_status"] == "ready_for_oracle_verification"
    assert recovered["circuit_breaker"]["consecutive_llm_failures"] == 0


def test_valid_abstention_and_no_pair_mass_are_explicit_degraded_iterations(tmp_path):
    breaker = LLMAuxiliaryCircuitBreaker(LLMAuxiliaryResilienceConfig())
    checkpoint_path = tmp_path / "ppo-checkpoint.pt"
    checkpoint_path.write_bytes(b"committed-primary-ppo")
    checkpoint = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    acquisition = run_optional_llm_acquisition(
        iteration=1,
        primary_ppo_checkpoint_path=checkpoint_path,
        primary_ppo_checkpoint_sha256=checkpoint,
        breaker=breaker,
        acquire=lambda: {"status": "abstained", "payload": {}},
    )
    final = finalize_optional_auxiliary(acquisition, None)
    assert final["continue_training"] is True
    assert final["method_observed"] == "ppo_only_degraded"
    assert final["silent_fallback_used"] is False

    ready = run_optional_llm_acquisition(
        iteration=2,
        primary_ppo_checkpoint_path=checkpoint_path,
        primary_ppo_checkpoint_sha256=checkpoint,
        breaker=breaker,
        acquire=lambda: {"status": "validated", "payload": {}},
    )
    no_mass = finalize_optional_auxiliary(
        ready,
        {
            "status": "skipped_insufficient_soft_pair_mass",
            "continue_primary_training": True,
        },
    )
    assert no_mass["continue_training"] is True
    assert no_mass["auxiliary_status"] == "skipped_after_verification"
    assert no_mass["method_observed"] == "ppo_only_degraded"


def test_integrity_and_programming_errors_remain_fail_closed(tmp_path):
    breaker = LLMAuxiliaryCircuitBreaker(LLMAuxiliaryResilienceConfig())
    checkpoint_path = tmp_path / "ppo-checkpoint.pt"
    checkpoint_path.write_bytes(b"committed-primary-ppo")
    checkpoint = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    with pytest.raises(ContractViolation, match="checkpoint"):
        run_optional_llm_acquisition(
            iteration=1,
            primary_ppo_checkpoint_path=tmp_path / "missing.pt",
            primary_ppo_checkpoint_sha256="missing",
            breaker=breaker,
            acquire=lambda: {"status": "validated", "payload": {}},
        )
    with pytest.raises(ContractViolation, match="unvalidated"):
        run_optional_llm_acquisition(
            iteration=1,
            primary_ppo_checkpoint_path=checkpoint_path,
            primary_ppo_checkpoint_sha256=checkpoint,
            breaker=breaker,
            acquire=lambda: {"status": "invalid", "payload": {}},
        )
    with pytest.raises(RuntimeError, match="programming bug"):
        run_optional_llm_acquisition(
            iteration=1,
            primary_ppo_checkpoint_path=checkpoint_path,
            primary_ppo_checkpoint_sha256=checkpoint,
            breaker=breaker,
            acquire=lambda: (_ for _ in ()).throw(RuntimeError("programming bug")),
        )
