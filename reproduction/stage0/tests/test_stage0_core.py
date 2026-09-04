from __future__ import annotations

import random
import re

import numpy as np
import pytest

from RL_PPO.envs.chemistry import BlockMetadata
from RL_PPO.envs.config import DAPiGenEnvConfig
from RL_PPO.envs.core import BranchableDAPiGenCore
from RL_PPO.envs.evaluator import (
    BudgetedCachingTerminalEvaluator,
    OracleBudgetExceeded,
    PersistentDAPiGenBenchmarkEvaluator,
    TerminalRewardAdapter,
)
from RL_PPO.envs.runtime import DAPiGenEpisodeController
from RL_PPO.envs.types import DAPiGenAction, DAPiGenState, TerminalEvaluation


class FakeChemistry(object):
    backend_version = "fake-v2"

    def __init__(self):
        self.diagnostics = {}

    def canonicalize(self, smiles: str) -> str:
        if not smiles:
            raise ValueError("empty")
        return smiles

    def atom_count(self, smiles: str) -> int:
        return len(smiles.replace("*", "").replace("!", ""))

    def attachment_count(self, smiles: str) -> int:
        return smiles.count("*")

    def attachment_labels(self, smiles: str):
        labels = [int(value) for value in re.findall(r"\[(\d+)\*\]", smiles)]
        if "*" in smiles and not labels:
            labels = [16]
        return frozenset(labels)

    def attachment_label_histogram(self, smiles: str, maximum_label: int):
        labels = [int(value) for value in re.findall(r"\[(\d+)\*\]", smiles)]
        if "*" in smiles and not labels:
            labels = [16] * smiles.count("*")
        counts = [0] * (int(maximum_label) + 1)
        for label in labels:
            counts[label] += 1
        return tuple(counts)

    def can_react(self, base_smiles: str, block_smiles: str) -> bool:
        return "*" in base_smiles and "*" in block_smiles

    def is_complete_dianhydride(self, smiles: str) -> bool:
        return smiles.startswith("D!") and "*" not in smiles

    def is_complete_diamine(self, smiles: str) -> bool:
        return smiles.startswith("A!") and "*" not in smiles

    def block_metadata(self, action_id: int, smiles: str, side: str):
        if side == "dianhydride":
            complete = self.is_complete_dianhydride(smiles)
        else:
            complete = self.is_complete_diamine(smiles)
        return BlockMetadata(
            action_id=action_id,
            raw_smiles=smiles,
            canonical_smiles=smiles,
            has_attachment="*" in smiles,
            attachment_labels=self.attachment_labels(smiles),
            attachment_count=self.attachment_count(smiles),
            atom_count=self.atom_count(smiles),
            is_complete_for_side=complete,
        )

    def assemble_candidates(self, base_smiles: str, block_smiles: str):
        if "bad" in block_smiles:
            return tuple()
        if "*finishD" == block_smiles:
            return ("D!" + base_smiles.replace("*", ""),)
        if "*finishA" == block_smiles:
            return ("A!" + base_smiles.replace("*", ""),)
        if "*" not in block_smiles:
            return (block_smiles,)
        suffix = block_smiles.replace("*", "")
        return (base_smiles + suffix + "x", base_smiles + suffix + "y")

    def final_polyimide_candidates(self, dianhydride_smiles: str, diamine_smiles: str):
        return (
            "PI-A:%s|%s" % (dianhydride_smiles, diamine_smiles),
            "PI-B:%s|%s" % (dianhydride_smiles, diamine_smiles),
        )


class ConstantEncoder(object):
    encoder_version = "constant-v1"

    def __call__(self, smiles: str):
        return np.asarray([len(smiles), smiles.count("*")], dtype=np.float32)

    def diagnostics(self):
        return {"encoder_version": self.encoder_version, "output_dim": 2}


class CountingEvaluator(object):
    evaluator_version = "counting-v1"

    def __init__(self, reverse=False):
        self.calls = []
        self.reverse = reverse

    def evaluate_batch(self, smiles):
        self.calls.extend(smiles)
        outputs = []
        for item in smiles:
            if self.reverse:
                value = 100.0 if item.startswith("PI-B") else 1.0
            else:
                value = 100.0 if item.startswith("PI-A") else 1.0
            outputs.append(
                TerminalEvaluation(
                    objective=value,
                    canonical_smiles=item,
                    evaluator_version=self.evaluator_version,
                )
            )
        return outputs


def make_core(
    max_steps=4,
    product_selection="seeded_uniform",
    max_atoms=100,
    minimum_growth=0,
    mask_mode="exact_cached",
    invalid_action_handling="terminate",
):
    return BranchableDAPiGenCore(
        dianhydride_blocks=("*grow", "*finishD", "D!direct", "*bad"),
        diamine_blocks=("*grow", "*finishA", "A!direct", "*bad"),
        initial_dianhydride_smiles="D*",
        initial_diamine_smiles="A*",
        chemistry=FakeChemistry(),
        encoder=ConstantEncoder(),
        config=DAPiGenEnvConfig(
            max_atoms=max_atoms,
            max_steps=max_steps,
            product_selection=product_selection,
            mask_mode=mask_mode,
            complete_block_policy="pristine_only",
            minimum_dianhydride_growth_steps=minimum_growth,
            minimum_diamine_growth_steps=minimum_growth,
            observation_mode="augmented_v3",
            invalid_action_handling=invalid_action_handling,
        ),
    )


def test_exact_replay_markov_observation_and_no_global_rng_consumption() -> None:
    core = make_core()
    initial = core.initial(0)
    np.random.seed(991)
    random.seed(992)
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    first = core.transition(initial.state, DAPiGenAction(0, 0), seed=123)
    replay = core.transition(initial.state, DAPiGenAction(0, 0), seed=123)
    numpy_after = np.random.get_state()
    python_after = random.getstate()
    assert first.state == replay.state
    assert first.info["selected_candidate_indices"] == replay.info["selected_candidate_indices"]
    assert np.array_equal(first.observation, replay.observation)
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert python_before == python_after
    # Two 2D embeddings, twelve metadata fields and two label histograms (0..16).
    assert initial.observation.shape == (50,)


def test_state_round_trip_and_stable_identifier() -> None:
    state = make_core().initial(0).state
    restored = DAPiGenState.from_dict(state.to_dict())
    assert restored == state
    assert restored.state_id == state.state_id


def test_augmented_observation_retains_transition_relevant_attachment_labels() -> None:
    core = make_core()
    base = core.initial(0).state
    label_8 = DAPiGenState(
        dianhydride_smiles="D[8*]",
        diamine_smiles=base.diamine_smiles,
        environment_id=core.environment_id,
        dianhydride_complete=False,
        diamine_complete=False,
        dianhydride_growth_steps=0,
        diamine_growth_steps=0,
        step_index=0,
        max_steps=base.max_steps,
    )
    label_16 = DAPiGenState(
        dianhydride_smiles="D[16*]",
        diamine_smiles=base.diamine_smiles,
        environment_id=core.environment_id,
        dianhydride_complete=False,
        diamine_complete=False,
        dianhydride_growth_steps=0,
        diamine_growth_steps=0,
        step_index=0,
        max_steps=base.max_steps,
    )
    assert not np.array_equal(core.observe(label_8), core.observe(label_16))


def test_exact_mask_removes_known_nonproductive_actions() -> None:
    initial = make_core(mask_mode="exact_cached").initial(0)
    assert not initial.action_mask.dianhydride[3]
    assert not initial.action_mask.diamine[3]


def test_mask_audit_exposes_compatibility_false_positives() -> None:
    core = make_core(mask_mode="compatibility")
    comparison = core.compare_compatibility_and_exact_masks(core.initial(0).state)
    assert comparison["compatibility"]["dianhydride"][3]
    assert not comparison["exact"]["dianhydride"][3]


def test_closure_exact_mask_removes_nonproductive_closure_without_full_mask():
    core = make_core(mask_mode="closure_exact_cached")
    initial = core.initial(0)
    comparison = core.compare_compatibility_and_exact_masks(initial.state)
    assert comparison["compatibility"]["dianhydride"][3]
    assert not comparison["refined"]["dianhydride"][3]
    assert comparison["refined"]["dianhydride"][0]
    assert np.array_equal(
        initial.action_mask.dianhydride,
        comparison["refined"]["dianhydride"],
    )


def test_exact_mask_is_independent_enough_to_expose_compatibility_false_negatives():
    class FalseNegativeChemistry(FakeChemistry):
        @staticmethod
        def labels_can_react(base_labels, block_labels):
            return False

    core = BranchableDAPiGenCore(
        dianhydride_blocks=("*grow",),
        diamine_blocks=("*grow",),
        initial_dianhydride_smiles="D*",
        initial_diamine_smiles="A*",
        chemistry=FalseNegativeChemistry(),
        encoder=ConstantEncoder(),
        config=DAPiGenEnvConfig(
            max_atoms=100,
            max_steps=2,
            mask_mode="exact_cached",
            complete_block_policy="never",
            observation_mode="augmented_v3",
            invalid_action_handling="raise",
        ),
    )
    comparison = core.compare_compatibility_and_exact_masks(core.initial(0).state)
    assert not comparison["compatibility"]["dianhydride"][0]
    assert not comparison["refined"]["dianhydride"][0]
    assert comparison["exact"]["dianhydride"][0]


def test_complete_block_cannot_erase_a_non_pristine_prefix() -> None:
    core = make_core()
    initial = core.initial(0)
    assert initial.action_mask.dianhydride[2]
    grown = core.transition(initial.state, DAPiGenAction(0, 0), seed=10)
    assert not grown.action_mask.dianhydride[2]
    assert not grown.action_mask.diamine[2]


def test_minimum_growth_can_create_controlled_long_horizon_variant() -> None:
    core = make_core(minimum_growth=2)
    initial = core.initial(0)
    assert not initial.action_mask.dianhydride[1]
    assert not initial.action_mask.diamine[1]
    grown = core.transition(initial.state, DAPiGenAction(0, 0), seed=10)
    assert grown.action_mask.dianhydride[1]
    assert grown.action_mask.diamine[1]


def test_completed_side_requires_explicit_noop() -> None:
    core = make_core()
    initial = core.initial(0)
    half = core.transition(initial.state, DAPiGenAction(1, 0), seed=5)
    assert half.state.dianhydride_complete
    assert not half.state.diamine_complete
    assert half.state.dianhydride_growth_steps == 1
    assert half.action_mask.dianhydride.tolist() == [False, False, False, False, True]
    completed = core.transition(
        half.state,
        DAPiGenAction(core.dianhydride_noop_id, 1),
        seed=6,
    )
    assert completed.terminated
    assert completed.terminal_smiles is not None


def test_horizon_has_no_released_off_by_one() -> None:
    core = make_core(max_steps=2)
    current = core.initial(0)
    current = core.transition(current.state, DAPiGenAction(0, 0), seed=1)
    assert not current.terminated
    current = core.transition(current.state, DAPiGenAction(0, 0), seed=2)
    assert current.terminated
    assert current.state.step_index == 2
    assert current.state.termination_reason == "design_horizon_exhausted"


def test_terminal_evaluator_is_never_called_for_partial_or_failed_states() -> None:
    core = make_core()
    raw = CountingEvaluator()
    evaluator = BudgetedCachingTerminalEvaluator(raw)
    rewards = TerminalRewardAdapter(evaluator, failure_reward=0.0)
    current = core.initial(0)
    current = core.transition(current.state, DAPiGenAction(0, 0), seed=1)
    assert rewards.apply(current, "ppo").reward == 0.0
    assert raw.calls == []
    failed = core.transition(current.state, DAPiGenAction(2, 2), seed=2)
    assert failed.terminated
    assert rewards.apply(failed, "ppo").reward == 0.0
    assert raw.calls == []


def test_terminal_product_selection_is_reward_independent() -> None:
    core = make_core(product_selection="canonical_first")
    initial = core.initial(0)
    terminal = core.transition(initial.state, DAPiGenAction(1, 1), seed=9)
    assert terminal.terminal_smiles.startswith("PI-A")

    eval_a = BudgetedCachingTerminalEvaluator(CountingEvaluator(reverse=False))
    eval_b = BudgetedCachingTerminalEvaluator(CountingEvaluator(reverse=True))
    reward_a = TerminalRewardAdapter(eval_a).apply(terminal, "a")
    reward_b = TerminalRewardAdapter(eval_b).apply(terminal, "b")
    assert reward_a.core.terminal_smiles == reward_b.core.terminal_smiles
    assert reward_a.core.terminal_smiles.startswith("PI-A")


def test_persistent_objective_uses_frozen_paper_equation() -> None:
    objective = PersistentDAPiGenBenchmarkEvaluator.score_objective(
        transmittance=80.0,
        cte=40.0,
        strength=250.0,
        tg=350.0,
        sa_score=3.5,
    )
    assert objective == 0.48


def test_persistent_batch_failure_is_isolated_to_the_failing_molecule() -> None:
    class Molecule(object):
        def __init__(self, smiles):
            self.smiles = smiles

    class Chemistry(object):
        @staticmethod
        def MolFromSmiles(smiles):
            return Molecule(smiles)

    class IsolatingEvaluator(PersistentDAPiGenBenchmarkEvaluator):
        def __init__(self):
            self.evaluator_version = "isolating-test-v1"

        @staticmethod
        def _chem():
            return Chemistry

        @staticmethod
        def _graph(smiles):
            return smiles

        def _predict(self, output_name, graphs):
            if "bad" in graphs:
                raise RuntimeError("synthetic item failure")
            values = {
                "transmittance": 80.0,
                "cte": 40.0,
                "strength": 250.0,
                "tg": 350.0,
            }
            return np.asarray([values[output_name]] * len(graphs))

        @staticmethod
        def _calculate_sa(molecule):
            return 3.5

    results = IsolatingEvaluator().evaluate_batch(["good", "bad"])
    assert results[0].valid
    assert results[0].objective == 0.48
    assert not results[1].valid
    assert "synthetic item failure" in results[1].failure_reason


def test_oracle_budget_cache_and_source_ledger_are_auditable(tmp_path) -> None:
    raw = CountingEvaluator()
    evaluator = BudgetedCachingTerminalEvaluator(
        raw, maximum_requested_calls=3, maximum_unique_calls=2
    )
    evaluator.evaluate_batch(["PI-A:x"], source="ppo")
    evaluator.evaluate_batch(["PI-A:x"], source="mcc")
    evaluator.evaluate_batch(["PI-B:y"], source="mcc")
    ledger = evaluator.ledger()
    assert ledger["requested_calls"] == 3
    assert ledger["unique_calls"] == 2
    assert ledger["cache_hits"] == 1
    assert ledger["requested_by_source"] == {"ppo": 1, "mcc": 2}
    audit_path = tmp_path / "audit.json"
    evaluator.write_audit(str(audit_path))
    assert audit_path.exists()
    with pytest.raises(OracleBudgetExceeded):
        evaluator.evaluate_batch(["PI-C:z"], source="mcc")


def test_evaluator_rejects_fractional_budget_limits() -> None:
    with pytest.raises(TypeError, match="maximum_requested_calls"):
        BudgetedCachingTerminalEvaluator(
            CountingEvaluator(), maximum_requested_calls=2.5
        )
    with pytest.raises(TypeError, match="maximum_cache_entries"):
        BudgetedCachingTerminalEvaluator(
            CountingEvaluator(), maximum_cache_entries=2.5
        )


def test_invalid_backend_batch_never_partially_populates_cache() -> None:
    class ContractBreakingEvaluator(CountingEvaluator):
        def evaluate_batch(self, smiles):
            values = super(ContractBreakingEvaluator, self).evaluate_batch(smiles)
            values[-1] = TerminalEvaluation(
                objective=1.0,
                canonical_smiles=smiles[-1],
                evaluator_version="wrong-version",
            )
            return values

    evaluator = BudgetedCachingTerminalEvaluator(ContractBreakingEvaluator())
    with pytest.raises(ValueError, match="does not match service version"):
        evaluator.evaluate_batch(["PI-A:x", "PI-B:y"], source="test")
    ledger = evaluator.ledger()
    assert ledger["requested_calls"] == 2
    assert ledger["unique_calls"] == 2
    assert ledger["backend_calls"] == 2
    assert ledger["cache_entries"] == 0
    assert evaluator.audit_events()[-1]["status"] == "invalid_backend_result"


def test_environment_fingerprint_includes_task_and_encoder_semantics() -> None:
    first = make_core(max_steps=4)
    second = make_core(max_steps=4)
    third = make_core(max_steps=5)
    assert first.environment_id == second.environment_id
    assert first.environment_id != third.environment_id


def test_episode_controller_snapshot_restore_and_branch_do_not_mutate_main_state() -> None:
    core = make_core()
    evaluator = BudgetedCachingTerminalEvaluator(CountingEvaluator())
    controller = DAPiGenEpisodeController(core, TerminalRewardAdapter(evaluator), run_seed=5)
    initial = controller.reset()
    first_seed = controller.transition_seed(initial.state)
    main = controller.step(DAPiGenAction(0, 0), source="ppo")
    snapshot = controller.snapshot()
    branched = controller.branch(
        initial.state, DAPiGenAction(1, 1), first_seed, source="policy_cc"
    )
    assert branched.terminated
    assert controller.current.state == main.state
    controller.restore(snapshot)
    assert controller.current.state == main.state


def test_action_parser_rejects_fractional_and_boolean_ids() -> None:
    with pytest.raises(TypeError):
        DAPiGenAction.from_any((1.5, 2))
    with pytest.raises(TypeError):
        DAPiGenAction.from_any((True, 2))
    assert DAPiGenAction.from_any(np.asarray([1, 2])).as_tuple() == (1, 2)


def test_environment_config_rejects_fractional_and_nonfinite_values() -> None:
    with pytest.raises(TypeError, match="max_steps"):
        DAPiGenEnvConfig(max_steps=2.5)
    with pytest.raises(ValueError, match="failure_reward"):
        DAPiGenEnvConfig(failure_reward=float("nan"))


def test_core_rejects_a_nonfinite_policy_observation() -> None:
    class NonfiniteEncoder(ConstantEncoder):
        def __call__(self, smiles):
            return np.asarray([len(smiles), np.nan], dtype=np.float32)

    with pytest.raises(ValueError, match="finite values"):
        BranchableDAPiGenCore(
            dianhydride_blocks=("*grow",),
            diamine_blocks=("*grow",),
            initial_dianhydride_smiles="D*",
            initial_diamine_smiles="A*",
            chemistry=FakeChemistry(),
            encoder=NonfiniteEncoder(),
            config=DAPiGenEnvConfig(
                max_steps=2,
                mask_mode="exact_cached",
                complete_block_policy="never",
            ),
        )


def test_legacy_state_snapshot_is_rejected_instead_of_silently_inferred() -> None:
    payload = make_core().initial(0).state.to_dict()
    payload.pop("schema_version")
    payload.pop("dianhydride_growth_steps")
    payload.pop("diamine_growth_steps")
    with pytest.raises(ValueError, match="cannot be restored exactly"):
        DAPiGenState.from_dict(payload)


def test_successful_terminal_snapshot_restores_selected_product() -> None:
    core = make_core(product_selection="canonical_first")
    terminal = core.transition(
        core.initial(0).state, DAPiGenAction(1, 1), seed=9
    )
    restored = core.restore_transition(terminal.to_snapshot())
    assert restored.state == terminal.state
    assert restored.terminal_smiles == terminal.terminal_smiles
    assert restored.terminal_candidates == terminal.terminal_candidates


def test_successful_terminal_snapshot_rejects_tampered_candidate_set() -> None:
    core = make_core(product_selection="canonical_first")
    terminal = core.transition(
        core.initial(0).state, DAPiGenAction(1, 1), seed=9
    )
    snapshot = terminal.to_snapshot()
    snapshot["terminal_candidates"] = [terminal.terminal_smiles]
    with pytest.raises(ValueError, match="terminal chemistry result"):
        core.restore_transition(snapshot)


def test_snapshot_from_another_environment_is_rejected() -> None:
    source = make_core(product_selection="canonical_first")
    target = make_core(product_selection="seeded_uniform")
    snapshot = source.initial(0).to_snapshot()
    with pytest.raises(ValueError, match="environment_id"):
        target.restore_transition(snapshot)


def test_state_snapshot_rejects_non_boolean_completion_flags() -> None:
    payload = make_core().initial(0).state.to_dict()
    payload["dianhydride_complete"] = "false"
    with pytest.raises(TypeError, match="Boolean"):
        DAPiGenState.from_dict(payload)


def test_state_snapshot_rejects_unfinished_horizon_and_completed_pair() -> None:
    payload = make_core().initial(0).state.to_dict()
    payload["step_index"] = payload["max_steps"]
    with pytest.raises(ValueError, match="max_steps must be finished"):
        DAPiGenState.from_dict(payload)

    payload = make_core().initial(0).state.to_dict()
    payload["dianhydride_smiles"] = "D!direct"
    payload["diamine_smiles"] = "A!direct"
    payload["dianhydride_complete"] = True
    payload["diamine_complete"] = True
    with pytest.raises(ValueError, match="two completed monomers"):
        DAPiGenState.from_dict(payload)


def test_evaluator_checkpoint_restores_cache_budget_and_audit_state() -> None:
    raw = CountingEvaluator()
    evaluator = BudgetedCachingTerminalEvaluator(
        raw,
        maximum_requested_calls=5,
        maximum_unique_calls=2,
        cache_scope="checkpoint-test",
    )
    evaluator.evaluate_batch(["PI-A:x", "PI-A:x"], source="ppo")
    checkpoint = evaluator.state_dict()

    restored_raw = CountingEvaluator()
    restored = BudgetedCachingTerminalEvaluator(
        restored_raw,
        maximum_requested_calls=5,
        maximum_unique_calls=2,
        cache_scope="checkpoint-test",
    )
    restored.load_state_dict(checkpoint)
    assert restored.ledger() == evaluator.ledger()
    restored.evaluate_one("PI-A:x", source="ppo")
    assert restored_raw.calls == []
    assert restored.ledger()["backend_calls"] == 1

    malformed = dict(checkpoint)
    malformed["cache_hits"] = 0
    with pytest.raises(ValueError, match="counters are inconsistent"):
        restored.load_state_dict(malformed)


def test_unique_molecules_and_backend_calls_remain_distinct_after_eviction() -> None:
    raw = CountingEvaluator()
    evaluator = BudgetedCachingTerminalEvaluator(
        raw,
        maximum_unique_calls=2,
        maximum_cache_entries=1,
    )
    evaluator.evaluate_one("PI-A:x", source="ppo")
    evaluator.evaluate_one("PI-B:y", source="ppo")
    evaluator.evaluate_one("PI-A:x", source="ppo")
    ledger = evaluator.ledger()
    assert ledger["unique_calls"] == 2
    assert ledger["backend_calls"] == 3
    with pytest.raises(ValueError, match="provenance"):
        evaluator.reset_ledger(retain_cache=True)


def test_unregistered_evaluator_source_is_rejected() -> None:
    evaluator = BudgetedCachingTerminalEvaluator(
        CountingEvaluator(), allowed_sources=("registered",)
    )
    with pytest.raises(ValueError, match="Unregistered evaluator source"):
        evaluator.evaluate_batch(["PI-A:x"], source="other")
    assert evaluator.ledger()["requested_calls"] == 0


def test_time_limit_truncation_never_invokes_terminal_evaluator() -> None:
    core = BranchableDAPiGenCore(
        dianhydride_blocks=("*grow",),
        diamine_blocks=("*grow",),
        initial_dianhydride_smiles="D*",
        initial_diamine_smiles="A*",
        chemistry=FakeChemistry(),
        encoder=ConstantEncoder(),
        config=DAPiGenEnvConfig(
            max_atoms=100,
            max_steps=1,
            mask_mode="exact_cached",
            complete_block_policy="never",
            horizon_semantics="time_limit_truncation",
            invalid_action_handling="raise",
        ),
    )
    raw = CountingEvaluator()
    reward_adapter = TerminalRewardAdapter(BudgetedCachingTerminalEvaluator(raw))
    transition = core.transition(
        core.initial(0).state, DAPiGenAction(0, 0), seed=1
    )
    assert transition.truncated and not transition.terminated
    evaluated = reward_adapter.apply(transition, source="ppo")
    assert evaluated.reward == 0.0
    assert raw.calls == []


def test_concurrent_duplicate_requests_share_one_transactional_cache_entry() -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    class SlowEvaluator(CountingEvaluator):
        def evaluate_batch(self, smiles):
            time.sleep(0.01)
            return super(SlowEvaluator, self).evaluate_batch(smiles)

    raw = SlowEvaluator()
    evaluator = BudgetedCachingTerminalEvaluator(
        raw,
        maximum_requested_calls=8,
        maximum_unique_calls=1,
    )

    def call_once(_index):
        return evaluator.evaluate_one("PI-A:shared", source="concurrent").objective

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(call_once, range(8)))

    ledger = evaluator.ledger()
    assert values == [100.0] * 8
    assert raw.calls == ["PI-A:shared"]
    assert ledger["requested_calls"] == 8
    assert ledger["unique_calls"] == 1
    assert ledger["backend_calls"] == 1
    assert ledger["cache_hits"] == 7
