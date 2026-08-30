from __future__ import annotations

import copy
import random
import unittest
from pathlib import Path

import numpy as np

from reproduction.framework.contracts import ContractError
from reproduction.scicf.acquisition.base import AcquisitionCandidate
from reproduction.scicf.acquisition.metrics import acquisition_metrics
from reproduction.scicf.acquisition.pool import CandidatePoolBuilder
from reproduction.scicf.acquisition.strategies import (
    HeuristicAcquisition,
    PolicyProbabilityAcquisition,
    RandomAcquisition,
)
from reproduction.scicf.core.config import load_scicf_config, validate_scicf_config
from reproduction.scicf.core.oracle import CountingOracle, OracleLedger
from reproduction.scicf.core.records import Intervention
from reproduction.scicf.domains.dapigen import DAPiGenDomainAdapter


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]


class _Discrete:
    def __init__(self, size):
        self.n = size


class _FakeDAPiGenEnvironment:
    def __init__(self):
        self.action_space_dianhydride = _Discrete(4)
        self.action_space_diamine = _Discrete(3)
        self.building_blocks_dianhydride = ["C", "CC", "CCC", "CCCC"]
        self.building_blocks_diamine = ["N", "CN", "CCN"]
        self.scoring_function = lambda objects: [float(len(item)) for item in objects]
        self.np_random = np.random.RandomState(0)
        self.reset()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self.base_smiles_dianhydride = "C"
        self.base_smiles_diamine = "N"
        self.env_step = 0
        self.flag_dianhydride = False
        self.flag_diamine = False
        self.PI = "None"
        self.transmittance = 0.0
        self.cte = 0.0
        self.strength = 0.0
        self.tg = 0.0
        self.SaScore = 0.0
        return np.array([0.0, 0.0], dtype=np.float32)

    def step(self, action):
        self.prev_smiles_dianhydride = self.base_smiles_dianhydride
        self.prev_smiles_diamine = self.base_smiles_diamine
        self.prev_action = tuple(action)
        self.env_step += 1
        noise = random.random() + float(np.random.random())
        self.base_smiles_dianhydride += str(action[0])
        self.base_smiles_diamine += str(action[1])
        done = self.env_step >= 2
        reward = 0.0
        if done:
            self.flag_dianhydride = True
            self.flag_diamine = True
            scores = self.scoring_function(["polymer-a", "polymer-b", "polymer-c"])
            reward = max(scores) + noise
            self.PI = "polymer-{}-{}".format(action[0], action[1])
            self.transmittance = reward
            self.cte = -reward
            self.strength = reward / 2.0
            self.tg = reward * 2.0
            self.SaScore = 1.0
        observation = np.array([float(self.env_step), noise], dtype=np.float32)
        return observation, reward, done, {"prev_action": tuple(action)}


def _policy(observation, rng):
    return int(rng.randint(0, 4)), int(rng.randint(0, 3))


def _intervention(index):
    return Intervention(
        intervention_id="candidate-{}".format(index),
        trajectory_id="trajectory-1",
        timestep=0,
        component="dianhydride",
        factual_action=(0, 1),
        alternative_action=(index + 1, 1),
        factual_component_value=0,
        alternative_component_value=index + 1,
        alternative_structure="C" * (index + 2),
        metadata={"factual_structure": "C"},
    )


class SciCFCoreTests(unittest.TestCase):
    def test_scaffold_config_is_valid_and_gate1_fails_closed(self):
        config = load_scicf_config(
            REPRODUCTION_ROOT / "configs" / "scicf-gate1-dev-v1.json"
        )
        self.assertFalse(config["pairwise_refinement"]["enabled"])
        config = copy.deepcopy(config)
        config["pairwise_refinement"]["enabled"] = True
        with self.assertRaises(ContractError):
            validate_scicf_config(config)

    def test_oracle_calls_require_scope_and_count_atomic_objects(self):
        ledger = OracleLedger()
        oracle = CountingOracle(lambda values: list(values), ledger)
        with self.assertRaises(RuntimeError):
            oracle([1])
        with ledger.scope("evaluation"):
            self.assertEqual(oracle([1, 2, 3]), [1, 2, 3])
        self.assertEqual(ledger.snapshot().as_dict()["evaluation"], 3)
        self.assertEqual(ledger.snapshot().total, 3)

    def test_dapigen_snapshot_replay_and_identity_intervention(self):
        adapter = DAPiGenDomainAdapter(_FakeDAPiGenEnvironment())
        observation = adapter.reset(19)
        snapshot = adapter.capture_snapshot(observation)
        factual = adapter.continue_from_snapshot(
            snapshot,
            first_action=(1, 1),
            continuation_policy=_policy,
            policy_version="frozen-test-policy",
            continuation_seed=71,
            oracle_scope="factual",
            max_steps=3,
        )
        identity = adapter.continue_from_snapshot(
            snapshot,
            first_action=(1, 1),
            continuation_policy=_policy,
            policy_version="frozen-test-policy",
            continuation_seed=71,
            oracle_scope="counterfactual",
            max_steps=3,
        )
        self.assertEqual(factual.terminal_return, identity.terminal_return)
        self.assertEqual(factual.actions, identity.actions)
        self.assertEqual(factual.terminal_scientific_object, identity.terminal_scientific_object)
        self.assertEqual(factual.atomic_oracle_calls, 3)
        self.assertEqual(identity.atomic_oracle_calls, 3)

    def test_trajectory_contains_snapshots_and_atomic_oracle_count(self):
        adapter = DAPiGenDomainAdapter(_FakeDAPiGenEnvironment())
        record, snapshots = adapter.record_episode(
            policy=_policy,
            policy_version="frozen-test-policy",
            seed=23,
            max_steps=3,
        )
        self.assertEqual(record.environment_transitions, 2)
        self.assertEqual(record.atomic_oracle_calls, 3)
        self.assertEqual(len(record.steps), len(snapshots))
        self.assertTrue(record.steps[-1].terminated)
        self.assertIsNotNone(record.terminal_scientific_object)
        self.assertEqual(
            adapter.enumerate_interventions(record.trajectory_id, 0, (0, 0)), ()
        )
        adapter.restore_snapshot(snapshots[record.steps[0].snapshot_id])
        self.assertGreater(
            len(adapter.enumerate_interventions(record.trajectory_id, 0, (0, 0))), 0
        )

    def test_interventions_are_atomic_and_legal(self):
        adapter = DAPiGenDomainAdapter(_FakeDAPiGenEnvironment())
        interventions = adapter.enumerate_interventions("trajectory-1", 2, (1, 1))
        self.assertEqual(len(interventions), 5)
        self.assertTrue(all(item.factual_action != item.alternative_action for item in interventions))
        for item in interventions:
            applied = adapter.apply_intervention((1, 1), item)
            changed = sum(a != b for a, b in zip((1, 1), applied))
            self.assertEqual(changed, 1)

    def test_acquisition_strategies_share_one_fixed_pool(self):
        candidates = tuple(
            AcquisitionCandidate(
                intervention=_intervention(index),
                policy_score=float(6 - index),
                structural_score=float(index),
                heuristic_score=float(index % 3),
            )
            for index in range(6)
        )
        pool = CandidatePoolBuilder().build(
            source_candidates={
                "policy_near": candidates,
                "random_legal": candidates,
                "structural": candidates,
            },
            source_quotas={"policy_near": 2, "random_legal": 2, "structural": 2},
            requested_size=6,
            seed=11,
        )
        results = [
            RandomAcquisition().select(pool, budget=2, seed=5),
            PolicyProbabilityAcquisition().select(pool, budget=2, seed=5),
            HeuristicAcquisition().select(pool, budget=2, seed=5),
        ]
        self.assertEqual(len(pool.candidate_ids), len(set(pool.candidate_ids)))
        self.assertTrue(all(result.pool_id == pool.pool_id for result in results))
        self.assertTrue(all(result.candidate_ids == pool.candidate_ids for result in results))

    def test_gate1_metrics(self):
        metrics = acquisition_metrics(
            {"a": 3.0, "b": 1.0, "c": -1.0}, ["b", "c"], budget=2
        )
        self.assertEqual(metrics["hit_rate_at_b"], 0.5)
        self.assertEqual(metrics["best_gain_at_b"], 1.0)
        self.assertEqual(metrics["regret_at_b"], 2.0)
        self.assertGreater(metrics["ndcg_at_b"], 0.0)
        self.assertLess(metrics["ndcg_at_b"], 1.0)


if __name__ == "__main__":
    unittest.main()
