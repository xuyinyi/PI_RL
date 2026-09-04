#!/usr/bin/env python
"""Validate the Stage-0 chemistry core without polyBERT or QSPR checkpoints."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from RL_PPO.envs.chemistry import LegacyDAPiGenChemistryBackend
from RL_PPO.envs.config import DAPiGenEnvConfig
from RL_PPO.envs.core import BranchableDAPiGenCore
from RL_PPO.envs.embedding import MorganFingerprintEncoder
from RL_PPO.envs.evaluator import (
    BudgetedCachingTerminalEvaluator,
    TerminalRewardAdapter,
)
from RL_PPO.envs.sources import ENVIRONMENT_REGRESSION
from RL_PPO.envs.types import DAPiGenAction, TerminalEvaluation


class DeterministicTestEvaluator(object):
    evaluator_version = "stage0-deterministic-test-v1"
    objective_contract = "synthetic-validation-only-v1"

    def evaluate_batch(self, smiles_batch):
        return [
            TerminalEvaluation(
                objective=float((sum(item.encode("utf-8")) % 1000) / 1000.0),
                properties={"test_length": float(len(item))},
                valid=True,
                canonical_smiles=item,
                evaluator_version=self.evaluator_version,
            )
            for item in smiles_batch
        ]


def main() -> None:
    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=True)
    d_fragment = "[16*]c1ccc2c(c1)C(=O)OC2=O"
    a_fragment = "[16*]c1ccc(N)cc1"
    core = BranchableDAPiGenCore(
        dianhydride_blocks=(d_fragment,),
        diamine_blocks=(a_fragment,),
        initial_dianhydride_smiles=d_fragment,
        initial_diamine_smiles=a_fragment,
        chemistry=chemistry,
        encoder=MorganFingerprintEncoder(radius=2, number_of_bits=256),
        config=DAPiGenEnvConfig(
            max_atoms=100,
            max_steps=2,
            mask_mode="exact_cached",
            complete_block_policy="never",
            product_selection="seeded_uniform",
            observation_mode="augmented_v3",
            invalid_action_handling="raise",
        ),
    )

    initial = core.initial(seed=7)
    action = DAPiGenAction(0, 0)

    np.random.seed(2026)
    random.seed(2027)
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    first = core.transition(initial.state, action, seed=20260903)
    second = core.transition(initial.state, action, seed=20260903)
    numpy_after = np.random.get_state()
    python_after = random.getstate()

    assert first.state == second.state
    assert first.terminal_smiles == second.terminal_smiles
    assert first.info["selected_candidate_indices"] == second.info[
        "selected_candidate_indices"
    ]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert python_before == python_after
    assert first.terminated
    assert first.state.termination_reason == "success"
    assert first.terminal_smiles is not None and "*" not in first.terminal_smiles

    evaluator = BudgetedCachingTerminalEvaluator(
        DeterministicTestEvaluator(),
        maximum_requested_calls=2,
        maximum_unique_calls=1,
        canonicalizer=chemistry.canonicalize,
        allowed_sources=(ENVIRONMENT_REGRESSION,),
        cache_scope="validation_run",
    )
    reward_adapter = TerminalRewardAdapter(evaluator)
    evaluation_1 = reward_adapter.apply(first, source=ENVIRONMENT_REGRESSION)
    evaluation_2 = reward_adapter.apply(second, source=ENVIRONMENT_REGRESSION)
    ledger = evaluator.ledger()
    assert evaluation_1.reward == evaluation_2.reward
    assert ledger["requested_calls"] == 2
    assert ledger["unique_calls"] == 1
    assert ledger["backend_calls"] == 1
    assert ledger["cache_hits"] == 1

    checkpoint = evaluator.state_dict()
    restored_evaluator = BudgetedCachingTerminalEvaluator(
        DeterministicTestEvaluator(),
        maximum_requested_calls=2,
        maximum_unique_calls=1,
        canonicalizer=chemistry.canonicalize,
        allowed_sources=(ENVIRONMENT_REGRESSION,),
        cache_scope="validation_run",
    )
    restored_evaluator.load_state_dict(checkpoint)
    assert restored_evaluator.ledger() == ledger

    restored = core.restore_transition(first.to_snapshot())
    assert restored.state == first.state
    assert restored.terminal_smiles == first.terminal_smiles

    report = {
        "status": "passed",
        "environment": core.specification(),
        "terminal_smiles": first.terminal_smiles,
        "reward": evaluation_1.reward,
        "oracle_ledger": ledger,
        "core_diagnostics": core.diagnostics(),
        "checks": [
            "real RDKit BRICS assembly",
            "real DAPiGen PI reaction chain",
            "state/action/seed replay",
            "no global Python or NumPy RNG consumption",
            "terminal-only evaluation",
            "requested/unique/cache budget accounting",
            "evaluator cache and ledger checkpoint restoration",
            "successful terminal snapshot restoration",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
