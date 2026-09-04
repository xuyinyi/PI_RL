#!/usr/bin/env python
"""Stress real-chemistry replay and transition-snapshot restoration."""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", required=True)
    parser.add_argument("--transitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--mask-mode",
        choices=(
            "closure_exact_cached",
            "compatibility",
            "exact_cached",
            "all",
        ),
        default="closure_exact_cached",
    )
    parser.add_argument("--output", default="stage0_audit/replay_restore.json")
    return parser.parse_args()


def assert_transition_equal(left, right):
    assert left.state == right.state
    assert np.array_equal(left.observation, right.observation)
    assert np.array_equal(
        left.action_mask.dianhydride, right.action_mask.dianhydride
    )
    assert np.array_equal(left.action_mask.diamine, right.action_mask.diamine)
    assert left.terminated == right.terminated
    assert left.truncated == right.truncated
    assert left.terminal_smiles == right.terminal_smiles
    assert left.terminal_candidates == right.terminal_candidates


def main():
    args = parse_args()
    if args.transitions <= 0:
        raise ValueError("--transitions must be positive.")

    root = Path(args.dapigen_root).resolve()
    sys.path.insert(0, str(root))

    from RL_PPO.envs.chemistry import LegacyDAPiGenChemistryBackend
    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.core import BranchableDAPiGenCore
    from RL_PPO.envs.embedding import MorganFingerprintEncoder
    from RL_PPO.envs.factory import load_block_catalog
    from RL_PPO.envs.rng import derive_seed, named_index
    from RL_PPO.envs.types import DAPiGenAction

    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=False)
    block_dir = root / "RL_PPO" / "outputs" / "building_blocks"
    d_blocks, _ = load_block_catalog(
        str(block_dir / "blocks_dianhydride.csv"), chemistry=chemistry
    )
    a_blocks, _ = load_block_catalog(
        str(block_dir / "blocks_diamine.csv"), chemistry=chemistry
    )
    core = BranchableDAPiGenCore(
        dianhydride_blocks=d_blocks,
        diamine_blocks=a_blocks,
        initial_dianhydride_smiles="[16*]c1ccc2c(c1)C(=O)OC2=O",
        initial_diamine_smiles="[16*]c1ccc(N)cc1",
        chemistry=chemistry,
        encoder=MorganFingerprintEncoder(radius=2, number_of_bits=256),
        config=DAPiGenEnvConfig(
            max_atoms=60,
            max_steps=5,
            mask_mode=args.mask_mode,
            complete_block_policy="pristine_only",
            product_selection="seeded_uniform",
            observation_mode="augmented_v3",
            horizon_semantics="failure_terminal",
            invalid_action_handling="raise",
        ),
    )

    np.random.seed(991)
    random.seed(992)
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    episode_index = 0
    current = core.initial(seed=derive_seed(args.seed, "episode", episode_index))
    reasons = collections.Counter()
    state_ids = set()
    maximum_step_reached = 0
    successful_terminal_count = 0
    no_product_count = 0
    for transition_index in range(args.transitions):
        if current.state.done:
            episode_index += 1
            current = core.initial(
                seed=derive_seed(args.seed, "episode", episode_index)
            )
        state_ids.add(current.state.state_id)
        d_ids = [
            index
            for index, allowed in enumerate(current.action_mask.dianhydride)
            if allowed
        ]
        a_ids = [
            index
            for index, allowed in enumerate(current.action_mask.diamine)
            if allowed
        ]
        transition_seed = derive_seed(args.seed, "transition", transition_index)
        action = DAPiGenAction(
            d_ids[named_index(transition_seed, "dianhydride_action", len(d_ids))],
            a_ids[named_index(transition_seed, "diamine_action", len(a_ids))],
        )
        first = core.transition(current.state, action, seed=transition_seed)
        replay = core.transition(current.state, action, seed=transition_seed)
        assert_transition_equal(first, replay)
        assert first.info == replay.info

        restored = core.restore_transition(first.to_snapshot())
        assert_transition_equal(first, restored)
        maximum_step_reached = max(maximum_step_reached, first.state.step_index)
        if first.state.termination_reason is not None:
            reasons[first.state.termination_reason] += 1
        successful_terminal_count += int(
            first.state.termination_reason == "success"
        )
        no_product_count += int(
            first.state.termination_reason == "no_reaction_product"
        )
        current = first

    numpy_after = np.random.get_state()
    python_after = random.getstate()
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert python_before == python_after

    report = {
        "status": "passed",
        "requested_transitions": int(args.transitions),
        "replayed_transitions": int(args.transitions),
        "restored_transition_snapshots": int(args.transitions),
        "episode_count": episode_index + 1,
        "distinct_input_state_count": len(state_ids),
        "maximum_step_reached": maximum_step_reached,
        "successful_terminal_count": successful_terminal_count,
        "no_product_count": no_product_count,
        "termination_reasons": dict(sorted(reasons.items())),
        "global_numpy_rng_unchanged": True,
        "global_python_rng_unchanged": True,
        "environment": core.specification(),
        "diagnostics": core.diagnostics(),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
