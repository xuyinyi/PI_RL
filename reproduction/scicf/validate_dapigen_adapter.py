#!/usr/bin/env python3
"""Validate SciCF wrapping against the real DAPiGen runtime under Slurm."""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from reproduction.framework.config import load_config
from reproduction.framework.io import git_identity, write_json
from reproduction.run_algorithm import build_task, require_execution_provenance
from reproduction.scicf.domains.dapigen import DAPiGenDomainAdapter


KNOWN_VALID_ACTION = (173, 952)
BASE_SEED = 2023
REPLAY_SEED = 3107


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--framework-config",
        type=Path,
        default=Path("reproduction/configs/ppo-compat-framework-smoke-v1.json"),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def terminal_properties(environment: Any) -> Dict[str, float]:
    return {
        "transmittance": float(environment.transmittance),
        "cte": float(environment.cte),
        "strength": float(environment.strength),
        "tg": float(environment.tg),
        "sa_score": float(environment.SaScore),
    }


def direct_rollout(environment: Any, seed: int) -> Mapping[str, Any]:
    seed_everything(seed)
    if hasattr(environment, "seed"):
        environment.seed(seed)
    environment.reset()
    _, reward, done, _ = environment.step(KNOWN_VALID_ACTION)
    if not done or environment.PI in (None, "None"):
        raise RuntimeError("archived known-valid action did not produce a terminal polymer")
    return {
        "terminal_object": str(environment.PI),
        "terminal_return": float(reward),
        "terminal_properties": terminal_properties(environment),
    }


class _SingleActionPolicy:
    def __init__(self, action: Tuple[int, int]) -> None:
        self.action = action
        self.calls = 0

    def __call__(self, observation: Any, rng: np.random.RandomState) -> Tuple[int, int]:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("known-valid regression trajectory unexpectedly needs another action")
        return self.action


def _unused_policy(observation: Any, rng: np.random.RandomState) -> Tuple[int, int]:
    raise RuntimeError("known-valid identity replay unexpectedly needs continuation")


def assert_same_rollout(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    if left["terminal_object"] != right["terminal_object"]:
        raise AssertionError("SciCF wrapper changed the terminal polymer")
    if not math.isclose(
        float(left["terminal_return"]),
        float(right["terminal_return"]),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise AssertionError("SciCF wrapper changed the terminal reward")
    for key, value in left["terminal_properties"].items():
        if not math.isclose(
            float(value),
            float(right["terminal_properties"][key]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise AssertionError("SciCF wrapper changed property {}".format(key))


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real DAPiGen SciCF validation must run through Slurm")
    repo_root = args.repo_root.resolve()
    config_path = (
        args.framework_config
        if args.framework_config.is_absolute()
        else repo_root / args.framework_config
    )
    config = load_config(config_path)
    source = git_identity(repo_root)
    require_execution_provenance(repo_root, config, source)
    task = build_task(repo_root, config)

    direct_environment = task["env_class"](dict(task["env_config"]))
    direct = direct_rollout(direct_environment, BASE_SEED)

    wrapped_environment = task["env_class"](dict(task["env_config"]))
    adapter = DAPiGenDomainAdapter(wrapped_environment)
    trajectory, snapshots = adapter.record_episode(
        policy=_SingleActionPolicy(KNOWN_VALID_ACTION),
        policy_version="known-valid-archived-action-v1",
        seed=BASE_SEED,
        max_steps=1,
        trajectory_id="baseline-wrapper-regression-v1",
    )
    wrapped = {
        "terminal_object": trajectory.terminal_scientific_object,
        "terminal_return": trajectory.episode_return,
        "terminal_properties": trajectory.terminal_properties,
    }
    assert_same_rollout(direct, wrapped)
    if trajectory.atomic_oracle_calls < 1:
        raise AssertionError("terminal scientific objects were not counted")

    first_step = trajectory.steps[0]
    snapshot = snapshots[first_step.snapshot_id]
    factual = adapter.continue_from_snapshot(
        snapshot=snapshot,
        first_action=KNOWN_VALID_ACTION,
        continuation_policy=_unused_policy,
        policy_version="known-valid-archived-action-v1",
        continuation_seed=REPLAY_SEED,
        oracle_scope="factual",
        max_steps=1,
    )
    identity = adapter.continue_from_snapshot(
        snapshot=snapshot,
        first_action=KNOWN_VALID_ACTION,
        continuation_policy=_unused_policy,
        policy_version="known-valid-archived-action-v1",
        continuation_seed=REPLAY_SEED,
        oracle_scope="counterfactual",
        max_steps=1,
    )
    identity_delta = identity.terminal_return - factual.terminal_return
    if not math.isclose(identity_delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("identity intervention produced non-zero effect")
    if factual.terminal_scientific_object != identity.terminal_scientific_object:
        raise AssertionError("identity replay changed the terminal scientific object")
    if factual.atomic_oracle_calls != identity.atomic_oracle_calls:
        raise AssertionError("paired identity branches used different oracle counts")

    adapter.restore_snapshot(snapshot)
    interventions = adapter.enumerate_interventions(
        trajectory_id=trajectory.trajectory_id,
        timestep=0,
        factual_action=KNOWN_VALID_ACTION,
    )
    if not interventions:
        raise AssertionError("DAPiGen adapter enumerated no legal interventions")
    if any(
        sum(left != right for left, right in zip(item.factual_action, item.alternative_action))
        != 1
        for item in interventions
    ):
        raise AssertionError("DAPiGen adapter emitted a non-atomic intervention")

    report = {
        "schema_version": 1,
        "status": "passed",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "source": source,
        "baseline_tag": config["provenance"]["baseline_tag"],
        "known_valid_action": KNOWN_VALID_ACTION,
        "direct": direct,
        "wrapped": wrapped,
        "trajectory": {
            "id": trajectory.trajectory_id,
            "steps": trajectory.environment_transitions,
            "atomic_oracle_calls": trajectory.atomic_oracle_calls,
            "snapshot_ids": sorted(snapshots),
        },
        "identity_replay": {
            "continuation_seed": REPLAY_SEED,
            "delta": identity_delta,
            "factual_oracle_calls": factual.atomic_oracle_calls,
            "counterfactual_oracle_calls": identity.atomic_oracle_calls,
            "terminal_object_equal": True,
        },
        "candidate_enumeration": {
            "count": len(interventions),
            "all_atomic": True,
        },
        "oracle_ledger": adapter.ledger.snapshot().as_dict(),
    }
    write_json(args.output.resolve(), report)
    print("SciCF DAPiGen adapter validation passed: {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
