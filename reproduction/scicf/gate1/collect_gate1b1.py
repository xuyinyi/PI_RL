#!/usr/bin/env python3
"""Collect one structure-isolated, cross-timestep Gate 1B.1 stage split."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from reproduction.framework.config import load_config as load_framework_config
from reproduction.framework.contracts import AlgorithmContext, ContractError
from reproduction.framework.io import git_identity, write_json
from reproduction.framework.registry import load_adapter_class
from reproduction.run_algorithm import build_task, require_execution_provenance
from reproduction.scicf.acquisition.chemistry import annotate_candidates
from reproduction.scicf.core.oracle import OracleLedger
from reproduction.scicf.core.records import record_to_dict
from reproduction.scicf.core.verifier import PairedCounterfactualVerifier, VerificationConfig
from reproduction.scicf.domains.dapigen import DAPiGenDomainAdapter
from reproduction.scicf.gate1.gate1b1 import (
    STAGES,
    SPLITS,
    build_cross_timestep_pool,
    load_config,
    structure_key,
    structure_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--framework-config", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path)
    return parser.parse_args()


def continuation_seeds(episode_seed: int, timestep: int, replicates: int) -> list:
    return [episode_seed * 10000 + timestep * 100 + 71 + index for index in range(replicates)]


def candidate_to_dict(candidate: Any) -> Dict[str, Any]:
    intervention = candidate.intervention
    return {
        "candidate_id": candidate.candidate_id,
        "structure_key": structure_key(
            intervention.component, intervention.alternative_structure
        ),
        "intervention": record_to_dict(intervention),
        "policy_score": candidate.policy_score,
        "structural_score": candidate.structural_score,
        "heuristic_score": candidate.heuristic_score,
    }


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 1B.1 collection must run through Slurm")
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    framework_config = load_framework_config(args.framework_config.resolve())
    gate_config = load_config(args.gate_config.resolve())
    source = git_identity(repo_root)
    require_execution_provenance(repo_root, framework_config, source)
    if source.get("dirty") is not False:
        raise ContractError("Gate 1B.1 requires a clean Git worktree")
    test_seal = None
    if args.split == "test":
        if args.model_manifest is None:
            raise ValueError("test collection requires the frozen model manifest")
        model_path = args.model_manifest.resolve()
        model_manifest = json.loads(model_path.read_text(encoding="utf-8"))
        if (
            model_manifest.get("gate") != gate_config["gate"]
            or model_manifest.get("dev_entry_passed") is not True
            or model_manifest.get("test_collection_authorized") is not True
            or model_manifest.get("source", {}).get("commit") != source.get("commit")
        ):
            raise RuntimeError("test seal is absent, failed, or belongs to another source")
        test_seal = {
            "model_manifest": str(model_path),
            "model_manifest_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
    elif args.model_manifest is not None:
        raise ValueError("train/dev collection must not receive a model manifest")
    stage_root = args.output_root.resolve() / args.split / args.stage
    if stage_root.exists():
        raise FileExistsError("Gate 1B.1 stage split already exists")
    stage_root.mkdir(parents=True)

    checkpoint_path = (
        args.baseline_run_root.resolve() / gate_config["checkpoints"][args.stage]
    ).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Gate 1B.1 checkpoint is missing")
    task = build_task(repo_root, framework_config)
    resources = {
        "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
        "rollout_workers": 0,
        "gpus": 1,
    }
    context = AlgorithmContext(
        repo_root=repo_root,
        output_root=stage_root / "policy-runtime",
        env_class=task["env_class"],
        env_config=task["env_config"],
        legacy_algorithm_config=task["legacy_algorithm_config"],
        seed=int(gate_config["seeds"][args.split][0]),
        resources=resources,
        slurm_cpus=resources["cpus"],
    )
    context.output_root.mkdir(parents=True)
    policy_adapter = load_adapter_class(framework_config["algorithm"]["adapter"])()
    policy_adapter.initialize(context, framework_config["algorithm"].get("parameters", {}))
    restore_metadata = policy_adapter.restore_policy_weights(str(checkpoint_path))
    policy_version = "{}:{}:{}".format(args.split, args.stage, checkpoint_path.name)

    verification = gate_config["verification"]
    ledger = OracleLedger(max_total=int(verification["atomic_oracle_budget_per_stage_split"]))
    environment = task["env_class"](dict(task["env_config"]))
    domain = DAPiGenDomainAdapter(environment, ledger=ledger)
    verifier = PairedCounterfactualVerifier(
        domain,
        VerificationConfig(
            paired_replicates=int(verification["paired_replicates"]),
            confidence_rule=verification["confidence_rule"],
            max_steps=int(verification["max_steps"]),
            numerical_tolerance=float(verification["numerical_tolerance"]),
            sign_consistency_fraction=float(verification["sign_consistency_fraction"]),
        ),
    )
    pool_config = gate_config["candidate_pool"]
    trajectory_results = []

    def frozen_policy(observation: Any, rng: Any) -> Any:
        return policy_adapter.act(observation, explore=False)

    try:
        for episode_seed in gate_config["seeds"][args.split]:
            trajectory_id = "gate1b1-{}-{}-seed-{}".format(
                args.split, args.stage, episode_seed
            )
            trajectory, snapshots = domain.record_episode(
                policy=frozen_policy,
                policy_version=policy_version,
                seed=int(episode_seed),
                max_steps=int(verification["max_steps"]),
                trajectory_id=trajectory_id,
            )
            universe = []
            universe_counts = {}
            for step in trajectory.steps:
                snapshot = snapshots[step.snapshot_id]
                domain.restore_snapshot(snapshot)
                interventions = domain.enumerate_interventions(
                    trajectory_id=trajectory_id,
                    timestep=int(step.timestep),
                    factual_action=step.factual_action,
                )
                probabilities = policy_adapter.action_component_probabilities(
                    snapshot.observation
                )
                policy_scores = {
                    intervention.intervention_id: float(
                        probabilities[intervention.component][
                            intervention.alternative_component_value
                        ]
                    )
                    for intervention in interventions
                }
                annotated = annotate_candidates(
                    interventions, policy_scores=policy_scores
                )
                filtered = [
                    candidate
                    for candidate in annotated
                    if structure_split(
                        candidate.intervention.component,
                        candidate.intervention.alternative_structure,
                        gate_config,
                    )
                    == args.split
                ]
                if not filtered:
                    raise RuntimeError("structure split removed an entire timestep")
                universe.extend(filtered)
                universe_counts[str(step.timestep)] = len(filtered)
            pool = build_cross_timestep_pool(
                universe,
                source_quotas=pool_config["source_quotas"],
                requested_size=int(pool_config["size"]),
                seed=int(episode_seed),
            )
            verifications = []
            gains = {}
            oracle_calls = {}
            for candidate in pool.candidates:
                timestep = int(candidate.intervention.timestep)
                step = trajectory.steps[timestep]
                result = verifier.verify(
                    intervention=candidate.intervention,
                    snapshot=snapshots[step.snapshot_id],
                    continuation_policy=frozen_policy,
                    policy_version=policy_version,
                    continuation_seeds=continuation_seeds(
                        int(episode_seed), timestep, int(verification["paired_replicates"])
                    ),
                )
                gains[candidate.candidate_id] = result.mean_delta
                oracle_calls[candidate.candidate_id] = result.atomic_oracle_calls
                verifications.append(record_to_dict(result))
            selected_timestep_counts = {}
            for candidate in pool.candidates:
                key = str(candidate.intervention.timestep)
                selected_timestep_counts[key] = selected_timestep_counts.get(key, 0) + 1
            trajectory_results.append(
                {
                    "trajectory": record_to_dict(trajectory),
                    "cross_timestep": {
                        "observed_timesteps": sorted(int(key) for key in universe_counts),
                        "eligible_candidates_by_timestep": universe_counts,
                        "selected_candidates_by_timestep": selected_timestep_counts,
                    },
                    "pool": {
                        "pool_id": pool.pool_id,
                        "requested_size": pool.requested_size,
                        "source_selected_counts": dict(pool.source_selected_counts),
                        "source_shortfalls": dict(pool.source_shortfalls),
                        "candidates": [candidate_to_dict(candidate) for candidate in pool.candidates],
                    },
                    "verified_gains": gains,
                    "oracle_calls_by_candidate": oracle_calls,
                    "verifications": verifications,
                }
            )
            write_json(
                stage_root / "gate1b1-stage-report.partial.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "split": args.split,
                    "stage": args.stage,
                    "source": source,
                    "slurm_job_id": os.environ["SLURM_JOB_ID"],
                    "trajectories": trajectory_results,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "gate1b1_trajectory_complete",
                        "split": args.split,
                        "stage": args.stage,
                        "seed": episode_seed,
                        "timesteps": len(trajectory.steps),
                        "oracle_total": ledger.snapshot().total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        write_json(
            stage_root / "gate1b1-stage-report.json",
            {
                "schema_version": 1,
                "status": "complete",
                "split": args.split,
                "stage": args.stage,
                "seeds": gate_config["seeds"][args.split],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "policy_version": policy_version,
                "policy_restore": restore_metadata,
                "source": source,
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "config": gate_config,
                "test_seal": test_seal,
                "oracle_counts": ledger.snapshot().as_dict(),
                "trajectories": trajectory_results,
            },
        )
    finally:
        policy_adapter.close()


if __name__ == "__main__":
    main()
