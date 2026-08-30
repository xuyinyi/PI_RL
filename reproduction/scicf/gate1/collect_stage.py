#!/usr/bin/env python3
"""Collect one fixed-checkpoint SciCF Gate 1 dataset under Slurm."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from reproduction.framework.config import load_config
from reproduction.framework.contracts import AlgorithmContext, ContractError
from reproduction.framework.io import append_jsonl, git_identity, write_json
from reproduction.framework.registry import load_adapter_class
from reproduction.run_algorithm import build_task, require_execution_provenance
from reproduction.scicf.acquisition.chemistry import annotate_candidates
from reproduction.scicf.acquisition.pool import CandidatePoolBuilder
from reproduction.scicf.acquisition.strategies import (
    HeuristicAcquisition,
    PolicyProbabilityAcquisition,
    RandomAcquisition,
)
from reproduction.scicf.core.config import load_scicf_config
from reproduction.scicf.core.oracle import OracleLedger
from reproduction.scicf.core.records import record_to_dict
from reproduction.scicf.core.verifier import (
    PairedCounterfactualVerifier,
    VerificationConfig,
)
from reproduction.scicf.domains.dapigen import DAPiGenDomainAdapter
from reproduction.scicf.llm.prompt import build_acquisition_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--framework-config", type=Path, required=True)
    parser.add_argument("--scicf-config", type=Path, required=True)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("early", "middle", "late"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def continuation_seeds(episode_seed: int, replicates: int) -> list:
    return [episode_seed * 1000 + 701 + index for index in range(replicates)]


def candidate_to_dict(candidate: Any) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "intervention": record_to_dict(candidate.intervention),
        "policy_score": candidate.policy_score,
        "structural_score": candidate.structural_score,
        "heuristic_score": candidate.heuristic_score,
    }


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 1 collection must run through Slurm")
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    framework_config = load_config(args.framework_config.resolve())
    scicf_config = load_scicf_config(args.scicf_config.resolve())
    source = git_identity(repo_root)
    require_execution_provenance(repo_root, framework_config, source)
    if source.get("dirty") is not False:
        raise ContractError("Gate 1 requires a clean Git worktree")
    stage_root = args.output_root.resolve() / args.stage
    if stage_root.exists():
        raise FileExistsError("stage output already exists: {}".format(stage_root))
    stage_root.mkdir(parents=True)

    checkpoint_relative = scicf_config["gate1"]["checkpoints"][args.stage]
    checkpoint_path = (args.baseline_run_root.resolve() / checkpoint_relative).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("checkpoint is missing: {}".format(checkpoint_path))
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
        seed=int(scicf_config["gate1"]["seeds"][0]),
        resources=resources,
        slurm_cpus=resources["cpus"],
    )
    context.output_root.mkdir(parents=True)
    adapter_class = load_adapter_class(framework_config["algorithm"]["adapter"])
    policy_adapter = adapter_class()
    policy_adapter.initialize(context, framework_config["algorithm"].get("parameters", {}))
    restore_metadata = policy_adapter.restore_policy_weights(str(checkpoint_path))
    policy_version = "{}:{}".format(args.stage, checkpoint_path.name)

    ledger = OracleLedger(
        max_total=int(scicf_config["verification"]["atomic_oracle_budget"])
    )
    environment = task["env_class"](dict(task["env_config"]))
    domain = DAPiGenDomainAdapter(environment, ledger=ledger)
    verification_config = VerificationConfig(
        paired_replicates=int(scicf_config["verification"]["paired_replicates"]),
        confidence_rule=scicf_config["verification"]["confidence_rule"],
        max_steps=int(scicf_config["verification"]["max_steps"]),
        numerical_tolerance=float(
            scicf_config["verification"]["numerical_tolerance"]
        ),
        sign_consistency_fraction=float(
            scicf_config["verification"]["sign_consistency_fraction"]
        ),
    )
    verifier = PairedCounterfactualVerifier(domain, verification_config)
    pool_builder = CandidatePoolBuilder()
    budget = int(scicf_config["acquisition"]["budget"])
    requested_size = int(scicf_config["acquisition"]["pool_size"])
    source_quotas = scicf_config["acquisition"]["source_quotas"]
    decision_timestep = int(scicf_config["gate1"]["decision_timestep"])
    requests_path = stage_root / "llm-requests.jsonl"
    trajectory_results = []

    def frozen_policy(observation: Any, rng: Any) -> Any:
        return policy_adapter.act(observation, explore=False)

    try:
        for episode_seed in scicf_config["gate1"]["seeds"]:
            trajectory_id = "gate1-{}-seed-{}".format(args.stage, episode_seed)
            trajectory, snapshots = domain.record_episode(
                policy=frozen_policy,
                policy_version=policy_version,
                seed=int(episode_seed),
                max_steps=int(scicf_config["verification"]["max_steps"]),
                trajectory_id=trajectory_id,
            )
            step = trajectory.steps[decision_timestep]
            snapshot = snapshots[step.snapshot_id]
            domain.restore_snapshot(snapshot)
            interventions = domain.enumerate_interventions(
                trajectory_id=trajectory_id,
                timestep=decision_timestep,
                factual_action=step.factual_action,
            )
            component_probabilities = policy_adapter.action_component_probabilities(
                snapshot.observation
            )
            policy_scores = {}
            for intervention in interventions:
                policy_scores[intervention.intervention_id] = float(
                    component_probabilities[intervention.component][
                        intervention.alternative_component_value
                    ]
                )
            candidates = annotate_candidates(interventions, policy_scores=policy_scores)
            pool = pool_builder.build(
                source_candidates={
                    "policy_near": candidates,
                    "random_legal": candidates,
                    "structural": candidates,
                },
                source_quotas=source_quotas,
                requested_size=requested_size,
                seed=int(episode_seed),
            )
            if pool.shortfall:
                raise RuntimeError("fixed candidate pool has shortfall {}".format(pool.shortfall))
            request_id = "{}:{}".format(args.stage, trajectory_id)
            llm_request = build_acquisition_request(
                request_id=request_id,
                trajectory=trajectory,
                decision_timestep=decision_timestep,
                pool=pool,
                budget=budget,
            )
            append_jsonl(requests_path, llm_request)

            strategies = {}
            for strategy in (
                RandomAcquisition(),
                PolicyProbabilityAcquisition(),
                HeuristicAcquisition(),
            ):
                result = strategy.select(pool, budget=budget, seed=int(episode_seed))
                strategies[result.strategy] = dataclasses.asdict(result)

            verifications = []
            gain_table = {}
            oracle_calls_by_candidate = {}
            seeds = continuation_seeds(
                int(episode_seed), verification_config.paired_replicates
            )
            for candidate in pool.candidates:
                result = verifier.verify(
                    intervention=candidate.intervention,
                    snapshot=snapshot,
                    continuation_policy=frozen_policy,
                    policy_version=policy_version,
                    continuation_seeds=seeds,
                )
                gain_table[candidate.candidate_id] = result.mean_delta
                oracle_calls_by_candidate[candidate.candidate_id] = result.atomic_oracle_calls
                verifications.append(record_to_dict(result))

            trajectory_results.append(
                {
                    "trajectory": record_to_dict(trajectory),
                    "decision_timestep": decision_timestep,
                    "snapshot_id": snapshot.snapshot_id,
                    "pool": {
                        "pool_id": pool.pool_id,
                        "requested_size": pool.requested_size,
                        "source_selected_counts": dict(pool.source_selected_counts),
                        "source_shortfalls": dict(pool.source_shortfalls),
                        "candidates": [candidate_to_dict(item) for item in pool.candidates],
                    },
                    "llm_request": {
                        key: value
                        for key, value in llm_request.items()
                        if key != "messages"
                    },
                    "strategies": strategies,
                    "verified_gains": gain_table,
                    "oracle_calls_by_candidate": oracle_calls_by_candidate,
                    "verifications": verifications,
                }
            )
            partial_report = {
                "schema_version": 1,
                "status": "running",
                "stage": args.stage,
                "checkpoint": str(checkpoint_path),
                "policy_version": policy_version,
                "policy_restore": restore_metadata,
                "source": source,
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "config": scicf_config,
                "oracle_counts": ledger.snapshot().as_dict(),
                "trajectories": trajectory_results,
            }
            write_json(stage_root / "gate1-stage-report.partial.json", partial_report)
            print(
                json.dumps(
                    {
                        "event": "gate1_trajectory_complete",
                        "stage": args.stage,
                        "seed": episode_seed,
                        "oracle_total": ledger.snapshot().total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        report = {
            "schema_version": 1,
            "status": "complete",
            "stage": args.stage,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "policy_version": policy_version,
            "policy_restore": restore_metadata,
            "source": source,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "config": scicf_config,
            "oracle_counts": ledger.snapshot().as_dict(),
            "llm_requests": str(requests_path),
            "trajectories": trajectory_results,
        }
        write_json(stage_root / "gate1-stage-report.json", report)
    finally:
        policy_adapter.close()


if __name__ == "__main__":
    main()
