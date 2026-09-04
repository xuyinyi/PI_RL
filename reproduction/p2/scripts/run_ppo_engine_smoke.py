#!/usr/bin/env python
"""Governed real-Stage-0 smoke run for the public native PPO engine."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

from reproduction.p2.contracts import (
    CREDIT_ESTIMATOR_CONTRACT_ID,
    METHOD_EVALUATOR_SOURCES,
    PPO,
    PPO_ENGINE_CONTRACT_ID,
    MethodRunContract,
    canonical_sha256,
)
from reproduction.p2.engine import (
    PPOEngine,
    PPOEngineConfig,
    rollout_digest,
)
from reproduction.p2.gae import GAECreditEstimator
from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    sha256_path,
    verify_accepted_binding,
)


SMOKE_SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--accepted-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    required = {
        "base_seed",
        "benchmark_name",
        "expected_accepted_environment_id",
        "expected_evaluator_version",
        "expected_objective_contract",
        "expected_observation_dimension",
        "maximum_requested_calls",
        "maximum_unique_calls",
        "minimum_successful_terminals_per_iteration",
        "ppo",
        "schema_version",
    }
    if set(payload) != required:
        raise ValueError(
            "Smoke config keys differ; missing=%s extra=%s"
            % (sorted(required - set(payload)), sorted(set(payload) - required))
        )
    if payload["schema_version"] != SMOKE_SCHEMA_VERSION:
        raise ValueError("Unsupported PPO smoke schema version.")
    engine_config = PPOEngineConfig(**payload["ppo"])
    if payload["base_seed"] != engine_config.seed:
        raise ValueError("Top-level and PPO seeds must match.")
    if payload["maximum_requested_calls"] < 2 * engine_config.rollout_steps:
        raise ValueError("Evaluator budget cannot cover two PPO iterations.")
    if payload["minimum_successful_terminals_per_iteration"] <= 0:
        raise ValueError("Minimum successful-terminal count must be positive.")
    return payload


def build_engine(
    root: Path,
    accepted_manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    config: Mapping[str, Any],
):
    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.factory import build_stage0_components
    from RL_PPO.envs.gymnasium_wrapper import DAPiGenGymnasiumEnv
    from RL_PPO.envs.sources import PPO_ON_POLICY

    engine_config = PPOEngineConfig(**config["ppo"])
    components = build_stage0_components(
        dapigen_root=str(root),
        polybert_path=str(root / "RL_PPO" / "models"),
        config=DAPiGenEnvConfig.from_mapping(binding["accepted_task_config"]),
        device=engine_config.device,
        maximum_requested_calls=int(config["maximum_requested_calls"]),
        maximum_unique_calls=int(config["maximum_unique_calls"]),
        encoder_mode="polybert",
        evaluator_mode="persistent",
        evaluator_fail_fast=True,
        allowed_evaluator_sources=tuple(METHOD_EVALUATOR_SOURCES[PPO]),
        cache_scope="p2_native_ppo_smoke_v1",
        allow_rdkit_brics_fallback=False,
    )
    specification = components.core.specification()
    accepted_environment = accepted_manifest["environment"]
    compared = (
        "environment_id",
        "environment_version",
        "state_schema_version",
        "config",
        "observation_dimension",
        "number_of_dianhydride_actions",
        "number_of_diamine_actions",
        "dianhydride_noop_id",
        "diamine_noop_id",
        "dianhydride_catalog_sha256",
        "diamine_catalog_sha256",
        "chemistry_backend",
        "encoder_version",
    )
    mismatches = {
        name: {"accepted": accepted_environment.get(name), "actual": specification.get(name)}
        for name in compared
        if specification.get(name) != accepted_environment.get(name)
    }
    expected = {
        "environment_id": config["expected_accepted_environment_id"],
        "observation_dimension": config["expected_observation_dimension"],
        "evaluator_version": config["expected_evaluator_version"],
        "objective_contract": config["expected_objective_contract"],
    }
    observed = {
        "environment_id": specification["environment_id"],
        "observation_dimension": specification["observation_dimension"],
        "evaluator_version": components.evaluator.evaluator_version,
        "objective_contract": components.evaluator.objective_contract,
    }
    for name in expected:
        if observed[name] != expected[name]:
            mismatches[name] = {"accepted": expected[name], "actual": observed[name]}
    if mismatches:
        raise RuntimeError("Accepted Stage-0 stack mismatch: %s" % mismatches)

    environment = DAPiGenGymnasiumEnv(
        components.core,
        components.reward_adapter,
        source=PPO_ON_POLICY,
        seed=int(config["base_seed"]),
        include_ledger_in_step_info=False,
    )
    budget_contract = canonical_sha256(
        {
            "protocol": "terminal-requested-unique-backend-v2",
            "maximum_requested_calls": config["maximum_requested_calls"],
            "maximum_unique_calls": config["maximum_unique_calls"],
            "allowed_sources": sorted(METHOD_EVALUATOR_SOURCES[PPO]),
            "cache_scope": "p2_native_ppo_smoke_v1",
        }
    )
    run_contract = MethodRunContract(
        method=PPO,
        environment_id=specification["environment_id"],
        task_contract_id=accepted_manifest["task_contract_id"],
        budget_contract_id=budget_contract,
        evaluator_version=components.evaluator.evaluator_version,
        objective_contract=components.evaluator.objective_contract,
        ppo_engine_contract_id=PPO_ENGINE_CONTRACT_ID,
        ppo_hyperparameters_sha256=engine_config.sha256,
        credit_estimator_contract_id=CREDIT_ESTIMATOR_CONTRACT_ID,
        allowed_evaluator_sources=tuple(METHOD_EVALUATOR_SOURCES[PPO]),
    )
    engine = PPOEngine(
        environment=environment,
        run_contract=run_contract,
        credit_estimator=GAECreditEstimator(),
        observation_dimension=int(specification["observation_dimension"]),
        number_of_dianhydride_actions=int(
            specification["number_of_dianhydride_actions"]
        ),
        number_of_diamine_actions=int(specification["number_of_diamine_actions"]),
        config=engine_config,
    )
    return components, engine, specification, run_contract


def iteration_summary(result) -> Dict[str, Any]:
    transitions = tuple(result.rollout.transitions)
    successful = sum(
        int("terminal_evaluation" in dict(item.info)) for item in transitions
    )
    terminated = sum(int(item.terminated) for item in transitions)
    truncated = sum(int(item.truncated) for item in transitions)
    return {
        "batch_id": result.rollout.batch_id,
        "rollout_digest": rollout_digest(result.rollout),
        "transition_count": len(transitions),
        "successful_terminal_count": successful,
        "terminated_count": terminated,
        "truncated_count": truncated,
        "reward_sum": float(sum(item.reward for item in transitions)),
        "gae_sha256": result.rollout.gae_sha256,
        "critic_returns_sha256": result.rollout.critic_returns_sha256,
        "actor_advantages_sha256": result.credit.actor_advantages_sha256,
        "rollout_evaluator_delta": asdict(result.rollout_evaluator_delta),
        "credit_evaluator_delta": asdict(result.credit.evaluator_delta),
        "update_receipt": asdict(result.receipt),
        "update_metrics": dict(result.update_metrics),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(args.dapigen_root).resolve()
    config_path = Path(args.config).resolve()
    accepted_manifest_path = Path(args.accepted_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    accepted_manifest = json.loads(accepted_manifest_path.read_text())
    binding = verify_accepted_binding(
        root, accepted_manifest, config["expected_accepted_environment_id"]
    )

    import gymnasium
    import torch

    started = time.perf_counter()
    components, engine, specification, run_contract = build_engine(
        root, accepted_manifest, binding, config
    )
    build_seconds = time.perf_counter() - started
    first = engine.run_iteration()
    first_summary = iteration_summary(first)
    checkpoint_path = output_dir / "after_iteration_1.pt"
    engine.save_checkpoint(checkpoint_path)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    checkpoint_ledger = dict(engine.environment.oracle_ledger())

    expected_second = engine.run_iteration()
    expected_summary = iteration_summary(expected_second)
    expected_policy_sha256 = engine.policy_state_sha256
    expected_ledger = dict(engine.environment.oracle_ledger())
    expected_policy_version = engine.policy_version
    del expected_second, first, engine, components
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    restored_components, restored_engine, restored_specification, restored_contract = (
        build_engine(root, accepted_manifest, binding, config)
    )
    restored_engine.load_checkpoint(checkpoint_path)
    ledger_after_load = dict(restored_engine.environment.oracle_ledger())
    restored_second = restored_engine.run_iteration()
    restored_summary = iteration_summary(restored_second)
    restored_policy_sha256 = restored_engine.policy_state_sha256
    restored_ledger = dict(restored_engine.environment.oracle_ledger())

    exact_resume_checks = {
        "contract_equal": asdict(restored_contract) == asdict(run_contract),
        "specification_equal": restored_specification == specification,
        "checkpoint_ledger_equal_after_load": ledger_after_load == checkpoint_ledger,
        "rollout_digest_equal": restored_summary["rollout_digest"]
        == expected_summary["rollout_digest"],
        "batch_id_equal": restored_summary["batch_id"] == expected_summary["batch_id"],
        "gae_equal": restored_summary["gae_sha256"] == expected_summary["gae_sha256"],
        "critic_returns_equal": restored_summary["critic_returns_sha256"]
        == expected_summary["critic_returns_sha256"],
        "actor_credit_equal": restored_summary["actor_advantages_sha256"]
        == expected_summary["actor_advantages_sha256"],
        "receipt_equal": restored_summary["update_receipt"]
        == expected_summary["update_receipt"],
        "update_metrics_equal": restored_summary["update_metrics"]
        == expected_summary["update_metrics"],
        "policy_state_equal": restored_policy_sha256 == expected_policy_sha256,
        "policy_version_equal": restored_engine.policy_version
        == expected_policy_version,
        "evaluator_ledger_equal": restored_ledger == expected_ledger,
    }
    failures = [name for name, passed in exact_resume_checks.items() if not passed]
    minimum_success = int(config["minimum_successful_terminals_per_iteration"])
    for name, summary in (
        ("first_iteration", first_summary),
        ("expected_second_iteration", expected_summary),
        ("restored_second_iteration", restored_summary),
    ):
        if summary["successful_terminal_count"] < minimum_success:
            failures.append("%s_insufficient_successful_terminals" % name)
        if summary["actor_advantages_sha256"] != summary["gae_sha256"]:
            failures.append("%s_actor_credit_not_exact_gae" % name)
        if summary["credit_evaluator_delta"]["requested_calls"] != 0:
            failures.append("%s_credit_queried_evaluator" % name)
        source_counts = summary["rollout_evaluator_delta"]["requested_by_source"]
        if set(source_counts) - {"ppo/on_policy"}:
            failures.append("%s_wrong_evaluator_source" % name)
    if binding.get("profile_git_dirty"):
        failures.append("profile_checkout_dirty")
    if binding.get("stage0_source_changed_from_accepted"):
        failures.append("stage0_source_changed_from_accepted")
    if int(restored_ledger.get("invalid_results", 0)) != 0:
        failures.append("invalid_evaluator_result")

    report = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "benchmark_name": config["benchmark_name"],
        "status": "passed" if not failures else "failed",
        "classification": {
            "status": "passed" if not failures else "failed",
            "failures": failures,
            "scope": "native-ppo-two-iteration-smoke-with-exact-resume",
            "training_admission": "not_defined_smoke_only",
        },
        "native_ppo_invoked": True,
        "rllib_ppo_invoked": False,
        "policy_cc_invoked": False,
        "mcc_ppo_invoked": False,
        "external_api_invoked": False,
        "scientific_validation_performed": False,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "device": config["ppo"]["device"],
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        },
        "config": config,
        "config_sha256": sha256_path(config_path),
        "script_sha256": sha256_path(Path(__file__).resolve()),
        "accepted_manifest_sha256": sha256_path(accepted_manifest_path),
        "accepted_binding": binding,
        "stack_specification": specification,
        "run_contract": asdict(run_contract),
        "model_and_stack_build_seconds": build_seconds,
        "first_iteration": first_summary,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "ledger": checkpoint_ledger,
        },
        "expected_second_iteration": expected_summary,
        "restored_second_iteration": restored_summary,
        "exact_resume_checks": exact_resume_checks,
        "final_policy_sha256": restored_policy_sha256,
        "final_evaluator_ledger": restored_ledger,
        "elapsed_seconds": time.perf_counter() - started,
        "resources": {
            "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            / 1024.0,
            "cuda_max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            "cuda_max_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            ),
        },
    }
    report_path = output_dir / "ppo_engine_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(report_path),
                "first_iteration": first_summary,
                "restored_second_iteration": restored_summary,
                "exact_resume_checks": exact_resume_checks,
                "final_policy_sha256": restored_policy_sha256,
                "final_evaluator_ledger": restored_ledger,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
