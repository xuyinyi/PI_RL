#!/usr/bin/env python
"""Run one frozen P4-A native-PPO preflight or formal seed on Stage 0."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import resource
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from reproduction.framework.evaluator import CommonEvaluator
from reproduction.p2.contracts import (
    CREDIT_ESTIMATOR_CONTRACT_ID,
    METHOD_EVALUATOR_SOURCES,
    PPO,
    PPO_ENGINE_CONTRACT_ID,
    MethodRunContract,
    canonical_sha256,
)
from reproduction.p2.engine import PPOEngine, PPOEngineConfig, rollout_digest
from reproduction.p2.gae import GAECreditEstimator
from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    verify_accepted_binding,
)
from reproduction.p4.protocol import (
    evaluate_acceptance,
    file_sha256,
    load_protocol,
    resolved_run,
)


CSV_COLUMNS = (
    "PI",
    "A_1",
    "A_2",
    "transmittance",
    "cte",
    "strength",
    "tg",
    "SaScore",
    "reward",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(partial), str(path))


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def git_identity(root: Path) -> Dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], universal_newlines=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"],
        universal_newlines=True,
    )
    return {"commit": commit, "dirty": bool(status.strip())}


def ppo_config(protocol: Mapping[str, Any], seed: int) -> PPOEngineConfig:
    return PPOEngineConfig(seed=int(seed), **dict(protocol["ppo"]))


def build_training_engine(
    root: Path,
    accepted_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    seed: int,
    mode: str,
):
    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.factory import build_stage0_components
    from RL_PPO.envs.gymnasium_wrapper import DAPiGenGymnasiumEnv
    from RL_PPO.envs.sources import PPO_ON_POLICY

    config = ppo_config(protocol, seed)
    cache_scope = "%s:%s:%d:training" % (protocol["protocol_id"], mode, seed)
    components = build_stage0_components(
        dapigen_root=str(root),
        polybert_path=str(root / "RL_PPO" / "models"),
        config=DAPiGenEnvConfig.from_mapping(protocol["task"]["configuration"]),
        device=config.device,
        maximum_requested_calls=int(run_spec["maximum_training_requested_calls"]),
        maximum_unique_calls=int(run_spec["maximum_training_unique_calls"]),
        encoder_mode="polybert",
        evaluator_mode="persistent",
        evaluator_fail_fast=True,
        allowed_evaluator_sources=tuple(METHOD_EVALUATOR_SOURCES[PPO]),
        cache_scope=cache_scope,
        allow_rdkit_brics_fallback=False,
    )
    specification = components.core.specification()
    environment = DAPiGenGymnasiumEnv(
        components.core,
        components.reward_adapter,
        source=PPO_ON_POLICY,
        seed=seed,
        include_ledger_in_step_info=False,
    )
    budget_contract = canonical_sha256(
        {
            "protocol": "terminal-requested-unique-backend-v2",
            "maximum_requested_calls": run_spec["maximum_training_requested_calls"],
            "maximum_unique_calls": run_spec["maximum_training_unique_calls"],
            "allowed_sources": sorted(METHOD_EVALUATOR_SOURCES[PPO]),
            "cache_scope": cache_scope,
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
        ppo_hyperparameters_sha256=config.sha256,
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
        config=config,
    )
    return components, engine, specification, run_contract


@contextmanager
def preserve_process_rng():
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state().cpu()
    torch_cuda_states = []
    if torch.cuda.is_available():
        torch_cuda_states = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_cpu_state)
        if torch_cuda_states:
            torch.cuda.set_rng_state_all(torch_cuda_states)


def evaluate_policy(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    engine: PPOEngine,
    target_requested_calls: int,
    actual_training_requested_calls: int,
    seed: int,
    output_dir: Path,
    evaluation_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.factory import build_stage0_components
    from RL_PPO.envs.gymnasium_wrapper import DAPiGenGymnasiumEnv
    from RL_PPO.envs.sources import EVALUATION

    import torch

    evaluation_seed = int(seed + target_requested_calls + 1_000_000)
    samples = int(evaluation_spec["samples"])
    checkpoint_dir = output_dir / "evaluations" / (
        "target_%06d_actual_%06d" % (
            target_requested_calls,
            actual_training_requested_calls,
        )
    )
    output_csv = checkpoint_dir / "generate.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    process_rng_before = {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().cpu().clone(),
        "engine": copy.deepcopy(engine._rng.bit_generator.state),
    }
    policy_before = engine.policy_state_sha256
    version_before = engine.policy_version
    termination_reasons = Counter()
    started = time.perf_counter()

    with preserve_process_rng():
        components = build_stage0_components(
            dapigen_root=str(root),
            polybert_path=str(root / "RL_PPO" / "models"),
            config=DAPiGenEnvConfig.from_mapping(protocol["task"]["configuration"]),
            device=engine.config.device,
            maximum_requested_calls=samples,
            maximum_unique_calls=samples,
            encoder_mode="polybert",
            evaluator_mode="persistent",
            evaluator_fail_fast=True,
            allowed_evaluator_sources=(EVALUATION,),
            cache_scope="%s:seed-%d:eval-%d"
            % (protocol["protocol_id"], seed, target_requested_calls),
            allow_rdkit_brics_fallback=False,
        )
        environment = DAPiGenGymnasiumEnv(
            components.core,
            components.reward_adapter,
            source=EVALUATION,
            seed=evaluation_seed,
            include_ledger_in_step_info=False,
        )
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for sample_index in range(samples):
                reset = environment.reset(seed=evaluation_seed) if sample_index == 0 else environment.reset()
                observation = reset[0] if isinstance(reset, tuple) else reset
                dianhydride_actions = []
                diamine_actions = []
                terminal_evaluation = None
                reward = 0.0
                for _step in range(int(evaluation_spec["max_episode_steps"])):
                    action = engine.deterministic_action(observation)
                    dianhydride_actions.append(int(action[0]))
                    diamine_actions.append(int(action[1]))
                    observation, reward, terminated, truncated, info = environment.step(action)
                    if terminated or truncated:
                        terminal_evaluation = info.get("terminal_evaluation")
                        termination_reasons[str(environment.current_state.termination_reason)] += 1
                        break
                else:
                    raise RuntimeError("Evaluation episode exceeded the frozen horizon.")

                properties = (
                    {} if terminal_evaluation is None else terminal_evaluation.get("properties", {})
                )
                writer.writerow(
                    {
                        "PI": ""
                        if terminal_evaluation is None
                        else terminal_evaluation.get("canonical_smiles") or "",
                        "A_1": ",".join(map(str, dianhydride_actions)),
                        "A_2": ",".join(map(str, diamine_actions)),
                        "transmittance": properties.get("transmittance", ""),
                        "cte": properties.get("cte", ""),
                        "strength": properties.get("strength", ""),
                        "tg": properties.get("tg", ""),
                        "SaScore": properties.get("sa_score", ""),
                        "reward": float(reward),
                    }
                )
                completed = sample_index + 1
                if (
                    completed == 1
                    or completed % int(evaluation_spec["progress_every"]) == 0
                    or completed == samples
                ):
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "event": "p4_evaluation_progress",
                                "target_requested_calls": target_requested_calls,
                                "completed": completed,
                                "total": samples,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        ledger = dict(environment.oracle_ledger())
        common = CommonEvaluator(
            reference_csv=root / evaluation_spec["reference_csv"],
            reference_column=evaluation_spec["reference_column"],
            compute_paper_metrics=bool(evaluation_spec["compute_paper_metrics"]),
            protocol="dapigen-common-v1",
        )
        metrics = common.evaluate_csv(output_csv)
        del environment, components, common
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rng_checks = {
        "python_rng_unchanged": random.getstate() == process_rng_before["python"],
        "numpy_rng_unchanged": all(
            np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
            for left, right in zip(np.random.get_state(), process_rng_before["numpy"])
        ),
        "torch_rng_unchanged": bool(
            torch.equal(torch.get_rng_state().cpu(), process_rng_before["torch"])
        ),
        "engine_rng_unchanged": engine._rng.bit_generator.state
        == process_rng_before["engine"],
        "policy_unchanged": engine.policy_state_sha256 == policy_before,
        "policy_version_unchanged": engine.policy_version == version_before,
    }
    if not all(rng_checks.values()):
        raise RuntimeError("Checkpoint evaluation mutated training state: %s" % rng_checks)
    return {
        "target_requested_calls": int(target_requested_calls),
        "actual_training_requested_calls": int(actual_training_requested_calls),
        "policy_version": int(engine.policy_version),
        "policy_sha256": engine.policy_state_sha256,
        "evaluation_seed": evaluation_seed,
        "metrics": metrics,
        "evaluator_ledger": ledger,
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "canonical_duplicate_count": int(metrics["valid_samples"])
        - int(metrics["canonical_unique_valid_PI"]),
        "training_state_checks": rng_checks,
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_csv": str(output_csv),
        "output_csv_sha256": file_sha256(output_csv),
    }


def iteration_record(result, iteration: int, ledger: Mapping[str, Any]):
    transitions = tuple(result.rollout.transitions)
    termination_reasons = Counter()
    terminal_smiles = []
    for item in transitions:
        snapshot = item.state_snapshot_after or {}
        reason = snapshot.get("termination_reason")
        if reason is not None:
            termination_reasons[str(reason)] += 1
        terminal = dict(item.info).get("terminal_evaluation")
        if terminal and terminal.get("canonical_smiles"):
            terminal_smiles.append(str(terminal["canonical_smiles"]))
    return {
        "iteration": int(iteration),
        "policy_version": int(result.receipt.policy_version_after),
        "batch_id": result.rollout.batch_id,
        "rollout_digest": rollout_digest(result.rollout),
        "transition_count": len(transitions),
        "cumulative_environment_transitions": int(iteration * len(transitions)),
        "reward_sum": float(sum(item.reward for item in transitions)),
        "successful_terminal_count": len(terminal_smiles),
        "terminal_unique_count": len(set(terminal_smiles)),
        "termination_reasons": dict(sorted(termination_reasons.items())),
        "gae_sha256": result.rollout.gae_sha256,
        "critic_returns_sha256": result.rollout.critic_returns_sha256,
        "actor_advantages_sha256": result.credit.actor_advantages_sha256,
        "rollout_evaluator_delta": asdict(result.rollout_evaluator_delta),
        "credit_evaluator_delta": asdict(result.credit.evaluator_delta),
        "update_receipt": asdict(result.receipt),
        "update_metrics": dict(result.update_metrics),
        "training_evaluator_ledger": dict(ledger),
    }


def save_checkpoint(
    engine: PPOEngine,
    output_dir: Path,
    target_requested_calls: int,
    actual_requested_calls: int,
) -> Dict[str, Any]:
    path = output_dir / "checkpoints" / (
        "target_%06d_actual_%06d.pt" % (
            target_requested_calls,
            actual_requested_calls,
        )
    )
    engine.save_checkpoint(path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "target_requested_calls": int(target_requested_calls),
        "actual_training_requested_calls": int(actual_requested_calls),
        "policy_version": int(engine.policy_version),
        "policy_sha256": engine.policy_state_sha256,
    }


def write_artifact_manifest(output_dir: Path) -> Path:
    target = output_dir / "ARTIFACT_SHA256SUMS.txt"
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == target or path.name.endswith(".partial"):
            continue
        rows.append("%s  %s" % (file_sha256(path), path.relative_to(output_dir)))
    target.write_text("\n".join(rows) + "\n")
    return target


def main() -> None:
    args = parse_args()
    root = Path(args.dapigen_root).resolve()
    protocol_path = Path(args.protocol).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Output directory already exists: %s" % output_dir)
    output_dir.mkdir(parents=True)

    protocol = load_protocol(protocol_path)
    seed_argument: Optional[int] = args.seed
    if args.mode == "preflight" and seed_argument is not None:
        raise ValueError("The preflight seed is frozen and cannot be overridden.")
    seed, run_spec = resolved_run(protocol, args.mode, seed_argument)
    source = git_identity(root)
    report_path = output_dir / "run_report.json"
    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "mode": args.mode,
        "seed": seed,
        "source": source,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "run_specification": run_spec,
        "ppo": {**dict(protocol["ppo"]), "seed": seed},
        "sealed_test_accessed": False,
        "external_api_invoked": False,
        "classification": protocol["reporting"]["claim_boundary"],
        "started_at": utc_now(),
    }
    write_json(report_path, report)
    started = time.perf_counter()
    try:
        if protocol["provenance"]["require_clean_git"] and source["dirty"]:
            raise RuntimeError("P4-A requires a clean Git checkout.")
        if protocol["provenance"]["require_slurm"] and not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("P4-A requires Slurm execution.")
        if socket.gethostname() != protocol["provenance"]["required_host"]:
            raise RuntimeError("P4-A executed on the wrong host.")
        accepted_path = root / protocol["accepted_binding"]["accepted_manifest"]
        accepted_manifest = json.loads(accepted_path.read_text())
        binding = verify_accepted_binding(
            root, accepted_manifest, protocol["accepted_binding"]["environment_id"]
        )

        import gymnasium
        import torch

        build_started = time.perf_counter()
        components, engine, specification, run_contract = build_training_engine(
            root, accepted_manifest, protocol, run_spec, seed, args.mode
        )
        report.update(
            {
                "status": "running",
                "accepted_binding": binding,
                "accepted_manifest_sha256": file_sha256(accepted_path),
                "stack_specification": specification,
                "run_contract": asdict(run_contract),
                "torch": torch.__version__,
                "gymnasium": gymnasium.__version__,
                "device": str(engine.device),
                "model_and_stack_build_seconds": float(
                    time.perf_counter() - build_started
                ),
            }
        )
        write_json(report_path, report)
        initial_policy = engine.policy_state_sha256
        report["initial_policy_sha256"] = initial_policy
        report["iterations"] = []
        report["evaluations"] = []
        report["checkpoints"] = []
        training_metrics_path = output_dir / "training_metrics.jsonl"
        targets = list(run_spec["checkpoint_requested_calls"])

        def record_target(target: int) -> None:
            ledger = dict(engine.environment.oracle_ledger())
            actual = int(ledger["requested_calls"])
            checkpoint = save_checkpoint(engine, output_dir, target, actual)
            evaluation = evaluate_policy(
                root=root,
                protocol=protocol,
                engine=engine,
                target_requested_calls=target,
                actual_training_requested_calls=actual,
                seed=seed,
                output_dir=output_dir,
                evaluation_spec=run_spec["evaluation"],
            )
            report["checkpoints"].append(checkpoint)
            report["evaluations"].append(evaluation)
            write_json(report_path, report)

        record_target(targets[0])
        next_target_index = 1
        iteration = 0
        maximum_iterations = int(run_spec["maximum_iterations"])
        rollout_steps = int(protocol["ppo"]["rollout_steps"])
        stop_reason = None
        while True:
            ledger_before = dict(engine.environment.oracle_ledger())
            remaining = int(ledger_before["remaining_requested_calls"])
            if remaining < rollout_steps:
                stop_reason = "insufficient_remaining_budget_for_reserved_rollout"
                break
            if iteration >= maximum_iterations:
                stop_reason = "maximum_iterations"
                break
            result = engine.run_iteration()
            iteration += 1
            ledger_after = dict(engine.environment.oracle_ledger())
            record = iteration_record(result, iteration, ledger_after)
            report["iterations"].append(record)
            append_jsonl(training_metrics_path, record)
            print(
                json.dumps(
                    {
                        "event": "p4_training_iteration",
                        "mode": args.mode,
                        "seed": seed,
                        "iteration": iteration,
                        "requested_calls": ledger_after["requested_calls"],
                        "unique_calls": ledger_after["unique_calls"],
                        "environment_transitions": iteration * rollout_steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            while (
                next_target_index < len(targets) - 1
                and int(ledger_after["requested_calls"]) >= targets[next_target_index]
            ):
                record_target(targets[next_target_index])
                next_target_index += 1
            write_json(report_path, report)

        final_target = targets[-1]
        if next_target_index != len(targets) - 1:
            raise RuntimeError(
                "Training stopped before intermediate checkpoint targets were reached."
            )
        record_target(final_target)
        final_checkpoint = report["checkpoints"][-1]
        final_ledger = dict(engine.environment.oracle_ledger())
        final_policy = engine.policy_state_sha256
        final_version = engine.policy_version

        del components
        gc.collect()
        restored_components, restored_engine, restored_specification, restored_contract = (
            build_training_engine(root, accepted_manifest, protocol, run_spec, seed, args.mode)
        )
        restored_engine.load_checkpoint(Path(final_checkpoint["path"]))
        checkpoint_roundtrip = {
            "specification_equal": restored_specification == specification,
            "contract_equal": restored_contract == run_contract,
            "policy_sha256_equal": restored_engine.policy_state_sha256 == final_policy,
            "policy_version_equal": restored_engine.policy_version == final_version,
            "evaluator_ledger_equal": dict(restored_engine.environment.oracle_ledger())
            == final_ledger,
        }
        del restored_engine, restored_components
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        report.update(
            {
                "status": "candidate",
                "stop_reason": stop_reason,
                "training_evaluator_ledger": final_ledger,
                "final_policy_sha256": final_policy,
                "final_policy_version": final_version,
                "environment_transitions": iteration * rollout_steps,
                "checkpoint_roundtrip": checkpoint_roundtrip,
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
                "elapsed_seconds": float(time.perf_counter() - started),
                "completed_at": utc_now(),
            }
        )
        checks, accepted = evaluate_acceptance(report, protocol)
        report["acceptance_checks"] = checks
        report["status"] = "passed" if accepted else "failed_gate"
        write_json(report_path, report)
        manifest = write_artifact_manifest(output_dir)
        print(
            json.dumps(
                {
                    "event": "p4_run_complete",
                    "mode": args.mode,
                    "seed": seed,
                    "status": report["status"],
                    "requested_calls": final_ledger["requested_calls"],
                    "artifact_manifest": str(manifest),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not accepted:
            raise RuntimeError("P4-A acceptance gate failed: %s" % checks)
    except Exception as error:
        if report.get("status") != "failed_gate":
            report["status"] = "failed"
        report["failed_at"] = utc_now()
        report["error"] = "%s: %s" % (type(error).__name__, error)
        report["traceback"] = traceback.format_exc()
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
