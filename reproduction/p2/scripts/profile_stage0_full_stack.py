#!/usr/bin/env python
"""Profile accepted Stage-0 with real polyBERT and the evaluator ledger.

This is a single-worker engineering characterization, not PPO execution and not
a parallel-scaling or scientific-performance admission.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    canonical_digest,
    numeric_deltas,
    sha256_path,
    verify_accepted_binding,
)


PROFILE_SCHEMA_VERSION = 1
PROFILE_REQUIRED_KEYS = frozenset(
    (
        "base_seed",
        "benchmark_name",
        "duplicate_each_success_for_cache_measurement",
        "expected_accepted_environment_id",
        "expected_evaluator_version",
        "expected_objective_contract",
        "expected_observation_dimension",
        "maximum_requested_calls",
        "maximum_unique_calls",
        "minimum_successful_terminals",
        "schema_version",
        "transition_count",
    )
)
LEDGER_COUNTERS = (
    "requested_calls",
    "unique_calls",
    "backend_calls",
    "cache_hits",
    "invalid_results",
)
LEDGER_SOURCE_COUNTERS = (
    "requested_by_source",
    "unique_by_source",
    "backend_by_source",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--accepted-manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_profile_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if set(payload) != PROFILE_REQUIRED_KEYS:
        missing = sorted(PROFILE_REQUIRED_KEYS - set(payload))
        extra = sorted(set(payload) - PROFILE_REQUIRED_KEYS)
        raise ValueError("Profile config keys differ; missing=%s extra=%s" % (missing, extra))
    if payload["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError("Unsupported full-stack profile schema version.")
    for name in (
        "base_seed",
        "expected_observation_dimension",
        "maximum_requested_calls",
        "maximum_unique_calls",
        "minimum_successful_terminals",
        "transition_count",
    ):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("%s must be an integer." % name)
    if payload["transition_count"] <= 0:
        raise ValueError("transition_count must be positive.")
    if payload["minimum_successful_terminals"] <= 0:
        raise ValueError("minimum_successful_terminals must be positive.")
    if payload["maximum_requested_calls"] < 4 * payload["transition_count"]:
        raise ValueError("maximum_requested_calls cannot cover both measured passes.")
    if payload["maximum_unique_calls"] < payload["transition_count"]:
        raise ValueError("maximum_unique_calls cannot cover the first measured pass.")
    if payload["duplicate_each_success_for_cache_measurement"] is not True:
        raise ValueError("The v1 profile requires one duplicate evaluation per success.")
    for name in (
        "benchmark_name",
        "expected_accepted_environment_id",
        "expected_evaluator_version",
        "expected_objective_contract",
    ):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ValueError("%s must be a non-empty string." % name)
    return payload


def ledger_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    result = {}
    for name in LEDGER_COUNTERS:
        result[name] = int(after.get(name, 0)) - int(before.get(name, 0))
    for name in LEDGER_SOURCE_COUNTERS:
        left = dict(before.get(name, {}))
        right = dict(after.get(name, {}))
        result[name] = {
            source: int(right.get(source, 0)) - int(left.get(source, 0))
            for source in sorted(set(left).union(right))
            if int(right.get(source, 0)) - int(left.get(source, 0)) != 0
        }
    return result


def synchronize_device(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize(torch.device(device))


def execute_measured_pass(
    core,
    reward_adapter,
    evaluator,
    transition_count: int,
    base_seed: int,
    device: str,
) -> Dict[str, Any]:
    from RL_PPO.envs.rng import derive_seed, named_index
    from RL_PPO.envs.sources import ENVIRONMENT_REGRESSION, EVALUATION
    from RL_PPO.envs.types import DAPiGenAction

    ledger_before = evaluator.ledger()
    diagnostics_before = core.diagnostics()
    audit_event_count_before = len(evaluator.audit_events())
    reasons = collections.Counter()
    trajectory_digest = hashlib.sha256()
    observation_digest = hashlib.sha256()
    terminal_evaluations = []
    no_product_count = 0
    episode_index = 0
    current = None

    synchronize_device(device)
    cpu_started = time.process_time()
    started = time.perf_counter()
    for transition_index in range(int(transition_count)):
        if current is None or current.state.done:
            episode_seed = derive_seed(base_seed, "full_stack_episode", episode_index)
            current = core.initial(seed=episode_seed)
            episode_index += 1
        observation = np.ascontiguousarray(current.observation, dtype=np.float32)
        observation_digest.update(observation.tobytes())
        d_ids = np.flatnonzero(current.action_mask.dianhydride).tolist()
        a_ids = np.flatnonzero(current.action_mask.diamine).tolist()
        transition_seed = derive_seed(base_seed, "full_stack_transition", transition_index)
        action = DAPiGenAction(
            d_ids[named_index(transition_seed, "dianhydride_action", len(d_ids))],
            a_ids[named_index(transition_seed, "diamine_action", len(a_ids))],
        )
        next_transition = core.transition(
            current.state, action, seed=transition_seed
        )
        evaluated = reward_adapter.apply(
            next_transition, source=ENVIRONMENT_REGRESSION
        )
        if evaluated.evaluation is not None:
            duplicate = evaluator.evaluate_one(
                next_transition.terminal_smiles, source=EVALUATION
            )
            if duplicate.to_dict() != evaluated.evaluation.to_dict():
                raise RuntimeError("Cached evaluator result differs from first result.")
            terminal_evaluations.append(evaluated.evaluation.to_dict())
        reason = next_transition.state.termination_reason
        if reason is not None:
            reasons[str(reason)] += 1
        no_product_count += int(reason == "no_reaction_product")
        trajectory_digest.update(current.state.to_json().encode("utf-8"))
        trajectory_digest.update(json.dumps(action.as_tuple()).encode("ascii"))
        trajectory_digest.update(next_transition.state.to_json().encode("utf-8"))
        trajectory_digest.update(str(float(evaluated.reward)).encode("ascii"))
        current = next_transition
    synchronize_device(device)
    elapsed = time.perf_counter() - started
    cpu_elapsed = time.process_time() - cpu_started

    ledger_after = evaluator.ledger()
    diagnostics_after = core.diagnostics()
    audit_events = evaluator.audit_events()[audit_event_count_before:]
    evaluator_event_seconds = sum(
        float(event.get("elapsed_seconds", 0.0)) for event in audit_events
    )
    successful = len(terminal_evaluations)
    return {
        "status": "completed",
        "transition_count": int(transition_count),
        "episode_count": int(episode_index),
        "successful_terminal_count": int(successful),
        "termination_reasons": dict(sorted(reasons.items())),
        "no_product_count": int(no_product_count),
        "elapsed_seconds": float(elapsed),
        "cpu_seconds": float(cpu_elapsed),
        "transitions_per_second": float(transition_count / elapsed),
        "successful_terminals_per_second": float(successful / elapsed),
        "evaluator_event_seconds": float(evaluator_event_seconds),
        "evaluator_fraction_of_elapsed": float(evaluator_event_seconds / elapsed),
        "trajectory_digest": trajectory_digest.hexdigest(),
        "observation_digest": observation_digest.hexdigest(),
        "terminal_evaluations_digest": canonical_digest(terminal_evaluations),
        "ledger_before": ledger_before,
        "ledger_after": ledger_after,
        "ledger_delta": ledger_delta(ledger_before, ledger_after),
        "diagnostics_before": diagnostics_before,
        "diagnostics_after": diagnostics_after,
        "diagnostic_delta": numeric_deltas(diagnostics_before, diagnostics_after),
    }


def validate_stack_specification(
    actual: Mapping[str, Any],
    accepted_manifest: Mapping[str, Any],
    evaluator,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    accepted = accepted_manifest["environment"]
    compared_fields = (
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
    mismatches = []
    for name in compared_fields:
        if actual.get(name) != accepted.get(name):
            mismatches.append(
                {"field": name, "accepted": accepted.get(name), "actual": actual.get(name)}
            )
    expectations = {
        "environment_id": config["expected_accepted_environment_id"],
        "observation_dimension": config["expected_observation_dimension"],
        "evaluator_version": config["expected_evaluator_version"],
        "objective_contract": config["expected_objective_contract"],
    }
    observed = {
        "environment_id": actual.get("environment_id"),
        "observation_dimension": actual.get("observation_dimension"),
        "evaluator_version": evaluator.evaluator_version,
        "objective_contract": evaluator.objective_contract,
    }
    for name, expected in expectations.items():
        if observed[name] != expected:
            mismatches.append(
                {"field": name, "accepted": expected, "actual": observed[name]}
            )
    return {
        "status": "passed" if not mismatches else "failed",
        "mismatches": mismatches,
        "observed": observed,
    }


def classify_profile(
    first_pass: Mapping[str, Any],
    cache_replay: Mapping[str, Any],
    binding: Mapping[str, Any],
    specification: Mapping[str, Any],
    checkpoint_roundtrip_equal: bool,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    from RL_PPO.envs.sources import ENVIRONMENT_REGRESSION, EVALUATION

    failures = []
    if binding.get("profile_git_dirty"):
        failures.append("profile_checkout_dirty")
    if binding.get("stage0_source_changed_from_accepted"):
        failures.append("stage0_source_changed_from_accepted")
    if specification.get("status") != "passed":
        failures.append("accepted_stack_specification_mismatch")
    for name, phase in (("first_pass", first_pass), ("cache_replay", cache_replay)):
        successes = int(phase["successful_terminal_count"])
        delta = phase["ledger_delta"]
        if int(phase["transition_count"]) != int(config["transition_count"]):
            failures.append("%s_transition_count_mismatch" % name)
        if successes < int(config["minimum_successful_terminals"]):
            failures.append("%s_insufficient_successful_terminals" % name)
        if int(phase["no_product_count"]) != 0:
            failures.append("%s_no_reaction_product" % name)
        if int(delta["requested_calls"]) != 2 * successes:
            failures.append("%s_requested_ledger_mismatch" % name)
        expected_sources = {ENVIRONMENT_REGRESSION: successes, EVALUATION: successes}
        if dict(delta["requested_by_source"]) != expected_sources:
            failures.append("%s_source_ledger_mismatch" % name)
        if int(delta["invalid_results"]) != 0:
            failures.append("%s_invalid_evaluator_result" % name)
    first_delta = first_pass["ledger_delta"]
    replay_delta = cache_replay["ledger_delta"]
    if int(first_delta["unique_calls"]) != int(first_delta["backend_calls"]):
        failures.append("first_pass_unique_backend_mismatch")
    if int(replay_delta["unique_calls"]) != 0 or int(replay_delta["backend_calls"]) != 0:
        failures.append("cache_replay_reached_backend")
    if int(replay_delta["cache_hits"]) != int(replay_delta["requested_calls"]):
        failures.append("cache_replay_cache_hit_mismatch")
    for digest_name in (
        "trajectory_digest",
        "observation_digest",
        "terminal_evaluations_digest",
    ):
        if first_pass[digest_name] != cache_replay[digest_name]:
            failures.append("nondeterministic_%s" % digest_name)
    if first_pass["termination_reasons"] != cache_replay["termination_reasons"]:
        failures.append("nondeterministic_termination_reasons")
    if not checkpoint_roundtrip_equal:
        failures.append("evaluator_checkpoint_roundtrip_mismatch")
    return {
        "status": "passed" if not failures else "failed_functional_gate",
        "functional_failures": failures,
        "performance_admission": "not_defined_characterization_only",
        "worker_count": 1,
        "parallel_scaling_profiled": False,
    }


def gpu_memory(device: str) -> Dict[str, Any]:
    if not str(device).startswith("cuda"):
        return {"device": str(device), "available": False}
    import torch

    index = torch.device(device)
    return {
        "device": str(device),
        "available": bool(torch.cuda.is_available()),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.dapigen_root).resolve()
    config_path = Path(args.config).resolve()
    accepted_manifest_path = Path(args.accepted_manifest).resolve()
    output_path = Path(args.output).resolve()
    config = load_profile_config(config_path)
    accepted_manifest = json.loads(accepted_manifest_path.read_text())
    binding = verify_accepted_binding(
        root, accepted_manifest, config["expected_accepted_environment_id"]
    )

    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.factory import build_stage0_components
    from RL_PPO.envs.sources import ENVIRONMENT_REGRESSION, EVALUATION

    build_started = time.perf_counter()
    components = build_stage0_components(
        dapigen_root=str(root),
        polybert_path=str(root / "RL_PPO" / "models"),
        config=DAPiGenEnvConfig.from_mapping(binding["accepted_task_config"]),
        device=args.device,
        maximum_requested_calls=int(config["maximum_requested_calls"]),
        maximum_unique_calls=int(config["maximum_unique_calls"]),
        encoder_mode="polybert",
        evaluator_mode="persistent",
        evaluator_fail_fast=True,
        allowed_evaluator_sources=(ENVIRONMENT_REGRESSION, EVALUATION),
        cache_scope="p2_full_stack_profile_v1",
        allow_rdkit_brics_fallback=False,
    )
    synchronize_device(args.device)
    build_seconds = time.perf_counter() - build_started
    actual_specification = components.core.specification()
    specification = validate_stack_specification(
        actual_specification, accepted_manifest, components.evaluator, config
    )

    first_pass = execute_measured_pass(
        components.core,
        components.reward_adapter,
        components.evaluator,
        int(config["transition_count"]),
        int(config["base_seed"]),
        args.device,
    )
    cache_replay = execute_measured_pass(
        components.core,
        components.reward_adapter,
        components.evaluator,
        int(config["transition_count"]),
        int(config["base_seed"]),
        args.device,
    )
    evaluator_state = components.evaluator.state_dict()
    ledger_before_restore = components.evaluator.ledger()
    components.evaluator.load_state_dict(evaluator_state)
    ledger_after_restore = components.evaluator.ledger()
    checkpoint_roundtrip_equal = ledger_before_restore == ledger_after_restore
    classification = classify_profile(
        first_pass,
        cache_replay,
        binding,
        specification,
        checkpoint_roundtrip_equal,
        config,
    )

    report = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "benchmark_name": config["benchmark_name"],
        "status": classification["status"],
        "scope": "single-worker-real-polybert-persistent-evaluator-ledger",
        "timed_operation": "accepted-core-transition-observation-terminal-reward-ledger",
        "ppo_invoked": False,
        "credit_estimator_invoked": False,
        "external_api_invoked": False,
        "scientific_validation_performed": False,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "device": str(args.device),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "cpus_per_task": int(
                os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
            ),
        },
        "config": config,
        "config_sha256": sha256_path(config_path),
        "profiler_sha256": sha256_path(Path(__file__).resolve()),
        "accepted_manifest_sha256": sha256_path(accepted_manifest_path),
        "accepted_binding": binding,
        "model_build_seconds": float(build_seconds),
        "stack_specification": actual_specification,
        "specification_validation": specification,
        "evaluator": {
            "evaluator_version": components.evaluator.evaluator_version,
            "objective_contract": components.evaluator.objective_contract,
            "ledger_after_profile": components.evaluator.ledger(),
            "checkpoint_roundtrip_equal": checkpoint_roundtrip_equal,
        },
        "first_pass": first_pass,
        "cache_replay": cache_replay,
        "classification": classification,
        "resources": {
            "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            / 1024.0,
            "gpu": gpu_memory(args.device),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output_path),
                "model_build_seconds": report["model_build_seconds"],
                "first_pass": {
                    "transitions_per_second": first_pass["transitions_per_second"],
                    "successful_terminal_count": first_pass[
                        "successful_terminal_count"
                    ],
                    "ledger_delta": first_pass["ledger_delta"],
                },
                "cache_replay": {
                    "transitions_per_second": cache_replay["transitions_per_second"],
                    "successful_terminal_count": cache_replay[
                        "successful_terminal_count"
                    ],
                    "ledger_delta": cache_replay["ledger_delta"],
                },
                "classification": classification,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
