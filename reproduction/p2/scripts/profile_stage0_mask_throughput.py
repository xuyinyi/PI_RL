#!/usr/bin/env python
"""Profile accepted Stage-0 mask throughput across independent workers.

The timed region calls ``raw_action_mask_for_mode`` on deterministic states
generated with the accepted transition and chemistry implementation. Molecular
embeddings and terminal evaluation are intentionally outside the benchmark.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import platform
import resource
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--accepted-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(payload: Any) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_profile_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    required = {
        "schema_version",
        "benchmark_name",
        "base_seed",
        "mask_mode",
        "worker_counts",
        "repetitions",
        "transitions_per_worker",
        "minimum_max_worker_speedup",
        "minimum_max_worker_efficiency",
        "minimum_adjacent_throughput_ratio",
        "expected_accepted_environment_id",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("Profile config is missing fields: %s" % missing)
    if int(payload["schema_version"]) != 1:
        raise ValueError("Unsupported profile schema version.")
    counts = [int(value) for value in payload["worker_counts"]]
    if len(counts) < 2 or counts != sorted(set(counts)) or counts[0] != 1:
        raise ValueError("worker_counts must be unique, increasing and start at 1.")
    if any(value <= 0 for value in counts):
        raise ValueError("worker_counts must be positive.")
    if int(payload["repetitions"]) <= 0:
        raise ValueError("repetitions must be positive.")
    if int(payload["transitions_per_worker"]) <= 0:
        raise ValueError("transitions_per_worker must be positive.")
    for name in (
        "minimum_max_worker_speedup",
        "minimum_max_worker_efficiency",
        "minimum_adjacent_throughput_ratio",
    ):
        if float(payload[name]) <= 0:
            raise ValueError("%s must be positive." % name)
    if payload["mask_mode"] != "closure_exact_cached":
        raise ValueError("P2-A profiles only the accepted closure_exact_cached mask.")
    payload["worker_counts"] = counts
    return payload


def git_coordinate(root: Path) -> Dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], universal_newlines=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"],
        universal_newlines=True,
    )
    return {"commit": commit, "dirty": bool(status.strip())}


def verify_accepted_binding(
    root: Path, accepted_manifest: Mapping[str, Any], expected_environment_id: str
) -> Dict[str, Any]:
    repository = accepted_manifest["dapigen_repository"]
    environment = accepted_manifest["environment"]
    accepted_commit = str(repository["dapigen_git_commit"])
    if repository.get("dapigen_git_dirty") is not False:
        raise ValueError("Accepted manifest is not bound to a clean Git checkout.")
    if environment.get("environment_id") != expected_environment_id:
        raise ValueError("Accepted environment ID does not match the frozen profile.")
    source_paths = [
        "RL_PPO/envs",
        "RL_PPO/moldr/env_stage0.py",
        "reproduction/stage0/configs/stage0_environment.json",
    ]
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", accepted_commit, "--"]
        + source_paths
    )
    if diff.returncode not in (0, 1):
        raise RuntimeError("Unable to compare Stage-0 source with accepted commit.")
    coordinate = git_coordinate(root)
    if coordinate["dirty"]:
        raise RuntimeError("P2-A profiling requires a clean Git checkout.")
    return {
        "accepted_git_commit": accepted_commit,
        "accepted_environment_id": environment["environment_id"],
        "accepted_task_config": environment["config"],
        "accepted_chemistry_backend": environment["chemistry_backend"],
        "accepted_dianhydride_catalog_sha256": environment[
            "dianhydride_catalog_sha256"
        ],
        "accepted_diamine_catalog_sha256": environment["diamine_catalog_sha256"],
        "profile_git_commit": coordinate["commit"],
        "profile_git_dirty": coordinate["dirty"],
        "stage0_source_changed_from_accepted": diff.returncode == 1,
    }


class MaskProfileEncoder(object):
    """Minimal deterministic encoder excluded from the timed mask region."""

    encoder_version = "mask-profile-zero-encoder-v1"
    output_dim = 1

    def __call__(self, smiles: str):
        del smiles
        import numpy as np

        return np.zeros((1,), dtype=np.float32)

    def diagnostics(self):
        return {
            "encoder_version": self.encoder_version,
            "output_dim": self.output_dim,
            "timed_region": False,
        }


def build_profile_core(root: Path, task_config_payload: Mapping[str, Any]):
    from RL_PPO.envs.chemistry import LegacyDAPiGenChemistryBackend
    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.core import BranchableDAPiGenCore
    from RL_PPO.envs.factory import load_block_catalog

    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=False)
    block_dir = root / "RL_PPO" / "outputs" / "building_blocks"
    d_blocks, _ = load_block_catalog(
        str(block_dir / "blocks_dianhydride.csv"), chemistry=chemistry
    )


def numeric_deltas(before: Mapping[str, Any], after: Mapping[str, Any]):
    """Return nested numeric counter changes without treating bool as an int."""

    result = {}
    for key in sorted(set(before).intersection(after)):
        left = before[key]
        right = after[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            nested = numeric_deltas(left, right)
            if nested:
                result[key] = nested
        elif (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            result[key] = right - left
    return result
    a_blocks, _ = load_block_catalog(
        str(block_dir / "blocks_diamine.csv"), chemistry=chemistry
    )
    return BranchableDAPiGenCore(
        dianhydride_blocks=d_blocks,
        diamine_blocks=a_blocks,
        initial_dianhydride_smiles="[16*]c1ccc2c(c1)C(=O)OC2=O",
        initial_diamine_smiles="[16*]c1ccc(N)cc1",
        chemistry=chemistry,
        encoder=MaskProfileEncoder(),
        config=DAPiGenEnvConfig.from_mapping(task_config_payload),
    )


def generate_workload_states(
    core, transition_count: int, base_seed: int, worker_index: int
) -> Tuple[List[Any], Dict[str, Any]]:
    from RL_PPO.envs.rng import derive_seed, named_index
    from RL_PPO.envs.types import DAPiGenAction

    worker_seed = derive_seed(base_seed, "profile_worker", worker_index)
    episode_index = 0
    current = core.initial(seed=derive_seed(worker_seed, "episode", episode_index))
    states = []
    reasons = collections.Counter()
    no_product_count = 0
    for transition_index in range(int(transition_count)):
        if current.state.done:
            episode_index += 1
            current = core.initial(
                seed=derive_seed(worker_seed, "episode", episode_index)
            )
        states.append(current.state)
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
        transition_seed = derive_seed(
            worker_seed, "profile_transition", transition_index
        )
        action = DAPiGenAction(
            d_ids[named_index(transition_seed, "dianhydride_action", len(d_ids))],
            a_ids[named_index(transition_seed, "diamine_action", len(a_ids))],
        )
        current = core.transition(current.state, action, seed=transition_seed)
        reason = current.state.termination_reason
        if reason is not None:
            reasons[str(reason)] += 1
        no_product_count += int(reason == "no_reaction_product")
    return states, {
        "worker_seed": int(worker_seed),
        "episode_count": int(episode_index + 1),
        "termination_reasons": dict(sorted(reasons.items())),
        "no_product_count": int(no_product_count),
        "state_digest": canonical_digest([state.to_dict() for state in states]),
    }


def _profile_worker(
    root_text: str,
    task_config_payload: Mapping[str, Any],
    mask_mode: str,
    transition_count: int,
    base_seed: int,
    worker_index: int,
    repetition: int,
    barrier,
) -> Dict[str, Any]:
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = Path(root_text)

    setup_started = time.perf_counter()
    generator_core = build_profile_core(root, task_config_payload)
    states, workload = generate_workload_states(
        generator_core, transition_count, base_seed, worker_index
    )
    generator_specification = generator_core.specification()
    del generator_core

    profile_core = build_profile_core(root, task_config_payload)
    profile_specification = profile_core.specification()
    diagnostics_before = profile_core.diagnostics()
    setup_seconds = time.perf_counter() - setup_started
    barrier.wait(timeout=600)

    import numpy as np

    mask_digest = hashlib.sha256()
    allowed_dianhydride = 0
    allowed_diamine = 0
    cpu_started = time.process_time()
    started = time.monotonic()
    for state in states:
        mask = profile_core.raw_action_mask_for_mode(state, mask_mode)
        d_mask = np.asarray(mask["dianhydride"], dtype=np.uint8)
        a_mask = np.asarray(mask["diamine"], dtype=np.uint8)
        allowed_dianhydride += int(d_mask.sum())
        allowed_diamine += int(a_mask.sum())
        mask_digest.update(d_mask.tobytes())
        mask_digest.update(a_mask.tobytes())
    finished = time.monotonic()
    cpu_seconds = time.process_time() - cpu_started
    elapsed = finished - started
    diagnostics_after = profile_core.diagnostics()
    peak_rss_mib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    return {
        "status": "passed",
        "worker_index": int(worker_index),
        "repetition": int(repetition),
        "pid": int(os.getpid()),
        "transitions": int(transition_count),
        "mask_evaluations": len(states),
        "setup_seconds": float(setup_seconds),
        "start_monotonic": float(started),
        "finish_monotonic": float(finished),
        "elapsed_seconds": float(elapsed),
        "cpu_seconds": float(cpu_seconds),
        "masks_per_second": float(len(states) / elapsed),
        "peak_rss_mib": peak_rss_mib,
        "allowed_dianhydride_total": int(allowed_dianhydride),
        "allowed_diamine_total": int(allowed_diamine),
        "mask_digest": mask_digest.hexdigest(),
        "workload": workload,
        "generator_specification": generator_specification,
        "profile_specification": profile_specification,
        "profile_diagnostics_before": diagnostics_before,
        "profile_diagnostics_after": diagnostics_after,
        "profile_diagnostic_delta": numeric_deltas(
            diagnostics_before, diagnostics_after
        ),
    }


def aggregate_group(
    worker_count: int, repetition: int, results: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    ordered = sorted(results, key=lambda item: int(item["worker_index"]))
    starts = [float(item["start_monotonic"]) for item in ordered]
    finishes = [float(item["finish_monotonic"]) for item in ordered]
    elapsed = max(finishes) - min(starts)
    total = sum(int(item["mask_evaluations"]) for item in ordered)
    reasons = collections.Counter()
    for item in ordered:
        reasons.update(item["workload"]["termination_reasons"])
    return {
        "worker_count": int(worker_count),
        "repetition": int(repetition),
        "status": (
            "passed" if all(item["status"] == "passed" for item in ordered) else "failed"
        ),
        "mask_evaluations": int(total),
        "concurrent_elapsed_seconds": float(elapsed),
        "aggregate_masks_per_second": float(total / elapsed),
        "start_spread_seconds": float(max(starts) - min(starts)),
        "median_worker_masks_per_second": float(
            statistics.median(item["masks_per_second"] for item in ordered)
        ),
        "maximum_worker_setup_seconds": float(
            max(item["setup_seconds"] for item in ordered)
        ),
        "maximum_worker_peak_rss_mib": float(
            max(item["peak_rss_mib"] for item in ordered)
        ),
        "sum_worker_peak_rss_mib": float(
            sum(item["peak_rss_mib"] for item in ordered)
        ),
        "no_product_count": int(
            sum(item["workload"]["no_product_count"] for item in ordered)
        ),
        "termination_reasons": dict(sorted(reasons.items())),
        "workers": ordered,
    }


def summarize_scaling(
    groups: Sequence[Mapping[str, Any]], worker_counts: Sequence[int]
) -> Dict[str, Any]:
    by_count = {}
    for count in worker_counts:
        selected = [item for item in groups if int(item["worker_count"]) == count]
        throughputs = [float(item["aggregate_masks_per_second"]) for item in selected]
        by_count[str(count)] = {
            "worker_count": int(count),
            "repetitions": len(selected),
            "aggregate_masks_per_second": throughputs,
            "median_aggregate_masks_per_second": float(statistics.median(throughputs)),
            "minimum_aggregate_masks_per_second": float(min(throughputs)),
            "maximum_aggregate_masks_per_second": float(max(throughputs)),
        }
    baseline = by_count["1"]["median_aggregate_masks_per_second"]
    for count in worker_counts:
        summary = by_count[str(count)]
        speedup = summary["median_aggregate_masks_per_second"] / baseline
        summary["median_speedup_vs_one_worker"] = float(speedup)
        summary["median_parallel_efficiency"] = float(speedup / count)
    adjacent = []
    for previous, current in zip(worker_counts[:-1], worker_counts[1:]):
        ratio = (
            by_count[str(current)]["median_aggregate_masks_per_second"]
            / by_count[str(previous)]["median_aggregate_masks_per_second"]
        )
        adjacent.append(
            {
                "previous_worker_count": int(previous),
                "current_worker_count": int(current),
                "throughput_ratio": float(ratio),
            }
        )
    return {"by_worker_count": by_count, "adjacent_throughput_ratios": adjacent}


def determinism_violations(groups: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    expected = {}
    violations = []
    for group in groups:
        for worker in group["workers"]:
            index = int(worker["worker_index"])
            observed = (
                worker["workload"]["state_digest"],
                worker["mask_digest"],
                int(worker["allowed_dianhydride_total"]),
                int(worker["allowed_diamine_total"]),
            )
            if index not in expected:
                expected[index] = observed
            elif expected[index] != observed:
                violations.append(
                    {
                        "worker_index": index,
                        "worker_count": int(group["worker_count"]),
                        "repetition": int(group["repetition"]),
                    }
                )
    return violations


def classify_profile(
    groups: Sequence[Mapping[str, Any]],
    scaling: Mapping[str, Any],
    worker_counts: Sequence[int],
    config: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> Dict[str, Any]:
    violations = determinism_violations(groups)
    functional_failures = []
    if binding["profile_git_dirty"]:
        functional_failures.append("dirty_profile_checkout")
    if binding["stage0_source_changed_from_accepted"]:
        functional_failures.append("stage0_source_changed_from_accepted")
    if violations:
        functional_failures.append("nondeterministic_worker_digest")
    if any(group["status"] != "passed" for group in groups):
        functional_failures.append("worker_failure")
    if any(int(group["no_product_count"]) != 0 for group in groups):
        functional_failures.append("no_reaction_product")

    maximum = int(worker_counts[-1])
    maximum_summary = scaling["by_worker_count"][str(maximum)]
    speedup = float(maximum_summary["median_speedup_vs_one_worker"])
    efficiency = float(maximum_summary["median_parallel_efficiency"])
    minimum_adjacent = min(
        item["throughput_ratio"] for item in scaling["adjacent_throughput_ratios"]
    )
    throughput_failures = []
    if speedup < float(config["minimum_max_worker_speedup"]):
        throughput_failures.append("maximum_worker_speedup_below_threshold")
    if efficiency < float(config["minimum_max_worker_efficiency"]):
        throughput_failures.append("maximum_worker_efficiency_below_threshold")
    if minimum_adjacent < float(config["minimum_adjacent_throughput_ratio"]):
        throughput_failures.append("adjacent_throughput_ratio_below_threshold")

    if functional_failures:
        status = "failed_functional_gate"
    elif throughput_failures:
        status = "failed_throughput_admission"
    else:
        status = "passed"
    return {
        "status": status,
        "functional_failures": functional_failures,
        "throughput_failures": throughput_failures,
        "determinism_violations": violations,
        "maximum_worker_count": maximum,
        "maximum_worker_median_speedup": speedup,
        "maximum_worker_median_efficiency": efficiency,
        "minimum_adjacent_throughput_ratio": float(minimum_adjacent),
    }


def validate_profile_specification(
    groups: Sequence[Mapping[str, Any]], binding: Mapping[str, Any]
) -> Dict[str, Any]:
    mismatches = []
    accepted_config = binding["accepted_task_config"]
    for group in groups:
        for worker in group["workers"]:
            profile = worker["profile_specification"]
            generator = worker["generator_specification"]
            for label, specification in (("profile", profile), ("generator", generator)):
                if specification["config"] != accepted_config:
                    mismatches.append(
                        {
                            "worker_count": group["worker_count"],
                            "repetition": group["repetition"],
                            "worker_index": worker["worker_index"],
                            "component": label,
                            "field": "config",
                        }
                    )
                for field, expected in (
                    ("chemistry_backend", binding["accepted_chemistry_backend"]),
                    (
                        "dianhydride_catalog_sha256",
                        binding["accepted_dianhydride_catalog_sha256"],
                    ),
                    (
                        "diamine_catalog_sha256",
                        binding["accepted_diamine_catalog_sha256"],
                    ),
                ):
                    if specification[field] != expected:
                        mismatches.append(
                            {
                                "worker_count": group["worker_count"],
                                "repetition": group["repetition"],
                                "worker_index": worker["worker_index"],
                                "component": label,
                                "field": field,
                            }
                        )
    return {"status": "passed" if not mismatches else "failed", "mismatches": mismatches}


def main():
    args = parse_args()
    root = Path(args.dapigen_root).resolve()
    config_path = Path(args.config).resolve()
    manifest_path = Path(args.accepted_manifest).resolve()
    output_path = Path(args.output).resolve()
    config = load_profile_config(config_path)
    accepted_manifest = json.loads(manifest_path.read_text())
    binding = verify_accepted_binding(
        root, accepted_manifest, config["expected_accepted_environment_id"]
    )

    worker_counts = config["worker_counts"]
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if max(worker_counts) > allocated_cpus:
        raise RuntimeError(
            "Maximum worker count %d exceeds allocated CPUs %d."
            % (max(worker_counts), allocated_cpus)
        )

    task_config = binding["accepted_task_config"]
    groups = []
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        for worker_count in worker_counts:
            for repetition in range(int(config["repetitions"])):
                barrier = manager.Barrier(int(worker_count))
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=int(worker_count), mp_context=context
                ) as executor:
                    futures = [
                        executor.submit(
                            _profile_worker,
                            str(root),
                            task_config,
                            config["mask_mode"],
                            int(config["transitions_per_worker"]),
                            int(config["base_seed"]),
                            worker_index,
                            repetition,
                            barrier,
                        )
                        for worker_index in range(int(worker_count))
                    ]
                    results = [future.result() for future in futures]
                group = aggregate_group(worker_count, repetition, results)
                groups.append(group)
                print(
                    "workers=%d repetition=%d masks_per_second=%.6f"
                    % (
                        worker_count,
                        repetition,
                        group["aggregate_masks_per_second"],
                    ),
                    flush=True,
                )

    scaling = summarize_scaling(groups, worker_counts)
    specification = validate_profile_specification(groups, binding)
    classification = classify_profile(groups, scaling, worker_counts, config, binding)
    if specification["status"] != "passed":
        classification["status"] = "failed_functional_gate"
        classification["functional_failures"].append(
            "accepted_specification_mismatch"
        )

    report = {
        "schema_version": 1,
        "benchmark_name": config["benchmark_name"],
        "status": classification["status"],
        "scope": "accepted-mask-and-custom-chemistry-only",
        "timed_operation": "BranchableDAPiGenCore.raw_action_mask_for_mode",
        "terminal_evaluator_invoked": False,
        "ppo_invoked": False,
        "external_api_invoked": False,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "cpus_per_task": allocated_cpus,
        },
        "config": config,
        "config_sha256": sha256_path(config_path),
        "profiler_sha256": sha256_path(Path(__file__).resolve()),
        "accepted_manifest_path": str(manifest_path),
        "accepted_manifest_sha256": sha256_path(manifest_path),
        "accepted_binding": binding,
        "specification_validation": specification,
        "classification": classification,
        "scaling": scaling,
        "groups": groups,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "status": report["status"],
        "classification": classification,
        "scaling": scaling,
        "output": str(output_path),
    }, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
