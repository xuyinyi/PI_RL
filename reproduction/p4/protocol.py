"""Validation and acceptance logic for the frozen P4-A baseline protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_ID = "dapigen-p4a-native-ppo-standard-v1"
FROZEN_PPO = {
    "rollout_steps": 128,
    "gamma": 1.0,
    "gae_lambda": 0.95,
    "update_epochs": 4,
    "minibatch_size": 64,
    "clip_ratio": 0.2,
    "value_clip": 0.2,
    "value_loss_coefficient": 0.5,
    "entropy_coefficient": 0.01,
    "maximum_gradient_norm": 0.5,
    "target_kl": 0.02,
    "normalize_actor_advantages": True,
    "learning_rate": 0.0003,
    "adam_epsilon": 1e-8,
    "hidden_sizes": [256, 256],
    "deterministic_torch": True,
    "device": "cuda:0",
}
REQUIRED_METRICS = (
    "validity",
    "uniqueness",
    "mean_reward",
    "max_reward",
    "novelty",
    "diversity",
    "frag",
    "snn",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected, locator: str) -> None:
    observed = set(value)
    required = set(expected)
    if observed != required:
        raise ValueError(
            "%s keys differ; missing=%s extra=%s"
            % (locator, sorted(required - observed), sorted(observed - required))
        )


def _positive_integer(value: Any, locator: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer." % locator)
    return int(value)


def _validate_run_spec(spec: Mapping[str, Any], *, formal: bool, rollout_steps: int):
    common = {
        "maximum_training_requested_calls",
        "maximum_training_unique_calls",
        "minimum_training_requested_calls",
        "maximum_iterations",
        "checkpoint_requested_calls",
        "evaluation",
    }
    _exact_keys(spec, common | ({"seeds"} if formal else {"seed"}), "run spec")
    maximum_requested = _positive_integer(
        spec["maximum_training_requested_calls"],
        "maximum_training_requested_calls",
    )
    maximum_unique = _positive_integer(
        spec["maximum_training_unique_calls"],
        "maximum_training_unique_calls",
    )
    minimum_requested = _positive_integer(
        spec["minimum_training_requested_calls"],
        "minimum_training_requested_calls",
    )
    if maximum_unique != maximum_requested:
        raise ValueError("Requested and unique training caps must match in P4-A v1.")
    guaranteed_floor = maximum_requested - rollout_steps + 1
    if minimum_requested != guaranteed_floor:
        raise ValueError(
            "The minimum requested-call gate must equal cap-rollout_steps+1."
        )
    _positive_integer(spec["maximum_iterations"], "maximum_iterations")
    checkpoints = spec["checkpoint_requested_calls"]
    if (
        not isinstance(checkpoints, list)
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[0] != 0
        or checkpoints[-1] != maximum_requested
    ):
        raise ValueError(
            "Checkpoint requested calls must be sorted, unique and span 0 to the cap."
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in checkpoints
    ):
        raise ValueError("Checkpoint requested calls must be non-negative integers.")

    evaluation = spec["evaluation"]
    _exact_keys(
        evaluation,
        {
            "samples",
            "progress_every",
            "max_episode_steps",
            "explore",
            "compute_paper_metrics",
            "reference_csv",
            "reference_column",
            "minimum_valid_samples",
        },
        "evaluation",
    )
    samples = _positive_integer(evaluation["samples"], "evaluation.samples")
    _positive_integer(evaluation["progress_every"], "evaluation.progress_every")
    _positive_integer(evaluation["max_episode_steps"], "evaluation.max_episode_steps")
    minimum_valid = _positive_integer(
        evaluation["minimum_valid_samples"], "evaluation.minimum_valid_samples"
    )
    if minimum_valid > samples:
        raise ValueError("minimum_valid_samples cannot exceed evaluation.samples.")
    if evaluation["explore"] is not False:
        raise ValueError("P4-A evaluation must use explore=false.")
    if evaluation["compute_paper_metrics"] is not True:
        raise ValueError("P4-A evaluation must compute the frozen paper metrics.")
    if evaluation["reference_csv"] != "raw_data/PI.csv":
        raise ValueError("P4-A v1 fixes raw_data/PI.csv as the novelty reference.")
    if evaluation["reference_column"] != "smile":
        raise ValueError("P4-A v1 fixes the novelty reference column to smile.")

    seeds = spec["seeds"] if formal else [spec["seed"]]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("The run spec must declare at least one seed.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative integers.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique.")


def validate_protocol(value: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = copy.deepcopy(dict(value))
    _exact_keys(
        protocol,
        {
            "schema_version",
            "protocol_id",
            "classification",
            "frozen_before_results",
            "provenance",
            "accepted_binding",
            "data_boundary",
            "task",
            "ppo",
            "preflight",
            "formal",
            "resources",
            "reporting",
        },
        "protocol",
    )
    if protocol["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Unsupported P4 protocol schema version.")
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Unsupported P4 protocol identity.")
    if protocol["classification"] != "p4a_native_standard_baseline":
        raise ValueError("P4-A v1 classification changed.")
    if protocol["frozen_before_results"] is not True:
        raise ValueError("The protocol must be frozen before results.")

    provenance = protocol["provenance"]
    _exact_keys(
        provenance,
        {"require_clean_git", "require_slurm", "required_host"},
        "provenance",
    )
    if provenance != {
        "require_clean_git": True,
        "require_slurm": True,
        "required_host": "yanlih100n1",
    }:
        raise ValueError("P4-A provenance requirements changed.")

    binding = protocol["accepted_binding"]
    _exact_keys(
        binding,
        {
            "accepted_manifest",
            "environment_id",
            "accepted_p1_commit",
            "evaluator_version",
            "objective_contract",
            "observation_dimension",
        },
        "accepted_binding",
    )
    if binding["environment_id"] != "dapigen:8004c1fa2055a186b4a4f3ed":
        raise ValueError("The accepted Stage-0 environment changed.")
    if binding["accepted_p1_commit"] != "373b3291ac04dbf654c134bee2ca61a0c86d1a68":
        raise ValueError("The accepted P1 commit changed.")
    if binding["observation_dimension"] != 1246:
        raise ValueError("The accepted observation dimension changed.")

    boundary = protocol["data_boundary"]
    _exact_keys(
        boundary,
        {
            "allowed_inputs",
            "sealed_test_access_authorized",
            "external_api_authorized",
            "hyperparameter_retuning_after_preflight",
        },
        "data_boundary",
    )
    if boundary["sealed_test_access_authorized"] is not False:
        raise ValueError("P4-A must not authorize sealed-test access.")
    if boundary["external_api_authorized"] is not False:
        raise ValueError("P4-A must not authorize an external API.")
    if boundary["hyperparameter_retuning_after_preflight"] is not False:
        raise ValueError("Preflight outcomes cannot tune the frozen protocol.")

    task = protocol["task"]
    _exact_keys(task, {"variant", "configuration"}, "task")
    if task["variant"] != "stage0_standard_five_step":
        raise ValueError("P4-A v1 admits only the standard five-step arm.")
    if task["configuration"].get("max_steps") != 5:
        raise ValueError("P4-A standard task must use five steps.")

    ppo = dict(protocol["ppo"])
    if ppo != FROZEN_PPO:
        raise ValueError("The frozen P4-A PPO configuration changed.")
    rollout_steps = int(ppo["rollout_steps"])
    _validate_run_spec(protocol["preflight"], formal=False, rollout_steps=rollout_steps)
    _validate_run_spec(protocol["formal"], formal=True, rollout_steps=rollout_steps)
    if protocol["preflight"]["seed"] in protocol["formal"]["seeds"]:
        raise ValueError("The preflight seed must be disjoint from formal seeds.")
    if len(protocol["formal"]["seeds"]) < 5:
        raise ValueError("P4-A v1 requires at least five formal seeds.")

    resources = protocol["resources"]
    _exact_keys(resources, {"cpus", "gpus", "memory_gib", "walltime"}, "resources")
    if resources["gpus"] != 1:
        raise ValueError("P4-A v1 uses exactly one GPU per seed.")
    _positive_integer(resources["cpus"], "resources.cpus")
    _positive_integer(resources["memory_gib"], "resources.memory_gib")

    reporting = protocol["reporting"]
    _exact_keys(
        reporting,
        {
            "primary_budget_axis",
            "secondary_budget_axes",
            "required_metrics",
            "claim_boundary",
        },
        "reporting",
    )
    if reporting["primary_budget_axis"] != "training_requested_evaluator_calls":
        raise ValueError("P4-A primary budget axis changed.")
    if tuple(reporting["required_metrics"]) != REQUIRED_METRICS:
        raise ValueError("P4-A metric list changed.")
    return protocol


def load_protocol(path: Path) -> Dict[str, Any]:
    return validate_protocol(json.loads(Path(path).read_text()))


def resolved_run(protocol: Mapping[str, Any], mode: str, seed: Any = None):
    if mode not in ("preflight", "formal"):
        raise ValueError("mode must be preflight or formal.")
    spec = copy.deepcopy(dict(protocol[mode]))
    if mode == "preflight":
        resolved_seed = int(spec.pop("seed"))
    else:
        allowed = tuple(int(value) for value in spec.pop("seeds"))
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in allowed:
            raise ValueError("Formal seed is absent from the frozen protocol.")
        resolved_seed = int(seed)
    return resolved_seed, spec


def evaluate_acceptance(report: Mapping[str, Any], protocol: Mapping[str, Any]):
    mode = report["mode"]
    _seed, spec = resolved_run(protocol, mode, int(report["seed"]))
    ledger = report.get("training_evaluator_ledger", {})
    iterations = report.get("iterations", [])
    evaluations = report.get("evaluations", [])
    checkpoint_roundtrip = report.get("checkpoint_roundtrip", {})
    requested = int(ledger.get("requested_calls", -1))
    allowed_training_sources = {"ppo/on_policy"}
    requested_sources = {
        key for key, value in ledger.get("requested_by_source", {}).items() if value
    }

    finite_updates = bool(iterations) and all(
        int(item.get("transition_count", -1)) == int(protocol["ppo"]["rollout_steps"])
        and item.get("credit_evaluator_delta", {}).get("requested_calls") == 0
        and item.get("actor_advantages_sha256") == item.get("gae_sha256")
        and all(math.isfinite(float(value)) for value in item.get("update_metrics", {}).values())
        for item in iterations
    )
    expected_targets = list(spec["checkpoint_requested_calls"])
    observed_targets = [int(item.get("target_requested_calls", -1)) for item in evaluations]
    evaluation_rows_ok = len(evaluations) == len(expected_targets) and all(
        item.get("metrics", {}).get("samples") == spec["evaluation"]["samples"]
        and item.get("metrics", {}).get("valid_samples", 0)
        >= spec["evaluation"]["minimum_valid_samples"]
        and all(
            item.get("metrics", {}).get(name) is not None
            and math.isfinite(float(item["metrics"][name]))
            for name in REQUIRED_METRICS
        )
        and {
            key
            for key, value in item.get("evaluator_ledger", {})
            .get("requested_by_source", {})
            .items()
            if value
        }
        <= {"evaluation"}
        and item.get("evaluator_ledger", {}).get("invalid_results") == 0
        for item in evaluations
    )
    checks = {
        "clean_git": report.get("source", {}).get("dirty") is False,
        "slurm_execution": bool(report.get("slurm", {}).get("job_id")),
        "required_host": report.get("host") == protocol["provenance"]["required_host"],
        "accepted_stage0_source_unchanged": report.get("accepted_binding", {}).get(
            "stage0_source_changed_from_accepted"
        )
        is False,
        "accepted_p1_commit": report.get("accepted_binding", {}).get(
            "accepted_git_commit"
        )
        == protocol["accepted_binding"]["accepted_p1_commit"],
        "accepted_environment": report.get("stack_specification", {}).get("environment_id")
        == protocol["accepted_binding"]["environment_id"],
        "accepted_task_configuration": report.get("stack_specification", {}).get("config")
        == protocol["task"]["configuration"],
        "accepted_observation_dimension": report.get("stack_specification", {}).get(
            "observation_dimension"
        )
        == protocol["accepted_binding"]["observation_dimension"],
        "accepted_evaluator": report.get("training_evaluator_ledger", {}).get(
            "evaluator_version"
        )
        == protocol["accepted_binding"]["evaluator_version"],
        "accepted_objective": report.get("training_evaluator_ledger", {}).get(
            "objective_contract"
        )
        == protocol["accepted_binding"]["objective_contract"],
        "standard_five_step_task": report.get("stack_specification", {})
        .get("config", {})
        .get("max_steps")
        == 5,
        "training_budget_lower_bound": requested
        >= int(spec["minimum_training_requested_calls"]),
        "training_budget_upper_bound": requested
        <= int(spec["maximum_training_requested_calls"]),
        "training_source_isolated": requested_sources <= allowed_training_sources,
        "training_evaluator_valid": ledger.get("invalid_results") == 0,
        "finite_exact_gae_updates": finite_updates,
        "policy_changed": report.get("initial_policy_sha256")
        != report.get("final_policy_sha256"),
        "policy_version_matches_iterations": report.get("final_policy_version")
        == len(iterations),
        "checkpoint_targets_complete": observed_targets == expected_targets,
        "evaluation_metrics_complete": evaluation_rows_ok,
        "checkpoint_roundtrip": bool(checkpoint_roundtrip)
        and all(bool(value) for value in checkpoint_roundtrip.values()),
        "sealed_test_not_accessed": report.get("sealed_test_accessed") is False,
        "external_api_not_invoked": report.get("external_api_invoked") is False,
    }
    return checks, all(checks.values())
