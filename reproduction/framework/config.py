"""Load and validate versioned experiment configurations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from reproduction.framework.contracts import ContractError


PROTOCOL_NAME = "dapigen-common-v1"
ASSET_MODES = {"original", "reconstructed-afp-compatibility"}


def _mapping(value: Any, locator: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("{} must be an object".format(locator))
    return value


def _positive_int(value: Any, locator: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError("{} must be a positive integer".format(locator))
    return value


def _nonnegative_int(value: Any, locator: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError("{} must be a non-negative integer".format(locator))
    return value


def validate_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(dict(value))
    if config.get("schema_version") != 1:
        raise ContractError("schema_version must equal 1")

    provenance = _mapping(config.get("provenance"), "provenance")
    if not isinstance(provenance.get("baseline_tag"), str) or not provenance["baseline_tag"]:
        raise ContractError("provenance.baseline_tag is required")
    if provenance.get("require_clean_git") is not True:
        raise ContractError("provenance.require_clean_git must be true")
    if provenance.get("require_slurm") is not True:
        raise ContractError("provenance.require_slurm must be true")

    task = _mapping(config.get("task"), "task")
    if task.get("asset_mode") not in ASSET_MODES:
        raise ContractError("task.asset_mode is not supported")
    _positive_int(task.get("length"), "task.length")
    _positive_int(task.get("step_length"), "task.step_length")

    algorithm = _mapping(config.get("algorithm"), "algorithm")
    for key in ("name", "version", "adapter"):
        if not isinstance(algorithm.get(key), str) or not algorithm[key].strip():
            raise ContractError("algorithm.{} must be a non-empty string".format(key))
    _mapping(algorithm.get("parameters", {}), "algorithm.parameters")

    training = _mapping(config.get("training"), "training")
    maximum_steps = _positive_int(
        training.get("max_environment_steps"), "training.max_environment_steps"
    )
    _positive_int(training.get("max_iterations"), "training.max_iterations")
    if training.get("strict_environment_step_budget") is not True:
        raise ContractError("strict_environment_step_budget must be true")
    seed = training.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractError("training.seed must be an explicit integer")

    evaluation = _mapping(config.get("evaluation"), "evaluation")
    if evaluation.get("protocol") != PROTOCOL_NAME:
        raise ContractError("evaluation.protocol must equal {}".format(PROTOCOL_NAME))
    _positive_int(evaluation.get("samples"), "evaluation.samples")
    _positive_int(evaluation.get("progress_every"), "evaluation.progress_every")
    _positive_int(evaluation.get("max_episode_steps"), "evaluation.max_episode_steps")
    if not isinstance(evaluation.get("compute_paper_metrics"), bool):
        raise ContractError("evaluation.compute_paper_metrics must be boolean")
    steps = evaluation.get("checkpoint_environment_steps")
    if not isinstance(steps, list) or not steps:
        raise ContractError("evaluation.checkpoint_environment_steps must be a list")
    if any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps):
        raise ContractError("evaluation checkpoint steps must be non-negative integers")
    if steps != sorted(set(steps)):
        raise ContractError("evaluation checkpoint steps must be sorted and unique")
    if steps[0] != 0 or steps[-1] != maximum_steps:
        raise ContractError("evaluation checkpoints must start at 0 and end at the training budget")
    if evaluation.get("explore") is not False:
        raise ContractError("common evaluation requires explore=false")
    if evaluation.get("compute_paper_metrics"):
        for key in ("reference_csv", "reference_column"):
            if not isinstance(evaluation.get(key), str) or not evaluation[key]:
                raise ContractError("evaluation.{} is required".format(key))

    resources = _mapping(config.get("resources"), "resources")
    _positive_int(resources.get("cpus"), "resources.cpus")
    _nonnegative_int(resources.get("rollout_workers"), "resources.rollout_workers")
    gpu_count = resources.get("gpus")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, (int, float)) or gpu_count < 0:
        raise ContractError("resources.gpus must be non-negative")

    return config


def load_config(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ContractError("cannot read experiment config {}: {}".format(path, error))
    return validate_config(_mapping(value, "root"))
