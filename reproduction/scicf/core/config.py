"""Fail-closed validation for versioned SciCF experiment configurations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from reproduction.framework.contracts import ContractError


SCHEMA_VERSION = 1
METHOD_NAME = "scicf-ppo"
PHASES = {"scaffold", "offline-gate1", "online-gate2", "grounding-gate3"}


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


def validate_scicf_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(dict(value))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("SciCF schema_version must equal 1")
    if config.get("method") != METHOD_NAME:
        raise ContractError("SciCF method must equal {}".format(METHOD_NAME))
    if config.get("phase") not in PHASES:
        raise ContractError("unsupported SciCF phase")

    provenance = _mapping(config.get("provenance"), "provenance")
    for key in ("baseline_tag", "upstream_revision", "specification_sha256"):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise ContractError("provenance.{} is required".format(key))
    if provenance.get("require_clean_git") is not True:
        raise ContractError("provenance.require_clean_git must be true")
    if provenance.get("require_slurm") is not True:
        raise ContractError("provenance.require_slurm must be true")

    safety = _mapping(config.get("ppo_safety"), "ppo_safety")
    if safety.get("preserve_standard_clipped_objective") is not True:
        raise ContractError("standard PPO clipping must remain enabled")
    if safety.get("external_actions_in_ppo_clipping") is not False:
        raise ContractError("external actions must be excluded from PPO clipping")
    if safety.get("llm_as_reward_or_value") is not False:
        raise ContractError("LLM output cannot be used as reward or value")

    acquisition = _mapping(config.get("acquisition"), "acquisition")
    pool_size = _positive_int(acquisition.get("pool_size"), "acquisition.pool_size")
    budget = _positive_int(acquisition.get("budget"), "acquisition.budget")
    if budget > pool_size:
        raise ContractError("acquisition budget cannot exceed pool size")
    quotas = _mapping(acquisition.get("source_quotas"), "acquisition.source_quotas")
    for source in ("policy_near", "random_legal", "structural"):
        _nonnegative_int(quotas.get(source), "acquisition.source_quotas.{}".format(source))
    if sum(int(quotas[source]) for source in quotas) != pool_size:
        raise ContractError("candidate source quotas must sum to pool_size")

    verification = _mapping(config.get("verification"), "verification")
    _positive_int(verification.get("paired_replicates"), "verification.paired_replicates")
    _positive_int(
        verification.get("atomic_oracle_budget"),
        "verification.atomic_oracle_budget",
    )
    if verification.get("common_random_numbers") is not True:
        raise ContractError("paired verification requires common random numbers")
    if verification.get("confidence_rule") not in {
        "sign-consistency",
        "interval-excludes-zero",
    }:
        raise ContractError("unsupported confidence rule")

    llm = _mapping(config.get("llm"), "llm")
    if llm.get("trusted_learning_signal") is not False:
        raise ContractError("LLM output cannot be a trusted learning signal")
    if llm.get("enabled") is True and not llm.get("model_id"):
        raise ContractError("enabled LLM acquisition requires an explicit model_id")

    refinement = _mapping(config.get("pairwise_refinement"), "pairwise_refinement")
    gate1 = _mapping(config.get("gate1"), "gate1")
    gate1_status = gate1.get("status")
    if gate1_status not in {"pending", "passed", "failed"}:
        raise ContractError("gate1.status must be pending, passed, or failed")
    if refinement.get("enabled") is True and gate1_status != "passed":
        raise ContractError("pairwise refinement is forbidden before Gate 1 passes")
    if config["phase"] in {"scaffold", "offline-gate1"} and refinement.get("enabled"):
        raise ContractError("offline phases cannot enable pairwise refinement")

    seeds = gate1.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ContractError("gate1.seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ContractError("gate1 seeds must be explicit integers")
    return config


def load_scicf_config(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ContractError("cannot read SciCF config {}: {}".format(path, error))
    return validate_scicf_config(_mapping(value, "root"))
