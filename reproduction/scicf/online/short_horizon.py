"""Checkpoint and audit controls for bounded multi-iteration SciCF runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from reproduction.framework.io import write_json
from reproduction.p2.contracts import ContractViolation

from .resilience import LLMAuxiliaryCircuitBreaker, LLMWallClockBudget
from .contracts import SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID
from .pipeline import KL_NUMERICAL_NEGATIVE_TOLERANCE


SHORT_HORIZON_CONTROL_CHECKPOINT_SCHEMA = 1


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_bound(root: Path, record: Mapping[str, Any]) -> Path:
    if set(record) != {"path", "sha256"}:
        raise ValueError("invalid short-horizon bound-file record")
    root = root.resolve(strict=True)
    path = (root / record["path"]).resolve(strict=True)
    if root != path and root not in path.parents:
        raise ValueError("short-horizon bound file escaped repository")
    if sha256_path(path) != record["sha256"]:
        raise RuntimeError("short-horizon bound-file hash mismatch")
    return path


def load_short_horizon_protocol(path: Path, root: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "classification",
        "status",
        "required_host",
        "prerequisites",
        "horizon",
        "llm_wall_clock_budget",
        "circuit_breaker",
        "kl_numerics",
        "scientific_budget",
        "failure_policy",
        "authorization_state",
    }
    if set(payload) != required:
        raise ValueError("short-horizon protocol keys changed")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported short-horizon protocol schema")
    if payload["protocol_id"] != SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID:
        raise ValueError("short-horizon protocol identity changed")
    if payload["classification"] != (
        "runner_development_and_no_credential_preflight_only"
    ):
        raise ValueError("short-horizon protocol classification changed")
    if payload["status"] != "frozen_before_runner_preflight":
        raise ValueError("short-horizon protocol status changed")
    if payload["required_host"] != "yanlih100n1":
        raise ValueError("short-horizon host binding changed")
    if payload["horizon"] != {
        "maximum_iterations": 6,
        "checkpoint_every_iterations": 1,
        "primary_ppo_checkpoint_before_llm_each_iteration": True,
        "resume_requires_exact_control_hash": True,
        "automatic_resume": False,
    }:
        raise ValueError("short-horizon iteration boundary changed")
    if payload["llm_wall_clock_budget"] != {
        "maximum_seconds_per_iteration": 60.0,
        "maximum_seconds_total": 180.0,
        "minimum_transport_timeout_seconds": 0.01,
        "persist_across_checkpoints": True,
        "exhaustion_outcome": "ppo_only_degraded",
    }:
        raise ValueError("short-horizon LLM time boundary changed")
    if payload["circuit_breaker"] != {
        "maximum_consecutive_llm_failures": 2,
        "circuit_cooldown_iterations": 3,
        "persist_across_checkpoints": True,
        "skip_outcome": "ppo_only_degraded",
    }:
        raise ValueError("short-horizon circuit boundary changed")
    if payload["kl_numerics"] != {
        "negative_tolerance": KL_NUMERICAL_NEGATIVE_TOLERANCE,
        "within_tolerance_action": "clamp_to_zero",
        "below_negative_tolerance_action": "fatal_contract_violation",
    }:
        raise ValueError("short-horizon KL numerical policy changed")
    if payload["scientific_budget"] != {
        "replicates_per_selected_candidate": 5,
        "maximum_selected_candidates_per_iteration": 8,
        "maximum_verification_branches_per_iteration": 80,
        "maximum_pairwise_optimizer_steps_per_iteration": 1,
        "maximum_requested_evaluator_calls_total": 1248,
    }:
        raise ValueError("short-horizon scientific budget changed")
    if payload["failure_policy"] != {
        "recoverable_llm_failure_continues_ppo": True,
        "valid_abstention_continues_ppo": True,
        "insufficient_soft_mass_continues_ppo": True,
        "pairwise_kl_rollback_continues_ppo": True,
        "silent_fallback_forbidden": True,
        "integrity_failure_is_fatal": True,
        "automatic_rerun": False,
    }:
        raise ValueError("short-horizon failure policy changed")
    if payload["authorization_state"] != {
        "runner_implementation_authorized": True,
        "no_credential_full_runtime_preflight_authorized": True,
        "real_multi_iteration_execution_authorized": False,
        "credentials_loading_authorized": False,
        "external_api_requests_authorized": False,
        "ppo_execution_authorized": False,
        "oracle_execution_authorized": False,
        "automatic_rerun_authorized": False,
        "formal_training_authorized": False,
        "sealed_test_access_authorized": False,
        "algorithm_effectiveness_claim_authorized": False,
        "scientific_claim_authorized": False,
    }:
        raise ValueError("short-horizon authorization boundary changed")
    prerequisites = payload["prerequisites"]
    required_prerequisites = {
        "v3_protocol",
        "soft_pair_config",
        "v3_full_runtime_preflight_report",
        "v3_real_single_iteration_report",
        "required_v3_preflight_decision",
        "required_v3_real_decision",
        "required_v3_implementation_commit",
    }
    if set(prerequisites) != required_prerequisites:
        raise ValueError("short-horizon prerequisite keys changed")
    v3_protocol = _resolve_bound(root, prerequisites["v3_protocol"])
    soft_config = _resolve_bound(root, prerequisites["soft_pair_config"])
    preflight_report = _resolve_bound(
        root, prerequisites["v3_full_runtime_preflight_report"]
    )
    real_report = _resolve_bound(root, prerequisites["v3_real_single_iteration_report"])
    preflight = json.loads(preflight_report.read_text(encoding="utf-8"))
    real = json.loads(real_report.read_text(encoding="utf-8"))
    if preflight["module_decision"] != prerequisites[
        "required_v3_preflight_decision"
    ]:
        raise RuntimeError("v3 preflight prerequisite decision changed")
    if real["module_decision"] != prerequisites["required_v3_real_decision"]:
        raise RuntimeError("v3 real prerequisite decision changed")
    if real["source"]["commit"] != prerequisites[
        "required_v3_implementation_commit"
    ]:
        raise RuntimeError("v3 real implementation binding changed")
    result = dict(payload)
    result["resolved_prerequisites"] = {
        "v3_protocol": str(v3_protocol),
        "soft_pair_config": str(soft_config),
        "v3_full_runtime_preflight_report": str(preflight_report),
        "v3_real_single_iteration_report": str(real_report),
    }
    return result


def save_short_horizon_checkpoint(
    *,
    engine,
    breaker: LLMAuxiliaryCircuitBreaker,
    wall_budget: LLMWallClockBudget,
    engine_checkpoint_path: Path,
    control_checkpoint_path: Path,
    completed_iteration: int,
    protocol_sha256: str,
    history_digest: str,
    previous_control_sha256: Optional[str],
) -> Mapping[str, Any]:
    """Commit engine and controller state at one atomic iteration boundary."""

    if isinstance(completed_iteration, bool) or int(completed_iteration) < 1:
        raise ValueError("completed iteration must be positive")
    for name, value in (
        ("protocol_sha256", protocol_sha256),
        ("history_digest", history_digest),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("%s must be a SHA-256 digest" % name)
    if previous_control_sha256 is not None and (
        not isinstance(previous_control_sha256, str)
        or len(previous_control_sha256) != 64
    ):
        raise ValueError("previous control hash must be null or SHA-256")
    engine_path = Path(engine_checkpoint_path).resolve()
    control_path = Path(control_checkpoint_path).resolve()
    if engine_path == control_path:
        raise ValueError("engine and control checkpoint paths must differ")
    if engine_path.exists() or control_path.exists():
        raise FileExistsError("short-horizon checkpoint target already exists")
    engine.save_checkpoint(engine_path)
    engine_sha256 = sha256_path(engine_path)
    payload = {
        "schema_version": SHORT_HORIZON_CONTROL_CHECKPOINT_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "completed_iteration": int(completed_iteration),
        "engine_checkpoint": {
            "path": str(engine_path),
            "sha256": engine_sha256,
        },
        "circuit_breaker": breaker.state_dict(),
        "llm_wall_clock_budget": wall_budget.state_dict(),
        "history_digest": history_digest,
        "previous_control_sha256": previous_control_sha256,
        "credentials_persisted": False,
        "api_key_persisted": False,
    }
    write_json(control_path, payload)
    return {
        "control_checkpoint_path": str(control_path),
        "control_checkpoint_sha256": sha256_path(control_path),
        "engine_checkpoint_path": str(engine_path),
        "engine_checkpoint_sha256": engine_sha256,
        "completed_iteration": int(completed_iteration),
    }


def load_short_horizon_checkpoint(
    *,
    engine,
    breaker: LLMAuxiliaryCircuitBreaker,
    wall_budget: LLMWallClockBudget,
    control_checkpoint_path: Path,
    expected_protocol_sha256: str,
    expected_control_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    """Restore engine, circuit, and cumulative LLM time as one identity."""

    control_path = Path(control_checkpoint_path).resolve(strict=True)
    observed_control_sha256 = sha256_path(control_path)
    if (
        expected_control_sha256 is not None
        and observed_control_sha256 != expected_control_sha256
    ):
        raise ContractViolation("short-horizon control checkpoint hash mismatch")
    payload = json.loads(control_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_sha256",
        "completed_iteration",
        "engine_checkpoint",
        "circuit_breaker",
        "llm_wall_clock_budget",
        "history_digest",
        "previous_control_sha256",
        "credentials_persisted",
        "api_key_persisted",
    }
    if set(payload) != required:
        raise ContractViolation("short-horizon control checkpoint schema mismatch")
    if payload["schema_version"] != SHORT_HORIZON_CONTROL_CHECKPOINT_SCHEMA:
        raise ContractViolation("short-horizon control checkpoint version mismatch")
    if payload["protocol_sha256"] != expected_protocol_sha256:
        raise ContractViolation("short-horizon protocol identity mismatch")
    if payload["credentials_persisted"] is not False:
        raise ContractViolation("credentials cannot enter a control checkpoint")
    if payload["api_key_persisted"] is not False:
        raise ContractViolation("API key cannot enter a control checkpoint")
    completed = payload["completed_iteration"]
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 1:
        raise ContractViolation("invalid completed-iteration checkpoint value")
    if not isinstance(payload["history_digest"], str) or len(
        payload["history_digest"]
    ) != 64:
        raise ContractViolation("invalid short-horizon history digest")
    previous_control_sha256 = payload["previous_control_sha256"]
    if previous_control_sha256 is not None and (
        not isinstance(previous_control_sha256, str)
        or len(previous_control_sha256) != 64
    ):
        raise ContractViolation("invalid previous control-checkpoint digest")
    engine_record = payload["engine_checkpoint"]
    if set(engine_record) != {"path", "sha256"}:
        raise ContractViolation("invalid engine-checkpoint record")
    if not isinstance(engine_record["path"], str) or not isinstance(
        engine_record["sha256"], str
    ):
        raise ContractViolation("invalid engine-checkpoint field type")
    try:
        engine_path = Path(engine_record["path"]).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ContractViolation("engine checkpoint cannot be resolved") from error
    if len(engine_record["sha256"]) != 64 or sha256_path(engine_path) != engine_record[
        "sha256"
    ]:
        raise ContractViolation("short-horizon engine checkpoint hash mismatch")

    # Validate all controller state before mutating the supplied engine. A
    # malformed control payload must fail closed without a partial restore.
    staged_breaker = LLMAuxiliaryCircuitBreaker(breaker.config)
    staged_wall_budget = LLMWallClockBudget(
        wall_budget.config, clock=wall_budget.clock
    )
    staged_breaker.load_state_dict(payload["circuit_breaker"])
    staged_wall_budget.load_state_dict(payload["llm_wall_clock_budget"])
    attempted_iterations = staged_wall_budget.completed_attempt_iterations
    if attempted_iterations != sorted(set(attempted_iterations)) or any(
        value > completed for value in attempted_iterations
    ):
        raise ContractViolation("LLM attempted-iteration history is inconsistent")

    engine.load_checkpoint(engine_path)
    breaker.load_state_dict(payload["circuit_breaker"])
    wall_budget.load_state_dict(payload["llm_wall_clock_budget"])
    return {
        "control_checkpoint_path": str(control_path),
        "control_checkpoint_sha256": observed_control_sha256,
        "engine_checkpoint_path": str(engine_path),
        "engine_checkpoint_sha256": engine_record["sha256"],
        "completed_iteration": completed,
        "history_digest": payload["history_digest"],
        "previous_control_sha256": previous_control_sha256,
    }


def history_digest(iteration_records) -> str:
    serialized = json.dumps(
        list(iteration_records),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def chained_history_digest(
    previous_digest: Optional[str], iteration_record: Mapping[str, Any]
) -> str:
    previous = previous_digest or ("0" * 64)
    if not isinstance(previous, str) or len(previous) != 64:
        raise ValueError("previous history digest must be null or SHA-256")
    serialized = json.dumps(
        {
            "previous_history_digest": previous,
            "iteration_record": dict(iteration_record),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
