#!/usr/bin/env python
"""Run one authorization-gated PPO plus SciCF integration-v2 cycle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import resource
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from reproduction.framework.io import git_identity, write_json
from reproduction.p2.budget import RequestedCallBudgetManager, evaluator_ledger_delta
from reproduction.p2.engine import PPOEngineConfig
from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    sha256_path,
    verify_accepted_binding,
)
from reproduction.scicf.llm.api_client import APISettings, OpenAICompatibleClient

from .contracts import (
    SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID,
    PairwiseRefinementConfig,
    SchemaRecoveryConfig,
)
from .pipeline import (
    FrozenPolicySampler,
    build_online_candidate_pool,
    eligible_online_episode_ids,
    refine_verified_pairs,
    verify_selected_candidates,
)
from .prompt import build_online_acquisition_request
from .response_guard import complete_with_bounded_schema_repair
from .run_pairwise_stability import _head_hash
from .run_smoke import build_runtime, iteration_summary
from .stability import factorized_policy_drift, pairwise_preference_metrics


PROTOCOL_CLASSIFICATION = "single_iteration_engineering_integration_protocol_only"
AUTHORIZATION_OPERATIONS = {
    "credentials_loading_authorized": True,
    "external_api_requests_authorized": True,
    "single_iteration_ppo_authorized": True,
    "oracle_execution_authorized": True,
    "single_iteration_integration_rerun_authorized": True,
    "automatic_rerun_authorized": False,
    "multi_iteration_training_authorized": False,
    "formal_training_authorized": False,
    "sealed_test_access_authorized": False,
    "baseline_mutation_authorized": False,
    "algorithm_effectiveness_claim_authorized": False,
    "scientific_claim_authorized": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--polybert-path", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolve_bound(root: Path, relative: str, expected_sha256: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if root != path and root not in path.parents:
        raise ValueError("integration-v2 bound input escaped the repository root")
    observed = _sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(
            "integration-v2 input hash mismatch for %s: %s != %s"
            % (relative, observed, expected_sha256)
        )
    return path


def load_protocol(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "classification",
        "status",
        "required_host",
        "base_seed",
        "prerequisites",
        "allowed_implementation_delta",
        "accepted_binding",
        "budget",
        "acquisition",
        "failure_atomicity",
        "verification",
        "pairwise_refinement",
        "integration_gate",
        "ppo",
        "authorization_state",
    }
    if set(payload) != required:
        raise ValueError(
            "integration-v2 protocol keys differ; missing=%s extra=%s"
            % (sorted(required - set(payload)), sorted(set(payload) - required))
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported integration-v2 protocol schema")
    if payload["protocol_id"] != SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID:
        raise ValueError("unsupported integration-v2 protocol identity")
    if payload["classification"] != PROTOCOL_CLASSIFICATION:
        raise ValueError("integration-v2 protocol is not engineering-only")
    if payload["status"] != "frozen_unimplemented_unexecuted":
        raise ValueError("integration-v2 protocol freeze status changed")

    acquisition = payload["acquisition"]
    guard = SchemaRecoveryConfig(
        maximum_schema_attempts=acquisition["maximum_schema_attempts_per_pool"],
        transport_retries_per_attempt=acquisition[
            "transport_retries_per_schema_attempt"
        ],
        max_output_tokens=acquisition["max_output_tokens_per_schema_attempt"],
        timeout_seconds=acquisition["timeout_seconds_per_transport"],
    )
    if acquisition["maximum_semantic_repairs_per_pool"] != guard.maximum_semantic_repairs:
        raise ValueError("integration-v2 semantic-repair bound is inconsistent")
    if acquisition["maximum_http_transmissions_per_pool"] != guard.maximum_http_transmissions:
        raise ValueError("integration-v2 per-pool HTTP bound is inconsistent")
    if acquisition["maximum_http_transmissions_total"] != (
        int(acquisition["pool_count"]) * guard.maximum_http_transmissions
    ):
        raise ValueError("integration-v2 total HTTP bound is inconsistent")
    if acquisition["maximum_completed_response_contents_total"] != (
        int(acquisition["pool_count"]) * guard.maximum_schema_attempts
    ):
        raise ValueError("integration-v2 completed-response bound is inconsistent")
    if int(acquisition["pool_decision_count"]) != 2:
        raise ValueError("integration-v2 requires exactly two pool decisions")
    if int(acquisition["pool_count"]) != 2 or int(acquisition["pool_size"]) != 24:
        raise ValueError("integration-v2 freezes two 24-candidate pools")
    if int(acquisition["maximum_selected_per_pool"]) != 4:
        raise ValueError("integration-v2 freezes B=4")
    if acquisition["episode_selection"] != "outcome_blind_deterministic_shuffle":
        raise ValueError("integration-v2 episode selection changed")
    for name in (
        "complete_all_validated_decisions_before_oracle",
        "allow_partial_selection",
        "allow_abstention",
        "persist_request_before_provider_call",
        "persist_raw_content_before_validation",
        "persist_invalid_raw_content",
        "preserve_exact_candidate_allowlist_during_repair",
        "preserve_maximum_budget_during_repair",
    ):
        if acquisition.get(name) is not True:
            raise ValueError("integration-v2 acquisition must enable %s" % name)
    for name in (
        "replay_invalid_raw_text_to_model",
        "allow_fallback_selection",
        "allow_cached_response_substitution",
    ):
        if acquisition.get(name) is not False:
            raise ValueError("integration-v2 acquisition must disable %s" % name)

    expected_delta = {
        "replace_direct_completion_and_inline_validation_with_response_guard": True,
        "write_terminal_failure_report_on_guard_failure": True,
        "aggregate_all_attempt_and_transport_accounting": True,
        "change_prompt_content": False,
        "change_candidate_construction": False,
        "change_episode_selection": False,
        "change_seed": False,
        "change_ppo": False,
        "change_verification": False,
        "change_pairwise_refinement": False,
        "change_gate_thresholds": False,
    }
    if payload["allowed_implementation_delta"] != expected_delta:
        raise ValueError("integration-v2 implementation delta changed")
    for name, value in payload["authorization_state"].items():
        if name == "separate_execution_manifest_required":
            if value is not True:
                raise ValueError("integration-v2 requires a separate execution manifest")
        elif name == "next_scope_if_future_run_passes":
            if value != "separate_bounded_short_horizon_multi_iteration_protocol_freeze_only":
                raise ValueError("integration-v2 next-scope boundary changed")
        elif value is not False:
            raise ValueError("protocol freeze cannot authorize %s" % name)
    PPOEngineConfig(**payload["ppo"])
    pairwise = dict(payload["pairwise_refinement"])
    if pairwise.pop("maximum_optimizer_steps") != 1:
        raise ValueError("integration-v2 permits at most one pairwise optimizer step")
    PairwiseRefinementConfig(**pairwise)
    return payload


def load_execution_authorization(
    path: Path,
    *,
    protocol_sha256: str,
    implementation_commit: str,
    output_dir: Path,
) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "authorization_id",
        "protocol_id",
        "protocol_sha256",
        "implementation_commit",
        "authorized_output_directory",
        "maximum_slurm_runs",
        "authorized_operations",
    }
    if set(payload) != required:
        raise ValueError(
            "execution authorization keys differ; missing=%s extra=%s"
            % (sorted(required - set(payload)), sorted(set(payload) - required))
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported execution authorization schema")
    if not isinstance(payload["authorization_id"], str) or not payload["authorization_id"]:
        raise ValueError("execution authorization_id must be non-empty")
    if payload["protocol_id"] != SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID:
        raise ValueError("execution authorization protocol mismatch")
    if payload["protocol_sha256"] != protocol_sha256:
        raise ValueError("execution authorization protocol hash mismatch")
    if payload["implementation_commit"] != implementation_commit:
        raise ValueError("execution authorization implementation commit mismatch")
    if Path(payload["authorized_output_directory"]).resolve() != output_dir.resolve():
        raise ValueError("execution authorization output directory mismatch")
    if payload["maximum_slurm_runs"] != 1:
        raise ValueError("execution authorization must permit exactly one Slurm run")
    if payload["authorized_operations"] != AUTHORIZATION_OPERATIONS:
        raise ValueError("execution authorization scope is not the exact bounded scope")
    return payload


def verify_protocol_bindings(
    root: Path, protocol: Mapping[str, Any]
) -> Mapping[str, Any]:
    prerequisite = protocol["prerequisites"]
    v1_archive_path = _resolve_bound(
        root,
        prerequisite["v1_archive_decision"],
        prerequisite["v1_archive_decision_sha256"],
    )
    v1_config_path = _resolve_bound(
        root, prerequisite["v1_config"], prerequisite["v1_config_sha256"]
    )
    _resolve_bound(root, prerequisite["v1_runner"], prerequisite["v1_runner_sha256"])
    _resolve_bound(
        root, prerequisite["response_guard"], prerequisite["response_guard_sha256"]
    )
    schema_path = _resolve_bound(
        root,
        prerequisite["schema_robustness_report"],
        prerequisite["schema_robustness_report_sha256"],
    )
    pairwise_path = _resolve_bound(
        root,
        prerequisite["pairwise_stability_report"],
        prerequisite["pairwise_stability_report_sha256"],
    )
    accepted_manifest_path = _resolve_bound(
        root,
        prerequisite["accepted_stage0_manifest"],
        prerequisite["accepted_stage0_manifest_sha256"],
    )
    archive = json.loads(v1_archive_path.read_text(encoding="utf-8"))
    if archive["decision"] != prerequisite["v1_required_decision"]:
        raise RuntimeError("v1 archive no-go binding changed")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema["module_decision"] != prerequisite["schema_robustness_required_decision"]:
        raise RuntimeError("schema-robustness decision binding changed")
    if schema["source"]["commit"] != prerequisite["schema_robustness_source_commit"]:
        raise RuntimeError("schema-robustness source binding changed")
    if schema["classification"]["next_scope_authorized"] is not True:
        raise RuntimeError("schema-robustness protocol-freeze scope is closed")
    if schema["classification"]["single_iteration_integration_rerun_authorized"] is not False:
        raise RuntimeError("schema report unexpectedly authorizes integration execution")
    pairwise_report = json.loads(pairwise_path.read_text(encoding="utf-8"))
    if pairwise_report["module_decision"] != "go_single_iteration_integration_smoke":
        raise RuntimeError("pairwise-stability prerequisite is closed")
    if pairwise_report["classification"]["next_scope_authorized"] is not True:
        raise RuntimeError("pairwise-stability next scope is closed")

    v1 = json.loads(v1_config_path.read_text(encoding="utf-8"))
    if protocol["base_seed"] != v1["base_seed"]:
        raise RuntimeError("integration-v2 seed differs from v1")
    if protocol["accepted_binding"] != v1["accepted_binding"]:
        raise RuntimeError("integration-v2 Stage-0 binding differs from v1")
    if protocol["ppo"] != v1["ppo"]:
        raise RuntimeError("integration-v2 PPO configuration differs from v1")
    if protocol["verification"] != v1["verification"]:
        raise RuntimeError("integration-v2 verification configuration differs from v1")
    v2_pairwise = dict(protocol["pairwise_refinement"])
    v2_pairwise.pop("maximum_optimizer_steps")
    if v2_pairwise != v1["pairwise_refinement"]:
        raise RuntimeError("integration-v2 pairwise configuration differs from v1")
    v2_gate = dict(protocol["integration_gate"])
    v2_gate.pop("slurm_completed_is_not_gate_pass")
    if v2_gate != v1["integration_gate"]:
        raise RuntimeError("integration-v2 thresholds differ from v1")
    for name in ("maximum_requested_calls", "maximum_unique_calls"):
        if protocol["budget"][name] != v1["budget"][name]:
            raise RuntimeError("integration-v2 evaluator budget differs from v1")
    mapping = {
        "pool_count": "pool_count",
        "pool_size": "pool_size",
        "maximum_selected_per_pool": "maximum_selected_per_pool",
        "pool_seed_stride": "pool_seed_stride",
        "episode_selection": "episode_selection",
    }
    for v2_name, v1_name in mapping.items():
        if protocol["acquisition"][v2_name] != v1["acquisition"][v1_name]:
            raise RuntimeError("integration-v2 acquisition invariant changed: %s" % v2_name)
    return {
        "v1_archive": archive,
        "schema_report": schema,
        "pairwise_report": pairwise_report,
        "accepted_manifest_path": accepted_manifest_path,
        "accepted_manifest": json.loads(
            accepted_manifest_path.read_text(encoding="utf-8")
        ),
    }


def schema_recovery_config(protocol: Mapping[str, Any]) -> SchemaRecoveryConfig:
    acquisition = protocol["acquisition"]
    return SchemaRecoveryConfig(
        maximum_schema_attempts=acquisition["maximum_schema_attempts_per_pool"],
        transport_retries_per_attempt=acquisition[
            "transport_retries_per_schema_attempt"
        ],
        max_output_tokens=acquisition["max_output_tokens_per_schema_attempt"],
        timeout_seconds=acquisition["timeout_seconds_per_transport"],
    )


def _attempt_usage(pool_dir: Path) -> Mapping[str, Any]:
    records = []
    for path in sorted(pool_dir.glob("attempt-*-raw-response.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    token_complete = bool(records) and all(
        item.get("prompt_tokens") is not None
        and item.get("completion_tokens") is not None
        for item in records
    )
    return {
        "returned_content_count": len(records),
        "http_transmissions_observed": sum(
            int(item["transport_retries_used"]) + 1 for item in records
        ),
        "token_accounting_complete": token_complete,
        "prompt_tokens": (
            sum(int(item["prompt_tokens"]) for item in records)
            if token_complete
            else None
        ),
        "completion_tokens": (
            sum(int(item["completion_tokens"]) for item in records)
            if token_complete
            else None
        ),
    }


def run_guarded_pool_decisions(
    *,
    client,
    provider: Mapping[str, Any],
    prepared_pools: Sequence[Mapping[str, Any]],
    output_root: Path,
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = int(protocol["acquisition"]["pool_count"])
    if len(prepared_pools) != expected:
        raise ValueError("guarded acquisition requires exactly two prepared pools")
    if output_root.exists():
        raise FileExistsError("guarded acquisition output already exists")
    output_root.mkdir(parents=True)
    outcomes = []
    usages = []
    selected_ids = []
    guard_config = schema_recovery_config(protocol)
    for item in prepared_pools:
        pool_index = int(item["pool_index"])
        pool_dir = output_root / ("pool-%02d" % pool_index)
        try:
            outcome = complete_with_bounded_schema_repair(
                client=client,
                request=item["request"],
                provider=provider,
                output_dir=pool_dir,
                config=guard_config,
                seed=int(item["pool_seed"]),
            )
        except ValueError as error:
            outcome = {
                "schema_version": 1,
                "status": "fail_closed_provider_or_guard_value_error",
                "request_id": item["request"]["request_id"],
                "attempt_count": len(list(pool_dir.glob("attempt-*-request.json"))),
                "semantic_repair_count": max(
                    0, len(list(pool_dir.glob("attempt-*-request.json"))) - 1
                ),
                "selected_intervention_ids": [],
                "oracle_selection_authorized": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "raw_capture_unavailable_only_if_no_model_content_extracted": True,
            }
            write_json(pool_dir / "outcome.json", outcome)
        usage = _attempt_usage(pool_dir)
        usages.append(usage)
        outcomes.append(
            {
                "pool_index": pool_index,
                "request_id": item["request"]["request_id"],
                "status": outcome["status"],
                "attempt_count": int(outcome["attempt_count"]),
                "semantic_repair_count": int(outcome["semantic_repair_count"]),
                "selected_intervention_ids": list(
                    outcome["selected_intervention_ids"]
                ),
                "oracle_selection_authorized": bool(
                    outcome["oracle_selection_authorized"]
                ),
                "outcome_sha256": _sha256_path(pool_dir / "outcome.json"),
                "usage": usage,
            }
        )
        if outcome["status"] != "validated" or not outcome["oracle_selection_authorized"]:
            break
        selected_ids.extend(outcome["selected_intervention_ids"])

    all_validated = len(outcomes) == expected and all(
        item["status"] == "validated" and item["oracle_selection_authorized"]
        for item in outcomes
    )
    token_complete = all(item["token_accounting_complete"] for item in usages)
    transport_exact = all(
        item["status"] == "validated"
        or item["status"] == "fail_closed_schema_exhausted"
        for item in outcomes
    )
    transmission_upper_bound = sum(
        usage["http_transmissions_observed"]
        + (
            0
            if outcome["status"]
            in {"validated", "fail_closed_schema_exhausted"}
            else int(guard_config.transport_retries_per_attempt) + 1
        )
        for outcome, usage in zip(outcomes, usages)
    )
    summary = {
        "status": "validated_all_pools" if all_validated else "fail_closed",
        "pool_decision_count_expected": expected,
        "pool_decision_count_started": len(outcomes),
        "all_pool_decisions_validated": all_validated,
        "oracle_selection_authorized": all_validated,
        "selected_intervention_ids": selected_ids if all_validated else [],
        "semantic_attempt_count": sum(item["attempt_count"] for item in outcomes),
        "semantic_repair_count": sum(
            item["semantic_repair_count"] for item in outcomes
        ),
        "returned_content_count": sum(
            item["returned_content_count"] for item in usages
        ),
        "http_transmissions_observed": sum(
            item["http_transmissions_observed"] for item in usages
        ),
        "http_transmissions_observed_is_exact": transport_exact,
        "http_transmissions_used": (
            sum(item["http_transmissions_observed"] for item in usages)
            if transport_exact
            else None
        ),
        "http_transmissions_upper_bound_for_started_decisions": (
            transmission_upper_bound
        ),
        "maximum_http_transmissions_total": int(
            protocol["acquisition"]["maximum_http_transmissions_total"]
        ),
        "token_accounting_complete": token_complete,
        "reported_tokens": {
            "prompt": (
                sum(int(item["prompt_tokens"]) for item in usages)
                if token_complete
                else None
            ),
            "completion": (
                sum(int(item["completion_tokens"]) for item in usages)
                if token_complete
                else None
            ),
        },
        "pool_outcomes": outcomes,
    }
    write_json(output_root / "acquisition-summary.json", summary)
    return summary


def _guard_failure_report(
    *,
    source: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    authorization: Mapping[str, Any],
    ppo_result,
    acquisition: Mapping[str, Any],
    pool_records: Sequence[Mapping[str, Any]],
    final_ledger: Mapping[str, Any],
    started: float,
) -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID,
        "execution_status": "failed",
        "module_decision": "no_go_multi_iteration_protocol_freeze",
        "classification": {
            "scope": "single-iteration-ppo-plus-scicf-engineering-smoke-v2",
            "failure_phase": "guarded_acquisition_before_oracle_verification",
            "next_scope_authorized": False,
            "multi_iteration_training_authorized": False,
            "formal_training_authorized": False,
            "algorithm_effectiveness_established": False,
            "scientific_claim_authorized": False,
        },
        "source": source,
        "protocol_sha256": protocol_sha256,
        "authorization_id": authorization["authorization_id"],
        "standard_ppo": iteration_summary(ppo_result),
        "pool_records": list(pool_records),
        "acquisition": acquisition,
        "verification": {
            "started": False,
            "selected_count": 0,
            "verified_count": 0,
            "post_guard_requested_calls": 0,
        },
        "pairwise_refinement": {"started": False, "optimizer_steps": 0},
        "final_evaluator_ledger": dict(final_ledger),
        "checkpoint": None,
        "external_api_invoked": True,
        "local_model_invoked": False,
        "sealed_test_accessed": False,
        "automatic_rerun_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _pairwise_config(protocol: Mapping[str, Any]) -> PairwiseRefinementConfig:
    payload = dict(protocol["pairwise_refinement"])
    payload.pop("maximum_optimizer_steps")
    return PairwiseRefinementConfig(**payload)


def main() -> None:
    args = parse_args()
    root = args.dapigen_root.resolve()
    protocol_path = args.protocol.resolve()
    authorization_path = args.execution_authorization.resolve()
    output_dir = args.output_dir.resolve()
    protocol = load_protocol(protocol_path)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("integration-v2 requires Slurm")
    if socket.gethostname() != protocol["required_host"]:
        raise RuntimeError("integration-v2 is bound to n001")
    source = git_identity(root)
    if source.get("dirty") is not False:
        raise RuntimeError("integration-v2 requires a clean Git worktree")
    protocol_sha256 = _sha256_path(protocol_path)
    authorization = load_execution_authorization(
        authorization_path,
        protocol_sha256=protocol_sha256,
        implementation_commit=source["commit"],
        output_dir=output_dir,
    )
    bindings = verify_protocol_bindings(root, protocol)
    accepted_manifest_path = bindings["accepted_manifest_path"]
    accepted_manifest = bindings["accepted_manifest"]
    binding = verify_accepted_binding(
        root, accepted_manifest, protocol["accepted_binding"]["environment_id"]
    )
    if binding["stage0_source_changed_from_accepted"]:
        raise RuntimeError("accepted Stage-0 source changed before integration-v2")
    if output_dir.exists():
        raise FileExistsError("integration-v2 output already exists")
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "schema_version": 1,
            "status": "declared_after_authorization_and_binding_before_credentials",
            "protocol_id": SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "protocol_sha256": protocol_sha256,
            "authorization_id": authorization["authorization_id"],
            "authorization_sha256": _sha256_path(authorization_path),
            "accepted_manifest_sha256": _sha256_path(accepted_manifest_path),
            "credentials_loaded_at_intent_time": False,
            "credentials_path_logged": False,
            "api_key_logged": False,
            "automatic_rerun_authorized": False,
            "multi_iteration_training_authorized": False,
        },
    )

    settings = APISettings.from_private_file(args.credentials_file.resolve())
    provider = settings.public_identity()
    provider["external_api"] = True
    write_json(
        output_dir / "provider-public-identity.json",
        {"provider": provider, "api_key_logged": False, "credentials_path_logged": False},
    )
    polybert_path = args.polybert_path.resolve()
    if not polybert_path.is_dir():
        raise FileNotFoundError("polyBERT path is not a directory")
    started = time.perf_counter()
    components, engine, specification, run_contract = build_runtime(
        root, accepted_manifest, binding, protocol, polybert_path
    )
    initial_policy_sha256 = engine.policy_state_sha256
    behavior_policy = FrozenPolicySampler(engine.model, protocol["ppo"]["device"])
    ppo_result = engine.run_iteration(query_requested_calls=0)
    post_ppo_policy_sha256 = engine.policy_state_sha256
    if behavior_policy.state_sha256 != ppo_result.rollout.frozen_policy.state_sha256:
        raise RuntimeError("behavior-policy copy differs from PPO rollout policy")

    eligible = list(
        eligible_online_episode_ids(ppo_result.rollout, prefer_successful=False)
    )
    random.Random(int(protocol["base_seed"])).shuffle(eligible)
    pool_count = int(protocol["acquisition"]["pool_count"])
    if len(eligible) < pool_count:
        raise RuntimeError("rollout lacks two complete cross-timestep episodes")
    chosen_episode_ids = tuple(eligible[:pool_count])
    prepared_pools = []
    pool_records = []
    all_candidates = []
    for pool_index, episode_id in enumerate(chosen_episode_ids):
        pool_seed = int(protocol["base_seed"]) + int(
            protocol["acquisition"]["pool_seed_stride"]
        ) * pool_index
        candidates, trajectory_context = build_online_candidate_pool(
            rollout=ppo_result.rollout,
            core=components.core,
            behavior_policy=behavior_policy,
            pool_size=int(protocol["acquisition"]["pool_size"]),
            seed=pool_seed,
            episode_id=episode_id,
        )
        request_id = "%s-pool-%02d-job-%s" % (
            SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID,
            pool_index,
            os.environ["SLURM_JOB_ID"],
        )
        request = build_online_acquisition_request(
            request_id=request_id,
            trajectory_context=trajectory_context,
            candidates=candidates,
            maximum_budget=int(
                protocol["acquisition"]["maximum_selected_per_pool"]
            ),
        )
        write_json(
            output_dir / "pools" / ("pool-%02d-candidates.json" % pool_index),
            {
                "pool_index": pool_index,
                "episode_id": episode_id,
                "trajectory_context": trajectory_context,
                "candidates": [item.audit_row() for item in candidates],
            },
        )
        write_json(
            output_dir / "requests" / ("pool-%02d-request.json" % pool_index),
            request,
        )
        prepared_pools.append(
            {
                "pool_index": pool_index,
                "pool_seed": pool_seed,
                "request": request,
            }
        )
        pool_records.append(
            {
                "pool_index": pool_index,
                "episode_id": episode_id,
                "candidate_ids": list(request["candidate_ids"]),
                "candidate_timesteps": sorted(
                    {item.timestep for item in candidates}
                ),
                "presented_ids_match_pool": set(request["presented_candidate_ids"])
                == set(request["candidate_ids"]),
                "reward_truth_exposed": bool(
                    request["candidate_reward_truth_exposed"]
                    or request["factual_reward_truth_exposed"]
                ),
                "policy_score_exposed": bool(request["policy_score_exposed"]),
            }
        )
        all_candidates.extend(candidates)

    client = OpenAICompatibleClient(settings)
    acquisition = run_guarded_pool_decisions(
        client=client,
        provider=provider,
        prepared_pools=prepared_pools,
        output_root=output_dir / "guarded-acquisition",
        protocol=protocol,
    )
    if not acquisition["oracle_selection_authorized"]:
        report = _guard_failure_report(
            source=source,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            authorization=authorization,
            ppo_result=ppo_result,
            acquisition=acquisition,
            pool_records=pool_records,
            final_ledger=engine.environment.oracle_ledger(),
            started=started,
        )
        write_json(output_dir / "integration-v2-report.json", report)
        print(json.dumps({"execution_status": "failed", "failure_phase": report["classification"]["failure_phase"]}, sort_keys=True))
        raise SystemExit(1)

    selected_ids = tuple(acquisition["selected_intervention_ids"])
    candidate_map = {item.candidate_id: item for item in all_candidates}
    if len(candidate_map) != len(all_candidates):
        raise RuntimeError("candidate ids are not unique across integration-v2 pools")
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("selected candidate ids repeat across integration-v2 pools")
    selected_by_pool = {
        item["pool_index"]: item["selected_intervention_ids"]
        for item in acquisition["pool_outcomes"]
    }
    for pool in pool_records:
        pool["selected_ids"] = list(selected_by_pool[pool["pool_index"]])
        pool["abstained"] = len(pool["selected_ids"]) == 0

    query_budget = RequestedCallBudgetManager(engine.environment.oracle_ledger)
    maximum_verification_calls = (
        2 * len(selected_ids) * int(protocol["verification"]["replicates"])
    )
    reservation = query_budget.reserve(maximum_verification_calls)
    ledger_before_verification = dict(engine.environment.oracle_ledger())
    try:
        verifications = verify_selected_candidates(
            environment=engine.environment,
            policy=behavior_policy,
            candidates_by_id=candidate_map,
            selected_ids=selected_ids,
            replicates=int(protocol["verification"]["replicates"]),
            seed=int(protocol["base_seed"]),
            delta_tolerance=float(
                protocol["pairwise_refinement"]["delta_tolerance"]
            ),
        )
        ledger_after_verification = dict(engine.environment.oracle_ledger())
        verification_delta = evaluator_ledger_delta(
            ledger_before_verification, ledger_after_verification
        )
        query_budget.reconcile(reservation, verification_delta.requested_calls)
    except Exception:
        query_budget.release(reservation)
        raise
    write_json(
        output_dir / "oracle-verification.json",
        {
            "selected_ids": list(selected_ids),
            "unselected_candidates_verified": False,
            "replicates": int(protocol["verification"]["replicates"]),
            "maximum_branch_count": maximum_verification_calls,
            "ledger_delta": asdict(verification_delta),
            "records": [item.to_dict() for item in verifications],
        },
    )

    accepted = tuple(item for item in verifications if item.accepted)
    before_pairwise_model = copy.deepcopy(engine.model).to(engine.device).eval()
    value_head_before = _head_hash(engine.model, "value_head")
    pairwise_before = (
        pairwise_preference_metrics(
            model=engine.model,
            candidates_by_id=candidate_map,
            verifications=accepted,
            device=engine.device,
        )
        if accepted
        else None
    )
    refinement = refine_verified_pairs(
        engine=engine,
        candidates_by_id=candidate_map,
        verifications=verifications,
        config=_pairwise_config(protocol),
    )
    value_head_after = _head_hash(engine.model, "value_head")
    pairwise_after = (
        pairwise_preference_metrics(
            model=engine.model,
            candidates_by_id=candidate_map,
            verifications=accepted,
            device=engine.device,
        )
        if accepted
        else None
    )
    pairwise_drift = factorized_policy_drift(
        before_model=before_pairwise_model,
        after_model=engine.model,
        candidates=all_candidates,
        device=engine.device,
    )
    final_policy_sha256 = engine.policy_state_sha256
    checkpoint_path = output_dir / "scicf-single-iteration-v2-checkpoint.pt"
    engine.save_checkpoint(checkpoint_path)
    checkpoint_sha256 = sha256_path(checkpoint_path)
    final_ledger = dict(engine.environment.oracle_ledger())

    integrity_failures = []
    integration_failures = []
    if not acquisition["all_pool_decisions_validated"]:
        integrity_failures.append("guarded_acquisition_not_fully_validated")
    if acquisition["pool_decision_count_started"] != 2:
        integrity_failures.append("pool_decision_count_mismatch")
    if acquisition["semantic_attempt_count"] > 4:
        integrity_failures.append("schema_attempt_bound_exceeded")
    if acquisition["http_transmissions_observed"] > 8:
        integrity_failures.append("http_transmission_bound_exceeded")
    if any(len(pool["candidate_ids"]) != 24 for pool in pool_records):
        integrity_failures.append("candidate_pool_size_mismatch")
    if any(len(pool["candidate_timesteps"]) < 2 for pool in pool_records):
        integrity_failures.append("cross_timestep_pool_missing")
    if any(not pool["presented_ids_match_pool"] for pool in pool_records):
        integrity_failures.append("presented_candidate_pool_mismatch")
    if any(
        pool["reward_truth_exposed"] or pool["policy_score_exposed"]
        for pool in pool_records
    ):
        integrity_failures.append("llm_prompt_information_leakage")
    if any(len(item.paired_seeds) != 2 for item in verifications):
        integrity_failures.append("matched_replicate_count_mismatch")
    if len(verifications) != len(selected_ids):
        integrity_failures.append("selected_verification_count_mismatch")
    if any(
        len(item.factual_terminal_records)
        != int(protocol["verification"]["replicates"])
        or len(item.counterfactual_terminal_records)
        != int(protocol["verification"]["replicates"])
        for item in verifications
    ):
        integrity_failures.append("verification_branch_record_count_mismatch")
    if ppo_result.credit.actor_advantages_sha256 != ppo_result.rollout.gae_sha256:
        integrity_failures.append("ppo_actor_credit_not_exact_gae")
    if ppo_result.receipt.critic_returns_sha256 != ppo_result.rollout.critic_returns_sha256:
        integrity_failures.append("ppo_critic_returns_not_from_environment_returns")
    if ppo_result.credit.evaluator_delta.requested_calls != 0:
        integrity_failures.append("ppo_credit_phase_queried_evaluator")
    if ppo_result.credit.diagnostics.get("counterfactual_actions_in_ppo_clipping") is not False:
        integrity_failures.append("counterfactual_action_entered_ppo_clipping")
    if initial_policy_sha256 == post_ppo_policy_sha256:
        integrity_failures.append("standard_ppo_policy_did_not_update")
    if int(ppo_result.receipt.policy_version_after) != 1:
        integrity_failures.append("standard_ppo_policy_version_mismatch")
    observed_sources = set(verification_delta.requested_by_source)
    if not observed_sources.issubset({"scicf_ppo/factual", "scicf_ppo/counterfactual"}):
        integrity_failures.append("unexpected_verification_evaluator_source")
    expected_factual_calls = sum(
        record.get("terminal_evaluation") is not None
        for item in verifications
        for record in item.factual_terminal_records
    )
    expected_counterfactual_calls = sum(
        record.get("terminal_evaluation") is not None
        for item in verifications
        for record in item.counterfactual_terminal_records
    )
    expected_evaluator_calls = expected_factual_calls + expected_counterfactual_calls
    if verification_delta.requested_calls != expected_evaluator_calls:
        integrity_failures.append("verification_requested_call_count_mismatch")
    expected_source_calls = {
        source_name: count
        for source_name, count in (
            ("scicf_ppo/counterfactual", expected_counterfactual_calls),
            ("scicf_ppo/factual", expected_factual_calls),
        )
        if count
    }
    if dict(verification_delta.requested_by_source) != expected_source_calls:
        integrity_failures.append("verification_source_call_count_mismatch")
    if int(final_ledger.get("invalid_results", 0)) != 0:
        integrity_failures.append("invalid_evaluator_result_recorded")
    if not checkpoint_path.is_file() or not checkpoint_sha256:
        integrity_failures.append("checkpoint_missing")
    if refinement.get("counterfactual_actions_in_ppo_clipping") is not False:
        integrity_failures.append("pairwise_action_entered_ppo_clipping")
    if refinement.get("llm_confidence_used_for_weight") is not False:
        integrity_failures.append("llm_confidence_entered_pairwise_loss")
    if value_head_before != value_head_after:
        integrity_failures.append("value_head_parameters_changed_in_pairwise_step")

    if len(accepted) < int(protocol["verification"]["minimum_accepted_pairs"]):
        integration_failures.append("insufficient_sign_consistent_pairs")
    if refinement["status"] != "applied" or int(refinement["optimizer_steps"]) != 1:
        integration_failures.append("pairwise_update_not_applied_once")
    if final_policy_sha256 == post_ppo_policy_sha256:
        integration_failures.append("pairwise_update_did_not_change_post_ppo_policy")
    tolerance = float(protocol["integration_gate"]["comparison_tolerance"])
    pairwise_margin_improvement = None
    if pairwise_before is not None and pairwise_after is not None:
        pairwise_margin_improvement = float(
            pairwise_after["mean_signed_margin"]
            - pairwise_before["mean_signed_margin"]
        )
        if pairwise_margin_improvement <= float(
            protocol["integration_gate"]["minimum_pairwise_mean_margin_improvement"]
        ) + tolerance:
            integration_failures.append("pairwise_mean_margin_not_improved")
        if (
            protocol["integration_gate"]["require_non_decreasing_pairwise_accuracy"]
            and pairwise_after["preference_accuracy"] + tolerance
            < pairwise_before["preference_accuracy"]
        ):
            integration_failures.append("pairwise_preference_accuracy_decreased")
    if pairwise_drift["maximum_joint_kl"] > float(
        protocol["integration_gate"]["maximum_full_support_joint_kl"]
    ) + tolerance:
        integration_failures.append("pairwise_full_support_joint_kl_exceeded")
    if pairwise_drift["maximum_non_target_factor_kl"] > float(
        protocol["integration_gate"]["maximum_non_target_factor_kl"]
    ) + tolerance:
        integration_failures.append("pairwise_non_target_factor_kl_exceeded")
    if pairwise_drift["maximum_absolute_value_drift"] > float(
        protocol["integration_gate"]["maximum_absolute_value_drift"]
    ) + tolerance:
        integration_failures.append("pairwise_critic_value_drift_exceeded")

    integrity_failures = sorted(set(integrity_failures))
    integration_failures = sorted(set(integration_failures))
    passed = not integrity_failures and not integration_failures
    report = {
        "schema_version": 1,
        "protocol_id": SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID,
        "execution_status": "passed" if not integrity_failures else "failed",
        "module_decision": (
            "go_separate_bounded_short_horizon_multi_iteration_protocol_freeze"
            if passed
            else "no_go_multi_iteration_protocol_freeze"
        ),
        "classification": {
            "scope": "single-iteration-ppo-plus-scicf-engineering-smoke-v2",
            "integrity_failures": integrity_failures,
            "integration_failures": integration_failures,
            "next_scope_authorized": passed,
            "multi_iteration_training_authorized": False,
            "formal_training_authorized": False,
            "algorithm_effectiveness_established": False,
            "scientific_claim_authorized": False,
        },
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        },
        "source": source,
        "protocol_sha256": protocol_sha256,
        "authorization_id": authorization["authorization_id"],
        "accepted_binding": binding,
        "stack_specification": specification,
        "run_contract": asdict(run_contract),
        "standard_ppo": iteration_summary(ppo_result),
        "acquisition": acquisition,
        "pools": pool_records,
        "verification": {
            "selected_only": True,
            "selected_count": len(selected_ids),
            "verified_count": len(verifications),
            "replicates": 2,
            "accepted_pair_count": len(accepted),
            "positive_pair_count": sum(
                item.accepted and item.mean_delta > 0.0 for item in verifications
            ),
            "negative_pair_count": sum(
                item.accepted and item.mean_delta < 0.0 for item in verifications
            ),
            "maximum_branch_count": maximum_verification_calls,
            "completed_branch_count": sum(
                len(item.factual_terminal_records)
                + len(item.counterfactual_terminal_records)
                for item in verifications
            ),
            "terminal_evaluation_branch_count": expected_evaluator_calls,
            "ledger_delta": asdict(verification_delta),
        },
        "pairwise_refinement": {
            "receipt": refinement,
            "metrics_before": pairwise_before,
            "metrics_after": pairwise_after,
            "mean_margin_improvement": pairwise_margin_improvement,
            "full_candidate_support_drift": pairwise_drift,
            "value_head_sha256_before": value_head_before,
            "value_head_sha256_after": value_head_after,
            "value_head_parameters_changed": value_head_before != value_head_after,
            "same_iteration_llm_confidence_used": False,
        },
        "policy_hashes": {
            "initial": initial_policy_sha256,
            "behavior": behavior_policy.state_sha256,
            "after_standard_ppo": post_ppo_policy_sha256,
            "after_pairwise": final_policy_sha256,
        },
        "policy_versions": {
            "initial": 0,
            "after_standard_ppo": int(ppo_result.receipt.policy_version_after),
            "after_pairwise": int(engine.policy_version),
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
        "final_evaluator_ledger": final_ledger,
        "external_api_invoked": True,
        "local_model_invoked": False,
        "sealed_test_accessed": False,
        "automatic_rerun_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
        "resources": {
            "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            / 1024.0
        },
    }
    write_json(output_dir / "integration-v2-report.json", report)
    print(
        json.dumps(
            {
                "execution_status": report["execution_status"],
                "module_decision": report["module_decision"],
                "integrity_failures": integrity_failures,
                "integration_failures": integration_failures,
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    if integrity_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
