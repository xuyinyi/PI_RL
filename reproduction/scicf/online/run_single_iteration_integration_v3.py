#!/usr/bin/env python
"""Run one PPO-primary iteration with an optional K=5 SciCF auxiliary stage."""

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
from typing import Any, Dict, Mapping, Sequence, Tuple

from reproduction.framework.io import git_identity, write_json
from reproduction.p2.budget import RequestedCallBudgetManager, evaluator_ledger_delta
from reproduction.p2.contracts import ContractViolation
from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    verify_accepted_binding,
)
from reproduction.scicf.llm.api_client import APISettings, OpenAICompatibleClient

from .contracts import (
    SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID,
    PairwiseRefinementConfig,
)
from .evaluator_asset import (
    load_evaluator_asset_binding,
    validate_evaluator_asset,
    verify_evaluator_route_source_delta,
)
from .model_asset import load_polybert_asset_binding, validate_polybert_asset
from .pipeline import (
    FrozenPolicySampler,
    build_online_candidate_pool,
    eligible_online_episode_ids,
    verify_selected_candidates,
)
from .prompt import build_online_acquisition_request
from .resilience import (
    LLMAuxiliaryCircuitBreaker,
    LLMAuxiliaryResilienceConfig,
    RecoverableLLMError,
    finalize_optional_auxiliary,
    run_optional_llm_acquisition,
)
from .run_pairwise_stability import _head_hash
from .run_single_iteration_integration_v2 import (
    _sha256_path,
    load_protocol as load_v2_protocol,
    run_guarded_pool_decisions,
    verify_protocol_bindings,
)
from .run_smoke import build_runtime, iteration_summary
from .soft_pair import (
    SoftPairAggregationConfig,
    aggregate_soft_verifications,
    refine_soft_pairs,
    weighted_pairwise_preference_metrics,
)
from .stability import factorized_policy_drift


AUTHORIZATION_OPERATIONS_V3 = {
    "credentials_loading_authorized": True,
    "external_api_requests_authorized": True,
    "single_iteration_ppo_authorized": True,
    "oracle_execution_authorized": True,
    "k5_soft_verification_authorized": True,
    "optional_llm_degradation_authorized": True,
    "single_iteration_integration_v3_authorized": True,
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
    parser.add_argument("--evaluator-asset-path", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve_bound(root: Path, record: Mapping[str, Any]) -> Path:
    if set(record) != {"path", "sha256"}:
        raise ValueError("invalid v3 bound-file record")
    root = root.resolve(strict=True)
    path = (root / record["path"]).resolve(strict=True)
    if root != path and root not in path.parents:
        raise ValueError("v3 bound file escaped repository")
    if _sha256_path(path) != record["sha256"]:
        raise RuntimeError("v3 bound-file hash mismatch: %s" % record["path"])
    return path


def load_v3_protocol(
    path: Path, root: Path
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "classification",
        "status",
        "required_host",
        "base_v2_protocol",
        "soft_pair_development",
        "primary_transaction",
        "verification",
        "llm_failure_policy",
        "integrity_policy",
        "authorization_state",
    }
    if set(payload) != required:
        raise ValueError("integration-v3 protocol keys changed")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported integration-v3 protocol schema")
    if payload["protocol_id"] != SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID:
        raise ValueError("integration-v3 protocol identity changed")
    if payload["classification"] != (
        "single_iteration_optional_llm_soft_pair_engineering_protocol_only"
    ):
        raise ValueError("integration-v3 protocol classification changed")
    if payload["status"] != "frozen_unimplemented_unexecuted":
        raise ValueError("integration-v3 frozen status changed")
    if payload["required_host"] != "yanlih100n1":
        raise ValueError("integration-v3 host binding changed")
    if payload["primary_transaction"] != {
        "standard_ppo_is_primary": True,
        "checkpoint_before_llm": True,
        "recoverable_llm_failure_cannot_rollback_ppo": True,
        "maximum_primary_ppo_iterations": 1,
    }:
        raise ValueError("integration-v3 primary transaction changed")
    if payload["verification"] != {
        "candidate_scope": "llm_selected_only",
        "replicates": 5,
        "maximum_selected_candidates": 8,
        "maximum_branches": 80,
    }:
        raise ValueError("integration-v3 verification boundary changed")
    if payload["llm_failure_policy"] != {
        "recoverable_outcome": "ppo_only_degraded",
        "silent_fallback_forbidden": True,
        "continue_after_provider_timeout": True,
        "continue_after_provider_rate_limit": True,
        "continue_after_provider_unavailable": True,
        "continue_after_transport_exhaustion": True,
        "continue_after_schema_exhaustion": True,
        "continue_after_valid_abstention": True,
        "continue_after_insufficient_soft_pair_mass": True,
    }:
        raise ValueError("integration-v3 LLM failure policy changed")
    if payload["integrity_policy"] != {
        "authorization_failure_is_fatal": True,
        "asset_binding_failure_is_fatal": True,
        "budget_overrun_is_fatal": True,
        "information_leakage_is_fatal": True,
        "unvalidated_output_cannot_reach_oracle": True,
        "unexpected_programming_error_is_fatal": True,
    }:
        raise ValueError("integration-v3 integrity policy changed")
    if payload["authorization_state"] != {
        "runner_implementation_authorized": True,
        "no_credential_full_runtime_preflight_authorized": True,
        "real_single_iteration_authorized": False,
        "automatic_rerun_authorized": False,
        "multi_iteration_training_authorized": False,
        "formal_training_authorized": False,
        "sealed_test_access_authorized": False,
        "baseline_mutation_authorized": False,
        "algorithm_effectiveness_claim_authorized": False,
        "scientific_claim_authorized": False,
    }:
        raise ValueError("integration-v3 authorization boundary changed")

    base_path = _resolve_bound(root, payload["base_v2_protocol"])
    base_protocol = load_v2_protocol(base_path)
    soft_record = dict(payload["soft_pair_development"])
    component_report_record = {
        "path": soft_record.pop("component_report"),
        "sha256": soft_record.pop("component_report_sha256"),
    }
    required_decision = soft_record.pop("required_component_decision")
    soft_path = _resolve_bound(root, soft_record)
    soft_payload = json.loads(soft_path.read_text(encoding="utf-8"))
    component_report_path = _resolve_bound(root, component_report_record)
    component_report = json.loads(
        component_report_path.read_text(encoding="utf-8")
    )
    if component_report["decision"] != required_decision:
        raise RuntimeError("soft-pair component preflight decision changed")
    aggregation = SoftPairAggregationConfig(**soft_payload["soft_pair_aggregation"])
    if aggregation.replicates != payload["verification"]["replicates"]:
        raise RuntimeError("v3 and soft-pair replicate contracts differ")
    return payload, base_protocol, soft_payload


def load_execution_authorization_v3(
    path: Path,
    *,
    protocol_sha256: str,
    soft_pair_config_sha256: str,
    implementation_commit: str,
    output_dir: Path,
    polybert_path: Path,
    polybert_asset_binding_sha256: str,
    polybert_checkpoint_fingerprint: str,
    evaluator_asset_path: Path,
    evaluator_asset_binding_sha256: str,
    evaluator_asset_fingerprint: str,
) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "authorization_id",
        "protocol_id",
        "protocol_sha256",
        "soft_pair_config_sha256",
        "implementation_commit",
        "authorized_output_directory",
        "authorized_polybert_path",
        "polybert_asset_binding_sha256",
        "polybert_checkpoint_fingerprint",
        "authorized_evaluator_asset_path",
        "evaluator_asset_binding_sha256",
        "evaluator_asset_fingerprint",
        "maximum_slurm_runs",
        "authorized_operations",
    }
    if set(payload) != required:
        raise ValueError("integration-v3 authorization keys changed")
    if payload["schema_version"] != 4:
        raise ValueError("integration-v3 requires authorization schema 4")
    if payload["protocol_id"] != SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID:
        raise ValueError("integration-v3 authorization protocol mismatch")
    checks = (
        (payload["protocol_sha256"], protocol_sha256, "protocol hash"),
        (
            payload["soft_pair_config_sha256"],
            soft_pair_config_sha256,
            "soft-pair config hash",
        ),
        (
            payload["implementation_commit"],
            implementation_commit,
            "implementation commit",
        ),
        (
            payload["polybert_asset_binding_sha256"],
            polybert_asset_binding_sha256,
            "polyBERT binding",
        ),
        (
            payload["polybert_checkpoint_fingerprint"],
            polybert_checkpoint_fingerprint,
            "polyBERT fingerprint",
        ),
        (
            payload["evaluator_asset_binding_sha256"],
            evaluator_asset_binding_sha256,
            "evaluator binding",
        ),
        (
            payload["evaluator_asset_fingerprint"],
            evaluator_asset_fingerprint,
            "evaluator fingerprint",
        ),
    )
    for observed, expected, name in checks:
        if observed != expected:
            raise ValueError("integration-v3 authorization %s mismatch" % name)
    path_checks = (
        (payload["authorized_output_directory"], output_dir, "output"),
        (payload["authorized_polybert_path"], polybert_path, "polyBERT path"),
        (
            payload["authorized_evaluator_asset_path"],
            evaluator_asset_path,
            "evaluator path",
        ),
    )
    for observed, expected, name in path_checks:
        if Path(observed).resolve() != expected.resolve():
            raise ValueError("integration-v3 authorization %s mismatch" % name)
    if payload["maximum_slurm_runs"] != 1:
        raise ValueError("integration-v3 authorization must permit one run")
    if payload["authorized_operations"] != AUTHORIZATION_OPERATIONS_V3:
        raise ValueError("integration-v3 authorization operation scope mismatch")
    return payload


def _prepare_pools(
    *,
    output_dir: Path,
    protocol_id: str,
    base_protocol: Mapping[str, Any],
    ppo_result,
    components,
    behavior_policy: FrozenPolicySampler,
) -> Tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], Sequence[Any]]:
    eligible = list(
        eligible_online_episode_ids(ppo_result.rollout, prefer_successful=False)
    )
    random.Random(int(base_protocol["base_seed"])).shuffle(eligible)
    pool_count = int(base_protocol["acquisition"]["pool_count"])
    if len(eligible) < pool_count:
        raise RuntimeError("rollout lacks two complete cross-timestep episodes")
    prepared = []
    records = []
    all_candidates = []
    for pool_index, episode_id in enumerate(eligible[:pool_count]):
        pool_seed = int(base_protocol["base_seed"]) + int(
            base_protocol["acquisition"]["pool_seed_stride"]
        ) * pool_index
        candidates, trajectory_context = build_online_candidate_pool(
            rollout=ppo_result.rollout,
            core=components.core,
            behavior_policy=behavior_policy,
            pool_size=int(base_protocol["acquisition"]["pool_size"]),
            seed=pool_seed,
            episode_id=episode_id,
        )
        request = build_online_acquisition_request(
            request_id="%s-pool-%02d-job-%s"
            % (protocol_id, pool_index, os.environ["SLURM_JOB_ID"]),
            trajectory_context=trajectory_context,
            candidates=candidates,
            maximum_budget=int(
                base_protocol["acquisition"]["maximum_selected_per_pool"]
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
        prepared.append(
            {"pool_index": pool_index, "pool_seed": pool_seed, "request": request}
        )
        records.append(
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
    return tuple(prepared), tuple(records), tuple(all_candidates)


def main() -> None:
    args = parse_args()
    root = args.dapigen_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    protocol, base_protocol, soft_payload = load_v3_protocol(protocol_path, root)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("integration-v3 requires Slurm")
    if socket.gethostname() != protocol["required_host"]:
        raise RuntimeError("integration-v3 is bound to n001")
    source = git_identity(root)
    if source["dirty"] is not False:
        raise RuntimeError("integration-v3 requires a clean worktree")
    if output_dir.exists():
        raise FileExistsError("integration-v3 output already exists")

    bindings = verify_protocol_bindings(root, base_protocol)
    accepted_manifest = bindings["accepted_manifest"]
    accepted_binding = verify_accepted_binding(
        root,
        accepted_manifest,
        base_protocol["accepted_binding"]["environment_id"],
    )
    polybert_binding = load_polybert_asset_binding(root)
    polybert_path = args.polybert_path.resolve(strict=True)
    model_asset = validate_polybert_asset(polybert_path, polybert_binding)
    if (
        accepted_manifest["environment"].get("encoder_version")
        != model_asset["encoder_version"]
    ):
        raise RuntimeError("polyBERT asset differs from accepted Stage-0 encoder")
    evaluator_binding = load_evaluator_asset_binding(root)
    evaluator_asset_path = args.evaluator_asset_path.resolve(strict=True)
    evaluator_asset = validate_evaluator_asset(
        evaluator_asset_path, evaluator_binding
    )
    if accepted_manifest.get("evaluator_version") != evaluator_asset[
        "evaluator_version"
    ]:
        raise RuntimeError("AFP evaluator asset differs from accepted Stage-0 evaluator")
    source_delta = verify_evaluator_route_source_delta(
        root, accepted_binding["accepted_git_commit"], evaluator_binding
    )
    soft_config_path = _resolve_bound(
        root,
        {
            "path": protocol["soft_pair_development"]["path"],
            "sha256": protocol["soft_pair_development"]["sha256"],
        },
    )
    authorization = load_execution_authorization_v3(
        args.execution_authorization.resolve(strict=True),
        protocol_sha256=_sha256_path(protocol_path),
        soft_pair_config_sha256=_sha256_path(soft_config_path),
        implementation_commit=source["commit"],
        output_dir=output_dir,
        polybert_path=polybert_path,
        polybert_asset_binding_sha256=model_asset["asset_binding_sha256"],
        polybert_checkpoint_fingerprint=model_asset["checkpoint_fingerprint"],
        evaluator_asset_path=evaluator_asset_path,
        evaluator_asset_binding_sha256=evaluator_asset["asset_binding_sha256"],
        evaluator_asset_fingerprint=evaluator_asset["asset_fingerprint"],
    )

    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "schema_version": 1,
            "status": "declared_after_authorization_and_assets_before_runtime",
            "protocol_id": SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "protocol_sha256": _sha256_path(protocol_path),
            "soft_pair_config_sha256": _sha256_path(soft_config_path),
            "authorization_id": authorization["authorization_id"],
            "authorization_sha256": _sha256_path(
                args.execution_authorization.resolve(strict=True)
            ),
            "polybert_asset": model_asset,
            "evaluator_asset": evaluator_asset,
            "evaluator_route_source_delta": source_delta,
            "credentials_loaded_at_intent_time": False,
            "primary_ppo_checkpoint_written_at_intent_time": False,
            "automatic_rerun_authorized": False,
            "multi_iteration_training_authorized": False,
        },
    )
    started = time.perf_counter()
    components, engine, specification, run_contract = build_runtime(
        root,
        accepted_manifest,
        accepted_binding,
        base_protocol,
        polybert_path,
        polybert_checkpoint_fingerprint=model_asset["checkpoint_fingerprint"],
        evaluator_asset_path=evaluator_asset_path,
    )
    initial_policy_sha256 = engine.policy_state_sha256
    behavior_policy = FrozenPolicySampler(engine.model, base_protocol["ppo"]["device"])
    ppo_result = engine.run_iteration(query_requested_calls=0)
    post_ppo_policy_sha256 = engine.policy_state_sha256
    if behavior_policy.state_sha256 != ppo_result.rollout.frozen_policy.state_sha256:
        raise ContractViolation("behavior-policy copy differs from PPO rollout policy")
    if initial_policy_sha256 == post_ppo_policy_sha256:
        raise ContractViolation("primary PPO policy did not update")
    primary_checkpoint = output_dir / "primary-ppo-checkpoint-before-llm.pt"
    engine.save_checkpoint(primary_checkpoint)
    primary_checkpoint_sha256 = _sha256_path(primary_checkpoint)
    write_json(
        output_dir / "primary-ppo-receipt.json",
        {
            "standard_ppo": iteration_summary(ppo_result),
            "checkpoint": {
                "path": str(primary_checkpoint),
                "sha256": primary_checkpoint_sha256,
            },
            "credentials_loaded": False,
            "external_api_invoked": False,
            "durable_before_llm": True,
        },
    )

    prepared, pool_records, all_candidates = _prepare_pools(
        output_dir=output_dir,
        protocol_id=SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID,
        base_protocol=base_protocol,
        ppo_result=ppo_result,
        components=components,
        behavior_policy=behavior_policy,
    )
    acquisition_holder: Dict[str, Any] = {}
    provider_holder: Dict[str, Any] = {}

    resilience_payload = dict(soft_payload["llm_resilience"])
    resilience_payload.pop("recoverable_failure_codes")
    breaker = LLMAuxiliaryCircuitBreaker(
        LLMAuxiliaryResilienceConfig(**resilience_payload)
    )

    def acquire() -> Mapping[str, Any]:
        try:
            settings = APISettings.from_private_file(
                args.credentials_file.resolve(strict=True)
            )
        except (OSError, ValueError) as error:
            raise RecoverableLLMError(
                "provider_unavailable", type(error).__name__
            )
        provider = settings.public_identity()
        provider["external_api"] = True
        provider_holder["identity"] = provider
        write_json(
            output_dir / "provider-public-identity.json",
            {
                "provider": provider,
                "api_key_logged": False,
                "credentials_path_logged": False,
            },
        )
        summary = run_guarded_pool_decisions(
            client=OpenAICompatibleClient(settings),
            provider=provider,
            prepared_pools=prepared,
            output_root=output_dir / "guarded-acquisition",
            protocol=base_protocol,
        )
        acquisition_holder["summary"] = summary
        if not summary["oracle_selection_authorized"]:
            statuses = {item["status"] for item in summary["pool_outcomes"]}
            if "fail_closed_schema_exhausted" in statuses:
                raise RecoverableLLMError("schema_exhausted")
            if "fail_closed_transport_error" in statuses:
                raise RecoverableLLMError("transport_exhausted")
            raise ContractViolation("unexpected guarded-acquisition failure status")
        if not summary["selected_intervention_ids"]:
            return {"status": "abstained", "payload": summary}
        return {"status": "validated", "payload": summary}

    acquisition_receipt = run_optional_llm_acquisition(
        iteration=1,
        primary_ppo_checkpoint_path=primary_checkpoint,
        primary_ppo_checkpoint_sha256=primary_checkpoint_sha256,
        breaker=breaker,
        acquire=acquire,
    )
    write_json(output_dir / "llm-auxiliary-receipt.json", acquisition_receipt)

    verifications = tuple()
    soft_evidence = tuple()
    verification_delta = None
    pairwise_before = None
    pairwise_after = None
    pairwise_drift = None
    value_head_before = _head_hash(engine.model, "value_head")
    value_head_after = value_head_before
    refinement = None
    final_receipt = acquisition_receipt

    if acquisition_receipt["auxiliary_status"] == "ready_for_oracle_verification":
        acquisition = acquisition_holder["summary"]
        selected_ids = tuple(acquisition["selected_intervention_ids"])
        if len(selected_ids) != len(set(selected_ids)):
            raise ContractViolation("v3 selected candidate ids repeat across pools")
        if len(selected_ids) > int(
            protocol["verification"]["maximum_selected_candidates"]
        ):
            raise ContractViolation("v3 selected-candidate ceiling exceeded")
        candidate_map = {item.candidate_id: item for item in all_candidates}
        if len(candidate_map) != len(all_candidates):
            raise ContractViolation("v3 candidate ids are not globally unique")
        maximum_branches = 2 * len(selected_ids) * int(
            protocol["verification"]["replicates"]
        )
        if maximum_branches > int(protocol["verification"]["maximum_branches"]):
            raise ContractViolation("v3 verification branch ceiling exceeded")
        budget = RequestedCallBudgetManager(engine.environment.oracle_ledger)
        reservation = budget.reserve(maximum_branches)
        ledger_before = dict(engine.environment.oracle_ledger())
        try:
            verifications = tuple(
                verify_selected_candidates(
                    environment=engine.environment,
                    policy=behavior_policy,
                    candidates_by_id=candidate_map,
                    selected_ids=selected_ids,
                    replicates=int(protocol["verification"]["replicates"]),
                    seed=int(base_protocol["base_seed"]),
                    delta_tolerance=float(
                        soft_payload["soft_pair_aggregation"][
                            "practical_delta_tolerance"
                        ]
                    ),
                )
            )
            ledger_after = dict(engine.environment.oracle_ledger())
            verification_delta = evaluator_ledger_delta(ledger_before, ledger_after)
            budget.reconcile(reservation, verification_delta.requested_calls)
        except Exception:
            budget.release(reservation)
            raise
        aggregation_config = SoftPairAggregationConfig(
            **soft_payload["soft_pair_aggregation"]
        )
        soft_evidence = aggregate_soft_verifications(
            verifications, aggregation_config
        )
        write_json(
            output_dir / "oracle-soft-verification.json",
            {
                "selected_ids": list(selected_ids),
                "selected_only": True,
                "replicates": aggregation_config.replicates,
                "maximum_branch_count": maximum_branches,
                "ledger_delta": asdict(verification_delta),
                "records": [item.to_dict() for item in soft_evidence],
            },
        )
        weighted = [item for item in soft_evidence if item.training_weight > 0.0]
        before_pairwise_model = copy.deepcopy(engine.model).to(engine.device).eval()
        if weighted:
            pairwise_before = weighted_pairwise_preference_metrics(
                model=engine.model,
                candidates_by_id=candidate_map,
                evidence=soft_evidence,
                device=engine.device,
            )
        refinement = refine_soft_pairs(
            engine=engine,
            candidates_by_id=candidate_map,
            evidence=soft_evidence,
            aggregation_config=aggregation_config,
            refinement_config=PairwiseRefinementConfig(
                **soft_payload["pairwise_refinement"]
            ),
        )
        value_head_after = _head_hash(engine.model, "value_head")
        if weighted:
            pairwise_after = weighted_pairwise_preference_metrics(
                model=engine.model,
                candidates_by_id=candidate_map,
                evidence=soft_evidence,
                device=engine.device,
            )
        pairwise_drift = factorized_policy_drift(
            before_model=before_pairwise_model,
            after_model=engine.model,
            candidates=all_candidates,
            device=engine.device,
        )
        final_receipt = finalize_optional_auxiliary(
            acquisition_receipt, refinement
        )
        write_json(output_dir / "llm-auxiliary-receipt.json", final_receipt)

    final_checkpoint = output_dir / "final-single-iteration-v3-checkpoint.pt"
    engine.save_checkpoint(final_checkpoint)
    final_checkpoint_sha256 = _sha256_path(final_checkpoint)
    integrity_failures = []
    if _sha256_path(primary_checkpoint) != primary_checkpoint_sha256:
        integrity_failures.append("primary_ppo_checkpoint_changed_after_llm")
    if final_receipt.get("continue_training") is not True:
        integrity_failures.append("optional_llm_blocked_primary_training")
    if final_receipt.get("silent_fallback_used") is not False:
        integrity_failures.append("silent_fallback_used")
    if refinement is not None:
        if refinement.get("llm_confidence_used_for_weight") is not False:
            integrity_failures.append("llm_confidence_entered_loss")
        if refinement.get("counterfactual_actions_in_ppo_clipping") is not False:
            integrity_failures.append("counterfactual_action_entered_ppo_clipping")
        if value_head_before != value_head_after:
            integrity_failures.append("value_head_changed_in_auxiliary_step")
    if any(len(item.deltas) != 5 for item in soft_evidence):
        integrity_failures.append("soft_verification_not_k5")
    acquisition = acquisition_holder.get("summary")
    if acquisition is not None:
        selected_by_pool = {
            item["pool_index"]: item["selected_intervention_ids"]
            for item in acquisition["pool_outcomes"]
        }
        for pool in pool_records:
            pool["selected_ids"] = list(
                selected_by_pool.get(pool["pool_index"], [])
            )
            pool["abstained"] = len(pool["selected_ids"]) == 0
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
    integrity_failures = sorted(set(integrity_failures))
    final_ledger = dict(engine.environment.oracle_ledger())
    report = {
        "schema_version": 1,
        "protocol_id": SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID,
        "execution_status": "passed" if not integrity_failures else "failed",
        "module_decision": (
            "primary_ppo_committed_auxiliary_applied"
            if not integrity_failures and final_receipt.get("scicf_applied") is True
            else "primary_ppo_committed_auxiliary_degraded"
            if not integrity_failures
            else "fatal_integrity_failure_primary_checkpoint_retained"
        ),
        "classification": {
            "scope": "single-iteration-v3-engineering-only",
            "integrity_failures": integrity_failures,
            "continue_training": not integrity_failures,
            "method_observed": final_receipt.get("method_observed"),
            "real_run_does_not_authorize_next_scope": True,
            "multi_iteration_training_authorized": False,
            "formal_training_authorized": False,
            "algorithm_effectiveness_established": False,
            "scientific_claim_authorized": False,
        },
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "source": source,
        "protocol_sha256": _sha256_path(protocol_path),
        "soft_pair_config_sha256": _sha256_path(soft_config_path),
        "authorization_id": authorization["authorization_id"],
        "accepted_binding": accepted_binding,
        "polybert_asset": model_asset,
        "evaluator_asset": evaluator_asset,
        "evaluator_route_source_delta": source_delta,
        "stack_specification": specification,
        "run_contract": asdict(run_contract),
        "standard_ppo": iteration_summary(ppo_result),
        "primary_ppo_checkpoint": {
            "path": str(primary_checkpoint),
            "sha256": primary_checkpoint_sha256,
            "written_before_credentials_and_llm": True,
        },
        "candidate_pools": pool_records,
        "acquisition": acquisition,
        "provider_public_identity_written": "identity" in provider_holder,
        "auxiliary_receipt": final_receipt,
        "verification": {
            "selected_only": True,
            "replicates": 5,
            "verified_count": len(verifications),
            "soft_weighted_count": sum(
                item.training_weight > 0.0 for item in soft_evidence
            ),
            "ledger_delta": (
                asdict(verification_delta) if verification_delta is not None else None
            ),
        },
        "pairwise_refinement": {
            "receipt": refinement,
            "weighted_metrics_before": pairwise_before,
            "weighted_metrics_after": pairwise_after,
            "full_candidate_support_drift": pairwise_drift,
            "value_head_sha256_before": value_head_before,
            "value_head_sha256_after": value_head_after,
        },
        "policy_hashes": {
            "initial": initial_policy_sha256,
            "behavior": behavior_policy.state_sha256,
            "after_standard_ppo": post_ppo_policy_sha256,
            "final": engine.policy_state_sha256,
        },
        "final_checkpoint": {
            "path": str(final_checkpoint),
            "sha256": final_checkpoint_sha256,
        },
        "final_evaluator_ledger": final_ledger,
        "external_api_invoked": "summary" in acquisition_holder,
        "local_model_invoked": False,
        "sealed_test_accessed": False,
        "automatic_rerun_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
        "resources": {
            "peak_rss_mib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            / 1024.0
        },
    }
    write_json(output_dir / "integration-v3-report.json", report)
    print(
        json.dumps(
            {
                "execution_status": report["execution_status"],
                "module_decision": report["module_decision"],
                "method_observed": final_receipt.get("method_observed"),
                "integrity_failures": integrity_failures,
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    if integrity_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
