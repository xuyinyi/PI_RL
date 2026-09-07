#!/usr/bin/env python
"""Run a bounded PPO-primary multi-iteration SciCF engineering smoke."""

from __future__ import annotations

import argparse
import copy
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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from reproduction.framework.io import git_identity, write_json
from reproduction.p2.budget import RequestedCallBudgetManager, evaluator_ledger_delta
from reproduction.p2.contracts import ContractViolation
from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    verify_accepted_binding,
)
from reproduction.scicf.llm.api_client import APISettings, OpenAICompatibleClient

from .contracts import (
    SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID,
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
    DeadlineBoundClient,
    LLMAuxiliaryCircuitBreaker,
    LLMAuxiliaryResilienceConfig,
    LLMWallClockBudget,
    LLMWallClockBudgetConfig,
    LLMWallTimeBudgetExhausted,
    RecoverableLLMError,
    finalize_optional_auxiliary,
    run_optional_llm_acquisition,
)
from .run_pairwise_stability import _head_hash
from .run_single_iteration_integration_v2 import (
    run_guarded_pool_decisions,
    verify_protocol_bindings,
)
from .run_single_iteration_integration_v3 import load_v3_protocol
from .run_smoke import build_runtime, iteration_summary
from .short_horizon import (
    chained_history_digest,
    load_short_horizon_checkpoint,
    load_short_horizon_protocol,
    save_short_horizon_checkpoint,
    sha256_path,
)
from .soft_pair import (
    SoftPairAggregationConfig,
    aggregate_soft_verifications,
    refine_soft_pairs,
    weighted_pairwise_preference_metrics,
)
from .stability import factorized_policy_drift


AUTHORIZATION_OPERATIONS_SHORT_HORIZON = {
    "credentials_loading_authorized": True,
    "external_api_requests_authorized": True,
    "ppo_execution_authorized": True,
    "oracle_execution_authorized": True,
    "k5_soft_verification_authorized": True,
    "optional_llm_degradation_authorized": True,
    "short_horizon_multi_iteration_authorized": True,
    "automatic_resume_authorized": False,
    "automatic_rerun_authorized": False,
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
    parser.add_argument("--resume-control-checkpoint", type=Path)
    return parser.parse_args()


def _runtime_protocol(base_protocol, short_protocol):
    payload = copy.deepcopy(base_protocol)
    maximum = int(
        short_protocol["scientific_budget"][
            "maximum_requested_evaluator_calls_total"
        ]
    )
    payload["budget"]["maximum_requested_calls"] = maximum
    payload["budget"]["maximum_unique_calls"] = maximum
    payload["budget"]["cache_scope"] = "scicf_short_horizon_multi_iteration_v1"
    return payload


def load_execution_authorization_short_horizon(
    path: Path,
    *,
    protocol_sha256: str,
    implementation_commit: str,
    output_dir: Path,
    polybert_path: Path,
    polybert_asset_binding_sha256: str,
    polybert_checkpoint_fingerprint: str,
    evaluator_asset_path: Path,
    evaluator_asset_binding_sha256: str,
    evaluator_asset_fingerprint: str,
    resume_control_checkpoint: Optional[Path],
    resume_control_checkpoint_sha256: Optional[str],
) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "authorization_id",
        "protocol_id",
        "protocol_sha256",
        "implementation_commit",
        "authorized_output_directory",
        "authorized_polybert_path",
        "polybert_asset_binding_sha256",
        "polybert_checkpoint_fingerprint",
        "authorized_evaluator_asset_path",
        "evaluator_asset_binding_sha256",
        "evaluator_asset_fingerprint",
        "authorized_start_iteration",
        "authorized_end_iteration",
        "authorized_resume_control_checkpoint",
        "resume_control_checkpoint_sha256",
        "maximum_slurm_runs",
        "authorized_operations",
    }
    if set(payload) != required:
        raise ValueError("short-horizon authorization keys changed")
    if payload["schema_version"] != 5:
        raise ValueError("short-horizon runner requires authorization schema 5")
    if not isinstance(payload["authorization_id"], str) or not payload[
        "authorization_id"
    ]:
        raise ValueError("short-horizon authorization id must be non-empty")
    if payload["protocol_id"] != SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID:
        raise ValueError("short-horizon authorization protocol mismatch")
    checks = (
        (payload["protocol_sha256"], protocol_sha256, "protocol hash"),
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
            raise ValueError("short-horizon authorization %s mismatch" % name)
    path_checks = (
        (payload["authorized_output_directory"], output_dir, "output"),
        (payload["authorized_polybert_path"], polybert_path, "polyBERT"),
        (
            payload["authorized_evaluator_asset_path"],
            evaluator_asset_path,
            "evaluator",
        ),
    )
    for observed, expected, name in path_checks:
        if not isinstance(observed, str) or Path(
            observed
        ).resolve() != expected.resolve():
            raise ValueError("short-horizon authorization %s path mismatch" % name)
    observed_resume = payload["authorized_resume_control_checkpoint"]
    expected_resume = (
        str(resume_control_checkpoint.resolve())
        if resume_control_checkpoint is not None
        else None
    )
    if observed_resume != expected_resume:
        raise ValueError("short-horizon resume path mismatch")
    if payload["resume_control_checkpoint_sha256"] != (
        resume_control_checkpoint_sha256 if resume_control_checkpoint else None
    ):
        raise ValueError("short-horizon resume hash mismatch")
    start = payload["authorized_start_iteration"]
    end = payload["authorized_end_iteration"]
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 1
        or end < start
        or end > 6
    ):
        raise ValueError("short-horizon authorized iteration range is invalid")
    if payload["maximum_slurm_runs"] != 1:
        raise ValueError("short-horizon authorization must permit one submission")
    if payload["authorized_operations"] != AUTHORIZATION_OPERATIONS_SHORT_HORIZON:
        raise ValueError("short-horizon authorization operation scope mismatch")
    return payload


def _prepare_iteration_pools(
    *,
    iteration: int,
    output_dir: Path,
    base_protocol: Mapping[str, Any],
    ppo_result,
    components,
    behavior_policy: FrozenPolicySampler,
) -> Tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], Sequence[Any]]:
    iteration_seed = int(base_protocol["base_seed"]) + 1000003 * int(iteration)
    eligible = list(
        eligible_online_episode_ids(ppo_result.rollout, prefer_successful=False)
    )
    random.Random(iteration_seed).shuffle(eligible)
    pool_count = int(base_protocol["acquisition"]["pool_count"])
    if len(eligible) < pool_count:
        raise ContractViolation("rollout lacks complete cross-timestep episodes")
    prepared = []
    records = []
    all_candidates = []
    for pool_index, episode_id in enumerate(eligible[:pool_count]):
        pool_seed = iteration_seed + int(
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
            request_id="%s-iter-%02d-pool-%02d-job-%s"
            % (
                SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID,
                iteration,
                pool_index,
                os.environ["SLURM_JOB_ID"],
            ),
            trajectory_context=trajectory_context,
            candidates=candidates,
            maximum_budget=int(
                base_protocol["acquisition"]["maximum_selected_per_pool"]
            ),
        )
        write_json(
            output_dir / "pools" / ("pool-%02d-candidates.json" % pool_index),
            {
                "iteration": iteration,
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


def _iteration(
    *,
    iteration: int,
    iteration_dir: Path,
    engine,
    components,
    base_protocol,
    short_protocol,
    soft_payload,
    breaker,
    llm_wall_budget,
    credentials_file: Path,
) -> Mapping[str, Any]:
    iteration_dir.mkdir(parents=True)
    initial_policy_sha256 = engine.policy_state_sha256
    behavior_policy = FrozenPolicySampler(engine.model, base_protocol["ppo"]["device"])
    ppo_result = engine.run_iteration(query_requested_calls=0)
    post_ppo_policy_sha256 = engine.policy_state_sha256
    if behavior_policy.state_sha256 != ppo_result.rollout.frozen_policy.state_sha256:
        raise ContractViolation("behavior-policy copy differs from PPO rollout policy")
    if initial_policy_sha256 == post_ppo_policy_sha256:
        raise ContractViolation("primary PPO policy did not update")
    primary_checkpoint = iteration_dir / "primary-ppo-before-llm.pt"
    engine.save_checkpoint(primary_checkpoint)
    primary_checkpoint_sha256 = sha256_path(primary_checkpoint)
    write_json(
        iteration_dir / "primary-ppo-receipt.json",
        {
            "iteration": iteration,
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

    prepared = tuple()
    pool_records = tuple()
    all_candidates = tuple()
    if breaker.allow_request(iteration):
        prepared, pool_records, all_candidates = _prepare_iteration_pools(
            iteration=iteration,
            output_dir=iteration_dir,
            base_protocol=base_protocol,
            ppo_result=ppo_result,
            components=components,
            behavior_policy=behavior_policy,
        )
    acquisition_holder: Dict[str, Any] = {}
    provider_holder: Dict[str, Any] = {}
    wall_receipt_holder: Dict[str, Any] = {}

    def acquire() -> Mapping[str, Any]:
        lease = None
        try:
            try:
                lease = llm_wall_budget.begin_iteration(iteration)
            except LLMWallTimeBudgetExhausted as error:
                raise RecoverableLLMError("llm_wall_time_exhausted", str(error))
            try:
                settings = APISettings.from_private_file(
                    credentials_file.resolve(strict=True)
                )
            except (OSError, ValueError) as error:
                raise RecoverableLLMError(
                    "provider_unavailable", type(error).__name__
                )
            provider = settings.public_identity()
            provider["external_api"] = True
            provider_holder["identity"] = provider
            write_json(
                iteration_dir / "provider-public-identity.json",
                {
                    "provider": provider,
                    "api_key_logged": False,
                    "credentials_path_logged": False,
                },
            )
            summary = run_guarded_pool_decisions(
                client=DeadlineBoundClient(
                    OpenAICompatibleClient(settings), lease
                ),
                provider=provider,
                prepared_pools=prepared,
                output_root=iteration_dir / "guarded-acquisition",
                protocol=base_protocol,
            )
            acquisition_holder["summary"] = summary
            if lease.remaining_seconds() <= 0.0:
                raise RecoverableLLMError("llm_wall_time_exhausted")
            if not summary["oracle_selection_authorized"]:
                if lease.exhausted:
                    raise RecoverableLLMError("llm_wall_time_exhausted")
                statuses = {item["status"] for item in summary["pool_outcomes"]}
                if "fail_closed_schema_exhausted" in statuses:
                    raise RecoverableLLMError("schema_exhausted")
                if "fail_closed_transport_error" in statuses:
                    raise RecoverableLLMError("transport_exhausted")
                raise ContractViolation("unexpected guarded-acquisition status")
            if not summary["selected_intervention_ids"]:
                return {"status": "abstained", "payload": summary}
            return {"status": "validated", "payload": summary}
        finally:
            if lease is not None and not lease.closed:
                wall_receipt_holder["receipt"] = lease.close()
                write_json(
                    iteration_dir / "llm-wall-clock-receipt.json",
                    wall_receipt_holder["receipt"],
                )

    acquisition_receipt = run_optional_llm_acquisition(
        iteration=iteration,
        primary_ppo_checkpoint_path=primary_checkpoint,
        primary_ppo_checkpoint_sha256=primary_checkpoint_sha256,
        breaker=breaker,
        acquire=acquire,
    )
    final_receipt = acquisition_receipt
    verifications = tuple()
    soft_evidence = tuple()
    verification_delta = None
    refinement = None
    pairwise_before = None
    pairwise_after = None
    pairwise_drift = None
    value_head_before = _head_hash(engine.model, "value_head")
    value_head_after = value_head_before

    if acquisition_receipt["auxiliary_status"] == "ready_for_oracle_verification":
        acquisition = acquisition_holder["summary"]
        selected_ids = tuple(acquisition["selected_intervention_ids"])
        if len(selected_ids) != len(set(selected_ids)):
            raise ContractViolation("selected candidate ids repeat across pools")
        if len(selected_ids) > int(
            short_protocol["scientific_budget"][
                "maximum_selected_candidates_per_iteration"
            ]
        ):
            raise ContractViolation("selected-candidate ceiling exceeded")
        candidate_map = {item.candidate_id: item for item in all_candidates}
        if len(candidate_map) != len(all_candidates):
            raise ContractViolation("candidate ids are not globally unique")
        maximum_branches = 2 * len(selected_ids) * int(
            short_protocol["scientific_budget"][
                "replicates_per_selected_candidate"
            ]
        )
        if maximum_branches > int(
            short_protocol["scientific_budget"][
                "maximum_verification_branches_per_iteration"
            ]
        ):
            raise ContractViolation("verification branch ceiling exceeded")
        evaluator_budget = RequestedCallBudgetManager(engine.environment.oracle_ledger)
        reservation = evaluator_budget.reserve(maximum_branches)
        ledger_before = dict(engine.environment.oracle_ledger())
        try:
            verifications = tuple(
                verify_selected_candidates(
                    environment=engine.environment,
                    policy=behavior_policy,
                    candidates_by_id=candidate_map,
                    selected_ids=selected_ids,
                    replicates=5,
                    seed=int(base_protocol["base_seed"]) + 1000003 * iteration,
                    delta_tolerance=float(
                        soft_payload["soft_pair_aggregation"][
                            "practical_delta_tolerance"
                        ]
                    ),
                )
            )
            ledger_after = dict(engine.environment.oracle_ledger())
            verification_delta = evaluator_ledger_delta(ledger_before, ledger_after)
            evaluator_budget.reconcile(
                reservation, verification_delta.requested_calls
            )
        except Exception:
            evaluator_budget.release(reservation)
            raise
        aggregation_config = SoftPairAggregationConfig(
            **soft_payload["soft_pair_aggregation"]
        )
        soft_evidence = aggregate_soft_verifications(
            verifications, aggregation_config
        )
        write_json(
            iteration_dir / "oracle-soft-verification.json",
            {
                "iteration": iteration,
                "selected_ids": list(selected_ids),
                "selected_only": True,
                "replicates": 5,
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
    write_json(iteration_dir / "llm-auxiliary-receipt.json", final_receipt)

    failures = []
    if sha256_path(primary_checkpoint) != primary_checkpoint_sha256:
        failures.append("primary_ppo_checkpoint_changed")
    if final_receipt.get("continue_training") is not True:
        failures.append("optional_llm_blocked_primary_training")
    if final_receipt.get("silent_fallback_used") is not False:
        failures.append("silent_fallback_used")
    if refinement is not None:
        if refinement.get("llm_confidence_used_for_weight") is not False:
            failures.append("llm_confidence_entered_loss")
        if refinement.get("counterfactual_actions_in_ppo_clipping") is not False:
            failures.append("counterfactual_action_entered_ppo_clipping")
        if value_head_before != value_head_after:
            failures.append("value_head_changed_in_auxiliary_step")
        for name in ("mean_joint_kl", "maximum_joint_kl"):
            if float(refinement[name]) < 0.0:
                failures.append("negative_kl_after_numerical_correction")
    if any(len(item.deltas) != 5 for item in soft_evidence):
        failures.append("soft_verification_not_k5")
    acquisition = acquisition_holder.get("summary")
    if acquisition is not None:
        if acquisition["semantic_attempt_count"] > 4:
            failures.append("schema_attempt_bound_exceeded")
        if acquisition["http_transmissions_observed"] > 8:
            failures.append("http_transmission_bound_exceeded")
    if pool_records:
        if any(len(pool["candidate_ids"]) != 24 for pool in pool_records):
            failures.append("candidate_pool_size_mismatch")
        if any(len(pool["candidate_timesteps"]) < 2 for pool in pool_records):
            failures.append("cross_timestep_pool_missing")
        if any(not pool["presented_ids_match_pool"] for pool in pool_records):
            failures.append("presented_candidate_pool_mismatch")
        if any(
            pool["reward_truth_exposed"] or pool["policy_score_exposed"]
            for pool in pool_records
        ):
            failures.append("llm_prompt_information_leakage")
    failures = sorted(set(failures))
    record = {
        "iteration": iteration,
        "execution_status": "passed" if not failures else "failed",
        "integrity_failures": failures,
        "standard_ppo": iteration_summary(ppo_result),
        "primary_ppo_checkpoint": {
            "path": str(primary_checkpoint),
            "sha256": primary_checkpoint_sha256,
            "written_before_llm": True,
        },
        "candidate_pools": pool_records,
        "acquisition": acquisition,
        "provider_public_identity_written": "identity" in provider_holder,
        "llm_wall_clock": wall_receipt_holder.get("receipt"),
        "auxiliary_receipt": final_receipt,
        "verification": {
            "verified_count": len(verifications),
            "soft_weighted_count": sum(
                item.training_weight > 0.0 for item in soft_evidence
            ),
            "ledger_delta": (
                asdict(verification_delta) if verification_delta else None
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
            "after_standard_ppo": post_ppo_policy_sha256,
            "final": engine.policy_state_sha256,
        },
        "circuit_breaker_after_iteration": breaker.state_dict(),
        "llm_wall_budget_after_iteration": llm_wall_budget.state_dict(),
        "evaluator_ledger_after_iteration": dict(
            engine.environment.oracle_ledger()
        ),
    }
    if failures:
        write_json(iteration_dir / "iteration-report.json", record)
        raise ContractViolation("short-horizon iteration integrity failure: %s" % failures)
    return record


def main() -> None:
    args = parse_args()
    root = args.dapigen_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("short-horizon runner requires Slurm")
    short_protocol = load_short_horizon_protocol(protocol_path, root)
    if socket.gethostname() != short_protocol["required_host"]:
        raise RuntimeError("short-horizon runner is bound to n001")
    source = git_identity(root)
    if source["dirty"] is not False:
        raise RuntimeError("short-horizon runner requires a clean worktree")
    if output_dir.exists():
        raise FileExistsError("short-horizon output already exists")

    v3_protocol_path = Path(
        short_protocol["resolved_prerequisites"]["v3_protocol"]
    )
    _v3_protocol, base_protocol, soft_payload = load_v3_protocol(
        v3_protocol_path, root
    )
    runtime_protocol = _runtime_protocol(base_protocol, short_protocol)
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

    resume_path = (
        args.resume_control_checkpoint.resolve(strict=True)
        if args.resume_control_checkpoint
        else None
    )
    resume_sha256 = sha256_path(resume_path) if resume_path else None
    authorization = load_execution_authorization_short_horizon(
        args.execution_authorization.resolve(strict=True),
        protocol_sha256=sha256_path(protocol_path),
        implementation_commit=source["commit"],
        output_dir=output_dir,
        polybert_path=polybert_path,
        polybert_asset_binding_sha256=model_asset["asset_binding_sha256"],
        polybert_checkpoint_fingerprint=model_asset["checkpoint_fingerprint"],
        evaluator_asset_path=evaluator_asset_path,
        evaluator_asset_binding_sha256=evaluator_asset["asset_binding_sha256"],
        evaluator_asset_fingerprint=evaluator_asset["asset_fingerprint"],
        resume_control_checkpoint=resume_path,
        resume_control_checkpoint_sha256=resume_sha256,
    )

    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "schema_version": 1,
            "protocol_id": SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID,
            "source": source,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "protocol_sha256": sha256_path(protocol_path),
            "authorization_id": authorization["authorization_id"],
            "authorization_sha256": sha256_path(
                args.execution_authorization.resolve(strict=True)
            ),
            "authorized_iteration_range": [
                authorization["authorized_start_iteration"],
                authorization["authorized_end_iteration"],
            ],
            "resume_control_checkpoint": str(resume_path) if resume_path else None,
            "polybert_asset": model_asset,
            "evaluator_asset": evaluator_asset,
            "evaluator_route_source_delta": source_delta,
            "credentials_loaded_at_intent_time": False,
            "external_api_invoked_at_intent_time": False,
            "automatic_resume_authorized": False,
            "automatic_rerun_authorized": False,
            "formal_training_authorized": False,
        },
    )
    started = time.perf_counter()
    components, engine, specification, run_contract = build_runtime(
        root,
        accepted_manifest,
        accepted_binding,
        runtime_protocol,
        polybert_path,
        polybert_checkpoint_fingerprint=model_asset["checkpoint_fingerprint"],
        evaluator_asset_path=evaluator_asset_path,
    )
    breaker = LLMAuxiliaryCircuitBreaker(
        LLMAuxiliaryResilienceConfig(
            maximum_consecutive_llm_failures=int(
                short_protocol["circuit_breaker"][
                    "maximum_consecutive_llm_failures"
                ]
            ),
            circuit_cooldown_iterations=int(
                short_protocol["circuit_breaker"]["circuit_cooldown_iterations"]
            ),
        )
    )
    wall_config = dict(short_protocol["llm_wall_clock_budget"])
    wall_config.pop("persist_across_checkpoints")
    wall_config.pop("exhaustion_outcome")
    llm_wall_budget = LLMWallClockBudget(LLMWallClockBudgetConfig(**wall_config))
    previous_control_sha256 = None
    previous_history_digest = None
    restored = None
    if resume_path is not None:
        restored = load_short_horizon_checkpoint(
            engine=engine,
            breaker=breaker,
            wall_budget=llm_wall_budget,
            control_checkpoint_path=resume_path,
            expected_protocol_sha256=sha256_path(protocol_path),
            expected_control_sha256=resume_sha256,
        )
        previous_control_sha256 = restored["control_checkpoint_sha256"]
        previous_history_digest = restored["history_digest"]
        expected_start = int(restored["completed_iteration"]) + 1
    else:
        expected_start = 1
    if authorization["authorized_start_iteration"] != expected_start:
        raise ContractViolation("authorization start differs from resume state")

    iteration_records = []
    for iteration in range(
        int(authorization["authorized_start_iteration"]),
        int(authorization["authorized_end_iteration"]) + 1,
    ):
        iteration_dir = output_dir / "iterations" / ("iteration-%02d" % iteration)
        record = _iteration(
            iteration=iteration,
            iteration_dir=iteration_dir,
            engine=engine,
            components=components,
            base_protocol=runtime_protocol,
            short_protocol=short_protocol,
            soft_payload=soft_payload,
            breaker=breaker,
            llm_wall_budget=llm_wall_budget,
            credentials_file=args.credentials_file,
        )
        compact = {
            "iteration": iteration,
            "execution_status": record["execution_status"],
            "ppo_batch_id": record["standard_ppo"]["batch_id"],
            "ppo_rollout_digest": record["standard_ppo"]["rollout_digest"],
            "auxiliary_status": record["auxiliary_receipt"]["auxiliary_status"],
            "method_observed": record["auxiliary_receipt"]["method_observed"],
            "policy_sha256": record["policy_hashes"]["final"],
            "evaluator_requested_calls": record[
                "evaluator_ledger_after_iteration"
            ]["requested_calls"],
            "circuit_breaker": record["circuit_breaker_after_iteration"],
            "llm_wall_budget": record["llm_wall_budget_after_iteration"],
        }
        current_history_digest = chained_history_digest(
            previous_history_digest, compact
        )
        checkpoint = save_short_horizon_checkpoint(
            engine=engine,
            breaker=breaker,
            wall_budget=llm_wall_budget,
            engine_checkpoint_path=iteration_dir / "final-engine.pt",
            control_checkpoint_path=iteration_dir / "control-checkpoint.json",
            completed_iteration=iteration,
            protocol_sha256=sha256_path(protocol_path),
            history_digest=current_history_digest,
            previous_control_sha256=previous_control_sha256,
        )
        record["control_checkpoint"] = checkpoint
        record["history_digest"] = current_history_digest
        write_json(iteration_dir / "iteration-report.json", record)
        iteration_records.append(record)
        previous_control_sha256 = checkpoint["control_checkpoint_sha256"]
        previous_history_digest = current_history_digest

    final_ledger = dict(engine.environment.oracle_ledger())
    maximum_evaluator_calls = int(
        short_protocol["scientific_budget"][
            "maximum_requested_evaluator_calls_total"
        ]
    )
    integrity_failures = []
    if final_ledger["requested_calls"] > maximum_evaluator_calls:
        integrity_failures.append("cumulative_evaluator_budget_exceeded")
    if llm_wall_budget.consumed_seconds > float(
        short_protocol["llm_wall_clock_budget"]["maximum_seconds_total"]
    ) + 1.0:
        integrity_failures.append("cumulative_llm_wall_time_materially_exceeded")
    report = {
        "schema_version": 1,
        "protocol_id": SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID,
        "execution_status": "passed" if not integrity_failures else "failed",
        "module_decision": (
            "short_horizon_engineering_run_complete_no_further_scope_authorized"
            if not integrity_failures
            else "fatal_integrity_failure_no_further_scope"
        ),
        "classification": {
            "scope": "bounded-short-horizon-engineering-only",
            "integrity_failures": integrity_failures,
            "automatic_rerun_authorized": False,
            "formal_training_authorized": False,
            "algorithm_effectiveness_established": False,
            "scientific_claim_authorized": False,
        },
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "source": source,
        "protocol_sha256": sha256_path(protocol_path),
        "authorization_id": authorization["authorization_id"],
        "restored_from": restored,
        "stack_specification": specification,
        "run_contract": asdict(run_contract),
        "iteration_count": len(iteration_records),
        "iterations": iteration_records,
        "final_control_checkpoint_sha256": previous_control_sha256,
        "final_history_digest": previous_history_digest,
        "final_circuit_breaker": breaker.state_dict(),
        "final_llm_wall_clock_budget": llm_wall_budget.state_dict(),
        "final_evaluator_ledger": final_ledger,
        "credentials_path_logged": False,
        "api_key_logged": False,
        "local_llm_loaded": False,
        "sealed_test_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "resources": {
            "peak_rss_mib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            / 1024.0
        },
    }
    write_json(output_dir / "short-horizon-report.json", report)
    print(
        json.dumps(
            {
                "execution_status": report["execution_status"],
                "module_decision": report["module_decision"],
                "iteration_count": report["iteration_count"],
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
