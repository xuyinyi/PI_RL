#!/usr/bin/env python
"""Run one bounded PPO plus SciCF integration cycle through Slurm."""

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
from typing import Any, Dict

from reproduction.framework.io import git_identity, write_json
from reproduction.p2.budget import RequestedCallBudgetManager, evaluator_ledger_delta
from reproduction.p2.engine import PPOEngineConfig
from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    sha256_path,
    verify_accepted_binding,
)
from reproduction.scicf.llm.api_client import APISettings, OpenAICompatibleClient

from .contracts import (
    SINGLE_ITERATION_INTEGRATION_PROTOCOL_ID,
    PairwiseRefinementConfig,
)
from .pipeline import (
    FrozenPolicySampler,
    build_online_candidate_pool,
    eligible_online_episode_ids,
    refine_verified_pairs,
    verify_selected_candidates,
)
from .prompt import build_online_acquisition_request, validate_online_ranked_response
from .run_pairwise_stability import _head_hash, _resolve_input
from .run_smoke import _extract_json, build_runtime, iteration_summary
from .stability import factorized_policy_drift, pairwise_preference_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--polybert-path", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "classification",
        "required_host",
        "base_seed",
        "prerequisite",
        "accepted_binding",
        "budget",
        "acquisition",
        "verification",
        "pairwise_refinement",
        "integration_gate",
        "ppo",
        "claim_boundary",
    }
    if set(payload) != required:
        raise ValueError(
            "single-iteration config keys differ; missing=%s extra=%s"
            % (sorted(required - set(payload)), sorted(set(payload) - required))
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported single-iteration integration schema")
    if payload["protocol_id"] != SINGLE_ITERATION_INTEGRATION_PROTOCOL_ID:
        raise ValueError("unsupported single-iteration integration protocol")
    if payload["classification"] != "single_iteration_engineering_integration_smoke_only":
        raise ValueError("integration classification is not engineering-only")
    acquisition = payload["acquisition"]
    verification = payload["verification"]
    if acquisition.get("external_api_authorized") is not True:
        raise ValueError("external API use was not declared")
    if int(acquisition["request_count"]) != 2 or int(acquisition["pool_count"]) != 2:
        raise ValueError("v1 freezes exactly two candidate pools and API requests")
    if int(acquisition["pool_size"]) != 24:
        raise ValueError("v1 freezes 24 candidates per pool")
    if int(acquisition["maximum_selected_per_pool"]) != 4:
        raise ValueError("v1 freezes maximum B=4 per pool")
    if acquisition["episode_selection"] != "outcome_blind_deterministic_shuffle":
        raise ValueError("outcome-blind episode selection is required")
    if int(verification["replicates"]) != 2:
        raise ValueError("v1 freezes verification at K=2")
    if verification["candidate_scope"] != "llm_selected_only":
        raise ValueError("integration smoke verifies selected candidates only")
    for name in (
        "scientific_claim_authorized",
        "algorithm_effectiveness_established",
        "formal_training_authorized",
        "multi_iteration_training_authorized",
        "sealed_test_access_authorized",
        "baseline_mutation_authorized",
    ):
        if payload["claim_boundary"].get(name) is not False:
            raise ValueError("claim boundary must disable %s" % name)
    PPOEngineConfig(**payload["ppo"])
    PairwiseRefinementConfig(**payload["pairwise_refinement"])
    return payload


def main() -> None:
    args = parse_args()
    root = args.dapigen_root.resolve()
    config_path = args.config.resolve()
    manifest_path = args.accepted_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("output directory already exists: %s" % output_dir)
    config = load_config(config_path)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("single-iteration integration requires Slurm")
    if socket.gethostname() != config["required_host"]:
        raise RuntimeError("single-iteration integration is bound to %s" % config["required_host"])
    source = git_identity(root)
    if source.get("dirty") is not False:
        raise RuntimeError("single-iteration integration requires a clean Git worktree")
    accepted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = verify_accepted_binding(
        root, accepted_manifest, config["accepted_binding"]["environment_id"]
    )
    if binding["stage0_source_changed_from_accepted"]:
        raise RuntimeError("accepted Stage-0 source changed before integration")

    prerequisite = config["prerequisite"]
    stability_path = _resolve_input(
        root,
        prerequisite["stability_report"],
        prerequisite["stability_report_sha256"],
    )
    stability_report = json.loads(stability_path.read_text(encoding="utf-8"))
    if stability_report["source"]["commit"] != prerequisite["expected_source_commit"]:
        raise RuntimeError("pairwise-stability source commit mismatch")
    if stability_report["module_decision"] != prerequisite["required_decision"]:
        raise RuntimeError("pairwise-stability prerequisite decision is closed")
    if stability_report["classification"]["next_scope_authorized"] is not True:
        raise RuntimeError("pairwise-stability next-scope flag is closed")

    settings = APISettings.from_private_file(args.credentials_file.resolve())
    provider = settings.public_identity()
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "status": "declared-before-rollout-and-before-first-api-request",
            "protocol_id": SINGLE_ITERATION_INTEGRATION_PROTOCOL_ID,
            "classification": config["classification"],
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "config_sha256": sha256_path(config_path),
            "accepted_manifest_sha256": sha256_path(manifest_path),
            "prerequisite_sha256": prerequisite["stability_report_sha256"],
            "provider": provider,
            "external_api_request_limit": 2,
            "credentials": {
                "private_file_used": True,
                "path_logged": False,
                "api_key_logged": False,
            },
            "claim_boundary": config["claim_boundary"],
        },
    )

    started = time.perf_counter()
    polybert_path = args.polybert_path.resolve()
    if not polybert_path.is_dir():
        raise FileNotFoundError("polyBERT path is not a local directory")
    components, engine, specification, run_contract = build_runtime(
        root, accepted_manifest, binding, config, polybert_path
    )
    initial_policy_sha256 = engine.policy_state_sha256
    behavior_policy = FrozenPolicySampler(engine.model, config["ppo"]["device"])
    ppo_result = engine.run_iteration(query_requested_calls=0)
    post_ppo_policy_sha256 = engine.policy_state_sha256
    if behavior_policy.state_sha256 != ppo_result.rollout.frozen_policy.state_sha256:
        raise RuntimeError("behavior-policy copy differs from PPO rollout policy")

    eligible = list(
        eligible_online_episode_ids(ppo_result.rollout, prefer_successful=False)
    )
    random.Random(int(config["base_seed"])).shuffle(eligible)
    pool_count = int(config["acquisition"]["pool_count"])
    if len(eligible) < pool_count:
        raise RuntimeError("rollout lacks two complete cross-timestep episodes")
    chosen_episode_ids = tuple(eligible[:pool_count])
    all_candidates = []
    pool_records = []
    api_records = []
    client = OpenAICompatibleClient(settings)
    prompt_tokens = 0
    completion_tokens = 0
    token_accounting_complete = True
    api_completion_times = []
    for pool_index, episode_id in enumerate(chosen_episode_ids):
        pool_seed = int(config["base_seed"]) + int(
            config["acquisition"]["pool_seed_stride"]
        ) * pool_index
        candidates, trajectory_context = build_online_candidate_pool(
            rollout=ppo_result.rollout,
            core=components.core,
            behavior_policy=behavior_policy,
            pool_size=int(config["acquisition"]["pool_size"]),
            seed=pool_seed,
            episode_id=episode_id,
        )
        request_id = "%s-pool-%02d-job-%s" % (
            SINGLE_ITERATION_INTEGRATION_PROTOCOL_ID,
            pool_index,
            os.environ["SLURM_JOB_ID"],
        )
        request = build_online_acquisition_request(
            request_id=request_id,
            trajectory_context=trajectory_context,
            candidates=candidates,
            maximum_budget=int(config["acquisition"]["maximum_selected_per_pool"]),
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
            output_dir / "requests" / ("pool-%02d-request.json" % pool_index), request
        )
        completion = client.complete(
            messages=request["messages"],
            max_tokens=int(config["acquisition"]["max_output_tokens"]),
            seed=pool_seed,
            timeout_seconds=float(config["acquisition"]["timeout_seconds"]),
            transport_retries=int(config["acquisition"]["transport_retries"]),
        )
        ranked = validate_online_ranked_response(
            _extract_json(completion.content),
            request["candidate_ids"],
            int(config["acquisition"]["maximum_selected_per_pool"]),
        )
        api_record = {
            "pool_index": pool_index,
            "request_id": request_id,
            "prompt_sha256": request["prompt_sha256"],
            "provider": provider,
            "provider_response_id": completion.provider_response_id,
            "system_fingerprint": completion.system_fingerprint,
            "provider_response_sha256": completion.response_sha256,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens,
            "transport_retries_used": completion.transport_retries_used,
            "raw_response": completion.content,
            "validated": ranked,
            "api_key_logged": False,
        }
        write_json(
            output_dir / "responses" / ("pool-%02d-response.json" % pool_index),
            api_record,
        )
        api_records.append(api_record)
        api_completion_times.append(time.perf_counter())
        if completion.prompt_tokens is None or completion.completion_tokens is None:
            token_accounting_complete = False
        else:
            prompt_tokens += int(completion.prompt_tokens)
            completion_tokens += int(completion.completion_tokens)
        pool_records.append(
            {
                "pool_index": pool_index,
                "episode_id": episode_id,
                "candidate_ids": list(request["candidate_ids"]),
                "candidate_timesteps": sorted({item.timestep for item in candidates}),
                "presented_ids_match_pool": set(request["presented_candidate_ids"])
                == set(request["candidate_ids"]),
                "reward_truth_exposed": bool(
                    request["candidate_reward_truth_exposed"]
                    or request["factual_reward_truth_exposed"]
                ),
                "policy_score_exposed": bool(request["policy_score_exposed"]),
                "selected_ids": list(ranked["selected_intervention_ids"]),
                "abstained": bool(ranked["abstain"]),
            }
        )
        all_candidates.extend(candidates)

    candidate_map = {item.candidate_id: item for item in all_candidates}
    if len(candidate_map) != len(all_candidates):
        raise RuntimeError("candidate ids are not unique across integration pools")
    selected_ids = tuple(
        identifier for pool in pool_records for identifier in pool["selected_ids"]
    )
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("selected candidate ids repeat across integration pools")
    query_budget = RequestedCallBudgetManager(engine.environment.oracle_ledger)
    maximum_verification_calls = (
        2 * len(selected_ids) * int(config["verification"]["replicates"])
    )
    reservation = query_budget.reserve(maximum_verification_calls)
    ledger_before_verification = dict(engine.environment.oracle_ledger())
    verification_started = time.perf_counter()
    all_requests_completed_before_verification = bool(
        len(api_completion_times) == int(config["acquisition"]["request_count"])
        and all(value <= verification_started for value in api_completion_times)
    )
    try:
        verifications = verify_selected_candidates(
            environment=engine.environment,
            policy=behavior_policy,
            candidates_by_id=candidate_map,
            selected_ids=selected_ids,
            replicates=int(config["verification"]["replicates"]),
            seed=int(config["base_seed"]),
            delta_tolerance=float(config["pairwise_refinement"]["delta_tolerance"]),
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
            "replicates": int(config["verification"]["replicates"]),
            "maximum_branch_count": maximum_verification_calls,
            "ledger_delta": asdict(verification_delta),
            "records": [item.to_dict() for item in verifications],
        },
    )

    accepted = tuple(item for item in verifications if item.accepted)
    before_pairwise_model = copy.deepcopy(engine.model).to(engine.device).eval()
    value_head_before = _head_hash(engine.model, "value_head")
    pairwise_before = None
    if accepted:
        pairwise_before = pairwise_preference_metrics(
            model=engine.model,
            candidates_by_id=candidate_map,
            verifications=accepted,
            device=engine.device,
        )
    refinement = refine_verified_pairs(
        engine=engine,
        candidates_by_id=candidate_map,
        verifications=verifications,
        config=PairwiseRefinementConfig(**config["pairwise_refinement"]),
    )
    value_head_after = _head_hash(engine.model, "value_head")
    pairwise_after = None
    if accepted:
        pairwise_after = pairwise_preference_metrics(
            model=engine.model,
            candidates_by_id=candidate_map,
            verifications=accepted,
            device=engine.device,
        )
    pairwise_drift = factorized_policy_drift(
        before_model=before_pairwise_model,
        after_model=engine.model,
        candidates=all_candidates,
        device=engine.device,
    )
    final_policy_sha256 = engine.policy_state_sha256
    checkpoint_path = output_dir / "scicf-single-iteration-checkpoint.pt"
    engine.save_checkpoint(checkpoint_path)
    checkpoint_sha256 = sha256_path(checkpoint_path)
    final_ledger = dict(engine.environment.oracle_ledger())

    integrity_failures = []
    integration_failures = []
    if len(api_records) != 2:
        integrity_failures.append("api_request_count_mismatch")
    if not all_requests_completed_before_verification:
        integrity_failures.append("verification_started_before_all_api_requests_completed")
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
        != int(config["verification"]["replicates"])
        or len(item.counterfactual_terminal_records)
        != int(config["verification"]["replicates"])
        for item in verifications
    ):
        integrity_failures.append("verification_branch_record_count_mismatch")
    if ppo_result.credit.actor_advantages_sha256 != ppo_result.rollout.gae_sha256:
        integrity_failures.append("ppo_actor_credit_not_exact_gae")
    if (
        ppo_result.receipt.critic_returns_sha256
        != ppo_result.rollout.critic_returns_sha256
    ):
        integrity_failures.append("ppo_critic_returns_not_from_rollout_environment_returns")
    if ppo_result.credit.evaluator_delta.requested_calls != 0:
        integrity_failures.append("ppo_credit_phase_queried_evaluator")
    if ppo_result.credit.diagnostics.get("counterfactual_actions_in_ppo_clipping") is not False:
        integrity_failures.append("counterfactual_action_entered_ppo_clipping")
    if initial_policy_sha256 == post_ppo_policy_sha256:
        integrity_failures.append("standard_ppo_policy_did_not_update")
    if int(ppo_result.receipt.policy_version_after) != 1:
        integrity_failures.append("standard_ppo_policy_version_mismatch")
    observed_sources = set(verification_delta.requested_by_source)
    if not observed_sources.issubset(
        {"scicf_ppo/factual", "scicf_ppo/counterfactual"}
    ):
        integrity_failures.append("unexpected_verification_evaluator_source")
    expected_factual_evaluator_calls = sum(
        record.get("terminal_evaluation") is not None
        for item in verifications
        for record in item.factual_terminal_records
    )
    expected_counterfactual_evaluator_calls = sum(
        record.get("terminal_evaluation") is not None
        for item in verifications
        for record in item.counterfactual_terminal_records
    )
    expected_evaluator_calls = (
        expected_factual_evaluator_calls + expected_counterfactual_evaluator_calls
    )
    if verification_delta.requested_calls != expected_evaluator_calls:
        integrity_failures.append("verification_requested_call_count_mismatch")
    expected_source_calls = {
        source: count
        for source, count in (
            ("scicf_ppo/counterfactual", expected_counterfactual_evaluator_calls),
            ("scicf_ppo/factual", expected_factual_evaluator_calls),
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

    if len(accepted) < int(config["verification"]["minimum_accepted_pairs"]):
        integration_failures.append("insufficient_sign_consistent_pairs")
    if refinement["status"] != "applied" or int(refinement["optimizer_steps"]) != 1:
        integration_failures.append("pairwise_update_not_applied_once")
    if final_policy_sha256 == post_ppo_policy_sha256:
        integration_failures.append("pairwise_update_did_not_change_post_ppo_policy")
    tolerance = float(config["integration_gate"]["comparison_tolerance"])
    pairwise_margin_improvement = None
    if pairwise_before is not None and pairwise_after is not None:
        pairwise_margin_improvement = float(
            pairwise_after["mean_signed_margin"]
            - pairwise_before["mean_signed_margin"]
        )
        if pairwise_margin_improvement <= float(
            config["integration_gate"]["minimum_pairwise_mean_margin_improvement"]
        ) + tolerance:
            integration_failures.append("pairwise_mean_margin_not_improved")
        if (
            config["integration_gate"]["require_non_decreasing_pairwise_accuracy"]
            and pairwise_after["preference_accuracy"] + tolerance
            < pairwise_before["preference_accuracy"]
        ):
            integration_failures.append("pairwise_preference_accuracy_decreased")
    if pairwise_drift["maximum_joint_kl"] > float(
        config["integration_gate"]["maximum_full_support_joint_kl"]
    ) + tolerance:
        integration_failures.append("pairwise_full_support_joint_kl_exceeded")
    if pairwise_drift["maximum_non_target_factor_kl"] > float(
        config["integration_gate"]["maximum_non_target_factor_kl"]
    ) + tolerance:
        integration_failures.append("pairwise_non_target_factor_kl_exceeded")
    if pairwise_drift["maximum_absolute_value_drift"] > float(
        config["integration_gate"]["maximum_absolute_value_drift"]
    ) + tolerance:
        integration_failures.append("pairwise_critic_value_drift_exceeded")

    integration_failures = sorted(set(integration_failures))
    next_scope_authorized = not integrity_failures and not integration_failures
    report = {
        "schema_version": 1,
        "protocol_id": SINGLE_ITERATION_INTEGRATION_PROTOCOL_ID,
        "execution_status": "passed" if not integrity_failures else "failed",
        "module_decision": (
            "go_bounded_short_horizon_multi_iteration_engineering_smoke"
            if next_scope_authorized
            else "no_go_bounded_short_horizon_multi_iteration_engineering_smoke"
        ),
        "classification": {
            "scope": "single-iteration-ppo-plus-scicf-engineering-smoke",
            "integrity_failures": integrity_failures,
            "integration_failures": integration_failures,
            "next_scope_authorized": next_scope_authorized,
            "formal_training_authorized": False,
            "multi_iteration_training_authorized": False,
            "algorithm_effectiveness_established": False,
            "scientific_claim_authorized": False,
        },
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        },
        "source": source,
        "accepted_binding": binding,
        "stack_specification": specification,
        "run_contract": asdict(run_contract),
        "config": config,
        "standard_ppo": iteration_summary(ppo_result),
        "acquisition": {
            "pool_count": len(pool_records),
            "pool_size": 24,
            "chosen_episode_ids": list(chosen_episode_ids),
            "outcome_blind_episode_selection": True,
            "pools": pool_records,
            "request_count": len(api_records),
            "provider": provider,
            "all_requests_completed_before_verification": (
                all_requests_completed_before_verification
            ),
            "token_accounting_complete": token_accounting_complete,
            "reported_tokens": {
                "prompt": prompt_tokens if token_accounting_complete else None,
                "completion": completion_tokens if token_accounting_complete else None,
                "total": (
                    prompt_tokens + completion_tokens
                    if token_accounting_complete
                    else None
                ),
            },
        },
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
            "labels_generated_after_standard_ppo_update": True,
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
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "final_evaluator_ledger": final_ledger,
        "external_api_invoked": True,
        "local_model_invoked": False,
        "sealed_test_accessed": False,
        "historical_gate_1b3_relabelled": False,
        "elapsed_seconds": time.perf_counter() - started,
        "resources": {
            "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            / 1024.0,
        },
    }
    write_json(output_dir / "integration-report.json", report)
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
