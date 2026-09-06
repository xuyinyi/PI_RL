#!/usr/bin/env python
"""Run the frozen, development-only SciCF pairwise stability protocol."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from reproduction.framework.io import git_identity, write_json
from reproduction.p2.budget import evaluator_ledger_delta
from reproduction.p2.contracts import canonical_sha256
from reproduction.p2.engine import PPOEngineConfig, rollout_digest, torch_state_sha256
from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    sha256_path,
    verify_accepted_binding,
)

from .contracts import (
    PAIRWISE_STABILITY_PROTOCOL_ID,
    OnlineVerification,
    PairwiseRefinementConfig,
)
from .pipeline import (
    FrozenPolicySampler,
    build_online_candidate_pool,
    eligible_online_episode_ids,
    refine_verified_pairs,
)
from .run_smoke import build_runtime
from .stability import factorized_policy_drift, pairwise_preference_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--polybert-path", type=Path, required=True)
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
        "accepted_binding",
        "input_artifacts",
        "budget",
        "reconstruction",
        "pairwise_refinement",
        "stability_design",
        "ppo",
        "claim_boundary",
    }
    if set(payload) != required:
        raise ValueError(
            "pairwise stability config keys differ; missing=%s extra=%s"
            % (sorted(required - set(payload)), sorted(set(payload) - required))
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported pairwise stability schema")
    if payload["protocol_id"] != PAIRWISE_STABILITY_PROTOCOL_ID:
        raise ValueError("unsupported pairwise stability protocol")
    if payload["classification"] != "pairwise_optimizer_stability_development_only":
        raise ValueError("pairwise stability classification is not development-only")
    reconstruction = payload["reconstruction"]
    design = payload["stability_design"]
    if int(reconstruction["counterfactual_oracle_calls_authorized"]) != 0:
        raise ValueError("fresh counterfactual labels must remain disabled")
    if int(reconstruction["external_api_requests_authorized"]) != 0:
        raise ValueError("external API calls must remain disabled")
    if int(design["fold_count"]) != 5 or int(design["replicas_per_scenario"]) != 2:
        raise ValueError("v1 freezes five folds and two exact replicas")
    expected_runs = (1 + int(design["fold_count"])) * int(
        design["replicas_per_scenario"]
    )
    if int(design["expected_update_runs"]) != expected_runs:
        raise ValueError("expected update-run count is inconsistent")
    for name in (
        "scientific_claim_authorized",
        "formal_training_authorized",
        "multi_iteration_ppo_authorized",
        "sealed_test_access_authorized",
        "baseline_mutation_authorized",
        "fresh_llm_requests_authorized",
        "fresh_counterfactual_labels_authorized",
    ):
        if payload["claim_boundary"].get(name) is not False:
            raise ValueError("claim boundary must disable %s" % name)
    PPOEngineConfig(**payload["ppo"])
    PairwiseRefinementConfig(**payload["pairwise_refinement"])
    return payload


def _resolve_input(root: Path, relative: str, expected_sha256: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if root != path and root not in path.parents:
        raise ValueError("input artifact escaped the repository root")
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(
            "input artifact hash mismatch for %s: %s != %s"
            % (relative, observed, expected_sha256)
        )
    return path


def _verification(value: Mapping[str, Any]) -> OnlineVerification:
    return OnlineVerification(
        candidate_id=value["candidate_id"],
        paired_seeds=tuple(int(item) for item in value["paired_seeds"]),
        factual_returns=tuple(float(item) for item in value["factual_returns"]),
        counterfactual_returns=tuple(
            float(item) for item in value["counterfactual_returns"]
        ),
        deltas=tuple(float(item) for item in value["deltas"]),
        accepted=bool(value["accepted"]),
        preferred=value["preferred"],
        rejection_reason=value["rejection_reason"],
        factual_terminal_records=tuple(value["factual_terminal_records"]),
        counterfactual_terminal_records=tuple(
            value["counterfactual_terminal_records"]
        ),
    )


def _pool_id(candidate_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(candidate_ids)).encode("utf-8")).hexdigest()


def _stratified_folds(
    verifications: Sequence[OnlineVerification], fold_count: int
) -> Tuple[Tuple[str, ...], ...]:
    positive = sorted(
        (item.candidate_id for item in verifications if item.mean_delta > 0.0),
        key=lambda identifier: hashlib.sha256(
            (PAIRWISE_STABILITY_PROTOCOL_ID + "|positive|" + identifier).encode("utf-8")
        ).hexdigest(),
    )
    negative = sorted(
        (item.candidate_id for item in verifications if item.mean_delta < 0.0),
        key=lambda identifier: hashlib.sha256(
            (PAIRWISE_STABILITY_PROTOCOL_ID + "|negative|" + identifier).encode("utf-8")
        ).hexdigest(),
    )
    folds = [[] for _ in range(int(fold_count))]
    for index, identifier in enumerate(positive):
        folds[index % int(fold_count)].append(identifier)
    for index, identifier in enumerate(negative):
        folds[index % int(fold_count)].append(identifier)
    result = tuple(tuple(sorted(fold)) for fold in folds)
    if any(not fold for fold in result):
        raise RuntimeError("stratified pairwise fold is empty")
    if set().union(*(set(fold) for fold in result)) != {
        item.candidate_id for item in verifications
    }:
        raise RuntimeError("stratified folds do not cover accepted pairs")
    if sum(len(fold) for fold in result) != len(verifications):
        raise RuntimeError("stratified folds overlap")
    return result


def _head_hash(model, head_name: str) -> str:
    head = getattr(model, head_name)
    return torch_state_sha256(
        {"weight": head.weight.detach(), "bias": head.bias.detach()}
    )


def _delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> float:
    return float(after[key]) - float(before[key])


def _run_update_probe(
    *,
    engine,
    initial_model_state,
    initial_optimizer_state,
    initial_policy_version: int,
    candidates_by_id,
    full_candidates,
    train_verifications,
    evaluation_verifications,
    all_accepted_verifications,
    refinement_config,
    scenario: str,
    replica: int,
) -> Mapping[str, Any]:
    engine.model.load_state_dict(initial_model_state, strict=True)
    engine.optimizer.load_state_dict(initial_optimizer_state)
    engine.policy_version = int(initial_policy_version)
    before_model = copy.deepcopy(engine.model).to(engine.device).eval()
    before_hash = engine.policy_state_sha256
    value_head_before = _head_hash(engine.model, "value_head")
    train_before = pairwise_preference_metrics(
        model=engine.model,
        candidates_by_id=candidates_by_id,
        verifications=train_verifications,
        device=engine.device,
    )
    evaluation_before = pairwise_preference_metrics(
        model=engine.model,
        candidates_by_id=candidates_by_id,
        verifications=evaluation_verifications,
        device=engine.device,
    )
    all_before = pairwise_preference_metrics(
        model=engine.model,
        candidates_by_id=candidates_by_id,
        verifications=all_accepted_verifications,
        device=engine.device,
    )
    receipt = refine_verified_pairs(
        engine=engine,
        candidates_by_id=candidates_by_id,
        verifications=train_verifications,
        config=refinement_config,
    )
    post_hash = engine.policy_state_sha256
    value_head_after = _head_hash(engine.model, "value_head")
    train_after = pairwise_preference_metrics(
        model=engine.model,
        candidates_by_id=candidates_by_id,
        verifications=train_verifications,
        device=engine.device,
    )
    evaluation_after = pairwise_preference_metrics(
        model=engine.model,
        candidates_by_id=candidates_by_id,
        verifications=evaluation_verifications,
        device=engine.device,
    )
    all_after = pairwise_preference_metrics(
        model=engine.model,
        candidates_by_id=candidates_by_id,
        verifications=all_accepted_verifications,
        device=engine.device,
    )
    drift = factorized_policy_drift(
        before_model=before_model,
        after_model=engine.model,
        candidates=full_candidates,
        device=engine.device,
    )
    result = {
        "scenario": scenario,
        "replica": int(replica),
        "train_pair_ids": [item.candidate_id for item in train_verifications],
        "evaluation_pair_ids": [
            item.candidate_id for item in evaluation_verifications
        ],
        "policy_sha256_before": before_hash,
        "policy_sha256_after": post_hash,
        "value_head_sha256_before": value_head_before,
        "value_head_sha256_after": value_head_after,
        "value_head_parameters_changed": value_head_before != value_head_after,
        "receipt": dict(receipt),
        "train_before": train_before,
        "train_after": train_after,
        "train_mean_margin_improvement": _delta(
            train_before, train_after, "mean_signed_margin"
        ),
        "evaluation_before": evaluation_before,
        "evaluation_after": evaluation_after,
        "evaluation_mean_margin_improvement": _delta(
            evaluation_before, evaluation_after, "mean_signed_margin"
        ),
        "all_accepted_before": all_before,
        "all_accepted_after": all_after,
        "all_accepted_mean_margin_improvement": _delta(
            all_before, all_after, "mean_signed_margin"
        ),
        "full_support_drift": drift,
    }
    result["replica_signature"] = canonical_sha256(
        {key: value for key, value in result.items() if key not in {"replica", "replica_signature"}}
    )
    engine.model.load_state_dict(initial_model_state, strict=True)
    engine.optimizer.load_state_dict(initial_optimizer_state)
    engine.policy_version = int(initial_policy_version)
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(float(value) for value in values) / len(values))


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
        raise RuntimeError("pairwise stability development requires Slurm")
    if socket.gethostname() != config["required_host"]:
        raise RuntimeError("pairwise stability is bound to %s" % config["required_host"])
    source = git_identity(root)
    if source.get("dirty") is not False:
        raise RuntimeError("pairwise stability requires a clean Git worktree")
    accepted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = verify_accepted_binding(
        root, accepted_manifest, config["accepted_binding"]["environment_id"]
    )
    if binding["stage0_source_changed_from_accepted"]:
        raise RuntimeError("accepted Stage-0 source changed before pairwise stability")

    inputs = config["input_artifacts"]
    report_path = _resolve_input(
        root, inputs["module_report"], inputs["module_report_sha256"]
    )
    oracle_path = _resolve_input(
        root, inputs["oracle_verification"], inputs["oracle_verification_sha256"]
    )
    evaluation_path = _resolve_input(
        root, inputs["matched_evaluation"], inputs["matched_evaluation_sha256"]
    )
    prior_report = json.loads(report_path.read_text(encoding="utf-8"))
    oracle_payload = json.loads(oracle_path.read_text(encoding="utf-8"))
    evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if prior_report["source"]["commit"] != inputs["expected_source_commit"]:
        raise RuntimeError("prior acquisition source commit mismatch")
    if prior_report["module_decision"] != "go_pairwise_stability_development":
        raise RuntimeError("prior gate did not authorize pairwise stability")
    if prior_report["classification"]["next_module_authorized"] is not True:
        raise RuntimeError("prior gate next-module flag is closed")
    if oracle_payload.get("labels_used_for_training") is not False:
        raise RuntimeError("input label provenance is not the frozen evaluation corpus")

    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "status": "declared-before-policy-reconstruction-and-before-update-probes",
            "protocol_id": PAIRWISE_STABILITY_PROTOCOL_ID,
            "classification": config["classification"],
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "config_sha256": sha256_path(config_path),
            "accepted_manifest_sha256": sha256_path(manifest_path),
            "input_hashes": {
                "module_report": inputs["module_report_sha256"],
                "oracle_verification": inputs["oracle_verification_sha256"],
                "matched_evaluation": inputs["matched_evaluation_sha256"],
            },
            "external_api_request_limit": 0,
            "fresh_counterfactual_oracle_call_limit": 0,
            "claim_boundary": config["claim_boundary"],
        },
    )

    started = time.perf_counter()
    polybert_path = args.polybert_path.resolve()
    if not polybert_path.is_dir():
        raise FileNotFoundError("polyBERT path is not a local directory")
    components, reconstruction_engine, specification, run_contract = build_runtime(
        root, accepted_manifest, binding, config, polybert_path
    )
    if reconstruction_engine.policy_state_sha256 != inputs["expected_policy_sha256"]:
        raise RuntimeError("reconstructed initial policy hash mismatch")
    reconstruction_policy = FrozenPolicySampler(
        reconstruction_engine.model, config["ppo"]["device"]
    )
    ledger_before = dict(reconstruction_engine.environment.oracle_ledger())
    frozen = reconstruction_engine.freeze_policy()
    rollout = reconstruction_engine.collect_rollout(frozen)
    ledger_after = dict(reconstruction_engine.environment.oracle_ledger())
    reconstruction_delta = evaluator_ledger_delta(ledger_before, ledger_after)
    observed_rollout_digest = rollout_digest(rollout)
    if observed_rollout_digest != inputs["expected_rollout_digest"]:
        raise RuntimeError("reconstructed rollout digest mismatch")
    if set(reconstruction_delta.requested_by_source) - {"scicf_ppo/on_policy"}:
        raise RuntimeError("reconstruction used a non-on-policy evaluator source")

    eligible = list(eligible_online_episode_ids(rollout, prefer_successful=False))
    import random

    random.Random(int(config["base_seed"])).shuffle(eligible)
    expected_episode_ids = tuple(prior_report["rollout"]["chosen_episode_ids"])
    chosen_episode_ids = tuple(eligible[: int(config["reconstruction"]["pool_count"])])
    if chosen_episode_ids != expected_episode_ids:
        raise RuntimeError("outcome-blind episode reconstruction mismatch")
    candidate_map = {}
    full_candidates = []
    reconstructed_pools = []
    expected_pools = {row["pool_id"]: row for row in evaluation_payload}
    for pool_index, episode_id in enumerate(chosen_episode_ids):
        pool_seed = int(config["base_seed"]) + int(
            config["reconstruction"]["pool_seed_stride"]
        ) * pool_index
        candidates, _context = build_online_candidate_pool(
            rollout=rollout,
            core=components.core,
            behavior_policy=reconstruction_policy,
            pool_size=int(config["reconstruction"]["pool_size"]),
            seed=pool_seed,
            episode_id=episode_id,
        )
        candidate_ids = tuple(item.candidate_id for item in candidates)
        pool_identifier = _pool_id(candidate_ids)
        if pool_identifier not in expected_pools:
            raise RuntimeError("reconstructed pool id is absent from the frozen input")
        if set(candidate_ids) != set(expected_pools[pool_identifier]["verified_gains"]):
            raise RuntimeError("reconstructed candidate ids differ from the frozen pool")
        reconstructed_pools.append(
            {
                "pool_index": pool_index,
                "pool_id": pool_identifier,
                "episode_id": episode_id,
                "candidate_count": len(candidate_ids),
                "candidate_ids_sha256": canonical_sha256(sorted(candidate_ids)),
            }
        )
        for candidate in candidates:
            if candidate.candidate_id in candidate_map:
                raise RuntimeError("candidate id repeated across reconstructed pools")
            candidate_map[candidate.candidate_id] = candidate
            full_candidates.append(candidate)
    if len(candidate_map) != int(inputs["expected_candidate_count"]):
        raise RuntimeError("reconstructed candidate count mismatch")

    verifications = tuple(_verification(row) for row in oracle_payload["records"])
    if set(item.candidate_id for item in verifications) != set(candidate_map):
        raise RuntimeError("frozen verification ids differ from reconstructed candidates")
    accepted = tuple(item for item in verifications if item.accepted)
    positive = tuple(item for item in accepted if item.mean_delta > 0.0)
    negative = tuple(item for item in accepted if item.mean_delta < 0.0)
    if len(accepted) != int(inputs["expected_accepted_pair_count"]):
        raise RuntimeError("accepted pair count mismatch")
    if len(positive) != int(inputs["expected_positive_pair_count"]):
        raise RuntimeError("positive pair count mismatch")
    if len(negative) != int(inputs["expected_negative_pair_count"]):
        raise RuntimeError("negative pair count mismatch")
    folds = _stratified_folds(accepted, int(config["stability_design"]["fold_count"]))
    write_json(
        output_dir / "reconstruction-audit.json",
        {
            "rollout_digest": observed_rollout_digest,
            "initial_policy_sha256": reconstruction_engine.policy_state_sha256,
            "chosen_episode_ids": list(chosen_episode_ids),
            "pools": reconstructed_pools,
            "candidate_count": len(candidate_map),
            "accepted_pair_count": len(accepted),
            "positive_pair_count": len(positive),
            "negative_pair_count": len(negative),
            "folds": [list(fold) for fold in folds],
            "reconstruction_evaluator_delta": asdict(reconstruction_delta),
            "fresh_counterfactual_labels_generated": False,
        },
    )

    del reconstruction_policy
    del reconstruction_engine
    del components
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _training_components, training_engine, _training_spec, _training_contract = build_runtime(
        root, accepted_manifest, binding, config, polybert_path
    )
    if training_engine.policy_state_sha256 != inputs["expected_policy_sha256"]:
        raise RuntimeError("pairwise probe initial policy hash mismatch")
    initial_model_state = copy.deepcopy(training_engine.model.state_dict())
    initial_optimizer_state = copy.deepcopy(training_engine.optimizer.state_dict())
    initial_policy_version = int(training_engine.policy_version)
    refinement_config = PairwiseRefinementConfig(**config["pairwise_refinement"])
    accepted_map = {item.candidate_id: item for item in accepted}
    scenarios = [("full-corpus", accepted, accepted)]
    all_ids = set(accepted_map)
    for fold_index, holdout_ids in enumerate(folds):
        holdout = tuple(accepted_map[identifier] for identifier in holdout_ids)
        train = tuple(
            accepted_map[identifier] for identifier in sorted(all_ids - set(holdout_ids))
        )
        scenarios.append(("fold-%d" % fold_index, train, holdout))

    probes = []
    for scenario, train, evaluation in scenarios:
        for replica in range(int(config["stability_design"]["replicas_per_scenario"])):
            probes.append(
                _run_update_probe(
                    engine=training_engine,
                    initial_model_state=initial_model_state,
                    initial_optimizer_state=initial_optimizer_state,
                    initial_policy_version=initial_policy_version,
                    candidates_by_id=candidate_map,
                    full_candidates=full_candidates,
                    train_verifications=train,
                    evaluation_verifications=evaluation,
                    all_accepted_verifications=accepted,
                    refinement_config=refinement_config,
                    scenario=scenario,
                    replica=replica,
                )
            )
    write_json(output_dir / "update-probes.json", probes)

    integrity_failures = []
    stability_failures = []
    if len(probes) != int(config["stability_design"]["expected_update_runs"]):
        integrity_failures.append("update_probe_count_mismatch")
    grouped = {}
    for probe in probes:
        grouped.setdefault(probe["scenario"], []).append(probe)
    for scenario, rows in grouped.items():
        if len(rows) != 2:
            integrity_failures.append("replica_count_mismatch:%s" % scenario)
        elif len({row["policy_sha256_after"] for row in rows}) != 1 or len(
            {row["replica_signature"] for row in rows}
        ) != 1:
            integrity_failures.append("exact_replica_mismatch:%s" % scenario)
    if training_engine.policy_state_sha256 != inputs["expected_policy_sha256"]:
        integrity_failures.append("initial_policy_not_restored_after_probes")
    training_ledger = dict(training_engine.environment.oracle_ledger())
    if int(training_ledger["requested_calls"]) != 0:
        integrity_failures.append("pairwise_probes_queried_oracle")

    tolerance = float(config["stability_design"]["comparison_tolerance"])
    for probe in probes:
        scenario = probe["scenario"]
        receipt = probe["receipt"]
        drift = probe["full_support_drift"]
        if receipt["status"] != "applied" or int(receipt["optimizer_steps"]) != 1:
            stability_failures.append("update_not_applied:%s" % scenario)
        if probe["policy_sha256_before"] == probe["policy_sha256_after"]:
            stability_failures.append("policy_unchanged_after_update:%s" % scenario)
        if probe["value_head_parameters_changed"]:
            stability_failures.append("value_head_parameters_changed:%s" % scenario)
        if probe["train_mean_margin_improvement"] <= float(
            config["stability_design"]["minimum_train_mean_margin_improvement"]
        ) + tolerance:
            stability_failures.append("train_margin_not_improved:%s" % scenario)
        if drift["maximum_joint_kl"] > float(
            config["stability_design"]["maximum_full_support_joint_kl"]
        ) + tolerance:
            stability_failures.append("full_support_joint_kl_exceeded:%s" % scenario)
        if drift["maximum_non_target_factor_kl"] > float(
            config["stability_design"]["maximum_full_support_non_target_factor_kl"]
        ) + tolerance:
            stability_failures.append("non_target_factor_kl_exceeded:%s" % scenario)
        if drift["maximum_absolute_value_drift"] > float(
            config["stability_design"]["maximum_absolute_value_drift"]
        ) + tolerance:
            stability_failures.append("value_prediction_drift_exceeded:%s" % scenario)

    canonical_probes = [rows[0] for rows in grouped.values()]
    full_probe = grouped["full-corpus"][0]
    if full_probe["all_accepted_mean_margin_improvement"] <= float(
        config["stability_design"]["minimum_full_mean_margin_improvement"]
    ) + tolerance:
        stability_failures.append("full_corpus_margin_not_improved")
    if (
        full_probe["all_accepted_after"]["preference_accuracy"] + tolerance
        < full_probe["all_accepted_before"]["preference_accuracy"]
    ):
        stability_failures.append("full_corpus_preference_accuracy_decreased")

    fold_probes = [grouped["fold-%d" % index][0] for index in range(len(folds))]
    fold_holdout_improvements = [
        float(probe["evaluation_mean_margin_improvement"]) for probe in fold_probes
    ]
    nonnegative_holdout_folds = sum(
        value + tolerance >= 0.0 for value in fold_holdout_improvements
    )
    mean_holdout_improvement = _mean(fold_holdout_improvements)
    if mean_holdout_improvement + tolerance < float(
        config["stability_design"]["minimum_fold_mean_holdout_margin_improvement"]
    ):
        stability_failures.append("mean_holdout_margin_decreased")
    if nonnegative_holdout_folds < int(
        config["stability_design"]["minimum_nonnegative_holdout_folds"]
    ):
        stability_failures.append("too_few_nonnegative_holdout_folds")

    stability_failures = sorted(set(stability_failures))
    next_scope_authorized = not integrity_failures and not stability_failures
    summary = {
        "full_corpus": {
            "mean_margin_before": full_probe["all_accepted_before"][
                "mean_signed_margin"
            ],
            "mean_margin_after": full_probe["all_accepted_after"][
                "mean_signed_margin"
            ],
            "mean_margin_improvement": full_probe[
                "all_accepted_mean_margin_improvement"
            ],
            "preference_accuracy_before": full_probe["all_accepted_before"][
                "preference_accuracy"
            ],
            "preference_accuracy_after": full_probe["all_accepted_after"][
                "preference_accuracy"
            ],
            "post_policy_sha256": full_probe["policy_sha256_after"],
            "drift": full_probe["full_support_drift"],
        },
        "folds": {
            "holdout_mean_margin_improvements": fold_holdout_improvements,
            "mean_holdout_margin_improvement": mean_holdout_improvement,
            "nonnegative_holdout_folds": nonnegative_holdout_folds,
            "fold_count": len(folds),
        },
        "all_scenarios": {
            "maximum_joint_kl": max(
                probe["full_support_drift"]["maximum_joint_kl"]
                for probe in canonical_probes
            ),
            "maximum_non_target_factor_kl": max(
                probe["full_support_drift"]["maximum_non_target_factor_kl"]
                for probe in canonical_probes
            ),
            "maximum_absolute_value_drift": max(
                probe["full_support_drift"]["maximum_absolute_value_drift"]
                for probe in canonical_probes
            ),
            "exact_replica_groups": sum(
                len({row["replica_signature"] for row in rows}) == 1
                for rows in grouped.values()
            ),
            "scenario_count": len(grouped),
        },
    }
    report = {
        "schema_version": 1,
        "protocol_id": PAIRWISE_STABILITY_PROTOCOL_ID,
        "execution_status": "passed" if not integrity_failures else "failed",
        "module_decision": (
            "go_single_iteration_integration_smoke"
            if next_scope_authorized
            else "no_go_single_iteration_integration_smoke"
        ),
        "classification": {
            "scope": "pairwise-optimizer-stability-development-only",
            "integrity_failures": integrity_failures,
            "stability_failures": stability_failures,
            "next_scope_authorized": next_scope_authorized,
            "formal_training_authorized": False,
            "multi_iteration_ppo_authorized": False,
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
        "inputs": {
            "prior_source_commit": prior_report["source"]["commit"],
            "rollout_digest": observed_rollout_digest,
            "candidate_count": len(candidate_map),
            "accepted_pair_count": len(accepted),
            "positive_pair_count": len(positive),
            "negative_pair_count": len(negative),
            "labels_reused_for_pairwise_development": True,
        },
        "reconstruction": {
            "evaluator_delta": asdict(reconstruction_delta),
            "fresh_counterfactual_labels_generated": False,
            "external_api_invoked": False,
            "ppo_optimizer_steps": 0,
        },
        "pairwise_probes": {
            "independent_one_step_update_count": len(probes),
            "sequential_training_iterations": 0,
            "policy_restored_between_probes": True,
            "pairwise_labels_used": True,
            "oracle_ledger": training_ledger,
            "summary": summary,
        },
        "final_policy_sha256": training_engine.policy_state_sha256,
        "initial_policy_sha256": inputs["expected_policy_sha256"],
        "final_policy_restored": training_engine.policy_state_sha256
        == inputs["expected_policy_sha256"],
        "sealed_test_accessed": False,
        "historical_gate_1b3_relabelled": False,
        "elapsed_seconds": time.perf_counter() - started,
        "resources": {
            "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            / 1024.0,
        },
    }
    write_json(output_dir / "stability-report.json", report)
    print(
        json.dumps(
            {
                "execution_status": report["execution_status"],
                "module_decision": report["module_decision"],
                "integrity_failures": integrity_failures,
                "stability_failures": stability_failures,
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    if integrity_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
