#!/usr/bin/env python
"""Run the development-only SciCF acquisition and verifier hardening gate."""

from __future__ import annotations

import argparse
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
from reproduction.p2.engine import PPOEngineConfig, rollout_digest
from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    sha256_path,
    verify_accepted_binding,
)
from reproduction.scicf.acquisition.chemistry import morgan_distance
from reproduction.scicf.acquisition.metrics import acquisition_metrics_with_abstention
from reproduction.scicf.llm.api_client import APISettings, OpenAICompatibleClient

from .contracts import ACQUISITION_VERIFIER_PROTOCOL_ID, OnlineCandidate
from .pipeline import (
    FrozenPolicySampler,
    build_online_candidate_pool,
    eligible_online_episode_ids,
    verify_selected_candidates,
)
from .prompt import build_online_acquisition_request, validate_online_ranked_response
from .run_smoke import _extract_json, build_runtime


STRATEGIES = ("deepseek_llm", "random", "policy_probability", "chemistry_heuristic")


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
        "accepted_binding",
        "budget",
        "acquisition",
        "verification",
        "ppo",
        "decision_gate",
        "claim_boundary",
    }
    if set(payload) != required:
        raise ValueError(
            "acquisition/verifier config keys differ; missing=%s extra=%s"
            % (sorted(required - set(payload)), sorted(set(payload) - required))
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported acquisition/verifier config schema")
    if payload["protocol_id"] != ACQUISITION_VERIFIER_PROTOCOL_ID:
        raise ValueError("unsupported acquisition/verifier protocol")
    if payload["classification"] != "development_module_hardening_only":
        raise ValueError("protocol classification cannot authorize training or claims")
    acquisition = payload["acquisition"]
    verification = payload["verification"]
    if acquisition.get("external_api_authorized") is not True:
        raise ValueError("external API use was not declared")
    if tuple(acquisition.get("strategies", ())) != STRATEGIES:
        raise ValueError("matched acquisition strategies differ from the frozen set")
    if int(acquisition["request_count"]) != int(acquisition["pool_count"]):
        raise ValueError("one LLM request is required per fixed candidate pool")
    if int(acquisition["pool_count"]) != 4:
        raise ValueError("v1 requires exactly four development pools")
    if int(acquisition["pool_size"]) != 24 or int(acquisition["maximum_selected"]) != 4:
        raise ValueError("v1 freezes 24 candidates and maximum B=4")
    if int(verification["replicates"]) != 2:
        raise ValueError("v1 freezes paired continuation replication at K=2")
    if verification.get("candidate_scope") != "all_candidates_for_evaluation_only":
        raise ValueError("v1 requires exhaustive development-pool verification")
    for name in (
        "scientific_claim_authorized",
        "formal_training_authorized",
        "sealed_test_access_authorized",
        "baseline_mutation_authorized",
        "pairwise_update_authorized",
    ):
        if payload["claim_boundary"].get(name) is not False:
            raise ValueError("claim boundary must disable %s" % name)
    PPOEngineConfig(**payload["ppo"])
    return payload


def _pool_id(candidates: Sequence[OnlineCandidate]) -> str:
    payload = "\n".join(sorted(item.candidate_id for item in candidates))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rank(scores: Mapping[str, float], budget: int) -> Sequence[str]:
    return tuple(
        sorted(scores, key=lambda identifier: (float(scores[identifier]), identifier), reverse=True)[
            : int(budget)
        ]
    )


def _chemistry_score(candidate: OnlineCandidate) -> float:
    factual = candidate.factual_block.get("canonical_smiles")
    alternative = candidate.alternative_block.get("canonical_smiles")
    if factual is None or alternative is None:
        # NOOP is legal but is not a chemical fragment whose fingerprint can be
        # compared; keep it behind every chemically defined alternative.
        return -1.0
    return float(morgan_distance(str(factual), str(alternative)))


def matched_selections(
    *,
    candidates: Sequence[OnlineCandidate],
    llm_selected: Sequence[str],
    budget: int,
    seed: int,
) -> Mapping[str, Mapping[str, Any]]:
    candidate_ids = tuple(item.candidate_id for item in candidates)
    random_ids = list(candidate_ids)
    random.Random(int(seed)).shuffle(random_ids)
    policy_scores = {
        item.candidate_id: float(item.behavior_policy_probability) for item in candidates
    }
    chemistry_scores = {item.candidate_id: _chemistry_score(item) for item in candidates}
    rows = {
        "deepseek_llm": {
            "selected_ids": list(llm_selected),
            "ranking_scores": None,
            "score_semantics": "provider_rank_only; reasoning_and_confidence_advisory",
        },
        "random": {
            "selected_ids": random_ids[: int(budget)],
            "ranking_scores": None,
            "score_semantics": "deterministic_seeded_permutation",
        },
        "policy_probability": {
            "selected_ids": list(_rank(policy_scores, budget)),
            "ranking_scores": policy_scores,
            "score_semantics": "frozen_behavior_policy_component_probability",
        },
        "chemistry_heuristic": {
            "selected_ids": list(_rank(chemistry_scores, budget)),
            "ranking_scores": chemistry_scores,
            "score_semantics": "morgan_tanimoto_distance; noop=-1",
        },
    }
    for name, row in rows.items():
        selected = row["selected_ids"]
        if len(selected) > int(budget) or len(selected) != len(set(selected)):
            raise RuntimeError("%s selection violates the matched maximum budget" % name)
        if not set(selected).issubset(candidate_ids):
            raise RuntimeError("%s selected outside the fixed candidate pool" % name)
    return rows


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
        raise RuntimeError("acquisition/verifier development requires Slurm")
    if socket.gethostname() != config["required_host"]:
        raise RuntimeError("development protocol is bound to %s" % config["required_host"])
    source = git_identity(root)
    if source.get("dirty") is not False:
        raise RuntimeError("development protocol requires a clean Git worktree")
    accepted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = verify_accepted_binding(
        root, accepted_manifest, config["accepted_binding"]["environment_id"]
    )
    if binding["stage0_source_changed_from_accepted"]:
        raise RuntimeError("accepted Stage-0 source changed before module hardening")
    settings = APISettings.from_private_file(args.credentials_file.resolve())
    provider = settings.public_identity()
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "status": "declared-before-first-api-request-and-before-development-labels",
            "protocol_id": ACQUISITION_VERIFIER_PROTOCOL_ID,
            "classification": config["classification"],
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "config_sha256": sha256_path(config_path),
            "accepted_manifest_sha256": sha256_path(manifest_path),
            "provider": provider,
            "external_api_request_limit": int(config["acquisition"]["request_count"]),
            "credential_file_path_logged": False,
            "api_key_logged": False,
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
    ledger_before_rollout = dict(engine.environment.oracle_ledger())
    frozen = engine.freeze_policy()
    rollout = engine.collect_rollout(frozen)
    ledger_after_rollout = dict(engine.environment.oracle_ledger())
    rollout_delta = evaluator_ledger_delta(ledger_before_rollout, ledger_after_rollout)
    if frozen.state_sha256 != behavior_policy.state_sha256:
        raise RuntimeError("frozen rollout and continuation policies differ")

    eligible_ids = list(eligible_online_episode_ids(rollout, prefer_successful=False))
    selection_rng = random.Random(int(config["base_seed"]))
    selection_rng.shuffle(eligible_ids)
    pool_count = int(config["acquisition"]["pool_count"])
    if len(eligible_ids) < pool_count:
        raise RuntimeError("rollout lacks four complete cross-timestep episodes")
    chosen_episode_ids = tuple(eligible_ids[:pool_count])
    all_candidates = []
    pool_records = []
    api_records = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    token_accounting_complete = True
    client = OpenAICompatibleClient(settings)

    for pool_index, episode_id in enumerate(chosen_episode_ids):
        pool_seed = int(config["base_seed"]) + 1009 * pool_index
        candidates, trajectory_context = build_online_candidate_pool(
            rollout=rollout,
            core=components.core,
            behavior_policy=behavior_policy,
            pool_size=int(config["acquisition"]["pool_size"]),
            seed=pool_seed,
            episode_id=episode_id,
        )
        pool_identifier = _pool_id(candidates)
        request_id = "%s-pool-%02d-job-%s" % (
            ACQUISITION_VERIFIER_PROTOCOL_ID,
            pool_index,
            os.environ["SLURM_JOB_ID"],
        )
        request = build_online_acquisition_request(
            request_id=request_id,
            trajectory_context=trajectory_context,
            candidates=candidates,
            maximum_budget=int(config["acquisition"]["maximum_selected"]),
        )
        write_json(
            output_dir / "pools" / ("pool-%02d-candidates.json" % pool_index),
            {
                "pool_index": pool_index,
                "pool_id": pool_identifier,
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
            int(config["acquisition"]["maximum_selected"]),
        )
        api_record = {
            "pool_index": pool_index,
            "pool_id": pool_identifier,
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
        if completion.prompt_tokens is None or completion.completion_tokens is None:
            token_accounting_complete = False
        else:
            total_prompt_tokens += int(completion.prompt_tokens)
            total_completion_tokens += int(completion.completion_tokens)
        selections = matched_selections(
            candidates=candidates,
            llm_selected=ranked["selected_intervention_ids"],
            budget=int(config["acquisition"]["maximum_selected"]),
            seed=pool_seed,
        )
        pool_records.append(
            {
                "pool_index": pool_index,
                "pool_id": pool_identifier,
                "episode_id": episode_id,
                "candidate_ids": [item.candidate_id for item in candidates],
                "candidate_timesteps": sorted({int(item.timestep) for item in candidates}),
                "request": {
                    "request_id": request_id,
                    "prompt_sha256": request["prompt_sha256"],
                    "candidate_ids_match_pool": set(request["candidate_ids"])
                    == {item.candidate_id for item in candidates},
                    "presented_ids_match_pool": set(request["presented_candidate_ids"])
                    == {item.candidate_id for item in candidates},
                    "reward_truth_exposed": bool(
                        request["candidate_reward_truth_exposed"]
                        or request["factual_reward_truth_exposed"]
                    ),
                    "policy_score_exposed": bool(request["policy_score_exposed"]),
                },
                "llm_abstained": bool(ranked["abstain"]),
                "selections": selections,
            }
        )
        all_candidates.extend(candidates)

    candidate_map = {item.candidate_id: item for item in all_candidates}
    if len(candidate_map) != len(all_candidates):
        raise RuntimeError("candidate ids are not unique across development pools")
    all_candidate_ids = tuple(item.candidate_id for item in all_candidates)
    maximum_verification_calls = (
        2 * len(all_candidate_ids) * int(config["verification"]["replicates"])
    )
    query_budget = RequestedCallBudgetManager(engine.environment.oracle_ledger)
    reservation = query_budget.reserve(maximum_verification_calls)
    ledger_before_verification = dict(engine.environment.oracle_ledger())
    try:
        verifications = verify_selected_candidates(
            environment=engine.environment,
            policy=behavior_policy,
            candidates_by_id=candidate_map,
            selected_ids=all_candidate_ids,
            replicates=int(config["verification"]["replicates"]),
            seed=int(config["base_seed"]),
            delta_tolerance=float(config["decision_gate"]["delta_tolerance"]),
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
            "candidate_scope": "all_candidates_for_evaluation_only",
            "candidate_count": len(all_candidate_ids),
            "replicates": int(config["verification"]["replicates"]),
            "maximum_branch_count": maximum_verification_calls,
            "ledger_delta": asdict(verification_delta),
            "records": [item.to_dict() for item in verifications],
            "labels_used_for_training": False,
        },
    )

    verification_map = {item.candidate_id: item for item in verifications}
    evaluation_rows = []
    for pool in pool_records:
        gains = {
            identifier: verification_map[identifier].mean_delta
            for identifier in pool["candidate_ids"]
        }
        strategy_rows = {}
        for strategy in STRATEGIES:
            selected = pool["selections"][strategy]["selected_ids"]
            metrics = acquisition_metrics_with_abstention(
                gains, selected, int(config["acquisition"]["maximum_selected"])
            )
            accepted_selected = [
                identifier for identifier in selected if verification_map[identifier].accepted
            ]
            positive_stable = [
                identifier
                for identifier in accepted_selected
                if verification_map[identifier].mean_delta > 0.0
            ]
            strategy_rows[strategy] = {
                "selected_ids": list(selected),
                "metrics": metrics,
                "accepted_selected_count": len(accepted_selected),
                "stable_positive_selected_count": len(positive_stable),
                "stable_positive_hit_rate_at_b": len(positive_stable)
                / float(config["acquisition"]["maximum_selected"]),
            }
        evaluation_rows.append(
            {
                "pool_index": pool["pool_index"],
                "pool_id": pool["pool_id"],
                "verified_gains": gains,
                "oracle_gain_top4": list(
                    _rank(gains, int(config["acquisition"]["maximum_selected"]))
                ),
                "strategies": strategy_rows,
            }
        )
    write_json(output_dir / "matched-acquisition-evaluation.json", evaluation_rows)

    aggregate = {}
    aggregate_keys = (
        "budget_utilization",
        "hit_rate_at_b",
        "positive_precision_selected",
        "effective_best_gain_at_b",
        "regret_at_b",
        "ndcg_at_b",
    )
    for strategy in STRATEGIES:
        aggregate[strategy] = {
            key: _mean(
                [row["strategies"][strategy]["metrics"][key] for row in evaluation_rows]
            )
            for key in aggregate_keys
        }
        aggregate[strategy]["stable_positive_hit_rate_at_b"] = _mean(
            [
                row["strategies"][strategy]["stable_positive_hit_rate_at_b"]
                for row in evaluation_rows
            ]
        )

    llm_ndcg = [
        row["strategies"]["deepseek_llm"]["metrics"]["ndcg_at_b"]
        for row in evaluation_rows
    ]
    random_ndcg = [
        row["strategies"]["random"]["metrics"]["ndcg_at_b"]
        for row in evaluation_rows
    ]
    llm_best = [
        row["strategies"]["deepseek_llm"]["metrics"]["effective_best_gain_at_b"]
        for row in evaluation_rows
    ]
    random_best = [
        row["strategies"]["random"]["metrics"]["effective_best_gain_at_b"]
        for row in evaluation_rows
    ]
    tolerance = float(config["decision_gate"]["comparison_tolerance"])
    comparisons = {
        "mean_llm_minus_random_ndcg": _mean(llm_ndcg) - _mean(random_ndcg),
        "mean_llm_minus_random_effective_best_gain": _mean(llm_best)
        - _mean(random_best),
        "paired_ndcg_nonlosses": sum(
            llm + tolerance >= baseline
            for llm, baseline in zip(llm_ndcg, random_ndcg)
        ),
        "pool_count": len(evaluation_rows),
    }

    accepted = [item for item in verifications if item.accepted]
    positive = [item for item in accepted if item.mean_delta > 0.0]
    negative = [item for item in accepted if item.mean_delta < 0.0]
    integrity_failures = []
    if len(pool_records) != pool_count:
        integrity_failures.append("pool_count_mismatch")
    if any(len(pool["candidate_ids"]) != 24 for pool in pool_records):
        integrity_failures.append("candidate_pool_size_mismatch")
    if any(len(pool["candidate_timesteps"]) < 2 for pool in pool_records):
        integrity_failures.append("cross_timestep_pool_missing")
    if len(api_records) != int(config["acquisition"]["request_count"]):
        integrity_failures.append("api_request_count_mismatch")
    if any(
        not pool["request"]["candidate_ids_match_pool"]
        or not pool["request"]["presented_ids_match_pool"]
        for pool in pool_records
    ):
        integrity_failures.append("llm_candidate_pool_mismatch")
    if any(
        pool["request"]["reward_truth_exposed"]
        or pool["request"]["policy_score_exposed"]
        for pool in pool_records
    ):
        integrity_failures.append("llm_prompt_information_leakage")
    if len(verifications) != 24 * pool_count:
        integrity_failures.append("exhaustive_verification_count_mismatch")
    if set(verification_map) != set(all_candidate_ids):
        integrity_failures.append("verification_candidate_set_mismatch")
    if any(len(item.paired_seeds) != 2 for item in verifications):
        integrity_failures.append("matched_replicate_count_mismatch")
    allowed_sources = {"scicf_ppo/factual", "scicf_ppo/counterfactual"}
    if not set(verification_delta.requested_by_source).issubset(allowed_sources):
        integrity_failures.append("unexpected_verification_evaluator_source")
    if engine.policy_state_sha256 != initial_policy_sha256:
        integrity_failures.append("policy_changed_during_evaluation_only_protocol")

    verifier_failures = []
    if len(accepted) < int(config["decision_gate"]["minimum_accepted_pairs"]):
        verifier_failures.append("insufficient_sign_consistent_pairs")
    if len(positive) < int(config["decision_gate"]["minimum_positive_pairs"]):
        verifier_failures.append("insufficient_positive_pairs")
    if len(negative) < int(config["decision_gate"]["minimum_negative_pairs"]):
        verifier_failures.append("insufficient_negative_pairs")

    acquisition_failures = []
    if comparisons["paired_ndcg_nonlosses"] < int(
        config["decision_gate"]["minimum_paired_ndcg_nonlosses"]
    ):
        acquisition_failures.append("llm_paired_ndcg_nonlosses_below_threshold")
    if comparisons["mean_llm_minus_random_ndcg"] + tolerance < float(
        config["decision_gate"]["minimum_mean_llm_minus_random_ndcg"]
    ):
        acquisition_failures.append("llm_mean_ndcg_below_random")
    if comparisons["mean_llm_minus_random_effective_best_gain"] + tolerance < float(
        config["decision_gate"]["minimum_mean_llm_minus_random_effective_best_gain"]
    ):
        acquisition_failures.append("llm_mean_best_gain_below_random")

    verifier_ready = not integrity_failures and not verifier_failures
    acquisition_ready = not integrity_failures and not acquisition_failures
    next_module_authorized = bool(verifier_ready and acquisition_ready)
    final_ledger = dict(engine.environment.oracle_ledger())
    report = {
        "schema_version": 1,
        "protocol_id": ACQUISITION_VERIFIER_PROTOCOL_ID,
        "execution_status": "passed" if not integrity_failures else "failed",
        "module_decision": (
            "go_pairwise_stability_development"
            if next_module_authorized
            else "no_go_pairwise_stability_development"
        ),
        "classification": {
            "scope": "development-acquisition-and-verifier-module-hardening",
            "integrity_failures": integrity_failures,
            "verifier_failures": verifier_failures,
            "acquisition_failures": acquisition_failures,
            "verifier_corpus_ready": verifier_ready,
            "llm_acquisition_ready": acquisition_ready,
            "next_module_authorized": next_module_authorized,
            "algorithm_effectiveness_established": False,
            "scientific_claim_authorized": False,
            "formal_training_authorized": False,
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
        "rollout": {
            "batch_id": rollout.batch_id,
            "rollout_digest": rollout_digest(rollout),
            "transition_count": len(rollout.transitions),
            "successful_terminal_count": sum(
                int("terminal_evaluation" in dict(item.info))
                for item in rollout.transitions
            ),
            "eligible_episode_count": len(eligible_ids),
            "chosen_episode_ids": list(chosen_episode_ids),
            "outcome_blind_episode_selection": True,
            "evaluator_delta": asdict(rollout_delta),
            "optimizer_steps": 0,
        },
        "acquisition": {
            "strategies": list(STRATEGIES),
            "pool_count": pool_count,
            "pool_size": 24,
            "maximum_budget_per_strategy": int(config["acquisition"]["maximum_selected"]),
            "api_request_count": len(api_records),
            "provider": provider,
            "token_accounting_complete": token_accounting_complete,
            "reported_tokens": {
                "prompt": total_prompt_tokens if token_accounting_complete else None,
                "completion": total_completion_tokens if token_accounting_complete else None,
                "total": (
                    total_prompt_tokens + total_completion_tokens
                    if token_accounting_complete
                    else None
                ),
            },
            "aggregate_metrics": aggregate,
            "llm_vs_random": comparisons,
        },
        "verification": {
            "candidate_scope": "all_candidates_for_evaluation_only",
            "verified_candidate_count": len(verifications),
            "replicates": 2,
            "maximum_branch_count": maximum_verification_calls,
            "accepted_pair_count": len(accepted),
            "positive_pair_count": len(positive),
            "negative_pair_count": len(negative),
            "rejected_pair_count": len(verifications) - len(accepted),
            "ledger_delta": asdict(verification_delta),
            "labels_used_for_training": False,
        },
        "policy": {
            "initial_sha256": initial_policy_sha256,
            "behavior_sha256": behavior_policy.state_sha256,
            "final_sha256": engine.policy_state_sha256,
            "policy_updated": engine.policy_state_sha256 != initial_policy_sha256,
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
    write_json(output_dir / "module-report.json", report)
    print(
        json.dumps(
            {
                "execution_status": report["execution_status"],
                "module_decision": report["module_decision"],
                "integrity_failures": integrity_failures,
                "verifier_failures": verifier_failures,
                "acquisition_failures": acquisition_failures,
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    if integrity_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
