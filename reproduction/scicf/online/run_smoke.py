#!/usr/bin/env python
"""Run one governed, claim-ineligible online SciCF-PPO architecture smoke."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

from reproduction.framework.io import git_identity, write_json
from reproduction.p2.budget import RequestedCallBudgetManager, evaluator_ledger_delta
from reproduction.p2.contracts import (
    CREDIT_ESTIMATOR_CONTRACT_ID,
    METHOD_EVALUATOR_SOURCES,
    SCICF_PPO,
    SCICF_PPO_ON_POLICY,
    PPO_ENGINE_CONTRACT_ID,
    MethodRunContract,
    canonical_sha256,
)
from reproduction.p2.engine import PPOEngine, PPOEngineConfig, rollout_digest
from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    sha256_path,
    verify_accepted_binding,
)
from reproduction.scicf.llm.api_client import APISettings, OpenAICompatibleClient

from .contracts import ONLINE_PROTOCOL_ID, PairwiseRefinementConfig
from .pipeline import (
    FrozenPolicySampler,
    SciCFGAECreditEstimator,
    build_online_candidate_pool,
    refine_verified_pairs,
    verify_selected_candidates,
)
from .prompt import build_online_acquisition_request, validate_online_ranked_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--polybert-path", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _extract_json(text: str) -> Any:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(stripped[index:])
            return value
        except ValueError:
            continue
    raise ValueError("model output contains no valid JSON object")


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
        "pairwise_refinement",
        "ppo",
        "claim_boundary",
    }
    if set(payload) != required:
        raise ValueError(
            "online smoke config keys differ; missing=%s extra=%s"
            % (sorted(required - set(payload)), sorted(set(payload) - required))
        )
    if payload["schema_version"] != 1 or payload["protocol_id"] != ONLINE_PROTOCOL_ID:
        raise ValueError("unsupported online smoke protocol")
    if payload["classification"] != "engineering_architecture_smoke_only":
        raise ValueError("online smoke classification cannot authorize a scientific run")
    if payload["claim_boundary"].get("scientific_claim_authorized") is not False:
        raise ValueError("scientific claims must be disabled")
    if payload["claim_boundary"].get("sealed_test_access_authorized") is not False:
        raise ValueError("sealed test access must be disabled")
    if payload["claim_boundary"].get("baseline_mutation_authorized") is not False:
        raise ValueError("baseline mutation must be disabled")
    if payload["acquisition"].get("external_api_authorized") is not True:
        raise ValueError("external API use was not declared")
    if int(payload["acquisition"]["request_count"]) != 1:
        raise ValueError("architecture smoke permits exactly one LLM request")
    if int(payload["verification"]["replicates"]) < 1:
        raise ValueError("verification replicates must be positive")
    PPOEngineConfig(**payload["ppo"])
    PairwiseRefinementConfig(**payload["pairwise_refinement"])
    return payload


def build_runtime(
    root: Path,
    accepted_manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    config: Mapping[str, Any],
    polybert_path: Path,
):
    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.factory import build_stage0_components
    from RL_PPO.envs.gymnasium_wrapper import DAPiGenGymnasiumEnv

    ppo_config = PPOEngineConfig(**config["ppo"])
    allowed_sources = tuple(METHOD_EVALUATOR_SOURCES[SCICF_PPO])
    components = build_stage0_components(
        dapigen_root=str(root),
        polybert_path=str(polybert_path),
        config=DAPiGenEnvConfig.from_mapping(binding["accepted_task_config"]),
        device=ppo_config.device,
        maximum_requested_calls=int(config["budget"]["maximum_requested_calls"]),
        maximum_unique_calls=int(config["budget"]["maximum_unique_calls"]),
        encoder_mode="polybert",
        evaluator_mode="persistent",
        evaluator_fail_fast=True,
        allowed_evaluator_sources=allowed_sources,
        cache_scope=str(config["budget"]["cache_scope"]),
        allow_rdkit_brics_fallback=False,
    )
    specification = components.core.specification()
    expected = config["accepted_binding"]
    observed = {
        "environment_id": specification["environment_id"],
        "observation_dimension": specification["observation_dimension"],
        "evaluator_version": components.evaluator.evaluator_version,
        "objective_contract": components.evaluator.objective_contract,
    }
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in observed
        if observed[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError("accepted Stage-0 binding mismatch: %s" % mismatches)
    environment = DAPiGenGymnasiumEnv(
        components.core,
        components.reward_adapter,
        source=SCICF_PPO_ON_POLICY,
        seed=int(config["base_seed"]),
        include_ledger_in_step_info=False,
    )
    budget_contract_id = canonical_sha256(
        {
            "protocol": "terminal-requested-unique-backend-v2",
            "maximum_requested_calls": config["budget"]["maximum_requested_calls"],
            "maximum_unique_calls": config["budget"]["maximum_unique_calls"],
            "allowed_sources": sorted(allowed_sources),
            "cache_scope": config["budget"]["cache_scope"],
        }
    )
    run_contract = MethodRunContract(
        method=SCICF_PPO,
        environment_id=specification["environment_id"],
        task_contract_id=accepted_manifest["task_contract_id"],
        budget_contract_id=budget_contract_id,
        evaluator_version=components.evaluator.evaluator_version,
        objective_contract=components.evaluator.objective_contract,
        ppo_engine_contract_id=PPO_ENGINE_CONTRACT_ID,
        ppo_hyperparameters_sha256=ppo_config.sha256,
        credit_estimator_contract_id=CREDIT_ESTIMATOR_CONTRACT_ID,
        allowed_evaluator_sources=allowed_sources,
    )
    engine = PPOEngine(
        environment=environment,
        run_contract=run_contract,
        credit_estimator=SciCFGAECreditEstimator(),
        observation_dimension=int(specification["observation_dimension"]),
        number_of_dianhydride_actions=int(
            specification["number_of_dianhydride_actions"]
        ),
        number_of_diamine_actions=int(specification["number_of_diamine_actions"]),
        config=ppo_config,
    )
    return components, engine, specification, run_contract


def iteration_summary(result) -> Dict[str, Any]:
    return {
        "batch_id": result.rollout.batch_id,
        "rollout_digest": rollout_digest(result.rollout),
        "transition_count": len(result.rollout.transitions),
        "successful_terminal_count": sum(
            int("terminal_evaluation" in dict(item.info))
            for item in result.rollout.transitions
        ),
        "gae_sha256": result.rollout.gae_sha256,
        "critic_returns_sha256": result.rollout.critic_returns_sha256,
        "actor_advantages_sha256": result.credit.actor_advantages_sha256,
        "rollout_evaluator_delta": asdict(result.rollout_evaluator_delta),
        "credit_evaluator_delta": asdict(result.credit.evaluator_delta),
        "update_receipt": asdict(result.receipt),
        "update_metrics": dict(result.update_metrics),
    }


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
        raise RuntimeError("online SciCF architecture smoke requires Slurm")
    if socket.gethostname() != config["required_host"]:
        raise RuntimeError("online smoke is bound to %s" % config["required_host"])
    source = git_identity(root)
    if source.get("dirty") is not False:
        raise RuntimeError("online smoke requires a clean Git worktree")
    accepted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = verify_accepted_binding(
        root,
        accepted_manifest,
        config["accepted_binding"]["environment_id"],
    )
    if binding["stage0_source_changed_from_accepted"]:
        raise RuntimeError("accepted Stage-0 source changed before online smoke")
    settings = APISettings.from_private_file(args.credentials_file.resolve())
    provider = settings.public_identity()
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "status": "declared-before-first-api-request",
            "protocol_id": ONLINE_PROTOCOL_ID,
            "classification": config["classification"],
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "config_sha256": sha256_path(config_path),
            "accepted_manifest_sha256": sha256_path(manifest_path),
            "provider": provider,
            "external_api_request_limit": 1,
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
    behavior_policy = FrozenPolicySampler(engine.model, config["ppo"]["device"])
    initial_policy_sha256 = engine.policy_state_sha256
    ppo_result = engine.run_iteration(query_requested_calls=0)
    ppo_policy_sha256 = engine.policy_state_sha256
    if behavior_policy.state_sha256 != ppo_result.rollout.frozen_policy.state_sha256:
        raise RuntimeError("saved behavior policy differs from the rollout policy")
    candidates, trajectory_context = build_online_candidate_pool(
        rollout=ppo_result.rollout,
        core=components.core,
        behavior_policy=behavior_policy,
        pool_size=int(config["acquisition"]["pool_size"]),
        seed=int(config["base_seed"]),
    )
    request_id = "%s-job-%s" % (ONLINE_PROTOCOL_ID, os.environ["SLURM_JOB_ID"])
    request = build_online_acquisition_request(
        request_id=request_id,
        trajectory_context=trajectory_context,
        candidates=candidates,
        maximum_budget=int(config["acquisition"]["maximum_selected"]),
    )
    write_json(output_dir / "candidate-pool-audit.json", [item.audit_row() for item in candidates])
    write_json(output_dir / "llm-request.json", request)

    completion = OpenAICompatibleClient(settings).complete(
        messages=request["messages"],
        max_tokens=int(config["acquisition"]["max_output_tokens"]),
        seed=int(config["base_seed"]),
        timeout_seconds=float(config["acquisition"]["timeout_seconds"]),
        transport_retries=int(config["acquisition"]["transport_retries"]),
    )
    parsed = _extract_json(completion.content)
    ranked = validate_online_ranked_response(
        parsed,
        request["candidate_ids"],
        int(config["acquisition"]["maximum_selected"]),
    )
    api_record = {
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
    write_json(output_dir / "llm-response.json", api_record)

    candidate_map = {item.candidate_id: item for item in candidates}
    selected_ids = ranked["selected_intervention_ids"]
    query_budget = RequestedCallBudgetManager(engine.environment.oracle_ledger)
    maximum_verification_calls = (
        2 * len(selected_ids) * int(config["verification"]["replicates"])
    )
    reservation = query_budget.reserve(maximum_verification_calls)
    ledger_before = dict(engine.environment.oracle_ledger())
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
        ledger_after = dict(engine.environment.oracle_ledger())
        verification_delta = evaluator_ledger_delta(ledger_before, ledger_after)
        query_budget.reconcile(reservation, verification_delta.requested_calls)
    except Exception:
        query_budget.release(reservation)
        raise
    write_json(
        output_dir / "oracle-verification.json",
        {
            "selected_ids": selected_ids,
            "unselected_candidates_verified": False,
            "replicates": int(config["verification"]["replicates"]),
            "ledger_delta": asdict(verification_delta),
            "records": [item.to_dict() for item in verifications],
        },
    )
    refinement = refine_verified_pairs(
        engine=engine,
        candidates_by_id=candidate_map,
        verifications=verifications,
        config=PairwiseRefinementConfig(**config["pairwise_refinement"]),
    )
    checkpoint_path = output_dir / "scicf-online-smoke-checkpoint.pt"
    engine.save_checkpoint(checkpoint_path)
    checkpoint_sha256 = sha256_path(checkpoint_path)
    final_ledger = dict(engine.environment.oracle_ledger())

    accepted_pairs = sum(int(item.accepted) for item in verifications)
    failures = []
    if ranked["abstain"]:
        failures.append("llm_abstained_before_verification")
    if len(selected_ids) > int(config["acquisition"]["maximum_selected"]):
        failures.append("llm_selection_exceeded_budget")
    if accepted_pairs < int(config["verification"]["minimum_verified_pairs"]):
        failures.append("insufficient_verified_pairs")
    if refinement["status"] != "applied" or refinement["optimizer_steps"] != 1:
        failures.append("pairwise_refinement_not_applied_once")
    observed_sources = set(verification_delta.requested_by_source)
    allowed_query_sources = {"scicf_ppo/factual", "scicf_ppo/counterfactual"}
    if not observed_sources.issubset(allowed_query_sources):
        failures.append("unexpected_verification_evaluator_source")
    if ppo_result.credit.actor_advantages_sha256 != ppo_result.rollout.gae_sha256:
        failures.append("ppo_actor_credit_not_exact_gae")
    if ppo_result.credit.evaluator_delta.requested_calls != 0:
        failures.append("ppo_credit_phase_queried_evaluator")
    if initial_policy_sha256 == ppo_policy_sha256:
        failures.append("ppo_policy_did_not_update")
    if refinement["status"] == "applied" and ppo_policy_sha256 == engine.policy_state_sha256:
        failures.append("pairwise_policy_did_not_update")

    report = {
        "schema_version": 1,
        "protocol_id": ONLINE_PROTOCOL_ID,
        "status": "passed" if not failures else "failed",
        "classification": {
            "scope": "single-iteration-online-architecture-smoke",
            "failures": failures,
            "algorithm_effectiveness_evaluated": False,
            "scientific_claim_authorized": False,
            "formal_training_admission": False,
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
        "ppo_iteration": iteration_summary(ppo_result),
        "acquisition": {
            "request_count": 1,
            "pool_size": len(candidates),
            "candidate_timesteps": sorted({item.timestep for item in candidates}),
            "selected_ids": selected_ids,
            "selected_count": len(selected_ids),
            "abstained": ranked["abstain"],
            "prompt_sha256": request["prompt_sha256"],
            "provider": provider,
            "reported_tokens": {
                "prompt": completion.prompt_tokens,
                "completion": completion.completion_tokens,
                "total": completion.total_tokens,
            },
        },
        "verification": {
            "selected_only": True,
            "verified_count": len(verifications),
            "accepted_pair_count": accepted_pairs,
            "ledger_delta": asdict(verification_delta),
        },
        "pairwise_refinement": refinement,
        "policy_hashes": {
            "initial": initial_policy_sha256,
            "behavior": behavior_policy.state_sha256,
            "after_standard_ppo": ppo_policy_sha256,
            "after_pairwise": engine.policy_state_sha256,
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
    write_json(output_dir / "smoke-report.json", report)
    print(json.dumps({"status": report["status"], "failures": failures, "output": str(output_dir)}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
