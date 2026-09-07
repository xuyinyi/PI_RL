#!/usr/bin/env python
"""No-credential Slurm preflight for K=5 soft pairs and LLM resilience."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from reproduction.framework.io import git_identity, write_json
from reproduction.scicf.online.contracts import (
    SOFT_PAIR_RESILIENCE_PROTOCOL_ID,
    OnlineVerification,
    PairwiseRefinementConfig,
)
from reproduction.scicf.online.resilience import (
    RECOVERABLE_LLM_FAILURE_CODES,
    LLMAuxiliaryCircuitBreaker,
    LLMAuxiliaryResilienceConfig,
    RecoverableLLMError,
    finalize_optional_auxiliary,
    run_optional_llm_acquisition,
)
from reproduction.scicf.online.soft_pair import (
    SoftPairAggregationConfig,
    aggregate_soft_verifications,
    soft_pair_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path, root: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "classification",
        "status",
        "source_evidence",
        "soft_pair_aggregation",
        "pairwise_refinement",
        "primary_transaction",
        "llm_resilience",
        "integrity_boundary",
        "future_live_budget_ceiling",
        "preflight",
        "authorization_state",
    }
    if set(payload) != required:
        raise ValueError("soft-pair protocol keys changed")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported soft-pair protocol schema")
    if payload["protocol_id"] != SOFT_PAIR_RESILIENCE_PROTOCOL_ID:
        raise ValueError("soft-pair protocol identity changed")
    if payload["classification"] != (
        "mock_only_soft_pair_and_optional_llm_resilience_development"
    ):
        raise ValueError("soft-pair protocol is not mock-only development")
    if payload["status"] != "frozen_for_no_credential_slurm_preflight":
        raise ValueError("soft-pair protocol status changed")
    evidence = payload["source_evidence"]
    report = (root / evidence["report"]).resolve(strict=True)
    if sha256_path(report) != evidence["report_sha256"]:
        raise RuntimeError("Job 4713 evidence hash mismatch")
    observed = json.loads(report.read_text(encoding="utf-8"))
    if observed["module_decision"] != evidence["decision"]:
        raise RuntimeError("Job 4713 decision binding changed")
    if observed["verification"]["accepted_pair_count"] != evidence[
        "accepted_pair_count"
    ]:
        raise RuntimeError("Job 4713 pair-count binding changed")
    SoftPairAggregationConfig(**payload["soft_pair_aggregation"])
    pairwise = PairwiseRefinementConfig(**payload["pairwise_refinement"])
    if pairwise.maximum_weight != payload["soft_pair_aggregation"][
        "maximum_training_weight"
    ]:
        raise ValueError("soft-pair maximum-weight contracts differ")
    if pairwise.delta_tolerance != payload["soft_pair_aggregation"][
        "practical_delta_tolerance"
    ]:
        raise ValueError("soft-pair delta-tolerance contracts differ")
    resilience = dict(payload["llm_resilience"])
    failure_codes = set(resilience.pop("recoverable_failure_codes"))
    if failure_codes != set(RECOVERABLE_LLM_FAILURE_CODES):
        raise ValueError("recoverable LLM failure-code set changed")
    LLMAuxiliaryResilienceConfig(**resilience)
    expected_primary = {
        "standard_ppo_is_primary": True,
        "checkpoint_before_llm": True,
        "llm_is_optional_auxiliary": True,
        "recoverable_llm_failure_cannot_rollback_ppo": True,
        "degraded_iteration_label": "ppo_only_degraded",
        "successful_auxiliary_label": "ppo_plus_scicf_soft_pair",
        "silent_fallback_forbidden": True,
    }
    if payload["primary_transaction"] != expected_primary:
        raise ValueError("primary PPO transaction boundary changed")
    expected_integrity = {
        "authorization_failure_is_fatal": True,
        "asset_binding_failure_is_fatal": True,
        "budget_overrun_is_fatal": True,
        "information_leakage_is_fatal": True,
        "unvalidated_llm_output_is_fatal": True,
        "unexpected_programming_error_is_fatal": True,
        "llm_confidence_may_weight_loss": False,
        "oracle_empirical_confidence_may_weight_loss": True,
        "counterfactual_actions_enter_ppo_clipping": False,
    }
    if payload["integrity_boundary"] != expected_integrity:
        raise ValueError("soft-pair integrity boundary changed")
    if payload["future_live_budget_ceiling"] != {
        "maximum_selected_candidates_per_iteration": 8,
        "maximum_matched_replicates_per_candidate": 5,
        "maximum_verification_branches_per_iteration": 80,
        "maximum_pairwise_optimizer_steps_per_iteration": 1,
    }:
        raise ValueError("future live budget ceiling changed")
    if payload["preflight"] != {
        "required_host": "yanlih100n1",
        "require_slurm": True,
        "require_clean_git": True,
        "credentials_permitted": False,
        "external_api_permitted": False,
        "ppo_permitted": False,
        "oracle_permitted": False,
        "local_model_permitted": False,
        "sealed_test_permitted": False,
    }:
        raise ValueError("mock preflight boundary changed")
    if any(value is not False for value in payload["authorization_state"].values()):
        raise ValueError("mock preflight cannot authorize live work")
    return payload


def verification(candidate_id: str, deltas: Sequence[float]) -> OnlineVerification:
    factual = tuple(0.5 for _ in deltas)
    counterfactual = tuple(0.5 + float(value) for value in deltas)
    strict_positive = all(float(value) > 0.005 for value in deltas)
    strict_negative = all(float(value) < -0.005 for value in deltas)
    accepted = strict_positive or strict_negative
    mean_delta = sum(deltas) / len(deltas)
    return OnlineVerification(
        candidate_id=candidate_id,
        paired_seeds=tuple(range(len(deltas))),
        factual_returns=factual,
        counterfactual_returns=counterfactual,
        deltas=tuple(float(value) for value in deltas),
        accepted=accepted,
        preferred=(
            "counterfactual"
            if accepted and mean_delta > 0.0
            else "factual" if accepted else None
        ),
        rejection_reason=None if accepted else "strict_rule_rejected",
        factual_terminal_records=tuple({} for _ in deltas),
        counterfactual_terminal_records=tuple({} for _ in deltas),
    )


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("preflight output already exists")
    config = load_config(config_path, root)
    preflight = config["preflight"]
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("soft-pair preflight must run through Slurm")
    if socket.gethostname() != preflight["required_host"]:
        raise RuntimeError("soft-pair preflight must run on n001")
    source = git_identity(root)
    if source["dirty"] is not False:
        raise RuntimeError("soft-pair preflight requires a clean worktree")
    output_dir.mkdir(parents=True)
    mock_checkpoint = output_dir / "mock-primary-ppo-checkpoint.bin"
    mock_checkpoint.write_bytes(b"mock-only-primary-ppo-checkpoint")
    checkpoint = sha256_path(mock_checkpoint)

    aggregation_config = SoftPairAggregationConfig(
        **config["soft_pair_aggregation"]
    )
    examples = aggregate_soft_verifications(
        (
            verification("cf-0000000000000001", (0.10, 0.08, 0.06, 0.04, 0.02)),
            verification("cf-0000000000000002", (0.08, 0.06, 0.04, 0.02, -0.01)),
            verification("cf-0000000000000003", (0.03, 0.02, 0.01, -0.01, -0.02)),
            verification("cf-0000000000000004", (0.04, 0.03, 0.02, 0.0, 0.0)),
            verification("cf-0000000000000005", (0.04, 0.03, -0.02, -0.01, 0.0)),
            verification("cf-0000000000000006", (0.0, 0.0, 0.0, 0.0, 0.0)),
        ),
        aggregation_config,
    )
    gate = soft_pair_gate(examples[:4], aggregation_config)
    zero_gate = soft_pair_gate(examples[4:], aggregation_config)

    resilience_payload = dict(config["llm_resilience"])
    resilience_payload.pop("recoverable_failure_codes")
    resilience_config = LLMAuxiliaryResilienceConfig(**resilience_payload)
    breaker = LLMAuxiliaryCircuitBreaker(resilience_config)
    timeout_receipt = run_optional_llm_acquisition(
        iteration=1,
        primary_ppo_checkpoint_path=mock_checkpoint,
        primary_ppo_checkpoint_sha256=checkpoint,
        breaker=breaker,
        acquire=lambda: (_ for _ in ()).throw(
            RecoverableLLMError("provider_timeout")
        ),
    )
    schema_receipt = run_optional_llm_acquisition(
        iteration=2,
        primary_ppo_checkpoint_path=mock_checkpoint,
        primary_ppo_checkpoint_sha256=checkpoint,
        breaker=breaker,
        acquire=lambda: (_ for _ in ()).throw(
            RecoverableLLMError("schema_exhausted")
        ),
    )
    circuit_receipt = run_optional_llm_acquisition(
        iteration=3,
        primary_ppo_checkpoint_path=mock_checkpoint,
        primary_ppo_checkpoint_sha256=checkpoint,
        breaker=breaker,
        acquire=lambda: (_ for _ in ()).throw(
            AssertionError("circuit-open acquisition must not execute")
        ),
    )
    recovered_receipt = run_optional_llm_acquisition(
        iteration=6,
        primary_ppo_checkpoint_path=mock_checkpoint,
        primary_ppo_checkpoint_sha256=checkpoint,
        breaker=breaker,
        acquire=lambda: {"status": "validated", "payload": {"ids": ["cf-a"]}},
    )
    no_mass = finalize_optional_auxiliary(
        recovered_receipt,
        {
            "status": "skipped_insufficient_soft_pair_mass",
            "continue_primary_training": True,
        },
    )

    checks: Dict[str, bool] = {
        "exact_k5": aggregation_config.replicates == 5,
        "five_votes_outweigh_four": examples[0].training_weight
        > examples[1].training_weight,
        "four_votes_outweigh_three": examples[1].training_weight
        > examples[2].training_weight,
        "three_of_five_is_low_nonzero_weight": 0.0
        < examples[2].training_weight
        < examples[1].training_weight,
        "positive_with_two_ties_retained": examples[3].training_weight > 0.0,
        "balanced_direction_abstains": examples[4].training_weight == 0.0,
        "all_ties_abstain": examples[5].training_weight == 0.0,
        "effective_mass_gate_passes": gate["eligible_for_optional_update"] is True,
        "no_mass_gate_skips": zero_gate["eligible_for_optional_update"] is False,
        "provider_timeout_keeps_training": timeout_receipt["continue_training"] is True,
        "schema_exhaustion_keeps_training": schema_receipt["continue_training"] is True,
        "circuit_opens": circuit_receipt["auxiliary_status"]
        == "skipped_circuit_open",
        "circuit_recovers": recovered_receipt["auxiliary_status"]
        == "ready_for_oracle_verification",
        "no_mass_is_explicit_degradation": no_mass["method_observed"]
        == "ppo_only_degraded",
        "silent_fallback_forbidden": no_mass["silent_fallback_used"] is False,
        "credentials_not_loaded": True,
        "external_api_not_invoked": True,
        "ppo_not_run": True,
        "oracle_not_run": True,
        "sealed_test_not_accessed": True,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "protocol_id": SOFT_PAIR_RESILIENCE_PROTOCOL_ID,
        "classification": config["classification"],
        "source": source,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "platform": platform.platform(),
        "config_sha256": sha256_path(config_path),
        "soft_pair_config": asdict(aggregation_config),
        "examples": [item.to_dict() for item in examples],
        "effective_mass_gate": gate,
        "zero_mass_gate": zero_gate,
        "resilience_receipts": {
            "timeout": timeout_receipt,
            "schema": schema_receipt,
            "circuit_open": circuit_receipt,
            "recovered": recovered_receipt,
            "no_mass": no_mass,
        },
        "checks": checks,
        "failed_checks": failures,
        "credential_file_argument_accepted": False,
        "credentials_loaded": False,
        "external_api_invoked": False,
        "ppo_iterations": 0,
        "oracle_calls": 0,
        "local_model_invoked": False,
        "sealed_test_accessed": False,
        "real_run_authorized": False,
        "multi_iteration_training_authorized": False,
        "decision": (
            "pass_components_only_no_real_run_authorized"
            if not failures
            else "fail_closed"
        ),
    }
    write_json(output_dir / "soft-pair-resilience-preflight-report.json", report)
    print(json.dumps({"decision": report["decision"], "failures": failures}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
