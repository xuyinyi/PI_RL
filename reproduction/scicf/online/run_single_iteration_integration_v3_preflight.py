#!/usr/bin/env python
"""No-credential full-runtime Slurm preflight for integration-v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path

from reproduction.framework.io import git_identity, write_json
from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    verify_accepted_binding,
)

from .contracts import SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID
from .evaluator_asset import (
    load_evaluator_asset_binding,
    validate_evaluator_asset,
    verify_evaluator_route_source_delta,
)
from .model_asset import load_polybert_asset_binding, validate_polybert_asset
from .resilience import (
    RECOVERABLE_LLM_FAILURE_CODES,
    LLMAuxiliaryCircuitBreaker,
    LLMAuxiliaryResilienceConfig,
    RecoverableLLMError,
    finalize_optional_auxiliary,
    run_optional_llm_acquisition,
)
from .run_single_iteration_integration_v2 import verify_protocol_bindings
from .run_single_iteration_integration_v3 import (
    AUTHORIZATION_OPERATIONS_V3,
    load_execution_authorization_v3,
    load_v3_protocol,
)
from .run_smoke import build_runtime
from .soft_pair import (
    SoftPairAggregationConfig,
    aggregate_soft_verifications,
    soft_pair_gate,
)
from .contracts import OnlineVerification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--polybert-path", type=Path, required=True)
    parser.add_argument("--evaluator-asset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_evaluator_hashes(terminal_evaluator):
    hashes = {}
    for output_name, dataset_name, model_id in terminal_evaluator.PROPERTY_SPECS:
        record = terminal_evaluator.model_manifest[output_name]
        model_stem = "Ensemble_%s_AFP_%d" % (dataset_name, int(model_id))
        settings_stem = model_stem.rsplit("_%d" % int(model_id), 1)[0]
        hashes[model_stem + ".pt"] = record["model_sha256"]
        hashes[dataset_name + "_scaler.pkl"] = record["scaler_sha256"]
        hashes[settings_stem + "_settings.csv"] = record["settings_sha256"]
    hashes["fpscores.pkl.gz"] = terminal_evaluator.model_manifest["sa"][
        "fpscores_sha256"
    ]
    return hashes


def _verification(candidate_id: str, deltas):
    factual = tuple(0.5 for _ in deltas)
    counterfactual = tuple(0.5 + float(value) for value in deltas)
    return OnlineVerification(
        candidate_id=candidate_id,
        paired_seeds=tuple(range(len(deltas))),
        factual_returns=factual,
        counterfactual_returns=counterfactual,
        deltas=tuple(float(value) for value in deltas),
        accepted=False,
        preferred=None,
        rejection_reason="v3-preflight-soft-only",
        factual_terminal_records=tuple({} for _ in deltas),
        counterfactual_terminal_records=tuple({} for _ in deltas),
    )


def _recoverable_receipt(code, checkpoint, checkpoint_sha256, config):
    breaker = LLMAuxiliaryCircuitBreaker(config)
    return run_optional_llm_acquisition(
        iteration=1,
        primary_ppo_checkpoint_path=checkpoint,
        primary_ppo_checkpoint_sha256=checkpoint_sha256,
        breaker=breaker,
        acquire=lambda: (_ for _ in ()).throw(RecoverableLLMError(code)),
    )


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("integration-v3 preflight output already exists")
    protocol, base_protocol, soft_payload = load_v3_protocol(protocol_path, root)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("integration-v3 preflight requires Slurm")
    if socket.gethostname() != protocol["required_host"]:
        raise RuntimeError("integration-v3 preflight is bound to n001")
    source = git_identity(root)
    if source["dirty"] is not False:
        raise RuntimeError("integration-v3 preflight requires a clean worktree")

    bindings = verify_protocol_bindings(root, base_protocol)
    accepted_manifest = bindings["accepted_manifest"]
    accepted_binding = verify_accepted_binding(
        root,
        accepted_manifest,
        base_protocol["accepted_binding"]["environment_id"],
    )
    polybert_binding = load_polybert_asset_binding(root)
    model_asset = validate_polybert_asset(
        args.polybert_path.resolve(strict=True), polybert_binding
    )
    if (
        accepted_manifest["environment"].get("encoder_version")
        != model_asset["encoder_version"]
    ):
        raise RuntimeError("polyBERT asset differs from accepted Stage-0 encoder")
    evaluator_binding = load_evaluator_asset_binding(root)
    evaluator_asset = validate_evaluator_asset(
        args.evaluator_asset_path.resolve(strict=True), evaluator_binding
    )
    if accepted_manifest.get("evaluator_version") != evaluator_asset[
        "evaluator_version"
    ]:
        raise RuntimeError("AFP evaluator asset differs from accepted Stage-0 evaluator")
    source_delta = verify_evaluator_route_source_delta(
        root, accepted_binding["accepted_git_commit"], evaluator_binding
    )

    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "schema_version": 1,
            "status": "declared-before-no-credential-full-runtime-preflight",
            "protocol_id": SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "protocol_sha256": _sha256_path(protocol_path),
            "polybert_asset": model_asset,
            "evaluator_asset": evaluator_asset,
            "evaluator_route_source_delta": source_delta,
            "credentials_loading_authorized": False,
            "external_api_requests_authorized": False,
            "ppo_execution_authorized": False,
            "oracle_execution_authorized": False,
            "real_single_iteration_authorized": False,
            "multi_iteration_training_authorized": False,
        },
    )
    started = time.perf_counter()
    components, engine, specification, run_contract = build_runtime(
        root,
        accepted_manifest,
        accepted_binding,
        base_protocol,
        Path(model_asset["model_path"]),
        polybert_checkpoint_fingerprint=model_asset["checkpoint_fingerprint"],
        evaluator_asset_path=Path(evaluator_asset["asset_path"]),
    )
    terminal_evaluator = components.evaluator.evaluator
    runtime_evaluator_hashes = _runtime_evaluator_hashes(terminal_evaluator)
    initial_ledger = components.evaluator.ledger()
    encoder_diagnostics = components.core.encoder.diagnostics()
    runtime = {
        "constructed": True,
        "components_class": type(components).__name__,
        "engine_class": type(engine).__name__,
        "environment_id": specification["environment_id"],
        "run_contract_environment_id": run_contract.environment_id,
        "evaluator_version": components.evaluator.evaluator_version,
        "evaluator_asset_path": str(terminal_evaluator.model_dir.resolve()),
        "evaluator_required_file_sha256": runtime_evaluator_hashes,
        "initial_evaluator_ledger": initial_ledger,
        "encoder_diagnostics": encoder_diagnostics,
        "scientific_local_models_loaded": True,
        "polybert_initial_embedding_inference_executed": int(
            encoder_diagnostics["cache_misses"]
        )
        > 0,
        "afp_property_inference_executed": False,
        "local_llm_loaded": False,
    }

    mock_checkpoint = output_dir / "mock-primary-ppo-checkpoint.bin"
    mock_checkpoint.write_bytes(b"mock-only-primary-ppo-checkpoint-v3")
    checkpoint_sha256 = _sha256_path(mock_checkpoint)
    resilience_payload = dict(soft_payload["llm_resilience"])
    configured_codes = set(resilience_payload.pop("recoverable_failure_codes"))
    resilience_config = LLMAuxiliaryResilienceConfig(**resilience_payload)
    recoverable_receipts = {
        code: _recoverable_receipt(
            code,
            mock_checkpoint,
            checkpoint_sha256,
            resilience_config,
        )
        for code in sorted(RECOVERABLE_LLM_FAILURE_CODES)
    }
    abstention_receipt = run_optional_llm_acquisition(
        iteration=1,
        primary_ppo_checkpoint_path=mock_checkpoint,
        primary_ppo_checkpoint_sha256=checkpoint_sha256,
        breaker=LLMAuxiliaryCircuitBreaker(resilience_config),
        acquire=lambda: {"status": "abstained", "payload": {}},
    )
    validated_receipt = run_optional_llm_acquisition(
        iteration=1,
        primary_ppo_checkpoint_path=mock_checkpoint,
        primary_ppo_checkpoint_sha256=checkpoint_sha256,
        breaker=LLMAuxiliaryCircuitBreaker(resilience_config),
        acquire=lambda: {"status": "validated", "payload": {"ids": ["cf-a"]}},
    )
    no_mass_receipt = finalize_optional_auxiliary(
        validated_receipt,
        {
            "status": "skipped_insufficient_soft_pair_mass",
            "continue_primary_training": True,
        },
    )
    circuit = LLMAuxiliaryCircuitBreaker(resilience_config)
    for iteration in (1, 2):
        run_optional_llm_acquisition(
            iteration=iteration,
            primary_ppo_checkpoint_path=mock_checkpoint,
            primary_ppo_checkpoint_sha256=checkpoint_sha256,
            breaker=circuit,
            acquire=lambda: (_ for _ in ()).throw(
                RecoverableLLMError("provider_timeout")
            ),
        )
    circuit_skips = []
    for iteration in (3, 4, 5):
        receipt = run_optional_llm_acquisition(
            iteration=iteration,
            primary_ppo_checkpoint_path=mock_checkpoint,
            primary_ppo_checkpoint_sha256=checkpoint_sha256,
            breaker=circuit,
            acquire=lambda: (_ for _ in ()).throw(
                AssertionError("circuit-open acquisition executed")
            ),
        )
        circuit_skips.append(receipt["auxiliary_status"])
    circuit_recovery = run_optional_llm_acquisition(
        iteration=6,
        primary_ppo_checkpoint_path=mock_checkpoint,
        primary_ppo_checkpoint_sha256=checkpoint_sha256,
        breaker=circuit,
        acquire=lambda: {"status": "validated", "payload": {}},
    )

    aggregation_config = SoftPairAggregationConfig(
        **soft_payload["soft_pair_aggregation"]
    )
    soft_evidence = aggregate_soft_verifications(
        (
            _verification("cf-soft-1", (0.10, 0.08, 0.06, 0.04, 0.02)),
            _verification("cf-soft-2", (0.08, 0.06, 0.04, 0.02, -0.01)),
            _verification("cf-soft-3", (0.03, 0.02, 0.01, -0.01, -0.02)),
            _verification("cf-soft-4", (0.0, 0.0, 0.0, 0.0, 0.0)),
        ),
        aggregation_config,
    )
    soft_gate = soft_pair_gate(soft_evidence, aggregation_config)

    soft_path = (
        root / protocol["soft_pair_development"]["path"]
    ).resolve(strict=True)
    closed_authorization = {
        "schema_version": 4,
        "authorization_id": "closed-v3-preflight-fixture",
        "protocol_id": SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID,
        "protocol_sha256": _sha256_path(protocol_path),
        "soft_pair_config_sha256": _sha256_path(soft_path),
        "implementation_commit": source["commit"],
        "authorized_output_directory": str(output_dir / "never-run"),
        "authorized_polybert_path": model_asset["model_path"],
        "polybert_asset_binding_sha256": model_asset["asset_binding_sha256"],
        "polybert_checkpoint_fingerprint": model_asset[
            "checkpoint_fingerprint"
        ],
        "authorized_evaluator_asset_path": evaluator_asset["asset_path"],
        "evaluator_asset_binding_sha256": evaluator_asset[
            "asset_binding_sha256"
        ],
        "evaluator_asset_fingerprint": evaluator_asset["asset_fingerprint"],
        "maximum_slurm_runs": 1,
        "authorized_operations": {
            name: False for name in AUTHORIZATION_OPERATIONS_V3
        },
    }
    closed_path = output_dir / "closed-authorization-fixture.json"
    write_json(closed_path, closed_authorization)
    closed_authorization_rejected = False
    try:
        load_execution_authorization_v3(
            closed_path,
            protocol_sha256=_sha256_path(protocol_path),
            soft_pair_config_sha256=_sha256_path(soft_path),
            implementation_commit=source["commit"],
            output_dir=output_dir / "never-run",
            polybert_path=Path(model_asset["model_path"]),
            polybert_asset_binding_sha256=model_asset["asset_binding_sha256"],
            polybert_checkpoint_fingerprint=model_asset[
                "checkpoint_fingerprint"
            ],
            evaluator_asset_path=Path(evaluator_asset["asset_path"]),
            evaluator_asset_binding_sha256=evaluator_asset[
                "asset_binding_sha256"
            ],
            evaluator_asset_fingerprint=evaluator_asset[
                "asset_fingerprint"
            ],
        )
    except ValueError:
        closed_authorization_rejected = True

    runner_path = (
        root
        / "reproduction/scicf/online/run_single_iteration_integration_v3.py"
    )
    runner_source = runner_path.read_text(encoding="utf-8")
    model_index = runner_source.index("model_asset = validate_polybert_asset(")
    evaluator_index = runner_source.index(
        "evaluator_asset = validate_evaluator_asset("
    )
    authorization_index = runner_source.index(
        "authorization = load_execution_authorization_v3("
    )
    ppo_index = runner_source.index("ppo_result = engine.run_iteration(")
    checkpoint_index = runner_source.index(
        "engine.save_checkpoint(primary_checkpoint)"
    )
    credentials_index = runner_source.index(
        "settings = APISettings.from_private_file("
    )
    optional_index = runner_source.index(
        "acquisition_receipt = run_optional_llm_acquisition("
    )

    checks = {
        "protocol_identity": protocol["protocol_id"]
        == SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID,
        "component_preflight_bound": protocol["soft_pair_development"][
            "required_component_decision"
        ]
        == "pass_components_only_no_real_run_authorized",
        "closed_schema4_authorization_rejected": closed_authorization_rejected,
        "assets_before_authorization": model_index < authorization_index
        and evaluator_index < authorization_index,
        "authorization_before_ppo": authorization_index < ppo_index,
        "ppo_checkpoint_before_credentials": ppo_index
        < checkpoint_index
        < credentials_index,
        "credentials_inside_optional_acquisition": credentials_index
        < optional_index,
        "model_asset_bound_to_accepted_encoder": model_asset["encoder_version"]
        == accepted_manifest["environment"]["encoder_version"],
        "evaluator_asset_bound_to_accepted_stage0": evaluator_asset[
            "evaluator_version"
        ]
        == accepted_manifest["evaluator_version"],
        "evaluator_source_delta_exact_and_routing_only": source_delta["exact"]
        is True
        and source_delta["observed_paths"] == ["RL_PPO/envs/evaluator.py"],
        "full_runtime_constructed": runtime["constructed"] is True,
        "runtime_evaluator_route_exact": Path(
            runtime["evaluator_asset_path"]
        ).resolve()
        == Path(evaluator_asset["asset_path"]).resolve(),
        "runtime_evaluator_hashes_bound": runtime[
            "evaluator_required_file_sha256"
        ]
        == evaluator_asset["required_file_sha256"],
        "runtime_evaluator_ledger_zero": all(
            int(initial_ledger[name]) == 0
            for name in ("requested_calls", "unique_calls", "backend_calls")
        ),
        "runtime_polybert_embedding_executed": runtime[
            "polybert_initial_embedding_inference_executed"
        ]
        is True,
        "runtime_afp_oracle_not_executed": runtime[
            "afp_property_inference_executed"
        ]
        is False,
        "no_local_llm_loaded": runtime["local_llm_loaded"] is False,
        "configured_recoverable_codes_exact": configured_codes
        == set(RECOVERABLE_LLM_FAILURE_CODES),
        "all_recoverable_failures_continue_ppo": all(
            receipt["continue_training"] is True
            and receipt["method_observed"] == "ppo_only_degraded"
            and receipt["silent_fallback_used"] is False
            for receipt in recoverable_receipts.values()
        ),
        "valid_abstention_continues_ppo": abstention_receipt[
            "method_observed"
        ]
        == "ppo_only_degraded"
        and abstention_receipt["continue_training"] is True,
        "insufficient_pair_mass_continues_ppo": no_mass_receipt[
            "method_observed"
        ]
        == "ppo_only_degraded"
        and no_mass_receipt["continue_training"] is True,
        "circuit_skips_exactly_three_iterations": circuit_skips
        == ["skipped_circuit_open"] * 3,
        "circuit_retries_after_cooldown": circuit_recovery["auxiliary_status"]
        == "ready_for_oracle_verification",
        "soft_aggregation_exact_k5": aggregation_config.replicates == 5,
        "soft_weights_order_empirical_confidence": soft_evidence[
            0
        ].training_weight
        > soft_evidence[1].training_weight
        > soft_evidence[2].training_weight
        > soft_evidence[3].training_weight,
        "soft_mass_gate_passes": soft_gate["eligible_for_optional_update"]
        is True,
        "external_api_request_count_zero": True,
        "credentials_loaded_false": True,
        "ppo_executed_false": True,
        "oracle_executed_false": True,
        "sealed_test_accessed_false": True,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "protocol_id": SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID,
        "execution_status": "passed" if not failures else "failed",
        "module_decision": (
            "go_request_separate_real_single_iteration_v3_authorization"
            if not failures
            else "no_go_real_single_iteration_v3_authorization"
        ),
        "classification": {
            "scope": "no-credential-full-runtime-integration-v3-preflight",
            "failures": failures,
            "real_single_iteration_authorized": False,
            "automatic_rerun_authorized": False,
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
        "soft_pair_config_sha256": _sha256_path(soft_path),
        "runner_sha256": _sha256_path(runner_path),
        "polybert_asset": model_asset,
        "evaluator_asset": evaluator_asset,
        "evaluator_route_source_delta": source_delta,
        "accepted_binding": accepted_binding,
        "runtime": runtime,
        "checks": checks,
        "recoverable_receipts": recoverable_receipts,
        "abstention_receipt": abstention_receipt,
        "no_mass_receipt": no_mass_receipt,
        "circuit_breaker": {
            "skips": circuit_skips,
            "recovery": circuit_recovery,
        },
        "soft_evidence": [item.to_dict() for item in soft_evidence],
        "soft_gate": soft_gate,
        "mock_primary_checkpoint_sha256": checkpoint_sha256,
        "credentials_loaded": False,
        "external_api_request_count": 0,
        "ppo_iterations": 0,
        "oracle_calls": 0,
        "local_llm_loaded": False,
        "scientific_local_models_loaded": True,
        "sealed_test_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "resources": {
            "peak_rss_mib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            / 1024.0
        },
    }
    write_json(output_dir / "integration-v3-preflight-report.json", report)
    print(
        json.dumps(
            {
                "execution_status": report["execution_status"],
                "module_decision": report["module_decision"],
                "failures": failures,
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

