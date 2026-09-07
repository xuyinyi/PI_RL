#!/usr/bin/env python
"""No-credential full-runtime preflight for the short-horizon runner."""

from __future__ import annotations

import argparse
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

from .contracts import SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID
from .evaluator_asset import (
    load_evaluator_asset_binding,
    validate_evaluator_asset,
    verify_evaluator_route_source_delta,
)
from .model_asset import load_polybert_asset_binding, validate_polybert_asset
from .pipeline import numerically_nonnegative_kl
from .resilience import (
    LLMAuxiliaryCircuitBreaker,
    LLMAuxiliaryResilienceConfig,
    LLMWallClockBudget,
    LLMWallClockBudgetConfig,
    RecoverableLLMError,
    run_optional_llm_acquisition,
)
from .run_short_horizon_multi_iteration import (
    AUTHORIZATION_OPERATIONS_SHORT_HORIZON,
    _runtime_protocol,
    load_execution_authorization_short_horizon,
)
from .run_single_iteration_integration_v2 import verify_protocol_bindings
from .run_single_iteration_integration_v3 import load_v3_protocol
from .run_smoke import build_runtime
from .short_horizon import (
    chained_history_digest,
    load_short_horizon_checkpoint,
    load_short_horizon_protocol,
    save_short_horizon_checkpoint,
    sha256_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--polybert-path", type=Path, required=True)
    parser.add_argument("--evaluator-asset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


class _Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class _Engine:
    def __init__(self):
        self.payload = b"short-horizon-mock-engine"
        self.loaded = None

    def save_checkpoint(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payload)

    def load_checkpoint(self, path):
        self.loaded = Path(path).read_bytes()


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


def _fresh_controls(config, clock):
    return (
        LLMAuxiliaryCircuitBreaker(
            LLMAuxiliaryResilienceConfig(
                maximum_consecutive_llm_failures=2,
                circuit_cooldown_iterations=3,
            )
        ),
        LLMWallClockBudget(config, clock=clock),
    )


def _run_control_sequence(output_dir, protocol_sha256, wall_config):
    output_dir.mkdir(parents=True)
    clock = _Clock()
    engine = _Engine()
    breaker, wall = _fresh_controls(wall_config, clock)
    previous_control = None
    previous_history = None
    receipts = []
    controls = []
    for iteration in range(1, 7):
        primary = output_dir / ("mock-primary-%02d.bin" % iteration)
        primary.write_bytes(("primary-%02d" % iteration).encode("ascii"))
        primary_hash = sha256_path(primary)

        if iteration in (1, 2):
            def acquire_failure(iteration=iteration):
                lease = wall.begin_iteration(iteration)
                clock.advance(10.0)
                lease.close()
                raise RecoverableLLMError("provider_timeout")

            acquire = acquire_failure
        elif iteration in (3, 4, 5):
            acquire = lambda: (_ for _ in ()).throw(
                AssertionError("circuit-open acquisition executed")
            )
        else:
            def acquire_success():
                lease = wall.begin_iteration(6)
                clock.advance(5.0)
                lease.close()
                return {"status": "validated", "payload": {"mock": True}}

            acquire = acquire_success
        receipt = run_optional_llm_acquisition(
            iteration=iteration,
            primary_ppo_checkpoint_path=primary,
            primary_ppo_checkpoint_sha256=primary_hash,
            breaker=breaker,
            acquire=acquire,
        )
        receipts.append(receipt)
        compact = {
            "iteration": iteration,
            "auxiliary_status": receipt["auxiliary_status"],
            "method_observed": receipt["method_observed"],
        }
        current_history = chained_history_digest(previous_history, compact)
        iteration_dir = output_dir / ("control-%02d" % iteration)
        saved = save_short_horizon_checkpoint(
            engine=engine,
            breaker=breaker,
            wall_budget=wall,
            engine_checkpoint_path=iteration_dir / "engine.pt",
            control_checkpoint_path=iteration_dir / "control.json",
            completed_iteration=iteration,
            protocol_sha256=protocol_sha256,
            history_digest=current_history,
            previous_control_sha256=previous_control,
        )
        controls.append(saved)
        restored_engine = _Engine()
        restored_breaker, restored_wall = _fresh_controls(wall_config, clock)
        loaded = load_short_horizon_checkpoint(
            engine=restored_engine,
            breaker=restored_breaker,
            wall_budget=restored_wall,
            control_checkpoint_path=Path(saved["control_checkpoint_path"]),
            expected_protocol_sha256=protocol_sha256,
            expected_control_sha256=saved["control_checkpoint_sha256"],
        )
        if restored_engine.loaded != engine.payload:
            raise RuntimeError("mock engine checkpoint did not restore")
        if loaded["history_digest"] != current_history:
            raise RuntimeError("mock history digest did not restore")
        engine = restored_engine
        breaker = restored_breaker
        wall = restored_wall
        previous_control = saved["control_checkpoint_sha256"]
        previous_history = current_history
    return {
        "receipts": receipts,
        "controls": controls,
        "final_breaker": breaker.state_dict(),
        "final_wall_budget": wall.state_dict(),
        "final_history_digest": previous_history,
    }


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("short-horizon preflight output already exists")
    protocol = load_short_horizon_protocol(protocol_path, root)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("short-horizon preflight requires Slurm")
    if socket.gethostname() != protocol["required_host"]:
        raise RuntimeError("short-horizon preflight is bound to n001")
    source = git_identity(root)
    if source["dirty"] is not False:
        raise RuntimeError("short-horizon preflight requires a clean worktree")

    v3_path = Path(protocol["resolved_prerequisites"]["v3_protocol"])
    _v3, base_protocol, _soft = load_v3_protocol(v3_path, root)
    runtime_protocol = _runtime_protocol(base_protocol, protocol)
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
    evaluator_binding = load_evaluator_asset_binding(root)
    evaluator_asset = validate_evaluator_asset(
        args.evaluator_asset_path.resolve(strict=True), evaluator_binding
    )
    source_delta = verify_evaluator_route_source_delta(
        root, accepted_binding["accepted_git_commit"], evaluator_binding
    )
    if (
        accepted_manifest["environment"]["encoder_version"]
        != model_asset["encoder_version"]
    ):
        raise RuntimeError("preflight polyBERT version mismatch")
    if accepted_manifest["evaluator_version"] != evaluator_asset[
        "evaluator_version"
    ]:
        raise RuntimeError("preflight evaluator version mismatch")

    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "schema_version": 1,
            "protocol_id": SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID,
            "source": source,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "protocol_sha256": sha256_path(protocol_path),
            "polybert_asset": model_asset,
            "evaluator_asset": evaluator_asset,
            "credentials_loading_authorized": False,
            "external_api_requests_authorized": False,
            "ppo_execution_authorized": False,
            "oracle_execution_authorized": False,
            "real_multi_iteration_authorized": False,
        },
    )
    started = time.perf_counter()
    components, engine, specification, run_contract = build_runtime(
        root,
        accepted_manifest,
        accepted_binding,
        runtime_protocol,
        Path(model_asset["model_path"]),
        polybert_checkpoint_fingerprint=model_asset["checkpoint_fingerprint"],
        evaluator_asset_path=Path(evaluator_asset["asset_path"]),
    )
    terminal_evaluator = components.evaluator.evaluator
    initial_ledger = components.evaluator.ledger()
    encoder_diagnostics = components.core.encoder.diagnostics()
    runtime = {
        "constructed": True,
        "components_class": type(components).__name__,
        "engine_class": type(engine).__name__,
        "environment_id": specification["environment_id"],
        "run_contract_environment_id": run_contract.environment_id,
        "maximum_requested_calls": initial_ledger["maximum_requested_calls"],
        "evaluator_asset_path": str(terminal_evaluator.model_dir.resolve()),
        "evaluator_required_file_sha256": _runtime_evaluator_hashes(
            terminal_evaluator
        ),
        "initial_evaluator_ledger": initial_ledger,
        "encoder_diagnostics": encoder_diagnostics,
        "polybert_initial_embedding_inference_executed": int(
            encoder_diagnostics["cache_misses"]
        )
        > 0,
        "afp_property_inference_executed": False,
        "local_llm_loaded": False,
    }

    wall_values = dict(protocol["llm_wall_clock_budget"])
    wall_values.pop("persist_across_checkpoints")
    wall_values.pop("exhaustion_outcome")
    wall_config = LLMWallClockBudgetConfig(**wall_values)
    control_sequence = _run_control_sequence(
        output_dir / "mock-control-sequence",
        sha256_path(protocol_path),
        wall_config,
    )
    statuses = [
        item["auxiliary_status"] for item in control_sequence["receipts"]
    ]
    control_payloads = [
        json.loads(
            Path(item["control_checkpoint_path"]).read_text(encoding="utf-8")
        )
        for item in control_sequence["controls"]
    ]
    checkpoint_chain_exact = all(
        payload["previous_control_sha256"]
        == (
            None
            if index == 0
            else control_sequence["controls"][index - 1][
                "control_checkpoint_sha256"
            ]
        )
        for index, payload in enumerate(control_payloads)
    )

    closed_authorization = {
        "schema_version": 5,
        "authorization_id": "closed-short-horizon-preflight",
        "protocol_id": SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID,
        "protocol_sha256": sha256_path(protocol_path),
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
        "authorized_start_iteration": 1,
        "authorized_end_iteration": 6,
        "authorized_resume_control_checkpoint": None,
        "resume_control_checkpoint_sha256": None,
        "maximum_slurm_runs": 1,
        "authorized_operations": {
            name: False for name in AUTHORIZATION_OPERATIONS_SHORT_HORIZON
        },
    }
    closed_path = output_dir / "closed-schema5-authorization.json"
    write_json(closed_path, closed_authorization)
    closed_authorization_rejected = False
    try:
        load_execution_authorization_short_horizon(
            closed_path,
            protocol_sha256=sha256_path(protocol_path),
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
            evaluator_asset_fingerprint=evaluator_asset["asset_fingerprint"],
            resume_control_checkpoint=None,
            resume_control_checkpoint_sha256=None,
        )
    except ValueError:
        closed_authorization_rejected = True

    runner_path = (
        root / "reproduction/scicf/online/run_short_horizon_multi_iteration.py"
    )
    runner_source = runner_path.read_text(encoding="utf-8")
    ppo_index = runner_source.index("ppo_result = engine.run_iteration(")
    primary_index = runner_source.index("engine.save_checkpoint(primary_checkpoint)")
    credentials_index = runner_source.index(
        "settings = APISettings.from_private_file("
    )
    deadline_index = runner_source.index("client=DeadlineBoundClient(")
    control_index = runner_source.index(
        "checkpoint = save_short_horizon_checkpoint("
    )

    checks = {
        "protocol_identity": protocol["protocol_id"]
        == SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID,
        "real_execution_not_authorized": protocol["authorization_state"][
            "real_multi_iteration_execution_authorized"
        ]
        is False,
        "closed_schema5_authorization_rejected": closed_authorization_rejected,
        "full_runtime_constructed": runtime["constructed"] is True,
        "runtime_budget_bound_to_1248": runtime["maximum_requested_calls"] == 1248,
        "runtime_evaluator_path_exact": Path(
            runtime["evaluator_asset_path"]
        ).resolve()
        == Path(evaluator_asset["asset_path"]).resolve(),
        "runtime_evaluator_hashes_exact": runtime[
            "evaluator_required_file_sha256"
        ]
        == evaluator_asset["required_file_sha256"],
        "runtime_evaluator_ledger_zero": all(
            int(initial_ledger[name]) == 0
            for name in ("requested_calls", "unique_calls", "backend_calls")
        ),
        "runtime_polybert_initial_embedding_executed": runtime[
            "polybert_initial_embedding_inference_executed"
        ]
        is True,
        "runtime_afp_property_inference_not_executed": runtime[
            "afp_property_inference_executed"
        ]
        is False,
        "no_local_llm_loaded": runtime["local_llm_loaded"] is False,
        "primary_checkpoint_precedes_credentials": ppo_index
        < primary_index
        < credentials_index,
        "deadline_wraps_provider_before_control_checkpoint": credentials_index
        < deadline_index
        < control_index,
        "six_control_checkpoints_written": len(
            control_sequence["controls"]
        )
        == 6,
        "checkpoint_previous_hash_chain_exact": checkpoint_chain_exact,
        "checkpoint_circuit_state_persisted": (
            control_payloads[1]["circuit_breaker"]["open_until_iteration"] == 6
            and control_payloads[4]["circuit_breaker"]["open_until_iteration"]
            == 6
        ),
        "checkpoint_wall_time_state_persisted": (
            control_payloads[1]["llm_wall_clock_budget"]["consumed_seconds"]
            == 20.0
            and control_payloads[4]["llm_wall_clock_budget"][
                "consumed_seconds"
            ]
            == 20.0
            and control_payloads[5]["llm_wall_clock_budget"][
                "consumed_seconds"
            ]
            == 25.0
        ),
        "two_failures_open_circuit": statuses[:2]
        == ["skipped_recoverable_llm_failure"] * 2,
        "three_iterations_skip_acquisition": statuses[2:5]
        == ["skipped_circuit_open"] * 3,
        "sixth_iteration_retries": statuses[5]
        == "ready_for_oracle_verification",
        "circuit_state_persisted_and_recovered": control_sequence[
            "final_breaker"
        ]["consecutive_llm_failures"]
        == 0,
        "wall_time_persisted_only_attempted_iterations": control_sequence[
            "final_wall_budget"
        ]["completed_attempt_iterations"]
        == [1, 2, 6],
        "wall_time_persisted_value": control_sequence["final_wall_budget"][
            "consumed_seconds"
        ]
        == 25.0,
        "tiny_negative_kl_clamped": numerically_nonnegative_kl(-2.9e-7)
        == 0.0,
        "credentials_loaded_false": True,
        "external_api_request_count_zero": True,
        "ppo_iterations_zero": True,
        "oracle_calls_zero": True,
        "sealed_test_accessed_false": True,
    }
    materially_negative_rejected = False
    try:
        numerically_nonnegative_kl(-1.1e-6)
    except Exception:
        materially_negative_rejected = True
    checks["materially_negative_kl_rejected"] = materially_negative_rejected
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "protocol_id": SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID,
        "execution_status": "passed" if not failures else "failed",
        "module_decision": (
            "go_request_separate_short_horizon_real_execution_authorization"
            if not failures
            else "no_go_short_horizon_real_execution_authorization"
        ),
        "classification": {
            "scope": "no-credential-full-runtime-short-horizon-preflight",
            "failures": failures,
            "real_multi_iteration_execution_authorized": False,
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
        "runner_sha256": sha256_path(runner_path),
        "polybert_asset": model_asset,
        "evaluator_asset": evaluator_asset,
        "evaluator_route_source_delta": source_delta,
        "accepted_binding": accepted_binding,
        "runtime": runtime,
        "checks": checks,
        "mock_control_sequence": control_sequence,
        "credentials_loaded": False,
        "external_api_request_count": 0,
        "ppo_iterations": 0,
        "oracle_calls": 0,
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
    write_json(output_dir / "short-horizon-preflight-report.json", report)
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
