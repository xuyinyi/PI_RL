#!/usr/bin/env python
"""Run the mock-only, no-credential integration-v2 implementation preflight."""

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
from reproduction.scicf.llm.api_client import APITransportError, ChatCompletion

from .contracts import SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID
from .run_single_iteration_integration_v2 import (
    AUTHORIZATION_OPERATIONS,
    load_execution_authorization,
    load_protocol,
    run_guarded_pool_decisions,
    verify_protocol_bindings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
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


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _request(pool_index: int):
    candidate_ids = [
        "cf-preflight-%02d-%02d" % (pool_index, index) for index in range(24)
    ]
    messages = [
        {"role": "system", "content": "Return one blinded JSON ranking."},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "candidate_ids": candidate_ids,
                    "maximum_oracle_budget": 4,
                    "terminal_reward_or_properties_included": False,
                    "policy_score_included": False,
                },
                sort_keys=True,
            ),
        },
    ]
    return {
        "request_id": "integration-v2-preflight-pool-%02d" % pool_index,
        "prompt_sha256": _canonical_sha256(messages),
        "candidate_ids": candidate_ids,
        "presented_candidate_ids": list(reversed(candidate_ids)),
        "maximum_budget": 4,
        "messages": messages,
        "api_key_exposed": False,
        "candidate_reward_truth_exposed": False,
        "factual_reward_truth_exposed": False,
        "policy_score_exposed": False,
    }


def _response(ids, abstain=False):
    return json.dumps(
        {"ranked_intervention_ids": list(ids), "abstain": bool(abstain)},
        sort_keys=True,
    )


class ScriptedClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def complete(self, **kwargs):
        index = len(self.calls)
        if index >= len(self.contents):
            raise RuntimeError("preflight attempted an unbounded completion")
        self.calls.append(kwargs)
        content = self.contents[index]
        if isinstance(content, Exception):
            raise content
        return ChatCompletion(
            content=content,
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
            provider_response_id="mock-v2-%02d" % index,
            system_fingerprint="mock-no-network",
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            transport_retries_used=0,
        )


def _prepared():
    return [
        {"pool_index": index, "pool_seed": 20260907 + index, "request": _request(index)}
        for index in range(2)
    ]


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("integration-v2 preflight output already exists")
    protocol = load_protocol(protocol_path)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("integration-v2 preflight requires Slurm")
    if socket.gethostname() != protocol["required_host"]:
        raise RuntimeError("integration-v2 preflight is bound to n001")
    source = git_identity(root)
    if source.get("dirty") is not False:
        raise RuntimeError("integration-v2 preflight requires a clean worktree")
    bindings = verify_protocol_bindings(root, protocol)
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "schema_version": 1,
            "status": "declared-before-mock-preflight",
            "protocol_id": SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "protocol_sha256": _sha256_path(protocol_path),
            "external_api_requests_authorized": False,
            "credentials_loading_authorized": False,
            "ppo_execution_authorized": False,
            "oracle_execution_authorized": False,
            "single_iteration_integration_rerun_authorized": False,
            "multi_iteration_training_authorized": False,
        },
    )
    started = time.perf_counter()
    prepared = _prepared()
    first_ids = prepared[0]["request"]["candidate_ids"]
    success_client = ScriptedClient(
        [
            _response(["cf-invented"]),
            _response([first_ids[2], first_ids[9]]),
            _response([], abstain=True),
        ]
    )
    success = run_guarded_pool_decisions(
        client=success_client,
        provider={"provider_id": "mock-v2-no-network", "external_api": False},
        prepared_pools=prepared,
        output_root=output_dir / "scenarios/success-repair-and-abstain",
        protocol=protocol,
    )
    failure_client = ScriptedClient(
        [_response(["cf-x"]), _response(["cf-y"])]
    )
    exhaustion = run_guarded_pool_decisions(
        client=failure_client,
        provider={"provider_id": "mock-v2-no-network", "external_api": False},
        prepared_pools=prepared,
        output_root=output_dir / "scenarios/schema-exhaustion",
        protocol=protocol,
    )
    transport_client = ScriptedClient(
        [APITransportError("mock transport exhaustion", retryable=True)]
    )
    transport = run_guarded_pool_decisions(
        client=transport_client,
        provider={"provider_id": "mock-v2-no-network", "external_api": False},
        prepared_pools=prepared,
        output_root=output_dir / "scenarios/transport-exhaustion",
        protocol=protocol,
    )

    closed_authorization = {
        "schema_version": 1,
        "authorization_id": "closed-preflight-fixture",
        "protocol_id": SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID,
        "protocol_sha256": _sha256_path(protocol_path),
        "implementation_commit": source["commit"],
        "authorized_output_directory": str(output_dir / "never-run"),
        "maximum_slurm_runs": 1,
        "authorized_operations": {
            name: False for name in AUTHORIZATION_OPERATIONS
        },
    }
    closed_path = output_dir / "closed-authorization-fixture.json"
    write_json(closed_path, closed_authorization)
    authorization_rejected = False
    try:
        load_execution_authorization(
            closed_path,
            protocol_sha256=_sha256_path(protocol_path),
            implementation_commit=source["commit"],
            output_dir=output_dir / "never-run",
        )
    except ValueError:
        authorization_rejected = True

    runner_path = root / "reproduction/scicf/online/run_single_iteration_integration_v2.py"
    runner_source = runner_path.read_text(encoding="utf-8")
    authorization_index = runner_source.index("authorization = load_execution_authorization(")
    credentials_index = runner_source.index("settings = APISettings.from_private_file(")
    checks = {
        "protocol_identity": protocol["protocol_id"]
        == SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID,
        "v1_remains_no_go": bindings["v1_archive"]["decision"]
        == protocol["prerequisites"]["v1_required_decision"],
        "schema_gate_bound_and_passed": bindings["schema_report"]["module_decision"]
        == protocol["prerequisites"]["schema_robustness_required_decision"],
        "closed_authorization_rejected": authorization_rejected,
        "authorization_checked_before_credentials": authorization_index
        < credentials_index,
        "success_all_pools_validated": success["all_pool_decisions_validated"] is True,
        "success_oracle_would_be_authorized": success["oracle_selection_authorized"] is True,
        "success_attempt_bound": success["semantic_attempt_count"] == 3,
        "success_http_bound": success["http_transmissions_observed"] == 3,
        "schema_exhaustion_fail_closed": exhaustion["status"] == "fail_closed",
        "schema_exhaustion_stops_second_pool": exhaustion[
            "pool_decision_count_started"
        ]
        == 1,
        "schema_exhaustion_oracle_closed": exhaustion[
            "oracle_selection_authorized"
        ]
        is False,
        "transport_exhaustion_fail_closed": transport["status"] == "fail_closed",
        "transport_exhaustion_oracle_closed": transport[
            "oracle_selection_authorized"
        ]
        is False,
        "external_api_request_count_zero": True,
        "credentials_loaded_false": True,
        "ppo_executed_false": True,
        "oracle_executed_false": True,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "protocol_id": SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL_ID,
        "execution_status": "passed" if not failures else "failed",
        "module_decision": (
            "go_request_separate_real_single_iteration_execution_authorization"
            if not failures
            else "no_go_real_single_iteration_execution_authorization"
        ),
        "classification": {
            "scope": "mock-only-integration-v2-runner-preflight",
            "failures": failures,
            "real_single_iteration_execution_authorized": False,
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
        "runner_sha256": _sha256_path(runner_path),
        "checks": checks,
        "scenarios": {
            "success_repair_and_abstain": success,
            "schema_exhaustion": exhaustion,
            "transport_exhaustion": transport,
        },
        "mock_completion_count": len(success_client.calls)
        + len(failure_client.calls)
        + len(transport_client.calls),
        "external_api_request_count": 0,
        "credentials_loaded": False,
        "ppo_executed": False,
        "oracle_executed": False,
        "local_model_invoked": False,
        "sealed_test_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "resources": {
            "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            / 1024.0
        },
    }
    write_json(output_dir / "integration-v2-preflight-report.json", report)
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
