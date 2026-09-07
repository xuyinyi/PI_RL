#!/usr/bin/env python
"""Run the frozen mock-only LLM response-schema robustness development gate."""

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
from typing import Any, Mapping, Sequence

from reproduction.framework.io import git_identity, write_json
from reproduction.scicf.llm.api_client import ChatCompletion

from .contracts import SCHEMA_ROBUSTNESS_PROTOCOL_ID, SchemaRecoveryConfig
from .response_guard import complete_with_bounded_schema_repair


SCENARIOS = (
    "valid_first_attempt",
    "invented_id_then_valid_repair",
    "malformed_json_then_valid_repair",
    "over_budget_then_abstain_repair",
    "two_invalid_attempts_fail_closed",
)
CANDIDATE_IDS = tuple("cf-%016x" % index for index in range(1, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
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


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve_input(root: Path, relative: str, expected_sha256: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if root != path and root not in path.parents:
        raise ValueError("schema robustness input escaped the repository root")
    observed = _sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(
            "schema robustness input hash mismatch: %s != %s"
            % (observed, expected_sha256)
        )
    return path


def load_config(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "classification",
        "required_host",
        "base_seed",
        "incident_binding",
        "limits",
        "retention",
        "repair_contract",
        "development_scenarios",
        "claim_boundary",
    }
    if set(payload) != required:
        raise ValueError(
            "schema robustness config keys differ; missing=%s extra=%s"
            % (sorted(required - set(payload)), sorted(set(payload) - required))
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported schema robustness config schema")
    if payload["protocol_id"] != SCHEMA_ROBUSTNESS_PROTOCOL_ID:
        raise ValueError("unsupported schema robustness protocol")
    if (
        payload["classification"]
        != "llm_response_schema_engineering_development_only"
    ):
        raise ValueError("schema robustness gate is not engineering-only")
    limits = payload["limits"]
    guard_config = SchemaRecoveryConfig(
        maximum_schema_attempts=limits["maximum_schema_attempts"],
        transport_retries_per_attempt=limits["transport_retries_per_attempt"],
        max_output_tokens=limits["max_output_tokens"],
        timeout_seconds=limits["timeout_seconds"],
    )
    if limits["maximum_semantic_repairs"] != guard_config.maximum_semantic_repairs:
        raise ValueError("semantic repair limit is inconsistent")
    if (
        limits["maximum_http_transmissions"]
        != guard_config.maximum_http_transmissions
    ):
        raise ValueError("HTTP transmission limit is inconsistent")
    if tuple(payload["development_scenarios"]) != SCENARIOS:
        raise ValueError("development scenario set or order changed")
    required_retention = {
        "persist_request_before_provider_call": True,
        "persist_raw_content_before_validation": True,
        "persist_invalid_raw_content": True,
        "persist_validation_receipt": True,
        "persist_provider_public_identity": True,
        "persist_api_key": False,
        "persist_credentials_path": False,
    }
    if payload["retention"] != required_retention:
        raise ValueError("response retention contract changed")
    required_repair = {
        "reuse_original_blinded_messages": True,
        "preserve_exact_candidate_allowlist": True,
        "preserve_original_maximum_budget": True,
        "allow_partial_selection": True,
        "allow_abstention": True,
        "replay_invalid_raw_text_to_model": False,
        "allow_fallback_selection": False,
        "allow_cached_response_substitution": False,
        "allow_reward_truth": False,
        "allow_policy_scores": False,
    }
    if payload["repair_contract"] != required_repair:
        raise ValueError("schema repair contract changed")
    for name in (
        "external_api_requests_authorized",
        "single_iteration_integration_rerun_authorized",
        "ppo_execution_authorized",
        "oracle_execution_authorized",
        "multi_iteration_training_authorized",
        "sealed_test_access_authorized",
        "algorithm_effectiveness_established",
        "scientific_claim_authorized",
    ):
        if payload["claim_boundary"].get(name) is not False:
            raise ValueError("claim boundary must disable %s" % name)
    return payload


class ScriptedClient:
    """In-memory completion source; it performs no network operation."""

    def __init__(self, contents: Sequence[str]) -> None:
        self.contents = tuple(contents)
        self.calls = []

    def complete(
        self,
        *,
        messages,
        max_tokens,
        seed,
        timeout_seconds,
        transport_retries,
    ) -> ChatCompletion:
        index = len(self.calls)
        if index >= len(self.contents):
            raise RuntimeError("scripted client exceeded its frozen completion list")
        self.calls.append(
            {
                "messages": [dict(item) for item in messages],
                "max_tokens": int(max_tokens),
                "seed": int(seed),
                "timeout_seconds": float(timeout_seconds),
                "transport_retries": int(transport_retries),
            }
        )
        content = self.contents[index]
        return ChatCompletion(
            content=content,
            prompt_tokens=100 + index,
            completion_tokens=20 + index,
            total_tokens=120 + 2 * index,
            provider_response_id="mock-response-%02d" % index,
            system_fingerprint="mock-schema-robustness-v1",
            response_sha256=hashlib.sha256(
                ("mock-envelope|" + content).encode("utf-8")
            ).hexdigest(),
            transport_retries_used=0,
        )


def _response(ranked, abstain=False, **extra) -> str:
    value = {
        "ranked_intervention_ids": list(ranked),
        "abstain": bool(abstain),
        "reasoning": "mock schema fixture",
    }
    value.update(extra)
    return json.dumps(value, sort_keys=True)


def _request(scenario: str) -> Mapping[str, Any]:
    messages = [
        {
            "role": "system",
            "content": "Return one blinded ranked-intervention JSON object.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "candidate_ids": list(CANDIDATE_IDS),
                    "maximum_oracle_budget": 2,
                    "terminal_reward_or_properties_included": False,
                    "policy_score_included": False,
                },
                sort_keys=True,
            ),
        },
    ]
    return {
        "schema_version": 1,
        "request_id": "schema-dev-" + scenario,
        "prompt_sha256": _canonical_sha256(messages),
        "candidate_ids": list(CANDIDATE_IDS),
        "presented_candidate_ids": list(reversed(CANDIDATE_IDS)),
        "maximum_budget": 2,
        "messages": messages,
        "api_key_exposed": False,
        "candidate_reward_truth_exposed": False,
        "factual_reward_truth_exposed": False,
        "policy_score_exposed": False,
    }


def _scenario_contents(name: str) -> Sequence[str]:
    valid = _response((CANDIDATE_IDS[1],))
    if name == "valid_first_attempt":
        return (valid,)
    if name == "invented_id_then_valid_repair":
        return (_response(("cf-281",)), valid)
    if name == "malformed_json_then_valid_repair":
        return ("not a JSON object", valid)
    if name == "over_budget_then_abstain_repair":
        return (
            _response((CANDIDATE_IDS[0], CANDIDATE_IDS[1], CANDIDATE_IDS[2])),
            _response((), abstain=True),
        )
    if name == "two_invalid_attempts_fail_closed":
        return (_response(("cf-281",)), _response(("cf-282",)))
    raise ValueError("unknown schema robustness scenario: %s" % name)


def _audit_scenario(
    *,
    name: str,
    request: Mapping[str, Any],
    client: ScriptedClient,
    outcome: Mapping[str, Any],
    scenario_dir: Path,
    config: SchemaRecoveryConfig,
) -> Mapping[str, Any]:
    expected_attempts = 1 if name == "valid_first_attempt" else 2
    expected_status = (
        "fail_closed_schema_exhausted"
        if name == "two_invalid_attempts_fail_closed"
        else "validated"
    )
    checks = {
        "status_matches": outcome["status"] == expected_status,
        "attempt_count_matches": int(outcome["attempt_count"]) == expected_attempts,
        "client_call_count_matches": len(client.calls) == expected_attempts,
        "attempt_limit_respected": len(client.calls)
        <= int(config.maximum_schema_attempts),
        "semantic_repair_limit_respected": int(outcome["semantic_repair_count"])
        <= int(config.maximum_semantic_repairs),
        "raw_capture_count_matches": len(list(scenario_dir.glob("*-raw-response.json")))
        == expected_attempts,
        "validation_count_matches": len(list(scenario_dir.glob("*-validation.json")))
        == expected_attempts,
        "request_receipt_count_matches": len(list(scenario_dir.glob("*-request.json")))
        == expected_attempts,
    }
    for attempt_index in range(expected_attempts):
        raw = json.loads(
            (scenario_dir / ("attempt-%02d-raw-response.json" % attempt_index)).read_text(
                encoding="utf-8"
            )
        )
        validation = json.loads(
            (scenario_dir / ("attempt-%02d-validation.json" % attempt_index)).read_text(
                encoding="utf-8"
            )
        )
        checks["raw_before_validation_%02d" % attempt_index] = bool(
            raw["capture_status"] == "captured_before_validation"
            and validation["raw_was_captured_before_validation"] is True
            and validation["raw_capture_sha256"]
            == _sha256_path(
                scenario_dir / ("attempt-%02d-raw-response.json" % attempt_index)
            )
        )
        checks["no_api_key_logged_%02d" % attempt_index] = (
            raw["api_key_logged"] is False
        )
    if expected_attempts == 2:
        repair_request = json.loads(
            (scenario_dir / "attempt-01-request.json").read_text(encoding="utf-8")
        )
        repair_payload = json.loads(repair_request["messages"][-1]["content"])
        checks["repair_reuses_original_messages"] = (
            repair_request["messages"][:-1] == request["messages"]
        )
        checks["repair_allowlist_exact"] = (
            repair_payload["allowed_candidate_ids_in_blinded_order"]
            == request["presented_candidate_ids"]
        )
        checks["repair_budget_unchanged"] = (
            repair_payload["maximum_oracle_budget"] == request["maximum_budget"]
        )
        checks["invalid_raw_not_replayed"] = all(
            content not in json.dumps(repair_request["messages"], sort_keys=True)
            for content in _scenario_contents(name)[:1]
        )
        checks["repair_leakage_flags_false"] = bool(
            repair_request["reward_truth_exposed"] is False
            and repair_request["policy_score_exposed"] is False
            and repair_request["api_key_exposed"] is False
        )
    if name == "two_invalid_attempts_fail_closed":
        checks["exhaustion_empty"] = outcome["selected_intervention_ids"] == []
        checks["oracle_closed"] = outcome["oracle_selection_authorized"] is False
        checks["no_fallback"] = outcome["fallback_selection_used"] is False
        checks["no_cache_substitution"] = (
            outcome["cached_response_substituted"] is False
        )
    else:
        checks["validated_selection_authorized"] = (
            outcome["oracle_selection_authorized"] is True
        )
    return {
        "scenario": name,
        "status": outcome["status"],
        "attempt_count": outcome["attempt_count"],
        "semantic_repair_count": outcome["semantic_repair_count"],
        "checks": checks,
        "passed": all(checks.values()),
        "outcome_sha256": _sha256_path(scenario_dir / "outcome.json"),
    }


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("schema robustness output already exists: %s" % output_dir)
    config = load_config(config_path)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("schema robustness development requires Slurm")
    if socket.gethostname() != config["required_host"]:
        raise RuntimeError("schema robustness development is bound to n001")
    source = git_identity(root)
    if source.get("dirty") is not False:
        raise RuntimeError("schema robustness development requires a clean worktree")
    incident = config["incident_binding"]
    archive_path = _resolve_input(
        root, incident["archive_decision"], incident["archive_decision_sha256"]
    )
    stderr_path = _resolve_input(
        root, incident["failed_stderr"], incident["failed_stderr_sha256"]
    )
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    if (
        archive["decision"]
        != "no_go_bounded_short_horizon_multi_iteration_engineering_smoke"
    ):
        raise RuntimeError("incident archive no longer records the required no-go")
    if "cf-281" not in stderr_path.read_text(encoding="utf-8"):
        raise RuntimeError("bound incident stderr lacks the rejected identifier")

    limits = config["limits"]
    guard_config = SchemaRecoveryConfig(
        maximum_schema_attempts=limits["maximum_schema_attempts"],
        transport_retries_per_attempt=limits["transport_retries_per_attempt"],
        max_output_tokens=limits["max_output_tokens"],
        timeout_seconds=limits["timeout_seconds"],
    )
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "run-intent.json",
        {
            "schema_version": 1,
            "status": "declared-before-mock-scenarios",
            "protocol_id": SCHEMA_ROBUSTNESS_PROTOCOL_ID,
            "classification": config["classification"],
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "config_sha256": _sha256_path(config_path),
            "incident_archive_sha256": incident["archive_decision_sha256"],
            "incident_stderr_sha256": incident["failed_stderr_sha256"],
            "external_api_requests_authorized": False,
            "credentials_loaded": False,
            "ppo_execution_authorized": False,
            "oracle_execution_authorized": False,
            "claim_boundary": config["claim_boundary"],
        },
    )

    started = time.perf_counter()
    provider = {
        "type": "scripted-in-memory-completion-source",
        "provider_id": "mock-only-no-network",
        "model_id": "schema-fixture-v1",
        "external_api": False,
    }
    scenario_reports = []
    for scenario_index, name in enumerate(SCENARIOS):
        request = _request(name)
        client = ScriptedClient(_scenario_contents(name))
        scenario_dir = output_dir / "scenarios" / name
        outcome = complete_with_bounded_schema_repair(
            client=client,
            request=request,
            provider=provider,
            output_dir=scenario_dir,
            config=guard_config,
            seed=int(config["base_seed"]) + scenario_index,
        )
        scenario_reports.append(
            _audit_scenario(
                name=name,
                request=request,
                client=client,
                outcome=outcome,
                scenario_dir=scenario_dir,
                config=guard_config,
            )
        )

    failures = [item["scenario"] for item in scenario_reports if not item["passed"]]
    passed = not failures
    report = {
        "schema_version": 1,
        "protocol_id": SCHEMA_ROBUSTNESS_PROTOCOL_ID,
        "execution_status": "passed" if passed else "failed",
        "module_decision": (
            "go_separate_single_iteration_integration_v2_protocol_freeze"
            if passed
            else "no_go_single_iteration_integration_v2_protocol_freeze"
        ),
        "classification": {
            "scope": "mock-only-llm-response-schema-robustness-development",
            "failed_scenarios": failures,
            "next_scope_authorized": passed,
            "single_iteration_integration_rerun_authorized": False,
            "multi_iteration_training_authorized": False,
            "algorithm_effectiveness_established": False,
            "scientific_claim_authorized": False,
        },
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "source": source,
        "config": config,
        "guard_config": {
            "maximum_schema_attempts": guard_config.maximum_schema_attempts,
            "maximum_semantic_repairs": guard_config.maximum_semantic_repairs,
            "transport_retries_per_attempt": (
                guard_config.transport_retries_per_attempt
            ),
            "maximum_http_transmissions": guard_config.maximum_http_transmissions,
            "max_output_tokens": guard_config.max_output_tokens,
            "timeout_seconds": guard_config.timeout_seconds,
        },
        "incident_binding": {
            "archive_decision": str(archive_path),
            "archive_decision_sha256": _sha256_path(archive_path),
            "failed_stderr": str(stderr_path),
            "failed_stderr_sha256": _sha256_path(stderr_path),
        },
        "scenarios": scenario_reports,
        "scenario_count": len(scenario_reports),
        "mock_completion_count": sum(
            int(item["attempt_count"]) for item in scenario_reports
        ),
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
    write_json(output_dir / "schema-robustness-report.json", report)
    print(
        json.dumps(
            {
                "execution_status": report["execution_status"],
                "module_decision": report["module_decision"],
                "failed_scenarios": failures,
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
