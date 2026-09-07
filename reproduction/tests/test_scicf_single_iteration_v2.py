from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reproduction.scicf.llm.api_client import APITransportError, ChatCompletion
from reproduction.scicf.online.run_single_iteration_integration_v2 import (
    AUTHORIZATION_OPERATIONS,
    load_execution_authorization,
    load_protocol,
    run_guarded_pool_decisions,
    verify_protocol_bindings,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "reproduction/scicf/online/configs/"
    / "single_iteration_integration_v2_protocol.json"
)


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _request(pool_index):
    candidate_ids = ["cf-%02d-%02d" % (pool_index, index) for index in range(24)]
    messages = [
        {"role": "system", "content": "Return one JSON object."},
        {
            "role": "user",
            "content": json.dumps(
                {"candidate_ids": candidate_ids, "maximum_oracle_budget": 4},
                sort_keys=True,
            ),
        },
    ]
    return {
        "request_id": "v2-test-pool-%02d" % pool_index,
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


class _Client:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def complete(self, **kwargs):
        index = len(self.calls)
        if index >= len(self.contents):
            raise AssertionError("v2 guard exceeded the scripted completion bound")
        self.calls.append(kwargs)
        content = self.contents[index]
        if isinstance(content, Exception):
            raise content
        return ChatCompletion(
            content=content,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            provider_response_id="mock-%02d" % index,
            system_fingerprint="v2-preflight",
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            transport_retries_used=0,
        )


def _prepared():
    return [
        {"pool_index": index, "pool_seed": 100 + index, "request": _request(index)}
        for index in range(2)
    ]


def test_frozen_v2_protocol_and_all_bound_inputs_validate():
    protocol = load_protocol(PROTOCOL)
    bindings = verify_protocol_bindings(ROOT, protocol)
    assert protocol["status"] == "frozen_unimplemented_unexecuted"
    assert bindings["v1_archive"]["decision"].startswith("no_go_")
    assert bindings["schema_report"]["external_api_request_count"] == 0
    assert protocol["authorization_state"]["external_api_requests_authorized"] is False
    assert protocol["authorization_state"]["multi_iteration_training_authorized"] is False


def test_execution_requires_exact_single_run_authorization(tmp_path):
    protocol_sha256 = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    output = tmp_path / "authorized-output"
    manifest = {
        "schema_version": 1,
        "authorization_id": "unit-test-only",
        "protocol_id": "dapigen-scicf-single-iteration-integration-smoke-v2",
        "protocol_sha256": protocol_sha256,
        "implementation_commit": "abc123",
        "authorized_output_directory": str(output),
        "maximum_slurm_runs": 1,
        "authorized_operations": dict(AUTHORIZATION_OPERATIONS),
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_execution_authorization(
        path,
        protocol_sha256=protocol_sha256,
        implementation_commit="abc123",
        output_dir=output,
    )
    assert loaded["authorization_id"] == "unit-test-only"

    manifest["authorized_operations"]["multi_iteration_training_authorized"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exact bounded scope"):
        load_execution_authorization(
            path,
            protocol_sha256=protocol_sha256,
            implementation_commit="abc123",
            output_dir=output,
        )


def test_guarded_two_pool_path_repairs_then_accepts_abstention(tmp_path):
    protocol = load_protocol(PROTOCOL)
    first_ids = _request(0)["candidate_ids"]
    client = _Client(
        [
            _response(["cf-invented"]),
            _response([first_ids[3], first_ids[7]]),
            _response([], abstain=True),
        ]
    )
    summary = run_guarded_pool_decisions(
        client=client,
        provider={"provider_id": "mock-only", "external_api": False},
        prepared_pools=_prepared(),
        output_root=tmp_path / "guarded",
        protocol=protocol,
    )
    assert len(client.calls) == 3
    assert summary["status"] == "validated_all_pools"
    assert summary["oracle_selection_authorized"] is True
    assert summary["selected_intervention_ids"] == [first_ids[3], first_ids[7]]
    assert summary["semantic_attempt_count"] == 3
    assert summary["semantic_repair_count"] == 1
    assert summary["http_transmissions_observed"] == 3
    assert summary["maximum_http_transmissions_total"] == 8
    assert summary["returned_content_count"] == 3


def test_schema_exhaustion_stops_before_second_pool_and_closes_oracle(tmp_path):
    protocol = load_protocol(PROTOCOL)
    client = _Client([_response(["cf-x"]), _response(["cf-y"])])
    summary = run_guarded_pool_decisions(
        client=client,
        provider={"provider_id": "mock-only", "external_api": False},
        prepared_pools=_prepared(),
        output_root=tmp_path / "guarded",
        protocol=protocol,
    )
    assert len(client.calls) == 2
    assert summary["status"] == "fail_closed"
    assert summary["pool_decision_count_started"] == 1
    assert summary["oracle_selection_authorized"] is False
    assert summary["selected_intervention_ids"] == []
    assert not (tmp_path / "guarded/pool-01").exists()


def test_transport_exhaustion_stops_without_semantic_repair(tmp_path):
    protocol = load_protocol(PROTOCOL)
    client = _Client([APITransportError("mock transport exhaustion", retryable=True)])
    summary = run_guarded_pool_decisions(
        client=client,
        provider={"provider_id": "mock-only", "external_api": False},
        prepared_pools=_prepared(),
        output_root=tmp_path / "guarded",
        protocol=protocol,
    )
    assert len(client.calls) == 1
    assert summary["status"] == "fail_closed"
    assert summary["semantic_attempt_count"] == 1
    assert summary["semantic_repair_count"] == 0
    assert summary["oracle_selection_authorized"] is False
    assert summary["http_transmissions_observed_is_exact"] is False
    assert summary["http_transmissions_used"] is None
    assert summary["http_transmissions_upper_bound_for_started_decisions"] == 2
    assert summary["token_accounting_complete"] is False
    assert summary["reported_tokens"] == {"prompt": None, "completion": None}


def test_protocol_rejects_expanded_http_or_training_scope(tmp_path):
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["acquisition"]["maximum_http_transmissions_total"] = 9
    changed = tmp_path / "changed-protocol.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="total HTTP bound"):
        load_protocol(changed)

    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["authorization_state"]["multi_iteration_training_authorized"] = True
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot authorize"):
        load_protocol(changed)
