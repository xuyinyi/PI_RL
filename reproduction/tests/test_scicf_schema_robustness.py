from __future__ import annotations

import hashlib
import json

import pytest

from reproduction.scicf.llm.api_client import ChatCompletion
from reproduction.scicf.online import response_guard
from reproduction.scicf.online.contracts import SchemaRecoveryConfig


CANDIDATES = ["cf-a", "cf-b", "cf-c", "cf-d"]


def _sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _request():
    messages = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "Rank the blinded candidates."},
    ]
    return {
        "request_id": "schema-guard-test",
        "prompt_sha256": _sha256(messages),
        "candidate_ids": list(CANDIDATES),
        "presented_candidate_ids": list(reversed(CANDIDATES)),
        "maximum_budget": 2,
        "messages": messages,
        "api_key_exposed": False,
        "candidate_reward_truth_exposed": False,
        "factual_reward_truth_exposed": False,
        "policy_score_exposed": False,
    }


def _content(ids, abstain=False, **extra):
    value = {
        "ranked_intervention_ids": list(ids),
        "abstain": abstain,
    }
    value.update(extra)
    return json.dumps(value, sort_keys=True)


class _Client:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def complete(self, **kwargs):
        index = len(self.calls)
        if index >= len(self.contents):
            raise AssertionError("schema guard made an unbounded extra call")
        self.calls.append(kwargs)
        content = self.contents[index]
        return ChatCompletion(
            content=content,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            provider_response_id="mock-%d" % index,
            system_fingerprint="fixture",
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            transport_retries_used=0,
        )


def _run(tmp_path, contents):
    client = _Client(contents)
    output = tmp_path / "attempts"
    outcome = response_guard.complete_with_bounded_schema_repair(
        client=client,
        request=_request(),
        provider={"provider_id": "mock-only", "external_api": False},
        output_dir=output,
        config=SchemaRecoveryConfig(
            maximum_schema_attempts=2,
            transport_retries_per_attempt=1,
            max_output_tokens=64,
            timeout_seconds=5.0,
        ),
        seed=7,
    )
    return client, output, outcome


def test_schema_recovery_config_hard_caps_attempts_and_transport_retries():
    config = SchemaRecoveryConfig()
    assert config.maximum_schema_attempts == 2
    assert config.maximum_semantic_repairs == 1
    assert config.maximum_http_transmissions == 4
    with pytest.raises(ValueError, match="maximum_schema_attempts"):
        SchemaRecoveryConfig(maximum_schema_attempts=3)
    with pytest.raises(ValueError, match="transport_retries_per_attempt"):
        SchemaRecoveryConfig(transport_retries_per_attempt=2)


def test_valid_first_attempt_is_captured_and_does_not_repair(tmp_path):
    output = tmp_path / "attempts"
    client = _Client([_content(["cf-b"])])
    unchecked_complete = client.complete

    def complete_after_request_receipt(**kwargs):
        assert (output / "attempt-00-request.json").is_file()
        return unchecked_complete(**kwargs)

    client.complete = complete_after_request_receipt
    outcome = response_guard.complete_with_bounded_schema_repair(
        client=client,
        request=_request(),
        provider={"provider_id": "mock-only", "external_api": False},
        output_dir=output,
        config=SchemaRecoveryConfig(max_output_tokens=64, timeout_seconds=5.0),
        seed=7,
    )
    assert len(client.calls) == 1
    assert outcome["status"] == "validated"
    assert outcome["attempt_count"] == 1
    assert outcome["semantic_repair_count"] == 0
    assert outcome["selected_intervention_ids"] == ["cf-b"]
    assert (output / "attempt-00-raw-response.json").is_file()
    assert not (output / "attempt-01-request.json").exists()


def test_invalid_id_is_persisted_before_validation_then_repaired(
    tmp_path, monkeypatch
):
    output = tmp_path / "attempts"
    client = _Client([_content(["cf-281"]), _content(["cf-b"])])
    original = response_guard.validate_online_ranked_response
    capture_seen = []

    def validate_after_capture(value, candidate_ids, maximum_budget):
        attempt = len(capture_seen)
        capture_seen.append(
            (output / ("attempt-%02d-raw-response.json" % attempt)).is_file()
        )
        return original(value, candidate_ids, maximum_budget)

    monkeypatch.setattr(
        response_guard, "validate_online_ranked_response", validate_after_capture
    )
    outcome = response_guard.complete_with_bounded_schema_repair(
        client=client,
        request=_request(),
        provider={"provider_id": "mock-only", "external_api": False},
        output_dir=output,
        config=SchemaRecoveryConfig(max_output_tokens=64, timeout_seconds=5.0),
        seed=7,
    )
    assert capture_seen == [True, True]
    assert outcome["status"] == "validated"
    assert outcome["attempt_count"] == 2
    invalid_raw = json.loads(
        (output / "attempt-00-raw-response.json").read_text(encoding="utf-8")
    )
    assert "cf-281" in invalid_raw["raw_response"]
    assert invalid_raw["capture_status"] == "captured_before_validation"
    repair = json.loads(
        (output / "attempt-01-request.json").read_text(encoding="utf-8")
    )
    serialized_messages = json.dumps(repair["messages"], sort_keys=True)
    assert "cf-281" not in serialized_messages
    repair_payload = json.loads(repair["messages"][-1]["content"])
    assert repair_payload["allowed_candidate_ids_in_blinded_order"] == list(
        reversed(CANDIDATES)
    )
    assert repair_payload["maximum_oracle_budget"] == 2
    assert repair["raw_invalid_response_replayed_to_model"] is False


def test_malformed_json_is_persisted_before_extraction_then_repaired(
    tmp_path, monkeypatch
):
    output = tmp_path / "attempts"
    client = _Client(["not JSON", _content(["cf-c"])])
    original = response_guard.extract_json_object
    capture_seen = []

    def extract_after_capture(text):
        attempt = len(capture_seen)
        capture_seen.append(
            (output / ("attempt-%02d-raw-response.json" % attempt)).is_file()
        )
        return original(text)

    monkeypatch.setattr(response_guard, "extract_json_object", extract_after_capture)
    outcome = response_guard.complete_with_bounded_schema_repair(
        client=client,
        request=_request(),
        provider={"provider_id": "mock-only", "external_api": False},
        output_dir=output,
        config=SchemaRecoveryConfig(max_output_tokens=64, timeout_seconds=5.0),
        seed=8,
    )
    assert capture_seen == [True, True]
    assert outcome["status"] == "validated"
    first_validation = json.loads(
        (output / "attempt-00-validation.json").read_text(encoding="utf-8")
    )
    assert first_validation["validation_error_code"] == (
        "json_object_extraction_failed"
    )


def test_two_invalid_attempts_stop_exactly_and_return_no_selection(tmp_path):
    client, output, outcome = _run(
        tmp_path, [_content(["cf-281"]), _content(["cf-282"])]
    )
    assert len(client.calls) == 2
    assert outcome["status"] == "fail_closed_schema_exhausted"
    assert outcome["attempt_count"] == 2
    assert outcome["semantic_repair_count"] == 1
    assert outcome["selected_intervention_ids"] == []
    assert outcome["oracle_selection_authorized"] is False
    assert outcome["fallback_selection_used"] is False
    assert outcome["cached_response_substituted"] is False
    assert len(list(output.glob("*-raw-response.json"))) == 2


def test_request_hash_and_leakage_flags_fail_before_any_provider_call(tmp_path):
    request = dict(_request())
    request["prompt_sha256"] = "0" * 64
    client = _Client([_content(["cf-b"])])
    with pytest.raises(ValueError, match="prompt hash"):
        response_guard.complete_with_bounded_schema_repair(
            client=client,
            request=request,
            provider={"provider_id": "mock-only"},
            output_dir=tmp_path / "hash-failure",
            config=SchemaRecoveryConfig(),
            seed=9,
        )
    assert client.calls == []

    with pytest.raises(ValueError, match="non-public fields"):
        response_guard.complete_with_bounded_schema_repair(
            client=client,
            request=_request(),
            provider={"provider_id": "mock-only", "api_key": "must-not-log"},
            output_dir=tmp_path / "provider-failure",
            config=SchemaRecoveryConfig(),
            seed=9,
        )
    assert client.calls == []

    request = dict(_request())
    request["candidate_reward_truth_exposed"] = True
    with pytest.raises(ValueError, match="leakage flag"):
        response_guard.complete_with_bounded_schema_repair(
            client=client,
            request=request,
            provider={"provider_id": "mock-only"},
            output_dir=tmp_path / "leak-failure",
            config=SchemaRecoveryConfig(),
            seed=9,
        )
    assert client.calls == []
