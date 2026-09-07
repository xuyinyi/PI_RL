"""Durable, bounded validation and repair for online LLM acquisition output."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from reproduction.framework.io import write_json
from reproduction.scicf.llm.api_client import APITransportError

from .contracts import (
    SCHEMA_ROBUSTNESS_PROTOCOL_ID,
    SchemaRecoveryConfig,
)
from .prompt import validate_online_ranked_response


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def extract_json_object(text: str) -> Any:
    """Extract the first JSON object from plain or fenced model output."""

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


def _validate_request(request: Mapping[str, Any]) -> None:
    required = {
        "request_id",
        "prompt_sha256",
        "candidate_ids",
        "presented_candidate_ids",
        "maximum_budget",
        "messages",
        "api_key_exposed",
        "candidate_reward_truth_exposed",
        "factual_reward_truth_exposed",
        "policy_score_exposed",
    }
    missing = sorted(required - set(request))
    if missing:
        raise ValueError("online request lacks required guard fields: %s" % missing)
    if not isinstance(request["request_id"], str) or not request["request_id"]:
        raise ValueError("guard request_id must be non-empty")
    candidate_ids = list(request["candidate_ids"])
    presented_ids = list(request["presented_candidate_ids"])
    if (
        not candidate_ids
        or any(not isinstance(value, str) or not value for value in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ValueError("guard requires a non-empty unique candidate pool")
    if len(presented_ids) != len(set(presented_ids)) or set(presented_ids) != set(
        candidate_ids
    ):
        raise ValueError("presented candidate IDs must equal the fixed pool")
    if isinstance(request["maximum_budget"], bool) or not isinstance(
        request["maximum_budget"], int
    ):
        raise ValueError("maximum budget must be an integer")
    maximum_budget = request["maximum_budget"]
    if maximum_budget < 1 or maximum_budget > len(candidate_ids):
        raise ValueError("maximum budget is outside the fixed candidate pool")
    if not isinstance(request["messages"], list) or not request["messages"]:
        raise ValueError("guard requires the original non-empty message list")
    if request["prompt_sha256"] != _canonical_sha256(request["messages"]):
        raise ValueError("original prompt hash does not match the guarded messages")
    for name in (
        "api_key_exposed",
        "candidate_reward_truth_exposed",
        "factual_reward_truth_exposed",
        "policy_score_exposed",
    ):
        if request[name] is not False:
            raise ValueError("response guard refuses request leakage flag %s" % name)


def _validate_public_provider(provider: Mapping[str, Any]) -> None:
    allowed = {
        "type",
        "provider_id",
        "endpoint_sha256",
        "model_id",
        "model_revision",
        "include_seed",
        "json_mode",
        "max_tokens_field",
        "thinking",
        "external_api",
    }
    unknown = sorted(set(provider) - allowed)
    if unknown:
        raise ValueError("provider identity contains non-public fields: %s" % unknown)
    if not provider.get("provider_id"):
        raise ValueError("provider public identity requires provider_id")


def build_schema_repair_messages(
    *,
    request: Mapping[str, Any],
    previous_response_text_sha256: str,
    validation_error_code: str,
) -> Sequence[Mapping[str, str]]:
    """Build a repair turn without replaying the untrusted invalid text."""

    _validate_request(request)
    payload = {
        "repair_protocol": "scicf-online-ranked-response-repair-v1",
        "task": "repair_response_schema_only",
        "previous_response_text_sha256": previous_response_text_sha256,
        "validation_error_code": validation_error_code,
        "allowed_candidate_ids_in_blinded_order": list(
            request["presented_candidate_ids"]
        ),
        "maximum_oracle_budget": int(request["maximum_budget"]),
        "required_output": {
            "ranked_intervention_ids": (
                "zero to maximum_oracle_budget unique IDs copied exactly from "
                "allowed_candidate_ids_in_blinded_order"
            ),
            "abstain": (
                "true requires an empty ranked_intervention_ids list; false requires "
                "one to maximum_oracle_budget IDs"
            ),
            "reasoning": "optional brief string; advisory only",
            "confidence": "optional number in [0,1]; advisory only",
        },
        "constraints": [
            "return one JSON object and no markdown",
            "do not invent, shorten, expand, or modify candidate IDs",
            (
                "do not add fields outside ranked_intervention_ids, abstain, "
                "reasoning, confidence"
            ),
            "do not provide reward values or policy scores",
        ],
    }
    messages = [dict(item) for item in request["messages"]]
    messages.append(
        {
            "role": "user",
            "content": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        }
    )
    return tuple(messages)


def complete_with_bounded_schema_repair(
    *,
    client,
    request: Mapping[str, Any],
    provider: Mapping[str, Any],
    output_dir: Path,
    config: SchemaRecoveryConfig,
    seed: int,
) -> Mapping[str, Any]:
    """Call, durably capture, validate, and at most once repair one response.

    Every returned model-content string is atomically persisted before JSON
    extraction or schema validation. Exhaustion returns an empty, fail-closed
    outcome instead of padding, substituting, or silently reusing a response.
    """

    _validate_request(request)
    _validate_public_provider(provider)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError("schema-attempt output already exists: %s" % target)
    target.mkdir(parents=True)
    original_messages = tuple(dict(item) for item in request["messages"])
    messages = original_messages
    attempt_records = []
    previous_response_text_sha256 = None
    next_error_code = None

    for attempt_index in range(int(config.maximum_schema_attempts)):
        attempt_kind = "initial" if attempt_index == 0 else "schema_repair"
        if attempt_index > 0:
            messages = build_schema_repair_messages(
                request=request,
                previous_response_text_sha256=str(previous_response_text_sha256),
                validation_error_code=str(next_error_code),
            )
        request_record = {
            "schema_version": 1,
            "protocol_id": SCHEMA_ROBUSTNESS_PROTOCOL_ID,
            "request_id": request["request_id"],
            "attempt_index": attempt_index,
            "attempt_kind": attempt_kind,
            "messages": [dict(item) for item in messages],
            "messages_sha256": _canonical_sha256(messages),
            "original_prompt_sha256": request["prompt_sha256"],
            "candidate_ids_sha256": _canonical_sha256(
                list(request["candidate_ids"])
            ),
            "maximum_budget": int(request["maximum_budget"]),
            "reward_truth_exposed": False,
            "policy_score_exposed": False,
            "api_key_exposed": False,
            "raw_invalid_response_replayed_to_model": False,
        }
        request_path = target / ("attempt-%02d-request.json" % attempt_index)
        write_json(request_path, request_record)
        try:
            completion = client.complete(
                messages=messages,
                max_tokens=int(config.max_output_tokens),
                seed=int(seed),
                timeout_seconds=float(config.timeout_seconds),
                transport_retries=int(config.transport_retries_per_attempt),
            )
        except APITransportError as error:
            failure = {
                "schema_version": 1,
                "status": "fail_closed_transport_error",
                "request_id": request["request_id"],
                "attempt_count": attempt_index + 1,
                "semantic_repair_count": max(0, attempt_index),
                "selected_intervention_ids": [],
                "oracle_selection_authorized": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "maximum_schema_attempts": int(config.maximum_schema_attempts),
                "maximum_http_transmissions": int(
                    config.maximum_http_transmissions
                ),
            }
            write_json(target / "outcome.json", failure)
            return failure

        raw_record = {
            "schema_version": 1,
            "capture_status": "captured_before_validation",
            "protocol_id": SCHEMA_ROBUSTNESS_PROTOCOL_ID,
            "request_id": request["request_id"],
            "attempt_index": attempt_index,
            "attempt_kind": attempt_kind,
            "provider": dict(provider),
            "provider_response_id": completion.provider_response_id,
            "system_fingerprint": completion.system_fingerprint,
            "provider_response_sha256": completion.response_sha256,
            "raw_response": completion.content,
            "raw_response_text_sha256": _sha256_bytes(
                completion.content.encode("utf-8")
            ),
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens,
            "transport_retries_used": completion.transport_retries_used,
            "api_key_logged": False,
        }
        raw_path = target / ("attempt-%02d-raw-response.json" % attempt_index)
        write_json(raw_path, raw_record)
        raw_artifact_sha256 = _sha256_path(raw_path)
        previous_response_text_sha256 = raw_record["raw_response_text_sha256"]

        validation_error = None
        validation_error_code = None
        validated = None
        try:
            parsed = extract_json_object(completion.content)
        except ValueError as error:
            parsed = None
            validation_error = str(error)
            validation_error_code = "json_object_extraction_failed"
        if validation_error is None:
            try:
                validated = validate_online_ranked_response(
                    parsed,
                    request["candidate_ids"],
                    int(request["maximum_budget"]),
                )
            except ValueError as error:
                validation_error = str(error)
                validation_error_code = "ranked_response_schema_invalid"

        validation_record = {
            "schema_version": 1,
            "request_id": request["request_id"],
            "attempt_index": attempt_index,
            "raw_capture_file": raw_path.name,
            "raw_capture_sha256": raw_artifact_sha256,
            "raw_was_captured_before_validation": True,
            "status": "validated" if validated is not None else "invalid",
            "validation_error_code": validation_error_code,
            "validation_error": validation_error,
            "validated": validated,
        }
        validation_path = target / ("attempt-%02d-validation.json" % attempt_index)
        write_json(validation_path, validation_record)
        attempt_records.append(
            {
                "attempt_index": attempt_index,
                "attempt_kind": attempt_kind,
                "request_file": request_path.name,
                "request_sha256": _sha256_path(request_path),
                "raw_capture_file": raw_path.name,
                "raw_capture_sha256": raw_artifact_sha256,
                "validation_file": validation_path.name,
                "validation_sha256": _sha256_path(validation_path),
                "status": validation_record["status"],
                "validation_error_code": validation_error_code,
            }
        )
        if validated is not None:
            outcome = {
                "schema_version": 1,
                "status": "validated",
                "request_id": request["request_id"],
                "attempt_count": attempt_index + 1,
                "semantic_repair_count": max(0, attempt_index),
                "selected_intervention_ids": list(
                    validated["selected_intervention_ids"]
                ),
                "validated": validated,
                "oracle_selection_authorized": True,
                "attempts": attempt_records,
                "maximum_schema_attempts": int(config.maximum_schema_attempts),
                "maximum_http_transmissions": int(
                    config.maximum_http_transmissions
                ),
                "raw_capture_complete": True,
            }
            write_json(target / "outcome.json", outcome)
            return outcome
        next_error_code = validation_error_code

    outcome = {
        "schema_version": 1,
        "status": "fail_closed_schema_exhausted",
        "request_id": request["request_id"],
        "attempt_count": int(config.maximum_schema_attempts),
        "semantic_repair_count": int(config.maximum_semantic_repairs),
        "selected_intervention_ids": [],
        "validated": None,
        "oracle_selection_authorized": False,
        "attempts": attempt_records,
        "maximum_schema_attempts": int(config.maximum_schema_attempts),
        "maximum_http_transmissions": int(config.maximum_http_transmissions),
        "raw_capture_complete": len(attempt_records)
        == int(config.maximum_schema_attempts),
        "fallback_selection_used": False,
        "cached_response_substituted": False,
    }
    write_json(target / "outcome.json", outcome)
    return outcome
