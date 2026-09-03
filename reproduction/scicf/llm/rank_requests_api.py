#!/usr/bin/env python3
"""Rank fixed Gate 1 pools through a private OpenAI-compatible API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from reproduction.framework.io import append_jsonl, git_identity, write_json
from reproduction.scicf.llm.api_client import APISettings, OpenAICompatibleClient
from reproduction.scicf.llm.schema import validate_ranked_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--requests", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--validation-retries", type=int, default=2)
    parser.add_argument("--transport-retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--expected-request-count", type=int, required=True)
    parser.add_argument("--expected-budget", type=int, required=True)
    parser.add_argument("--expected-pool-size", type=int, required=True)
    parser.add_argument("--expected-prompt-version", required=True)
    parser.add_argument("--expected-candidate-presentation", required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def iter_requests(paths: Iterable[Path]) -> Iterable[Mapping[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def extract_json(text: str) -> Any:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            return value
        except ValueError:
            continue
    raise ValueError("model output contains no valid JSON object")


def cache_key(
    request: Mapping[str, Any],
    provider: Mapping[str, Any],
    decoding: Mapping[str, Any],
) -> str:
    identity = {
        "prompt_sha256": request["prompt_sha256"],
        "provider": provider,
        "decoding": decoding,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _request_seed(base_seed: int, request_id: str) -> int:
    offset = int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16)
    return int((base_seed + offset) % 2147483647)


def _known_token_total(records: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(sum(int(record[key]) for record in records if record.get(key) is not None))


def run(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("LLM API acquisition must run through Slurm")
    if args.max_output_tokens < 1:
        raise ValueError("max-output-tokens must be positive")
    if args.validation_retries < 0 or args.transport_retries < 0:
        raise ValueError("retry counts cannot be negative")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")

    settings = APISettings.from_private_file(args.credentials_file)
    provider = settings.public_identity()
    source = git_identity(args.repo_root.resolve())
    if source.get("dirty") is not False:
        raise RuntimeError("formal LLM API acquisition requires a clean Git worktree")
    if args.output_root.exists():
        raise FileExistsError("output root already exists: {}".format(args.output_root))
    requests = list(iter_requests(args.requests))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        requests = requests[: args.limit]
    if not requests:
        raise ValueError("no LLM requests were supplied")
    if len(requests) != args.expected_request_count:
        raise ValueError("LLM request count does not match the declared run")
    request_ids = [str(item["request_id"]) for item in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("LLM request IDs must be unique")
    for request in requests:
        candidate_ids = list(request["candidate_ids"])
        presented_ids = list(request.get("presented_candidate_ids", []))
        if int(request["budget"]) != args.expected_budget:
            raise ValueError("LLM request budget does not match the declared run")
        if len(candidate_ids) != args.expected_pool_size:
            raise ValueError("LLM request pool size does not match the declared run")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("LLM candidate IDs must be unique")
        if len(presented_ids) != len(set(presented_ids)) or set(presented_ids) != set(
            candidate_ids
        ):
            raise ValueError("LLM presented candidate IDs must equal the fixed pool")

    decoding = {
        "generation_protocol": "openai-compatible-json-exact-budget-v1",
        "temperature": 0.0,
        "max_output_tokens": args.max_output_tokens,
        "validation_retries": args.validation_retries,
        "transport_retries": args.transport_retries,
        "timeout_seconds": args.timeout_seconds,
        "seed_base": args.seed,
        "include_seed": settings.include_seed,
        "json_mode": settings.json_mode,
        "max_tokens_field": settings.max_tokens_field,
        "thinking": settings.thinking,
    }
    prompt_versions = sorted({str(item["prompt_version"]) for item in requests})
    candidate_presentations = sorted(
        {str(item.get("candidate_presentation", "unspecified")) for item in requests}
    )
    if prompt_versions != [args.expected_prompt_version]:
        raise ValueError("LLM request prompt version does not match the declared run")
    if candidate_presentations != [args.expected_candidate_presentation]:
        raise ValueError(
            "LLM candidate presentation does not match the declared run"
        )
    args.output_root.mkdir(parents=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_root / "run-intent.json",
        {
            "schema_version": 1,
            "status": "declared-before-first-api-request",
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "provider": provider,
            "decoding": decoding,
            "prompt_versions": prompt_versions,
            "candidate_presentations": candidate_presentations,
            "request_ids": request_ids,
            "credentials": {
                "source": "private-file",
                "api_key_logged": False,
                "credentials_path_logged": False,
            },
        },
    )

    client = OpenAICompatibleClient(settings)
    response_path = args.output_root / "responses.jsonl"
    completed_records: List[Mapping[str, Any]] = []
    cache_hits = 0
    for request in requests:
        key = cache_key(request, provider, decoding)
        cache_path = args.cache_root / "{}.json".format(key)
        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            if record.get("cache_key") != key or record.get("provider") != provider:
                raise ValueError("cached API response identity mismatch")
            validated = validate_ranked_response(
                record["parsed"], request["candidate_ids"], int(request["budget"])
            )
            record["validated"] = validated
            record["cache_hit"] = True
            cache_hits += 1
        else:
            messages: List[Dict[str, str]] = [dict(item) for item in request["messages"]]
            errors = []
            raw_attempts = []
            api_attempts = []
            reported_input_tokens = 0
            reported_output_tokens = 0
            reported_total_tokens = 0
            usage_missing_attempts = 0
            record = None
            seed = _request_seed(args.seed, str(request["request_id"]))
            for attempt in range(args.validation_retries + 1):
                completion = client.complete(
                    messages=messages,
                    max_tokens=args.max_output_tokens,
                    seed=seed,
                    timeout_seconds=args.timeout_seconds,
                    transport_retries=args.transport_retries,
                )
                raw = completion.content
                raw_attempts.append(raw)
                api_attempts.append(
                    {
                        "provider_response_id": completion.provider_response_id,
                        "system_fingerprint": completion.system_fingerprint,
                        "provider_response_sha256": completion.response_sha256,
                        "transport_retries_used": completion.transport_retries_used,
                        "prompt_tokens": completion.prompt_tokens,
                        "completion_tokens": completion.completion_tokens,
                        "total_tokens": completion.total_tokens,
                    }
                )
                if (
                    completion.prompt_tokens is None
                    or completion.completion_tokens is None
                    or completion.total_tokens is None
                ):
                    usage_missing_attempts += 1
                if completion.prompt_tokens is not None:
                    reported_input_tokens += completion.prompt_tokens
                if completion.completion_tokens is not None:
                    reported_output_tokens += completion.completion_tokens
                if completion.total_tokens is not None:
                    reported_total_tokens += completion.total_tokens
                try:
                    parsed = extract_json(raw)
                    validated = validate_ranked_response(
                        parsed, request["candidate_ids"], int(request["budget"])
                    )
                    record = {
                        "schema_version": 1,
                        "request_id": request["request_id"],
                        "pool_id": request["pool_id"],
                        "prompt_sha256": request["prompt_sha256"],
                        "prompt_version": request["prompt_version"],
                        "provider": provider,
                        "model_id": settings.model_id,
                        "revision": settings.model_revision,
                        "decoding": decoding,
                        "request_seed": seed if settings.include_seed else None,
                        "attempt": attempt,
                        "raw_response": raw,
                        "parsed": parsed,
                        "validated": validated,
                        "reported_input_tokens": reported_input_tokens,
                        "reported_output_tokens": reported_output_tokens,
                        "reported_total_tokens": reported_total_tokens,
                        "usage_missing_attempts": usage_missing_attempts,
                        "api_calls": len(api_attempts),
                        "api_attempts": api_attempts,
                        "provider_response_id": completion.provider_response_id,
                        "system_fingerprint": completion.system_fingerprint,
                        "provider_response_sha256": completion.response_sha256,
                        "transport_retries_used": completion.transport_retries_used,
                        "cache_key": key,
                        "cache_hit": False,
                        "validation_errors": errors,
                    }
                    break
                except ValueError as error:
                    errors.append(str(error))
                    messages = [dict(item) for item in request["messages"]]
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Repair instruction {}: the prior response was invalid because "
                                "{}. Start with '{{'. Return one compact JSON object containing "
                                "exactly {} unique ranked_intervention_ids from this exact list, "
                                "then end with '}}': {}"
                            ).format(
                                attempt + 1,
                                error,
                                request["budget"],
                                json.dumps(
                                    request.get(
                                        "presented_candidate_ids",
                                        request["candidate_ids"],
                                    )
                                ),
                            ),
                        }
                    )
            if record is None:
                write_json(
                    args.output_root
                    / "failed-{}.json".format(
                        hashlib.sha256(
                            str(request["request_id"]).encode("utf-8")
                        ).hexdigest()[:16]
                    ),
                    {
                        "request_id": request["request_id"],
                        "prompt_sha256": request["prompt_sha256"],
                        "provider": provider,
                        "errors": errors,
                        "raw_attempts": raw_attempts,
                        "decoding": decoding,
                    },
                )
                raise RuntimeError(
                    "LLM response for {} failed validation after retries: {}".format(
                        request["request_id"], errors
                    )
                )
            write_json(cache_path, record)
        append_jsonl(response_path, record)
        completed_records.append(record)
        print(
            json.dumps(
                {
                    "event": "llm_api_request_complete",
                    "request_id": request["request_id"],
                    "cache_hit": record["cache_hit"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    write_json(
        args.output_root / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source": source,
            "provider": provider,
            "model_id": settings.model_id,
            "revision": settings.model_revision,
            "decoding": decoding,
            "prompt_versions": prompt_versions,
            "candidate_presentations": candidate_presentations,
            "requests": len(requests),
            "cache_hits": cache_hits,
            "api_calls_this_run": int(
                sum(
                    int(record.get("api_calls", 0))
                    for record in completed_records
                    if not record.get("cache_hit")
                )
            ),
            "origin_api_calls": int(
                sum(int(record.get("api_calls", 0)) for record in completed_records)
            ),
            "origin_reported_input_tokens": _known_token_total(
                completed_records, "reported_input_tokens"
            ),
            "origin_reported_output_tokens": _known_token_total(
                completed_records, "reported_output_tokens"
            ),
            "origin_reported_total_tokens": _known_token_total(
                completed_records, "reported_total_tokens"
            ),
            "origin_usage_missing_attempts": int(
                sum(
                    int(record.get("usage_missing_attempts", 0))
                    for record in completed_records
                )
            ),
            "responses": str(response_path.resolve()),
            "credentials": {
                "source": "private-file",
                "api_key_logged": False,
                "credentials_path_logged": False,
            },
        },
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
