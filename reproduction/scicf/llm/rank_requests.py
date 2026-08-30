#!/usr/bin/env python3
"""Rank fixed Gate 1 candidate pools with a pinned local Qwen model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproduction.framework.io import append_jsonl, write_json
from reproduction.scicf.llm.schema import validate_ranked_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2023)
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
    request: Mapping[str, Any], model_id: str, revision: str, decoding: Mapping[str, Any]
) -> str:
    identity = {
        "prompt_sha256": request["prompt_sha256"],
        "model_id": model_id,
        "revision": revision,
        "decoding": decoding,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("LLM acquisition must run through Slurm")
    if args.output_root.exists():
        raise FileExistsError("output root already exists: {}".format(args.output_root))
    args.output_root.mkdir(parents=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    decoding = {
        "generation_protocol": "greedy-attention-mask-exact-budget-v2",
        "do_sample": False,
        "attention_mask": "all-ones",
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_new_tokens": args.max_new_tokens,
        "max_retries": args.max_retries,
    }

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    response_path = args.output_root / "responses.jsonl"
    total_input_tokens = 0
    total_output_tokens = 0
    request_count = 0
    cache_hits = 0
    for request in iter_requests(args.requests):
        request_count += 1
        key = cache_key(request, args.model_id, args.revision, decoding)
        cache_path = args.cache_root / "{}.json".format(key)
        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
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
            record = None
            for attempt in range(args.max_retries + 1):
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)
                attention_mask = torch.ones_like(input_ids, device=model.device)
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = output_ids[0, input_ids.shape[1] :]
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                raw_attempts.append(raw)
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
                        "model_id": args.model_id,
                        "revision": args.revision,
                        "decoding": decoding,
                        "attempt": attempt,
                        "raw_response": raw,
                        "parsed": parsed,
                        "validated": validated,
                        "input_tokens": int(input_ids.shape[1]),
                        "output_tokens": int(generated.shape[0]),
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
                        hashlib.sha256(request["request_id"].encode("utf-8")).hexdigest()[:16]
                    ),
                    {
                        "request_id": request["request_id"],
                        "prompt_sha256": request["prompt_sha256"],
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
        total_input_tokens += int(record.get("input_tokens", 0))
        total_output_tokens += int(record.get("output_tokens", 0))
        append_jsonl(response_path, record)
        print(
            json.dumps(
                {
                    "event": "llm_request_complete",
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
            "model_id": args.model_id,
            "revision": args.revision,
            "model_path": str(args.model_path),
            "decoding": decoding,
            "requests": request_count,
            "cache_hits": cache_hits,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "responses": str(response_path),
        },
    )


if __name__ == "__main__":
    main()
