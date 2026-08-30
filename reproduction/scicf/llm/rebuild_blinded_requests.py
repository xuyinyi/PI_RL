#!/usr/bin/env python3
"""Rebuild gain-blind Gate 1 prompts with deterministic candidate shuffling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from reproduction.framework.io import append_jsonl
from reproduction.scicf.llm.prompt import (
    PRESENTATION_PROTOCOL,
    PROMPT_VERSION,
    acquisition_system_prompt,
    blind_candidate_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("prompt rebuilding must run through Slurm")
    if args.output.exists():
        raise FileExistsError("blinded request file already exists: {}".format(args.output))
    count = 0
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            request = json.loads(line)
            context = json.loads(request["messages"][1]["content"])
            blinded = blind_candidate_rows(
                request["request_id"],
                request["pool_id"],
                context["candidate_interventions"],
            )
            context["candidate_interventions"] = blinded
            context["required_output"]["ranked_intervention_ids"] = (
                "exactly {} candidate ID strings in best-first order".format(
                    request["budget"]
                )
            )
            messages = [
                {
                    "role": "system",
                    "content": acquisition_system_prompt(int(request["budget"])),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        context, sort_keys=True, separators=(",", ":")
                    ),
                },
            ]
            prompt_bytes = json.dumps(
                messages, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            rebuilt = dict(request)
            rebuilt.update(
                {
                    "prompt_version": PROMPT_VERSION,
                    "candidate_presentation": PRESENTATION_PROTOCOL,
                    "presented_candidate_ids": [
                        row["intervention_id"] for row in blinded
                    ],
                    "messages": messages,
                    "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                    "verified_gain_exposed": False,
                }
            )
            append_jsonl(args.output, rebuilt)
            count += 1
    print(
        json.dumps(
            {
                "event": "blinded_requests_ready",
                "input": str(args.input),
                "output": str(args.output),
                "requests": count,
                "prompt_version": PROMPT_VERSION,
                "candidate_presentation": PRESENTATION_PROTOCOL,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
