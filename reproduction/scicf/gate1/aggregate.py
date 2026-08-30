#!/usr/bin/env python3
"""Aggregate offline Gate 1 metrics and apply the pre-declared decision rule."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from reproduction.framework.io import git_identity, write_json
from reproduction.scicf.acquisition.metrics import (
    METRIC_DEFINITIONS,
    acquisition_metrics,
)
from reproduction.scicf.core.config import load_scicf_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scicf-config", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, action="append", required=True)
    parser.add_argument("--llm-responses", type=Path, required=True)
    parser.add_argument("--llm-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def interval(values: Sequence[float], confidence: float) -> Tuple[float, float]:
    alpha = 1.0 - confidence
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def bootstrap_mean(
    values: Sequence[float], resamples: int, confidence: float, seed: int
) -> Dict[str, Any]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(array), size=(resamples, len(array)))
    estimates = array[indices].mean(axis=1)
    lower, upper = interval(estimates, confidence)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "resamples": resamples,
    }


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    resamples: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equal non-empty samples")
    differences = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    result = bootstrap_mean(differences.tolist(), resamples, confidence, seed)
    result["direction"] = "left_minus_right"
    return result


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 1 aggregation must run through Slurm")
    if args.output.exists():
        raise FileExistsError("Gate 1 aggregate output already exists")
    config = load_scicf_config(args.scicf_config.resolve())
    aggregate_source = git_identity(args.repo_root.resolve())
    if config["provenance"].get("require_clean_git") and aggregate_source.get(
        "dirty"
    ) is not False:
        raise RuntimeError("Gate 1 aggregation requires a clean Git worktree")
    llm_execution = None
    if args.llm_manifest is not None:
        llm_execution = json.loads(
            args.llm_manifest.resolve().read_text(encoding="utf-8")
        )
        if llm_execution.get("status") != "complete":
            raise ValueError("LLM execution manifest is not complete")
        if llm_execution.get("source", {}).get("commit") != aggregate_source.get(
            "commit"
        ):
            raise ValueError("LLM ranking and aggregation source commits differ")
        if llm_execution.get("source", {}).get("dirty") is not False:
            raise ValueError("LLM execution manifest records a dirty source tree")
        if Path(llm_execution.get("responses", "")).resolve() != args.llm_responses.resolve():
            raise ValueError("LLM manifest response path mismatch")
        provider = llm_execution.get("provider")
        if not isinstance(provider, dict):
            raise ValueError("LLM execution manifest is missing provider identity")
        if provider.get("type") != config["llm"].get("provider"):
            raise ValueError("LLM provider type does not match Gate 1 config")
        if llm_execution.get("prompt_versions") != [config["llm"]["prompt_version"]]:
            raise ValueError("LLM prompt version does not match Gate 1 config")
        config = copy.deepcopy(config)
        config["llm"]["model_id"] = llm_execution.get("model_id")
        config["llm"]["revision"] = llm_execution.get("revision")
        config["llm"]["provider_identity"] = provider
        if not config["llm"]["model_id"] or not config["llm"]["revision"]:
            raise ValueError("LLM manifest must resolve model ID and revision")
    responses = {}
    for record in read_jsonl(args.llm_responses.resolve()):
        request_id = record["request_id"]
        if request_id in responses:
            raise ValueError("duplicate LLM response: {}".format(request_id))
        if llm_execution is not None:
            if record.get("provider") != llm_execution["provider"]:
                raise ValueError("LLM response provider identity mismatch")
            if record.get("model_id") != llm_execution["model_id"]:
                raise ValueError("LLM response model identity mismatch")
            if record.get("revision") != llm_execution["revision"]:
                raise ValueError("LLM response model revision mismatch")
        responses[request_id] = record

    rows = []
    expected_stages = set(config["gate1"]["checkpoint_stages"])
    observed_stages = set()
    for report_path in args.stage_report:
        report = json.loads(report_path.resolve().read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ValueError("stage report is not complete: {}".format(report_path))
        stage = report["stage"]
        if stage in observed_stages:
            raise ValueError("duplicate stage report: {}".format(stage))
        observed_stages.add(stage)
        for trajectory in report["trajectories"]:
            request_id = trajectory["llm_request"]["request_id"]
            response = responses.get(request_id)
            if response is None:
                raise ValueError("missing LLM response: {}".format(request_id))
            if response["pool_id"] != trajectory["pool"]["pool_id"]:
                raise ValueError("LLM response pool identity mismatch")
            gain_table = trajectory["verified_gains"]
            selections = {
                name: value["selected_ids"]
                for name, value in trajectory["strategies"].items()
            }
            selections["llm"] = response["validated"]["selected_intervention_ids"]
            budget = int(config["acquisition"]["budget"])
            strategy_metrics = {
                name: acquisition_metrics(gain_table, identifiers, budget)
                for name, identifiers in selections.items()
            }
            rows.append(
                {
                    "stage": stage,
                    "trajectory_id": trajectory["trajectory"]["trajectory_id"],
                    "pool_id": trajectory["pool"]["pool_id"],
                    "selections": selections,
                    "metrics": strategy_metrics,
                    "selected_atomic_oracle_calls": {
                        name: sum(
                            int(trajectory["oracle_calls_by_candidate"][identifier])
                            for identifier in identifiers
                        )
                        for name, identifiers in selections.items()
                    },
                }
            )
    if observed_stages != expected_stages:
        raise ValueError(
            "stage set mismatch: expected {}, observed {}".format(
                sorted(expected_stages), sorted(observed_stages)
            )
        )
    expected_requests = len(rows)
    if len(responses) != expected_requests:
        raise ValueError(
            "LLM response count {} does not match trajectory count {}".format(
                len(responses), expected_requests
            )
        )
    if llm_execution is not None and llm_execution.get("requests") != expected_requests:
        raise ValueError("LLM manifest request count mismatch")

    resamples = int(config["gate1"]["bootstrap_resamples"])
    confidence = float(config["gate1"]["confidence_level"])
    summaries: Dict[str, Any] = {}
    strategies = ("random", "policy_probability", "chemistry_heuristic", "llm")
    for stage_index, stage in enumerate(config["gate1"]["checkpoint_stages"]):
        stage_rows = [row for row in rows if row["stage"] == stage]
        summaries[stage] = {}
        for strategy_index, strategy in enumerate(strategies):
            summaries[stage][strategy] = {}
            for metric_index, metric in enumerate(config["gate1"]["metrics"]):
                values = [row["metrics"][strategy][metric] for row in stage_rows]
                summaries[stage][strategy][metric] = bootstrap_mean(
                    values,
                    resamples=resamples,
                    confidence=confidence,
                    seed=10000 + stage_index * 100 + strategy_index * 10 + metric_index,
                )
            oracle_values = [
                row["selected_atomic_oracle_calls"][strategy] for row in stage_rows
            ]
            summaries[stage][strategy]["selected_atomic_oracle_calls"] = {
                "total": int(sum(oracle_values)),
                "mean": float(np.mean(oracle_values)),
                "minimum": int(min(oracle_values)),
                "maximum": int(max(oracle_values)),
            }

    primary_metric = config["gate1"]["primary_metric"]
    required_comparators = config["gate1"]["decision_rule"][
        "required_comparators"
    ]
    comparisons = {}
    successful_stages = []
    for stage_index, stage in enumerate(config["gate1"]["checkpoint_stages"]):
        stage_rows = [row for row in rows if row["stage"] == stage]
        llm_values = [row["metrics"]["llm"][primary_metric] for row in stage_rows]
        comparisons[stage] = {}
        stage_success = True
        for comparator_index, comparator in enumerate(required_comparators):
            comparator_values = [
                row["metrics"][comparator][primary_metric] for row in stage_rows
            ]
            result = paired_bootstrap_difference(
                llm_values,
                comparator_values,
                resamples=resamples,
                confidence=confidence,
                seed=30000 + stage_index * 100 + comparator_index,
            )
            result["passes"] = result["ci_lower"] > 0.0
            comparisons[stage][comparator] = result
            stage_success = stage_success and result["passes"]
        comparisons[stage]["stage_success"] = stage_success
        if stage_success:
            successful_stages.append(stage)

    minimum_stages = int(
        config["gate1"]["decision_rule"]["minimum_successful_stages"]
    )
    gate_passed = len(successful_stages) >= minimum_stages
    result = {
        "schema_version": 1,
        "gate": "offline-acquisition-gate1",
        "status": "passed" if gate_passed else "failed",
        "source": aggregate_source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "configuration": config,
        "llm_execution": llm_execution,
        "metric_definitions": METRIC_DEFINITIONS,
        "rows": rows,
        "summaries": summaries,
        "primary_comparisons": comparisons,
        "decision": {
            "primary_metric": primary_metric,
            "required_comparators": required_comparators,
            "minimum_successful_stages": minimum_stages,
            "successful_stages": successful_stages,
            "passed": gate_passed,
            "pairwise_refinement_authorized": gate_passed,
        },
        "claim_boundary": (
            "This offline acquisition gate does not establish online PPO improvement, "
            "independent-oracle robustness, cross-domain generality, or wet-lab validity."
        ),
    }
    write_json(args.output.resolve(), result)
    print(json.dumps({"event": "gate1_complete", **result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
