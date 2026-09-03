#!/usr/bin/env python3
"""Audit fixed-pool acquisition headroom without new oracle or LLM calls."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from reproduction.framework.io import git_identity, write_json
from reproduction.scicf.acquisition.metrics import acquisition_metrics
from reproduction.scicf.core.config import load_scicf_config


AUDIT_VERSION = "scicf-gate1a-headroom-v1"
SUMMARY_FIELDS = (
    "positive_rate",
    "gain_mean",
    "gain_stddev",
    "gain_range",
    "oracle_best_gain_at_b",
    "expected_random_best_gain_at_b",
    "observed_random_best_gain_at_b",
    "oracle_minus_expected_random_best_gain_at_b",
    "oracle_ndcg_at_b",
    "expected_random_ndcg_at_b",
    "observed_random_ndcg_at_b",
    "oracle_minus_expected_random_ndcg_at_b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scicf-config", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def terminal_valid(trajectory: Mapping[str, Any]) -> bool:
    value = trajectory.get("terminal_scientific_object")
    return isinstance(value, str) and value.strip() not in {"", "None", "null"}


def _dcg(relevances: Sequence[float]) -> float:
    return sum(
        (2.0 ** relevance - 1.0) / math.log2(index + 2.0)
        for index, relevance in enumerate(relevances)
    )


def expected_random_best_gain(gains: Sequence[float], budget: int) -> float:
    """Exact expected maximum for an unordered sample without replacement."""

    if not gains or budget < 1 or budget > len(gains):
        raise ValueError("invalid gains or random acquisition budget")
    ordered = sorted(float(value) for value in gains)
    denominator = float(math.comb(len(ordered), budget))
    numerator = 0.0
    for index, value in enumerate(ordered):
        if index >= budget - 1:
            numerator += value * math.comb(index, budget - 1)
    return numerator / denominator


def expected_random_ndcg(gains: Sequence[float], budget: int) -> float:
    """Exact expectation for a uniformly random ordered top-B selection."""

    if not gains or budget < 1 or budget > len(gains):
        raise ValueError("invalid gains or random acquisition budget")
    relevances = [max(float(value), 0.0) for value in gains]
    ideal = sorted(relevances, reverse=True)[:budget]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg <= 0.0:
        return 0.0
    mean_numerator = sum(2.0 ** value - 1.0 for value in relevances) / len(
        relevances
    )
    discount_sum = sum(1.0 / math.log2(index + 2.0) for index in range(budget))
    return mean_numerator * discount_sum / ideal_dcg


def _bootstrap_mean_interval(
    values: Sequence[float], resamples: int, confidence: float, seed: int
) -> Tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    generator = random.Random(seed)
    size = len(values)
    estimates = []
    for _ in range(resamples):
        estimates.append(
            sum(float(values[generator.randrange(size)]) for _ in range(size)) / size
        )
    estimates.sort()
    alpha = 1.0 - confidence
    lower_index = max(0, int(math.floor((alpha / 2.0) * (resamples - 1))))
    upper_index = min(
        resamples - 1,
        int(math.ceil((1.0 - alpha / 2.0) * (resamples - 1))),
    )
    return float(estimates[lower_index]), float(estimates[upper_index])


def _metric_summary(
    values: Sequence[float], resamples: int, confidence: float, seed: int
) -> Dict[str, Any]:
    lower, upper = _bootstrap_mean_interval(values, resamples, confidence, seed)
    return {
        "mean": float(sum(values) / len(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "bootstrap_resamples": resamples,
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    resamples: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    if not rows:
        return {"status": "not-observed", "trajectories": 0}
    metrics = {}
    for index, field in enumerate(SUMMARY_FIELDS):
        metrics[field] = _metric_summary(
            [float(row[field]) for row in rows],
            resamples,
            confidence,
            seed + index,
        )
    best_gain_lower = metrics[
        "oracle_minus_expected_random_best_gain_at_b"
    ]["ci_lower"]
    ndcg_lower = metrics["oracle_minus_expected_random_ndcg_at_b"]["ci_lower"]
    return {
        "status": "complete",
        "trajectories": len(rows),
        "valid_trajectories": sum(bool(row["terminal_valid"]) for row in rows),
        "invalid_trajectories": sum(
            not bool(row["terminal_valid"]) for row in rows
        ),
        "no_positive_candidate_trajectories": sum(
            int(row["positive_count"]) == 0 for row in rows
        ),
        "metrics": metrics,
        "headroom_rule": {
            "best_gain_ci_lower_strictly_positive": best_gain_lower > 0.0,
            "ndcg_ci_lower_strictly_positive": ndcg_lower > 0.0,
            "headroom_detected": best_gain_lower > 0.0 and ndcg_lower > 0.0,
            "minimum_effect_size_predeclared": False,
        },
    }


def _iter_stage_reports(paths: Iterable[Path]) -> Iterable[Mapping[str, Any]]:
    for path in paths:
        report = json.loads(path.resolve().read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ValueError("stage report is not complete: {}".format(path))
        yield report


def audit(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 1A headroom audit must run through Slurm")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("headroom audit output already exists")
    config = load_scicf_config(args.scicf_config.resolve())
    source = git_identity(args.repo_root.resolve())
    if source.get("dirty") is not False:
        raise RuntimeError("Gate 1A headroom audit requires a clean Git worktree")

    budget = int(config["acquisition"]["budget"])
    expected_pool_size = int(config["acquisition"]["pool_size"])
    expected_stages = list(config["gate1"]["checkpoint_stages"])
    reports = list(_iter_stage_reports(args.stage_report))
    observed_stages = [str(report["stage"]) for report in reports]
    if len(observed_stages) != len(set(observed_stages)):
        raise ValueError("duplicate headroom stage report")
    if set(observed_stages) != set(expected_stages):
        raise ValueError("headroom stage set does not match configuration")

    rows: List[Dict[str, Any]] = []
    report_sources = {}
    for report in reports:
        stage = str(report["stage"])
        report_sources[stage] = {
            "checkpoint": report.get("checkpoint"),
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "source": report.get("source"),
            "slurm_job_id": report.get("slurm_job_id"),
        }
        for item in report["trajectories"]:
            candidates = item["pool"]["candidates"]
            candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
            if len(candidate_ids) != expected_pool_size:
                raise ValueError("candidate pool size mismatch")
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("candidate pool contains duplicate IDs")
            gain_table = item["verified_gains"]
            if set(gain_table) != set(candidate_ids):
                raise ValueError("verified gain table does not match candidate pool")
            gains = [float(gain_table[identifier]) for identifier in candidate_ids]
            if any(not math.isfinite(value) for value in gains):
                raise ValueError("verified gains must be finite")

            oracle_ids = sorted(
                candidate_ids,
                key=lambda identifier: (float(gain_table[identifier]), identifier),
                reverse=True,
            )[:budget]
            oracle_metrics = acquisition_metrics(gain_table, oracle_ids, budget)
            random_ids = item["strategies"]["random"]["selected_ids"]
            observed_random = acquisition_metrics(gain_table, random_ids, budget)
            expected_best = expected_random_best_gain(gains, budget)
            expected_ndcg = expected_random_ndcg(gains, budget)
            global_best = float(max(gains))
            positive_count = sum(value > 0.0 for value in gains)
            row = {
                "stage": stage,
                "trajectory_id": item["trajectory"]["trajectory_id"],
                "terminal_valid": terminal_valid(item["trajectory"]),
                "candidate_count": len(gains),
                "positive_count": positive_count,
                "positive_rate": positive_count / len(gains),
                "gain_mean": float(sum(gains) / len(gains)),
                "gain_stddev": float(statistics.pstdev(gains)),
                "gain_range": float(max(gains) - min(gains)),
                "oracle_best_gain_at_b": global_best,
                "expected_random_best_gain_at_b": expected_best,
                "observed_random_best_gain_at_b": float(
                    observed_random["best_gain_at_b"]
                ),
                "oracle_minus_expected_random_best_gain_at_b": global_best
                - expected_best,
                "oracle_ndcg_at_b": float(oracle_metrics["ndcg_at_b"]),
                "expected_random_ndcg_at_b": expected_ndcg,
                "observed_random_ndcg_at_b": float(observed_random["ndcg_at_b"]),
                "oracle_minus_expected_random_ndcg_at_b": float(
                    oracle_metrics["ndcg_at_b"]
                )
                - expected_ndcg,
            }
            rows.append(row)

    resamples = int(config["gate1"]["bootstrap_resamples"])
    confidence = float(config["gate1"]["confidence_level"])
    stages = {
        stage: summarize_rows(
            [row for row in rows if row["stage"] == stage],
            resamples,
            confidence,
            41000 + index * 100,
        )
        for index, stage in enumerate(expected_stages)
    }
    validity = {
        label: summarize_rows(
            [row for row in rows if bool(row["terminal_valid"]) == is_valid],
            resamples,
            confidence,
            42000 + index * 100,
        )
        for index, (label, is_valid) in enumerate(
            (("invalid", False), ("valid", True))
        )
    }
    stage_by_validity = {}
    for stage_index, stage in enumerate(expected_stages):
        for validity_index, (label, is_valid) in enumerate(
            (("invalid", False), ("valid", True))
        ):
            key = "{}-{}".format(stage, label)
            stage_by_validity[key] = summarize_rows(
                [
                    row
                    for row in rows
                    if row["stage"] == stage
                    and bool(row["terminal_valid"]) == is_valid
                ],
                resamples,
                confidence,
                43000 + stage_index * 100 + validity_index * 10,
            )

    stage_headroom = {
        stage: bool(summary["headroom_rule"]["headroom_detected"])
        for stage, summary in stages.items()
    }
    result = {
        "schema_version": 1,
        "audit": AUDIT_VERSION,
        "status": "complete",
        "source": source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "inputs": {
            "reports": report_sources,
            "pool_size": expected_pool_size,
            "budget": budget,
            "bootstrap_resamples": resamples,
            "confidence": confidence,
        },
        "definitions": {
            "terminal_valid": "terminal_scientific_object is a nonempty string",
            "positive_rate": "fraction of all fixed-pool candidate mean deltas greater than zero",
            "oracle_best_gain_at_b": "maximum verified mean delta in the fixed pool",
            "expected_random_best_gain_at_b": "exact expected maximum under uniform sampling of B candidates without replacement",
            "expected_random_ndcg_at_b": "exact expected NDCG for a uniformly random ordered top-B selection",
        },
        "rows": rows,
        "summaries": {
            "stages": stages,
            "validity": validity,
            "stage_by_validity": stage_by_validity,
        },
        "diagnostic": {
            "stage_headroom_detected": stage_headroom,
            "all_stages_headroom_detected": all(stage_headroom.values()),
            "gate1b_learnability_probe_eligible": all(stage_headroom.values()),
            "llm_gate1c_authorized": False,
            "pairwise_refinement_authorized": False,
            "minimum_effect_size_predeclared": False,
        },
        "claim_boundary": (
            "This read-only Gate 1A diagnostic tests candidate-pool acquisition "
            "headroom only. It does not test feature learnability, LLM acquisition, "
            "online PPO improvement, or scientific validity."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(
        json.dumps(
            {
                "event": "gate1a_headroom_complete",
                "output": str(output),
                "trajectories": len(rows),
                **result["diagnostic"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()
