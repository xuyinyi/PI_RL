#!/usr/bin/env python3
"""Run the single sealed Gate 1B.1 structure-isolated test evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from reproduction.framework.io import git_identity, write_json
from reproduction.scicf.gate1.gate1b1 import (
    STAGES,
    attach_predictions,
    evaluate_early,
    evaluate_late,
    evaluate_middle,
    file_sha256,
    headroom_fraction,
    load_config,
    load_split_reports,
    mean,
    structure_keys,
    trajectory_cross_timestep_fraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _bootstrap_interval(
    values: Sequence[float], resamples: int, confidence: float, seed: int
) -> Tuple[float, float]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        estimates.append(
            sum(float(values[generator.randrange(len(values))]) for _ in values)
            / float(len(values))
        )
    estimates.sort()
    alpha = 1.0 - confidence
    lower = max(0, int(math.floor((alpha / 2.0) * (resamples - 1))))
    upper = min(
        resamples - 1,
        int(math.ceil((1.0 - alpha / 2.0) * (resamples - 1))),
    )
    return float(estimates[lower]), float(estimates[upper])


def _permuted_rows(
    rows: Sequence[Mapping[str, Any]], target: str, generator: random.Random
) -> list:
    result = [dict(row) for row in rows]
    groups = {}
    for index, row in enumerate(result):
        groups.setdefault(str(row["trajectory_id"]), []).append(index)
    for indices in groups.values():
        values = [result[index][target] for index in indices]
        generator.shuffle(values)
        for index, value in zip(indices, values):
            result[index][target] = value
    return result


def _test_evidence(
    differences: Sequence[float],
    headrooms: Sequence[float],
    p_value: float,
    config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    rule = config["test_rule"]
    lower, upper = _bootstrap_interval(
        differences,
        int(rule["bootstrap_resamples"]),
        float(rule["confidence_level"]),
        seed,
    )
    difference = mean(differences)
    captured = headroom_fraction(differences, headrooms)
    checks = {
        "mean_difference_positive": difference > 0.0,
        "bootstrap_ci_lower_positive": lower > 0.0,
        "permutation_p_at_most_alpha": p_value <= float(rule["one_sided_alpha"]),
        "minimum_headroom_fraction_captured": captured
        >= float(rule["minimum_oracle_headroom_fraction_captured"]),
    }
    return {
        "mean_difference": difference,
        "bootstrap_ci_95": [lower, upper],
        "permutation_p_one_sided": p_value,
        "oracle_headroom_fraction_captured": captured,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _permutation_p_values(
    early_rows: Sequence[Mapping[str, Any]],
    middle_rows: Sequence[Mapping[str, Any]],
    budget: int,
    validity_threshold: float,
    resamples: int,
    seed: int,
) -> Tuple[float, float, float]:
    observed_early = mean(
        [float(row["descriptor_minus_random"]) for row in evaluate_early(early_rows, budget)]
    )
    observed_middle = evaluate_middle(middle_rows, budget, validity_threshold)
    observed_best = mean(
        [float(row["best_gain_descriptor_minus_random"]) for row in observed_middle]
    )
    observed_ndcg = mean(
        [float(row["ndcg_descriptor_minus_random"]) for row in observed_middle]
    )
    generator = random.Random(seed)
    early_exceedances = 0
    best_exceedances = 0
    ndcg_exceedances = 0
    for _ in range(resamples):
        permuted_early = _permuted_rows(
            early_rows, "counterfactual_valid_rate", generator
        )
        early_statistic = mean(
            [
                float(row["descriptor_minus_random"])
                for row in evaluate_early(permuted_early, budget)
            ]
        )
        permuted_middle = _permuted_rows(middle_rows, "gain", generator)
        middle_metrics = evaluate_middle(
            permuted_middle, budget, validity_threshold
        )
        best_statistic = mean(
            [
                float(row["best_gain_descriptor_minus_random"])
                for row in middle_metrics
            ]
        )
        ndcg_statistic = mean(
            [float(row["ndcg_descriptor_minus_random"]) for row in middle_metrics]
        )
        if early_statistic >= observed_early - 1e-15:
            early_exceedances += 1
        if best_statistic >= observed_best - 1e-15:
            best_exceedances += 1
        if ndcg_statistic >= observed_ndcg - 1e-15:
            ndcg_exceedances += 1
    denominator = float(resamples + 1)
    return (
        (early_exceedances + 1.0) / denominator,
        (best_exceedances + 1.0) / denominator,
        (ndcg_exceedances + 1.0) / denominator,
    )


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 1B.1 test evaluation must run through Slurm")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("sealed Gate 1B.1 test has already been evaluated")
    repo_root = args.repo_root.resolve()
    source = git_identity(repo_root)
    if source.get("dirty") is not False:
        raise RuntimeError("Gate 1B.1 test evaluation requires clean Git")
    config = load_config(args.config.resolve())
    model_path = args.model_manifest.resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if (
        model.get("gate") != config["gate"]
        or model.get("status") != "frozen"
        or model.get("dev_entry_passed") is not True
        or model.get("test_collection_authorized") is not True
        or model.get("test_evaluations_completed") != 0
        or model.get("source", {}).get("commit") != source.get("commit")
        or model.get("config_sha256") != file_sha256(args.config.resolve())
    ):
        raise RuntimeError("Gate 1B.1 model is not an unused, source-matched test seal")
    model_sha256 = file_sha256(model_path)
    test, test_sources = load_split_reports(args.test_report, "test", config)
    for stage in STAGES:
        seal = test_sources[stage]["test_seal"] or {}
        if (
            seal.get("model_manifest_sha256") != model_sha256
            or test_sources[stage]["source"] != source
        ):
            raise RuntimeError("test report was not collected under this model seal")
    seen_keys = set(model["train_structure_keys"]) | set(model["dev_structure_keys"])
    test_keys = structure_keys(test)
    overlap = sorted(seen_keys & test_keys)
    predicted = attach_predictions(test, model["validity_model"], model["gain_model"])
    budget = int(config["candidate_pool"]["budget"])
    validity_threshold = float(model["validity_threshold"])
    gain_threshold = float(model["late_gain_threshold"])
    early = evaluate_early(predicted["early"], budget)
    middle = evaluate_middle(predicted["middle"], budget, validity_threshold)
    late = evaluate_late(
        predicted["late"], budget, validity_threshold, gain_threshold
    )
    rule = config["test_rule"]
    early_p, best_p, ndcg_p = _permutation_p_values(
        predicted["early"],
        predicted["middle"],
        budget,
        validity_threshold,
        int(rule["permutation_resamples"]),
        int(rule["permutation_seed"]),
    )
    early_evidence = _test_evidence(
        [float(row["descriptor_minus_random"]) for row in early],
        [float(row["oracle_minus_random"]) for row in early],
        early_p,
        config,
        int(rule["bootstrap_seed"]),
    )
    best_evidence = _test_evidence(
        [float(row["best_gain_descriptor_minus_random"]) for row in middle],
        [float(row["best_gain_oracle_minus_random"]) for row in middle],
        best_p,
        config,
        int(rule["bootstrap_seed"]) + 100,
    )
    ndcg_evidence = _test_evidence(
        [float(row["ndcg_descriptor_minus_random"]) for row in middle],
        [float(row["ndcg_oracle_minus_random"]) for row in middle],
        ndcg_p,
        config,
        int(rule["bootstrap_seed"]) + 200,
    )
    no_positive = [row for row in late if not bool(row["has_positive_candidate"])]
    positive = [row for row in late if bool(row["has_positive_candidate"])]
    correct_abstention_rate = (
        mean([float(bool(row["correct_abstention"])) for row in no_positive])
        if no_positive
        else float("nan")
    )
    opportunity_response_rate = (
        mean([float(bool(row["opportunity_response"])) for row in positive])
        if positive
        else float("nan")
    )
    late_checks = {
        "both_late_classes_observed": bool(no_positive) and bool(positive),
        "correct_abstention_rate": correct_abstention_rate
        >= float(rule["late_minimum_correct_abstention_rate"]),
        "opportunity_response_rate": opportunity_response_rate
        >= float(rule["late_minimum_opportunity_response_rate"]),
    }
    evaluated = {"early": early, "middle": middle, "late": late}
    cross_timestep_fraction = trajectory_cross_timestep_fraction(evaluated)
    integrity_checks = {
        "test_structure_overlap_zero": len(overlap) == 0,
        "cross_timestep_fraction": cross_timestep_fraction
        >= float(rule["minimum_cross_timestep_trajectory_fraction"]),
    }
    gate_checks = {
        "early_rescue": bool(early_evidence["passed"]),
        "middle_best_gain": bool(best_evidence["passed"]),
        "middle_ndcg": bool(ndcg_evidence["passed"]),
        "late_abstention_safety": all(late_checks.values()),
        "structure_and_timestep_integrity": all(integrity_checks.values()),
    }
    passed = all(gate_checks.values())
    result = {
        "schema_version": 1,
        "gate": config["gate"],
        "status": "complete",
        "dataset_role": "sealed-structure-isolated-test",
        "test_evaluation_ordinal": 1,
        "source": source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "model_manifest": str(model_path),
        "model_manifest_sha256": model_sha256,
        "test_sources": test_sources,
        "structure_isolation": {
            "seen_train_dev_keys": len(seen_keys),
            "test_keys": len(test_keys),
            "overlap_count": len(overlap),
            "overlap": overlap,
        },
        "cross_timestep": {
            "trajectory_fraction": cross_timestep_fraction,
            "minimum_required": float(
                rule["minimum_cross_timestep_trajectory_fraction"]
            ),
        },
        "early": {"evidence": early_evidence, "trajectories": early},
        "middle": {
            "best_gain_evidence": best_evidence,
            "ndcg_evidence": ndcg_evidence,
            "trajectories": middle,
        },
        "late": {
            "gain_threshold": gain_threshold,
            "correct_abstention_rate": correct_abstention_rate,
            "opportunity_response_rate": opportunity_response_rate,
            "checks": late_checks,
            "trajectories": late,
        },
        "decision": {
            "checks": gate_checks,
            "gate1b1_passed": passed,
            "gate1c_eligible": passed,
            "gate1c_authorized": False,
            "pairwise_refinement_authorized": False,
            "ppo_integration_authorized": False,
        },
        "claim_boundary": (
            "This is the one sealed Gate 1B.1 test evaluation. It does not test "
            "LLM acquisition, PPO improvement, an independent scientific oracle, "
            "or wet-lab validity."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(
        json.dumps(
            {
                "event": "gate1b1_sealed_test_complete",
                "output": str(output),
                **result["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
