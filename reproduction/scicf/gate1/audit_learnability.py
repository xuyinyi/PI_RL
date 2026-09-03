#!/usr/bin/env python3
"""Gate 1B deterministic cheap-feature learnability audit on frozen Gate1-dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from reproduction.framework.io import git_identity, write_json
from reproduction.scicf.acquisition.chemistry import morgan_distance
from reproduction.scicf.acquisition.metrics import acquisition_metrics
from reproduction.scicf.gate1.audit_headroom import (
    expected_random_best_gain,
    expected_random_ndcg,
)


AUDIT_VERSION = "scicf-gate1b-cheap-feature-learnability-v1"
FEATURE_PROTOCOL = "request-time-rdkit-descriptors-v1"
STAGES = ("early", "middle", "late")
DESCRIPTOR_NAMES = (
    "mol_wt",
    "mol_log_p",
    "tpsa",
    "h_acceptors",
    "h_donors",
    "rotatable_bonds",
    "ring_count",
    "fraction_csp3",
    "heavy_atom_count",
    "aromatic_ring_count",
    "hetero_atom_count",
    "dummy_atom_count",
    "formal_charge",
)
STRUCTURE_PREFIXES = (
    "context_dianhydride",
    "context_diamine",
    "factual_fragment",
    "alternative_fragment",
    "fragment_delta",
    "fragment_abs_delta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scientific_object_valid(value: Any) -> bool:
    return isinstance(value, str) and value.strip() not in {"", "None", "null"}


def molecule_descriptors(smiles: str) -> np.ndarray:
    """Return the fixed, low-dimensional RDKit descriptor vector."""

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("descriptor input is not valid SMILES: {}".format(smiles))
    atoms = tuple(molecule.GetAtoms())
    values = np.asarray(
        (
            Descriptors.MolWt(molecule),
            Descriptors.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            Descriptors.NumHAcceptors(molecule),
            Descriptors.NumHDonors(molecule),
            Descriptors.NumRotatableBonds(molecule),
            Descriptors.RingCount(molecule),
            Descriptors.FractionCSP3(molecule),
            Descriptors.HeavyAtomCount(molecule),
            Descriptors.NumAromaticRings(molecule),
            sum(atom.GetAtomicNum() not in {0, 1, 6} for atom in atoms),
            sum(atom.GetAtomicNum() == 0 for atom in atoms),
            sum(atom.GetFormalCharge() for atom in atoms),
        ),
        dtype=float,
    )
    if values.shape != (len(DESCRIPTOR_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("RDKit descriptor vector is malformed or non-finite")
    return values


def feature_names() -> Tuple[str, ...]:
    names = []
    for prefix in STRUCTURE_PREFIXES:
        names.extend("{}__{}".format(prefix, name) for name in DESCRIPTOR_NAMES)
    names.extend(("component__dianhydride", "component__diamine", "morgan_distance"))
    return tuple(names)


def candidate_feature_vector(
    trajectory: Mapping[str, Any],
    decision_timestep: int,
    candidate: Mapping[str, Any],
) -> np.ndarray:
    """Build features solely from information available in the blinded request."""

    steps = trajectory.get("steps", [])
    if decision_timestep < 0 or decision_timestep >= len(steps):
        raise ValueError("decision timestep is outside the factual trajectory")
    pre_state = steps[decision_timestep]["pre_state"]
    intervention = candidate["intervention"]
    component = str(intervention["component"])
    if component not in {"dianhydride", "diamine"}:
        raise ValueError("unsupported intervention component")
    context_dianhydride = molecule_descriptors(pre_state["dianhydride_structure"])
    context_diamine = molecule_descriptors(pre_state["diamine_structure"])
    factual_fragment = molecule_descriptors(
        intervention["metadata"]["factual_structure"]
    )
    alternative_fragment = molecule_descriptors(intervention["alternative_structure"])
    delta = alternative_fragment - factual_fragment
    distance = morgan_distance(
        str(intervention["metadata"]["factual_structure"]),
        str(intervention["alternative_structure"]),
    )
    result = np.concatenate(
        (
            context_dianhydride,
            context_diamine,
            factual_fragment,
            alternative_fragment,
            delta,
            np.abs(delta),
            np.asarray(
                (
                    1.0 if component == "dianhydride" else 0.0,
                    1.0 if component == "diamine" else 0.0,
                    distance,
                )
            ),
        )
    )
    if result.shape != (len(feature_names()),) or not np.all(np.isfinite(result)):
        raise ValueError("candidate feature vector is malformed or non-finite")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_reports(paths: Iterable[Path]) -> Dict[str, Tuple[Path, Mapping[str, Any]]]:
    reports = {}
    for unresolved in paths:
        path = unresolved.resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        stage = str(report.get("stage"))
        if report.get("status") != "complete" or stage not in STAGES:
            raise ValueError("invalid or incomplete stage report: {}".format(path))
        if stage in reports:
            raise ValueError("duplicate stage report: {}".format(stage))
        reports[stage] = (path, report)
    if set(reports) != set(STAGES):
        raise ValueError("Gate 1B requires exactly early, middle, and late reports")
    return reports


def build_rows(
    reports: Mapping[str, Tuple[Path, Mapping[str, Any]]],
    expected_trajectories: int,
    expected_pool_size: int,
    decision_timestep: int,
) -> Dict[str, List[Dict[str, Any]]]:
    rows_by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for stage in STAGES:
        report = reports[stage][1]
        trajectories = report["trajectories"]
        if len(trajectories) != expected_trajectories:
            raise ValueError("{} trajectory count mismatch".format(stage))
        stage_rows = []
        for item in trajectories:
            trajectory = item["trajectory"]
            trajectory_id = str(trajectory["trajectory_id"])
            terminal_valid = scientific_object_valid(
                trajectory.get("terminal_scientific_object")
            )
            if stage == "early" and terminal_valid:
                raise ValueError("early stratum contains a valid factual trajectory")
            if stage in {"middle", "late"} and not terminal_valid:
                raise ValueError("{} stratum contains an invalid factual trajectory".format(stage))
            candidates = item["pool"]["candidates"]
            if len(candidates) != expected_pool_size:
                raise ValueError("{} pool size mismatch".format(trajectory_id))
            gains = item["verified_gains"]
            verifications = {
                str(entry["intervention"]["intervention_id"]): entry
                for entry in item["verifications"]
            }
            candidate_ids = [str(entry["candidate_id"]) for entry in candidates]
            if (
                len(candidate_ids) != len(set(candidate_ids))
                or set(candidate_ids) != set(gains)
                or set(candidate_ids) != set(verifications)
            ):
                raise ValueError("candidate, gain, and verification identities differ")
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                verification = verifications[candidate_id]
                outcomes = verification["paired_outcomes"]
                if not outcomes:
                    raise ValueError("verification has no paired outcome")
                factual_valid = [
                    scientific_object_valid(outcome.get("factual_terminal_object"))
                    for outcome in outcomes
                ]
                counterfactual_valid = [
                    scientific_object_valid(
                        outcome.get("counterfactual_terminal_object")
                    )
                    for outcome in outcomes
                ]
                rescue_rate = sum(counterfactual_valid) / float(len(outcomes))
                gain = float(gains[candidate_id])
                if not math.isclose(
                    gain,
                    float(verification["mean_delta"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError("gain table and verification mean differ")
                if not math.isfinite(gain):
                    raise ValueError("verified gain is non-finite")
                if stage == "early" and any(factual_valid):
                    raise ValueError("early rescue label has a valid factual replicate")
                features = candidate_feature_vector(
                    trajectory, decision_timestep, candidate
                )
                stage_rows.append(
                    {
                        "stage": stage,
                        "trajectory_id": trajectory_id,
                        "candidate_id": candidate_id,
                        "component": candidate["intervention"]["component"],
                        "features": features,
                        "rescue_rate": rescue_rate,
                        "any_rescue": rescue_rate > 0.0,
                        "full_rescue": rescue_rate == 1.0,
                        "gain": gain,
                        "policy_score": float(candidate["policy_score"]),
                        "morgan_distance": float(candidate["structural_score"]),
                    }
                )
        groups = {row["trajectory_id"] for row in stage_rows}
        if len(groups) != expected_trajectories:
            raise ValueError("{} group count mismatch".format(stage))
        rows_by_stage[stage] = stage_rows
    return rows_by_stage


def build_loto_operators(
    features: np.ndarray, groups: Sequence[str], alpha: float
) -> Tuple[Tuple[Mapping[str, Any], ...], Tuple[Mapping[str, Any], ...]]:
    """Precompute label-independent leave-one-trajectory-out ridge operators."""

    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(groups):
        raise ValueError("feature matrix and group vector are incompatible")
    if alpha <= 0.0 or not np.all(np.isfinite(matrix)):
        raise ValueError("ridge alpha and features must be finite and valid")
    unique_groups = tuple(sorted(set(str(group) for group in groups)))
    if len(unique_groups) < 2:
        raise ValueError("leave-one-group-out needs at least two groups")
    operators = []
    receipts = []
    group_array = np.asarray([str(group) for group in groups], dtype=object)
    for held_group in unique_groups:
        test_indices = np.flatnonzero(group_array == held_group)
        train_indices = np.flatnonzero(group_array != held_group)
        train = matrix[train_indices]
        test = matrix[test_indices]
        mean = np.mean(train, axis=0)
        scale = np.std(train, axis=0)
        constant_mask = scale < 1e-12
        scale[constant_mask] = 1.0
        train_standardized = (train - mean) / scale
        test_standardized = (test - mean) / scale
        train_augmented = np.column_stack(
            (np.ones(len(train_standardized)), train_standardized)
        )
        test_augmented = np.column_stack(
            (np.ones(len(test_standardized)), test_standardized)
        )
        penalty = np.eye(train_augmented.shape[1], dtype=float) * float(alpha)
        penalty[0, 0] = 0.0
        inverse = np.linalg.pinv(
            train_augmented.T.dot(train_augmented) + penalty
        )
        operator = test_augmented.dot(inverse).dot(train_augmented.T)
        operators.append(
            {
                "held_group": held_group,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "operator": operator,
            }
        )
        receipts.append(
            {
                "held_group": held_group,
                "train_groups": sorted(set(group_array[train_indices].tolist())),
                "train_candidates": int(len(train_indices)),
                "test_candidates": int(len(test_indices)),
                "constant_training_features": int(sum(constant_mask)),
            }
        )
    return tuple(operators), tuple(receipts)


def predict_with_operators(
    operators: Sequence[Mapping[str, Any]], targets: Sequence[float]
) -> np.ndarray:
    target_array = np.asarray(targets, dtype=float)
    predictions = np.full(target_array.shape, np.nan, dtype=float)
    for fold in operators:
        predictions[fold["test_indices"]] = fold["operator"].dot(
            target_array[fold["train_indices"]]
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("cross-validated ridge predictions are incomplete")
    return predictions


def _group_indices(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        result.setdefault(str(row["trajectory_id"]), []).append(index)
    return {key: result[key] for key in sorted(result)}


def _ranked_indices(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int], scores: Sequence[float]
) -> List[int]:
    return sorted(
        indices,
        key=lambda index: (-float(scores[index]), str(rows[index]["candidate_id"])),
    )


def binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = [index for index, value in enumerate(labels) if bool(value)]
    negatives = [index for index, value in enumerate(labels) if not bool(value)]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if float(scores[positive]) > float(scores[negative]):
                wins += 1.0
            elif float(scores[positive]) == float(scores[negative]):
                wins += 0.5
    return wins / float(len(positives) * len(negatives))


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(values[order[end]]) == float(
            values[order[start]]
        ):
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman_rho(values: Sequence[float], scores: Sequence[float]) -> float:
    if len(values) != len(scores) or len(values) < 2:
        raise ValueError("Spearman inputs must have equal nontrivial length")
    left = _average_ranks(values)
    right = _average_ranks(scores)
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = math.sqrt(float(left.dot(left) * right.dot(right)))
    if denominator <= 0.0:
        return float("nan")
    return float(left.dot(right) / denominator)


def early_trajectory_metrics(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float], budget: int
) -> List[Dict[str, Any]]:
    result = []
    policy_scores = [float(row["policy_score"]) for row in rows]
    distance_scores = [float(row["morgan_distance"]) for row in rows]
    for trajectory_id, indices in _group_indices(rows).items():
        values = [float(rows[index]["rescue_rate"]) for index in indices]
        any_labels = [bool(rows[index]["any_rescue"]) for index in indices]
        full_labels = [bool(rows[index]["full_rescue"]) for index in indices]

        def summarize(selection: Sequence[int], ranking_scores: Sequence[float]) -> Dict[str, Any]:
            selected = list(selection[:budget])
            return {
                "selected_ids": [str(rows[index]["candidate_id"]) for index in selected],
                "rescue_rate_at_4": sum(
                    float(rows[index]["rescue_rate"]) for index in selected
                )
                / float(budget),
                "any_rescue_hit_rate_at_4": sum(
                    bool(rows[index]["any_rescue"]) for index in selected
                )
                / float(budget),
                "full_rescue_hit_rate_at_4": sum(
                    bool(rows[index]["full_rescue"]) for index in selected
                )
                / float(budget),
                "trajectory_has_any_rescue_at_4": any(
                    bool(rows[index]["any_rescue"]) for index in selected
                ),
                "any_rescue_auc": binary_auc(
                    any_labels, [float(ranking_scores[index]) for index in indices]
                ),
            }

        descriptor_selection = _ranked_indices(rows, indices, scores)
        policy_selection = _ranked_indices(rows, indices, policy_scores)
        distance_selection = _ranked_indices(rows, indices, distance_scores)
        oracle_scores = [float(row["rescue_rate"]) for row in rows]
        oracle_selection = _ranked_indices(rows, indices, oracle_scores)
        positive_count = sum(any_labels)
        random_any_probability = 1.0
        if positive_count == 0:
            random_any_probability = 0.0
        elif len(indices) - positive_count >= budget:
            random_any_probability = 1.0 - math.comb(
                len(indices) - positive_count, budget
            ) / float(math.comb(len(indices), budget))
        expected_random = {
            "rescue_rate_at_4": sum(values) / float(len(values)),
            "any_rescue_hit_rate_at_4": positive_count / float(len(indices)),
            "full_rescue_hit_rate_at_4": sum(full_labels) / float(len(indices)),
            "trajectory_has_any_rescue_at_4": random_any_probability,
            "any_rescue_auc": 0.5 if 0 < positive_count < len(indices) else None,
        }
        descriptor = summarize(descriptor_selection, scores)
        oracle = summarize(oracle_selection, oracle_scores)
        result.append(
            {
                "trajectory_id": trajectory_id,
                "candidates": len(indices),
                "any_rescue_candidates": positive_count,
                "full_rescue_candidates": sum(full_labels),
                "partial_rescue_candidates": sum(
                    0.0 < value < 1.0 for value in values
                ),
                "strategies": {
                    "descriptor_ridge": descriptor,
                    "policy_probability": summarize(policy_selection, policy_scores),
                    "morgan_distance": summarize(distance_selection, distance_scores),
                    "oracle": oracle,
                    "expected_random": expected_random,
                },
                "descriptor_minus_expected_random_rescue_rate_at_4": descriptor[
                    "rescue_rate_at_4"
                ]
                - expected_random["rescue_rate_at_4"],
                "oracle_minus_expected_random_rescue_rate_at_4": oracle[
                    "rescue_rate_at_4"
                ]
                - expected_random["rescue_rate_at_4"],
            }
        )
    return result


def gain_trajectory_metrics(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float], budget: int
) -> List[Dict[str, Any]]:
    result = []
    policy_scores = [float(row["policy_score"]) for row in rows]
    distance_scores = [float(row["morgan_distance"]) for row in rows]
    oracle_scores = [float(row["gain"]) for row in rows]
    for trajectory_id, indices in _group_indices(rows).items():
        gain_table = {
            str(rows[index]["candidate_id"]): float(rows[index]["gain"])
            for index in indices
        }

        def summarize(selection: Sequence[int], ranking_scores: Sequence[float]) -> Dict[str, Any]:
            selected = list(selection[:budget])
            metrics = acquisition_metrics(
                gain_table,
                [str(rows[index]["candidate_id"]) for index in selected],
                budget,
            )
            return {
                "selected_ids": [str(rows[index]["candidate_id"]) for index in selected],
                **metrics,
                "spearman_rho": spearman_rho(
                    [float(rows[index]["gain"]) for index in indices],
                    [float(ranking_scores[index]) for index in indices],
                ),
            }

        descriptor_selection = _ranked_indices(rows, indices, scores)
        policy_selection = _ranked_indices(rows, indices, policy_scores)
        distance_selection = _ranked_indices(rows, indices, distance_scores)
        oracle_selection = _ranked_indices(rows, indices, oracle_scores)
        gains = [float(rows[index]["gain"]) for index in indices]
        expected_random = {
            "best_gain_at_b": expected_random_best_gain(gains, budget),
            "ndcg_at_b": expected_random_ndcg(gains, budget),
            "spearman_rho": 0.0,
        }
        descriptor = summarize(descriptor_selection, scores)
        oracle = summarize(oracle_selection, oracle_scores)
        result.append(
            {
                "trajectory_id": trajectory_id,
                "candidates": len(indices),
                "positive_candidates": sum(value > 0.0 for value in gains),
                "strategies": {
                    "descriptor_ridge": descriptor,
                    "policy_probability": summarize(policy_selection, policy_scores),
                    "morgan_distance": summarize(distance_selection, distance_scores),
                    "oracle": oracle,
                    "expected_random": expected_random,
                },
                "descriptor_minus_expected_random_best_gain_at_4": descriptor[
                    "best_gain_at_b"
                ]
                - expected_random["best_gain_at_b"],
                "oracle_minus_expected_random_best_gain_at_4": oracle[
                    "best_gain_at_b"
                ]
                - expected_random["best_gain_at_b"],
                "descriptor_minus_expected_random_ndcg_at_4": descriptor[
                    "ndcg_at_b"
                ]
                - expected_random["ndcg_at_b"],
                "oracle_minus_expected_random_ndcg_at_4": oracle["ndcg_at_b"]
                - expected_random["ndcg_at_b"],
            }
        )
    return result


def _bootstrap_mean_interval(
    values: Sequence[float], resamples: int, confidence: float, seed: int
) -> Tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
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


def metric_evidence(
    differences: Sequence[float],
    oracle_headroom: Sequence[float],
    permutation_p_value: float,
    config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    inference = config["inference"]
    lower, upper = _bootstrap_mean_interval(
        differences,
        int(inference["bootstrap_resamples"]),
        float(inference["confidence_level"]),
        seed,
    )
    mean_difference = sum(float(value) for value in differences) / float(
        len(differences)
    )
    mean_oracle_headroom = sum(float(value) for value in oracle_headroom) / float(
        len(oracle_headroom)
    )
    captured = (
        mean_difference / mean_oracle_headroom
        if mean_oracle_headroom > 0.0
        else float("nan")
    )
    minimum_fraction = float(
        config["gate_rule"]["minimum_oracle_headroom_fraction_captured"]
    )
    alpha = float(inference["one_sided_alpha"])
    checks = {
        "mean_difference_positive": mean_difference > 0.0,
        "bootstrap_ci_lower_positive": lower > 0.0,
        "permutation_p_at_most_alpha": permutation_p_value <= alpha,
        "minimum_headroom_fraction_captured": captured >= minimum_fraction,
    }
    return {
        "mean_descriptor_minus_expected_random": mean_difference,
        "trajectory_bootstrap_ci": {
            "lower": lower,
            "upper": upper,
            "confidence": float(inference["confidence_level"]),
            "resamples": int(inference["bootstrap_resamples"]),
        },
        "permutation": {
            "one_sided_p_value": permutation_p_value,
            "alpha": alpha,
            "resamples": int(inference["permutation_resamples"]),
            "scheme": inference["permutation_scheme"],
        },
        "mean_oracle_minus_expected_random_headroom": mean_oracle_headroom,
        "oracle_headroom_fraction_captured": captured,
        "minimum_required_fraction": minimum_fraction,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _early_difference(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float], budget: int
) -> float:
    values = []
    for indices in _group_indices(rows).values():
        selected = _ranked_indices(rows, indices, scores)[:budget]
        observed = sum(float(rows[index]["rescue_rate"]) for index in selected) / float(
            budget
        )
        expected = sum(float(rows[index]["rescue_rate"]) for index in indices) / float(
            len(indices)
        )
        values.append(observed - expected)
    return sum(values) / float(len(values))


def _gain_differences(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float], budget: int
) -> Tuple[float, float]:
    best_values = []
    ndcg_values = []
    for indices in _group_indices(rows).values():
        selected = _ranked_indices(rows, indices, scores)[:budget]
        gain_table = {
            str(rows[index]["candidate_id"]): float(rows[index]["gain"])
            for index in indices
        }
        observed = acquisition_metrics(
            gain_table,
            [str(rows[index]["candidate_id"]) for index in selected],
            budget,
        )
        gains = list(gain_table.values())
        best_values.append(
            float(observed["best_gain_at_b"])
            - expected_random_best_gain(gains, budget)
        )
        ndcg_values.append(
            float(observed["ndcg_at_b"])
            - expected_random_ndcg(gains, budget)
        )
    return (
        sum(best_values) / float(len(best_values)),
        sum(ndcg_values) / float(len(ndcg_values)),
    )


def _permuted_targets(
    targets: np.ndarray,
    group_indices: Mapping[str, Sequence[int]],
    generator: random.Random,
) -> np.ndarray:
    result = np.asarray(targets, dtype=float).copy()
    for indices in group_indices.values():
        shuffled = [float(result[index]) for index in indices]
        generator.shuffle(shuffled)
        for index, value in zip(indices, shuffled):
            result[index] = value
    return result


def early_permutation_p_value(
    rows: Sequence[Mapping[str, Any]],
    operators: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    budget: int,
    resamples: int,
    seed: int,
) -> float:
    observed_scores = predict_with_operators(operators, targets)
    observed = _early_difference(rows, observed_scores, budget)
    generator = random.Random(seed)
    groups = _group_indices(rows)
    exceedances = 0
    for _ in range(resamples):
        permuted = _permuted_targets(targets, groups, generator)
        scores = predict_with_operators(operators, permuted)
        if _early_difference(rows, scores, budget) >= observed - 1e-15:
            exceedances += 1
    return (exceedances + 1.0) / float(resamples + 1)


def gain_permutation_p_values(
    rows: Sequence[Mapping[str, Any]],
    operators: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    budget: int,
    resamples: int,
    seed: int,
) -> Tuple[float, float]:
    observed_scores = predict_with_operators(operators, targets)
    observed_best, observed_ndcg = _gain_differences(rows, observed_scores, budget)
    generator = random.Random(seed)
    groups = _group_indices(rows)
    best_exceedances = 0
    ndcg_exceedances = 0
    for _ in range(resamples):
        permuted = _permuted_targets(targets, groups, generator)
        scores = predict_with_operators(operators, permuted)
        best, ndcg = _gain_differences(rows, scores, budget)
        if best >= observed_best - 1e-15:
            best_exceedances += 1
        if ndcg >= observed_ndcg - 1e-15:
            ndcg_exceedances += 1
    denominator = float(resamples + 1)
    return (
        (best_exceedances + 1.0) / denominator,
        (ndcg_exceedances + 1.0) / denominator,
    )


def summarize_strategy_metric(
    trajectories: Sequence[Mapping[str, Any]],
    strategy: str,
    metric: str,
) -> float:
    values = []
    for trajectory in trajectories:
        value = trajectory["strategies"][strategy][metric]
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return sum(values) / float(len(values)) if values else float("nan")


def decide_gate1b(
    early_evidence: Mapping[str, Any],
    middle_best_evidence: Mapping[str, Any],
    middle_ndcg_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    checks = {
        "early_invalid_rescue_rate_at_4": bool(early_evidence["passed"]),
        "middle_valid_best_gain_at_4": bool(middle_best_evidence["passed"]),
        "middle_valid_ndcg_at_4": bool(middle_ndcg_evidence["passed"]),
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "gate1b_passed": passed,
        "gate1c_eligible": passed,
        "llm_gate1c_authorized": False,
        "pairwise_refinement_authorized": False,
        "ppo_integration_authorized": False,
    }


def audit(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 1B learnability audit must run through Slurm")
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Gate 1B output already exists: {}".format(output))
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config.get("gate") != AUDIT_VERSION:
        raise ValueError("Gate 1B configuration identity mismatch")
    if config["features"]["protocol"] != FEATURE_PROTOCOL:
        raise ValueError("Gate 1B feature protocol mismatch")
    source = git_identity(repo_root)
    if source.get("dirty") is not False:
        raise RuntimeError("Gate 1B requires a clean Git worktree")
    reports = _load_reports(args.stage_report)
    data_config = config["data"]
    rows_by_stage = build_rows(
        reports,
        expected_trajectories=int(data_config["trajectories_per_stage"]),
        expected_pool_size=int(data_config["candidate_pool_size"]),
        decision_timestep=int(data_config["decision_timestep"]),
    )
    budget = int(data_config["acquisition_budget"])
    alpha = float(config["model"]["alpha"])
    inference = config["inference"]
    resamples = int(inference["permutation_resamples"])
    permutation_seed = int(inference["permutation_seed"])
    bootstrap_seed = int(inference["bootstrap_seed"])

    stage_predictions = {}
    fold_receipts = {}
    operators_by_stage = {}
    targets_by_stage = {}
    for stage_index, stage in enumerate(STAGES):
        rows = rows_by_stage[stage]
        features = np.vstack([row["features"] for row in rows])
        targets = np.asarray(
            [
                float(row["rescue_rate"] if stage == "early" else row["gain"])
                for row in rows
            ],
            dtype=float,
        )
        groups = [str(row["trajectory_id"]) for row in rows]
        operators, receipts = build_loto_operators(features, groups, alpha)
        predictions = predict_with_operators(operators, targets)
        stage_predictions[stage] = predictions
        fold_receipts[stage] = receipts
        operators_by_stage[stage] = operators
        targets_by_stage[stage] = targets

    early_trajectories = early_trajectory_metrics(
        rows_by_stage["early"], stage_predictions["early"], budget
    )
    middle_trajectories = gain_trajectory_metrics(
        rows_by_stage["middle"], stage_predictions["middle"], budget
    )
    late_trajectories = gain_trajectory_metrics(
        rows_by_stage["late"], stage_predictions["late"], budget
    )

    early_p = early_permutation_p_value(
        rows_by_stage["early"],
        operators_by_stage["early"],
        targets_by_stage["early"],
        budget,
        resamples,
        permutation_seed,
    )
    middle_best_p, middle_ndcg_p = gain_permutation_p_values(
        rows_by_stage["middle"],
        operators_by_stage["middle"],
        targets_by_stage["middle"],
        budget,
        resamples,
        permutation_seed + 1000,
    )

    early_evidence = metric_evidence(
        [
            float(row["descriptor_minus_expected_random_rescue_rate_at_4"])
            for row in early_trajectories
        ],
        [
            float(row["oracle_minus_expected_random_rescue_rate_at_4"])
            for row in early_trajectories
        ],
        early_p,
        config,
        bootstrap_seed,
    )
    middle_best_evidence = metric_evidence(
        [
            float(row["descriptor_minus_expected_random_best_gain_at_4"])
            for row in middle_trajectories
        ],
        [
            float(row["oracle_minus_expected_random_best_gain_at_4"])
            for row in middle_trajectories
        ],
        middle_best_p,
        config,
        bootstrap_seed + 100,
    )
    middle_ndcg_evidence = metric_evidence(
        [
            float(row["descriptor_minus_expected_random_ndcg_at_4"])
            for row in middle_trajectories
        ],
        [
            float(row["oracle_minus_expected_random_ndcg_at_4"])
            for row in middle_trajectories
        ],
        middle_ndcg_p,
        config,
        bootstrap_seed + 200,
    )
    decision = decide_gate1b(
        early_evidence, middle_best_evidence, middle_ndcg_evidence
    )

    def stage_strategy_summary(
        trajectories: Sequence[Mapping[str, Any]], metrics: Sequence[str]
    ) -> Dict[str, Any]:
        strategies = (
            "descriptor_ridge",
            "policy_probability",
            "morgan_distance",
            "oracle",
            "expected_random",
        )
        return {
            strategy: {
                metric: summarize_strategy_metric(trajectories, strategy, metric)
                for metric in metrics
            }
            for strategy in strategies
        }

    report_sources = {
        stage: {
            "path": str(reports[stage][0]),
            "sha256": _sha256(reports[stage][0]),
            "checkpoint": reports[stage][1].get("checkpoint"),
            "checkpoint_sha256": reports[stage][1].get("checkpoint_sha256"),
            "slurm_job_id": reports[stage][1].get("slurm_job_id"),
        }
        for stage in STAGES
    }
    result = {
        "schema_version": 1,
        "audit": AUDIT_VERSION,
        "status": "complete",
        "dataset_role": config["dataset_role"],
        "independent_test_status": "not-created",
        "source": source,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "config": config,
        "inputs": {
            "reports": report_sources,
            "new_llm_calls": 0,
            "new_oracle_calls": 0,
            "ppo_updates": 0,
        },
        "feature_contract": {
            "protocol": FEATURE_PROTOCOL,
            "feature_count": len(feature_names()),
            "feature_names": feature_names(),
            "allowed_inputs": config["features"]["allowed_inputs"],
            "forbidden_inputs": config["features"]["forbidden_inputs"],
            "candidate_identity_used_only_for_stable_tie_breaking": True,
        },
        "cross_validation": {
            "protocol": config["model"]["cross_validation"],
            "folds": fold_receipts,
        },
        "early_invalid": {
            "role": "gate metric",
            "label": config["targets"]["early-invalid"],
            "strategy_summary": stage_strategy_summary(
                early_trajectories,
                (
                    "rescue_rate_at_4",
                    "any_rescue_hit_rate_at_4",
                    "full_rescue_hit_rate_at_4",
                    "trajectory_has_any_rescue_at_4",
                    "any_rescue_auc",
                ),
            ),
            "gate_metric": {"rescue_rate_at_4": early_evidence},
            "trajectories": early_trajectories,
        },
        "middle_valid": {
            "role": "gate metrics",
            "label": config["targets"]["middle-valid"],
            "strategy_summary": stage_strategy_summary(
                middle_trajectories,
                ("best_gain_at_b", "ndcg_at_b", "spearman_rho"),
            ),
            "gate_metrics": {
                "best_gain_at_4": middle_best_evidence,
                "ndcg_at_4": middle_ndcg_evidence,
            },
            "trajectories": middle_trajectories,
        },
        "late_valid": {
            "role": "saturation diagnostic only",
            "contributes_to_gate": False,
            "label": config["targets"]["late-valid"],
            "trajectories_without_positive_candidate": sum(
                int(row["positive_candidates"]) == 0 for row in late_trajectories
            ),
            "strategy_summary": stage_strategy_summary(
                late_trajectories,
                ("best_gain_at_b", "ndcg_at_b", "spearman_rho"),
            ),
            "trajectories": late_trajectories,
        },
        "decision": decision,
        "claim_boundary": (
            "Gate 1B is a trajectory-held-out learnability diagnostic on Gate1-dev. "
            "It does not establish independent-test generalization, LLM acquisition "
            "benefit, online PPO improvement, independent-oracle robustness, or "
            "wet-lab validity."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(
        json.dumps(
            {
                "event": "gate1b_learnability_complete",
                "output": str(output),
                **decision,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()
