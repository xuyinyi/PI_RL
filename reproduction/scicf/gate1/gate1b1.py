"""Shared contracts for structure-isolated, cross-timestep SciCF Gate 1B.1."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from reproduction.scicf.acquisition.base import AcquisitionCandidate, CandidatePool
from reproduction.scicf.acquisition.chemistry import morgan_distance
from reproduction.scicf.acquisition.metrics import acquisition_metrics
from reproduction.scicf.gate1.audit_headroom import (
    expected_random_best_gain,
    expected_random_ndcg,
)


GATE_VERSION = "scicf-gate1b1-structure-isolated-two-stage-v1"
FEATURE_PROTOCOL = "gate1b1-request-time-trajectory-descriptors-v1"
STAGES = ("early", "middle", "late")
SPLITS = ("train", "dev", "test")
DESCRIPTOR_NAMES = (
    "mol_wt",
    "tpsa",
    "aromatic_ring_count",
    "ring_count",
    "rotatable_bonds",
    "hetero_atom_count",
    "halogen_count",
    "fraction_csp3",
    "h_donors",
    "h_acceptors",
    "attachment_point_count",
    "formal_charge",
)


def load_config(path: Path) -> Mapping[str, Any]:
    config = json.loads(path.resolve().read_text(encoding="utf-8"))
    if config.get("gate") != GATE_VERSION:
        raise ValueError("Gate 1B.1 configuration identity mismatch")
    if config["features"]["protocol"] != FEATURE_PROTOCOL:
        raise ValueError("Gate 1B.1 feature protocol mismatch")
    if tuple(config["features"]["rdkit_descriptors_per_structure"]) != DESCRIPTOR_NAMES:
        raise ValueError("Gate 1B.1 descriptor list mismatch")
    seed_sets = {
        split: {int(seed) for seed in config["seeds"][split]} for split in SPLITS
    }
    if any(
        seed_sets[left] & seed_sets[right]
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    ):
        raise ValueError("Gate 1B.1 train/dev/test seeds must be disjoint")
    return config


def scientific_object_valid(value: Any) -> bool:
    return isinstance(value, str) and value.strip() not in {"", "None", "null"}


def canonical_structure(smiles: str) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("structure split requires valid SMILES")
    return str(Chem.MolToSmiles(molecule, canonical=True))


def structure_key(component: str, smiles: str) -> str:
    if component not in {"dianhydride", "diamine"}:
        raise ValueError("invalid structure-key component")
    return "{}|{}".format(component, canonical_structure(smiles))


def structure_split(component: str, smiles: str, config: Mapping[str, Any]) -> str:
    split_config = config["structure_split"]
    payload = "{}|{}|{}".format(
        split_config["salt"], component, canonical_structure(smiles)
    ).encode("utf-8")
    bucket = int(hashlib.sha256(payload).hexdigest()[:16], 16) % int(
        split_config["modulus"]
    )
    if bucket < int(split_config["train_upper_exclusive"]):
        return "train"
    if bucket < int(split_config["dev_upper_exclusive"]):
        return "dev"
    return "test"


def _round_robin(
    ordered_by_timestep: Mapping[int, Sequence[AcquisitionCandidate]],
    count: int,
    selected_ids: set,
) -> List[AcquisitionCandidate]:
    queues = {key: list(value) for key, value in ordered_by_timestep.items()}
    positions = {key: 0 for key in queues}
    result = []
    while len(result) < count:
        progressed = False
        for timestep in sorted(queues):
            queue = queues[timestep]
            while positions[timestep] < len(queue):
                candidate = queue[positions[timestep]]
                positions[timestep] += 1
                if candidate.candidate_id in selected_ids:
                    continue
                result.append(candidate)
                selected_ids.add(candidate.candidate_id)
                progressed = True
                break
            if len(result) == count:
                break
        if not progressed:
            break
    return result


def build_cross_timestep_pool(
    candidates: Sequence[AcquisitionCandidate],
    source_quotas: Mapping[str, int],
    requested_size: int,
    seed: int,
) -> CandidatePool:
    """Select one common pool while round-robining every observed timestep."""

    if set(source_quotas) != {"policy_near", "random_legal", "structural"}:
        raise ValueError("cross-timestep pool has invalid source quotas")
    if sum(int(value) for value in source_quotas.values()) != requested_size:
        raise ValueError("cross-timestep source quotas must sum to pool size")
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("cross-timestep candidate IDs are not unique")
    timesteps = sorted(
        {int(candidate.intervention.timestep) for candidate in candidates}
    )
    if not timesteps:
        raise ValueError("cross-timestep candidate universe is empty")
    selected = []
    selected_ids = set()
    counts = {}
    for source in ("policy_near", "random_legal", "structural"):
        grouped = {}
        for timestep in timesteps:
            current = [
                candidate
                for candidate in candidates
                if int(candidate.intervention.timestep) == timestep
            ]
            if source == "policy_near":
                current.sort(
                    key=lambda candidate: (
                        -float(candidate.policy_score), candidate.candidate_id
                    )
                )
            elif source == "structural":
                current.sort(
                    key=lambda candidate: (
                        -float(candidate.structural_score), candidate.candidate_id
                    )
                )
            else:
                random.Random(seed + timestep * 1009).shuffle(current)
            grouped[timestep] = current
        chosen = _round_robin(
            grouped, int(source_quotas[source]), selected_ids
        )
        selected.extend(chosen)
        counts[source] = len(chosen)
    if len(selected) < requested_size:
        remainder = [
            candidate
            for identifier, candidate in sorted(by_id.items())
            if identifier not in selected_ids
        ]
        random.Random(seed + 900001).shuffle(remainder)
        selected.extend(remainder[: requested_size - len(selected)])
    pool = CandidatePool(
        candidates=tuple(selected),
        requested_size=requested_size,
        source_selected_counts=counts,
        source_shortfalls={
            source: int(source_quotas[source]) - counts[source]
            for source in source_quotas
        },
    )
    if pool.shortfall:
        raise RuntimeError("cross-timestep candidate pool has a shortfall")
    selected_timesteps = {
        int(candidate.intervention.timestep) for candidate in pool.candidates
    }
    if selected_timesteps != set(timesteps):
        raise RuntimeError("cross-timestep pool did not cover every observed timestep")
    return pool


def molecule_descriptors(smiles: str) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError("descriptor input is not valid SMILES")
    atoms = tuple(molecule.GetAtoms())
    halogens = {9, 17, 35, 53, 85}
    values = np.asarray(
        (
            Descriptors.MolWt(molecule),
            Descriptors.TPSA(molecule),
            Descriptors.NumAromaticRings(molecule),
            Descriptors.RingCount(molecule),
            Descriptors.NumRotatableBonds(molecule),
            sum(atom.GetAtomicNum() not in {0, 1, 6} for atom in atoms),
            sum(atom.GetAtomicNum() in halogens for atom in atoms),
            Descriptors.FractionCSP3(molecule),
            Descriptors.NumHDonors(molecule),
            Descriptors.NumHAcceptors(molecule),
            sum(atom.GetAtomicNum() == 0 for atom in atoms),
            sum(atom.GetFormalCharge() for atom in atoms),
        ),
        dtype=float,
    )
    if values.shape != (len(DESCRIPTOR_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("invalid Gate 1B.1 descriptor vector")
    return values


def common_feature_names() -> Tuple[str, ...]:
    prefixes = (
        "context_dianhydride",
        "context_diamine",
        "factual_fragment",
        "alternative_fragment",
        "fragment_delta",
        "fragment_abs_delta",
    )
    names = []
    for prefix in prefixes:
        names.extend("{}__{}".format(prefix, name) for name in DESCRIPTOR_NAMES)
    names.extend(
        (
            "component__dianhydride",
            "component__diamine",
            "morgan_distance",
            "timestep_fraction",
            "trajectory_length",
            "pre_dianhydride_complete",
            "pre_diamine_complete",
            "factual_terminal_valid",
            "factual_episode_return",
            "factual_properties_available",
            "factual_transmittance",
            "factual_cte",
            "factual_strength",
            "factual_tg",
            "factual_sa_score",
        )
    )
    return tuple(names)


def validity_feature_names() -> Tuple[str, ...]:
    return common_feature_names()


def gain_feature_names() -> Tuple[str, ...]:
    return common_feature_names() + ("policy_probability", "log_policy_probability")


def candidate_features(
    trajectory: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    intervention = candidate["intervention"]
    timestep = int(intervention["timestep"])
    steps = trajectory["steps"]
    if timestep < 0 or timestep >= len(steps):
        raise ValueError("candidate timestep is outside trajectory")
    step = steps[timestep]
    pre_state = step["pre_state"]
    component = str(intervention["component"])
    context_dianhydride = molecule_descriptors(pre_state["dianhydride_structure"])
    context_diamine = molecule_descriptors(pre_state["diamine_structure"])
    factual = molecule_descriptors(intervention["metadata"]["factual_structure"])
    alternative = molecule_descriptors(intervention["alternative_structure"])
    delta = alternative - factual
    terminal_valid = scientific_object_valid(
        trajectory.get("terminal_scientific_object")
    )
    properties = trajectory.get("terminal_properties") or {}
    property_values = (
        float(properties.get("transmittance", 0.0)) if terminal_valid else 0.0,
        float(properties.get("cte", 0.0)) if terminal_valid else 0.0,
        float(properties.get("strength", 0.0)) if terminal_valid else 0.0,
        float(properties.get("tg", 0.0)) if terminal_valid else 0.0,
        float(properties.get("sa_score", 0.0)) if terminal_valid else 0.0,
    )
    timestep_fraction = timestep / float(max(1, len(steps) - 1))
    distance = morgan_distance(
        str(intervention["metadata"]["factual_structure"]),
        str(intervention["alternative_structure"]),
    )
    common = np.concatenate(
        (
            context_dianhydride,
            context_diamine,
            factual,
            alternative,
            delta,
            np.abs(delta),
            np.asarray(
                (
                    1.0 if component == "dianhydride" else 0.0,
                    1.0 if component == "diamine" else 0.0,
                    distance,
                    timestep_fraction,
                    float(len(steps)),
                    float(bool(pre_state["dianhydride_complete"])),
                    float(bool(pre_state["diamine_complete"])),
                    float(terminal_valid),
                    float(trajectory["episode_return"]),
                    float(terminal_valid),
                    *property_values,
                ),
                dtype=float,
            ),
        )
    )
    probability = float(candidate["policy_score"])
    gain = np.concatenate(
        (common, np.asarray((probability, math.log(max(probability, 1e-12)))))
    )
    if common.shape != (len(validity_feature_names()),):
        raise ValueError("validity feature length mismatch")
    if gain.shape != (len(gain_feature_names()),):
        raise ValueError("gain feature length mismatch")
    if not np.all(np.isfinite(common)) or not np.all(np.isfinite(gain)):
        raise ValueError("Gate 1B.1 features must be finite")
    return common, gain


def load_split_reports(
    paths: Iterable[Path], expected_split: str, config: Mapping[str, Any]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    if expected_split not in SPLITS:
        raise ValueError("invalid Gate 1B.1 split")
    by_stage = {}
    sources = {}
    expected_seeds = tuple(int(seed) for seed in config["seeds"][expected_split])
    expected_pool = int(config["candidate_pool"]["size"])
    for unresolved in paths:
        path = unresolved.resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        stage = str(report.get("stage"))
        if (
            report.get("status") != "complete"
            or report.get("split") != expected_split
            or stage not in STAGES
        ):
            raise ValueError("invalid Gate 1B.1 split report")
        if tuple(int(seed) for seed in report["seeds"]) != expected_seeds:
            raise ValueError("Gate 1B.1 seed manifest mismatch")
        if stage in by_stage:
            raise ValueError("duplicate Gate 1B.1 stage report")
        rows = []
        trajectories = report["trajectories"]
        if len(trajectories) != len(expected_seeds):
            raise ValueError("Gate 1B.1 trajectory count mismatch")
        for item in trajectories:
            trajectory = item["trajectory"]
            candidates = item["pool"]["candidates"]
            if len(candidates) != expected_pool:
                raise ValueError("Gate 1B.1 candidate pool size mismatch")
            gains = item["verified_gains"]
            verifications = {
                str(entry["intervention"]["intervention_id"]): entry
                for entry in item["verifications"]
            }
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                verification = verifications[candidate_id]
                outcomes = verification["paired_outcomes"]
                counterfactual_valid_rate = sum(
                    scientific_object_valid(
                        outcome.get("counterfactual_terminal_object")
                    )
                    for outcome in outcomes
                ) / float(len(outcomes))
                validity_features, gain_features = candidate_features(
                    trajectory, candidate
                )
                key = structure_key(
                    str(candidate["intervention"]["component"]),
                    str(candidate["intervention"]["alternative_structure"]),
                )
                if key != candidate["structure_key"]:
                    raise ValueError("stored structure key does not match candidate")
                if structure_split(
                    str(candidate["intervention"]["component"]),
                    str(candidate["intervention"]["alternative_structure"]),
                    config,
                ) != expected_split:
                    raise ValueError("candidate is assigned to the wrong structure split")
                gain = float(gains[candidate_id])
                if not math.isclose(
                    gain, float(verification["mean_delta"]), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError("verified gain identity mismatch")
                rows.append(
                    {
                        "split": expected_split,
                        "stage": stage,
                        "trajectory_id": str(trajectory["trajectory_id"]),
                        "candidate_id": candidate_id,
                        "structure_key": key,
                        "timestep": int(candidate["intervention"]["timestep"]),
                        "terminal_valid": scientific_object_valid(
                            trajectory.get("terminal_scientific_object")
                        ),
                        "counterfactual_valid_rate": counterfactual_valid_rate,
                        "gain": gain,
                        "validity_features": validity_features,
                        "gain_features": gain_features,
                    }
                )
        by_stage[stage] = rows
        if stage == "early" and any(bool(row["terminal_valid"]) for row in rows):
            raise ValueError("early split is not an all-invalid stratum")
        if stage in {"middle", "late"} and any(
            not bool(row["terminal_valid"]) for row in rows
        ):
            raise ValueError("{} split is not an all-valid stratum".format(stage))
        sources[stage] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "slurm_job_id": report.get("slurm_job_id"),
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "source": report.get("source"),
            "test_seal": report.get("test_seal"),
        }
    if set(by_stage) != set(STAGES):
        raise ValueError("Gate 1B.1 needs early, middle, and late reports")
    return by_stage, sources


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> Dict[str, Any]:
    matrix = np.asarray(features, dtype=float)
    values = np.asarray(targets, dtype=float)
    if matrix.ndim != 2 or values.shape != (matrix.shape[0],):
        raise ValueError("ridge features and targets have incompatible shapes")
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (matrix - mean) / scale
    augmented = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.eye(augmented.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(
        augmented.T.dot(augmented) + penalty
    ).dot(augmented.T).dot(values)
    return {
        "family": "ridge",
        "alpha": float(alpha),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
        "training_rows": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
    }


def predict_ridge(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(coefficients):
        raise ValueError("prediction feature count does not match ridge model")
    return float(model["intercept"]) + ((matrix - mean) / scale).dot(coefficients)


def flatten_rows(rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
    return [row for stage in STAGES for row in rows_by_stage[stage]]


def attach_predictions(
    rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]],
    validity_model: Mapping[str, Any],
    gain_model: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    result = {}
    for stage in STAGES:
        rows = rows_by_stage[stage]
        validity = predict_ridge(
            validity_model, np.vstack([row["validity_features"] for row in rows])
        )
        gain = predict_ridge(
            gain_model, np.vstack([row["gain_features"] for row in rows])
        )
        result[stage] = [
            {
                **{key: value for key, value in row.items() if not key.endswith("features")},
                "predicted_validity": float(validity[index]),
                "predicted_gain": float(gain[index]),
            }
            for index, row in enumerate(rows)
        ]
    return result


def group_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    result = {}
    for row in rows:
        result.setdefault(str(row["trajectory_id"]), []).append(row)
    return {key: result[key] for key in sorted(result)}


def rank_rows(rows: Sequence[Mapping[str, Any]], key) -> List[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: (*key(row), str(row["candidate_id"])))


def evaluate_early(
    rows: Sequence[Mapping[str, Any]], budget: int
) -> List[Dict[str, Any]]:
    results = []
    for trajectory_id, current in group_rows(rows).items():
        selected = rank_rows(current, lambda row: (-float(row["predicted_validity"]),))[
            :budget
        ]
        oracle = sorted(
            (float(row["counterfactual_valid_rate"]) for row in current), reverse=True
        )[:budget]
        observed = sum(float(row["counterfactual_valid_rate"]) for row in selected) / float(
            budget
        )
        expected = sum(float(row["counterfactual_valid_rate"]) for row in current) / float(
            len(current)
        )
        oracle_value = sum(oracle) / float(budget)
        results.append(
            {
                "trajectory_id": trajectory_id,
                "candidate_timesteps": sorted({int(row["timestep"]) for row in current}),
                "selected_ids": [str(row["candidate_id"]) for row in selected],
                "rescue_rate_at_4": observed,
                "expected_random_rescue_rate_at_4": expected,
                "oracle_rescue_rate_at_4": oracle_value,
                "descriptor_minus_random": observed - expected,
                "oracle_minus_random": oracle_value - expected,
            }
        )
    return results


def evaluate_middle(
    rows: Sequence[Mapping[str, Any]], budget: int, validity_threshold: float
) -> List[Dict[str, Any]]:
    results = []
    for trajectory_id, current in group_rows(rows).items():
        selected = rank_rows(
            current,
            lambda row: (
                -int(float(row["predicted_validity"]) >= validity_threshold),
                -float(row["predicted_gain"]),
                -float(row["predicted_validity"]),
            ),
        )[:budget]
        gain_table = {
            str(row["candidate_id"]): float(row["gain"]) for row in current
        }
        observed = acquisition_metrics(
            gain_table, [str(row["candidate_id"]) for row in selected], budget
        )
        gains = list(gain_table.values())
        random_best = expected_random_best_gain(gains, budget)
        random_ndcg = expected_random_ndcg(gains, budget)
        oracle_ids = [
            str(row["candidate_id"])
            for row in sorted(current, key=lambda row: float(row["gain"]), reverse=True)[
                :budget
            ]
        ]
        oracle = acquisition_metrics(gain_table, oracle_ids, budget)
        results.append(
            {
                "trajectory_id": trajectory_id,
                "candidate_timesteps": sorted({int(row["timestep"]) for row in current}),
                "selected_ids": [str(row["candidate_id"]) for row in selected],
                "selected_passing_validity_filter": sum(
                    float(row["predicted_validity"]) >= validity_threshold
                    for row in selected
                ),
                "best_gain_at_4": float(observed["best_gain_at_b"]),
                "expected_random_best_gain_at_4": random_best,
                "oracle_best_gain_at_4": float(oracle["best_gain_at_b"]),
                "best_gain_descriptor_minus_random": float(observed["best_gain_at_b"])
                - random_best,
                "best_gain_oracle_minus_random": float(oracle["best_gain_at_b"])
                - random_best,
                "ndcg_at_4": float(observed["ndcg_at_b"]),
                "expected_random_ndcg_at_4": random_ndcg,
                "oracle_ndcg_at_4": float(oracle["ndcg_at_b"]),
                "ndcg_descriptor_minus_random": float(observed["ndcg_at_b"])
                - random_ndcg,
                "ndcg_oracle_minus_random": float(oracle["ndcg_at_b"])
                - random_ndcg,
            }
        )
    return results


def calibrate_late_threshold(
    rows: Sequence[Mapping[str, Any]], validity_threshold: float
) -> Dict[str, Any]:
    examples = []
    for trajectory_id, current in group_rows(rows).items():
        eligible_scores = [
            float(row["predicted_gain"])
            for row in current
            if float(row["predicted_validity"]) >= validity_threshold
        ]
        maximum = max(eligible_scores) if eligible_scores else float("-inf")
        examples.append(
            {
                "trajectory_id": trajectory_id,
                "maximum_eligible_predicted_gain": maximum,
                "has_positive_candidate": any(float(row["gain"]) > 0.0 for row in current),
            }
        )
    positives = sum(bool(item["has_positive_candidate"]) for item in examples)
    negatives = len(examples) - positives
    if not positives or not negatives:
        return {
            "status": "not-calibratable",
            "positive_trajectories": positives,
            "no_positive_trajectories": negatives,
            "examples": examples,
        }
    finite_scores = sorted(
        set(
            float(item["maximum_eligible_predicted_gain"])
            for item in examples
            if math.isfinite(float(item["maximum_eligible_predicted_gain"]))
        )
    )
    margin = max(1.0, finite_scores[-1] - finite_scores[0])
    thresholds = [finite_scores[0] - margin]
    thresholds.extend(
        (finite_scores[index] + finite_scores[index + 1]) / 2.0
        for index in range(len(finite_scores) - 1)
    )
    thresholds.extend(finite_scores)
    thresholds.append(finite_scores[-1] + margin)
    candidates = []
    for threshold in sorted(set(thresholds)):
        true_positive = sum(
            bool(item["has_positive_candidate"])
            and float(item["maximum_eligible_predicted_gain"]) > threshold
            for item in examples
        )
        true_negative = sum(
            not bool(item["has_positive_candidate"])
            and float(item["maximum_eligible_predicted_gain"]) <= threshold
            for item in examples
        )
        sensitivity = true_positive / float(positives)
        specificity = true_negative / float(negatives)
        candidates.append(
            {
                "threshold": threshold,
                "balanced_accuracy": (sensitivity + specificity) / 2.0,
                "sensitivity": sensitivity,
                "specificity": specificity,
            }
        )
    chosen = max(
        candidates,
        key=lambda item: (float(item["balanced_accuracy"]), float(item["threshold"])),
    )
    return {
        "status": "complete",
        "positive_trajectories": positives,
        "no_positive_trajectories": negatives,
        "chosen": chosen,
        "examples": examples,
    }


def evaluate_late(
    rows: Sequence[Mapping[str, Any]],
    budget: int,
    validity_threshold: float,
    gain_threshold: float,
) -> List[Dict[str, Any]]:
    results = []
    for trajectory_id, current in group_rows(rows).items():
        eligible = [
            row
            for row in current
            if float(row["predicted_validity"]) >= validity_threshold
            and float(row["predicted_gain"]) > gain_threshold
        ]
        selected = rank_rows(
            eligible, lambda row: (-float(row["predicted_gain"]),)
        )[:budget]
        has_positive = any(float(row["gain"]) > 0.0 for row in current)
        best_gain = max((float(row["gain"]) for row in selected), default=0.0)
        oracle_best = max(0.0, max(float(row["gain"]) for row in current))
        results.append(
            {
                "trajectory_id": trajectory_id,
                "candidate_timesteps": sorted({int(row["timestep"]) for row in current}),
                "has_positive_candidate": has_positive,
                "abstained": len(selected) == 0,
                "selected_count": len(selected),
                "selected_ids": [str(row["candidate_id"]) for row in selected],
                "best_gain_with_abstention": best_gain,
                "oracle_best_gain_with_abstention": oracle_best,
                "regret_with_abstention": oracle_best - best_gain,
                "correct_abstention": (not has_positive and len(selected) == 0),
                "opportunity_response": (has_positive and len(selected) > 0),
            }
        )
    return results


def mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / float(len(values))


def headroom_fraction(differences: Sequence[float], headrooms: Sequence[float]) -> float:
    denominator = mean(headrooms)
    return mean(differences) / denominator if denominator > 0.0 else float("nan")


def trajectory_cross_timestep_fraction(
    stages: Mapping[str, Sequence[Mapping[str, Any]]]
) -> float:
    trajectories = [item for stage in STAGES for item in stages[stage]]
    return sum(len(item["candidate_timesteps"]) > 1 for item in trajectories) / float(
        len(trajectories)
    )


def structure_keys(rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]]) -> set:
    return {str(row["structure_key"]) for row in flatten_rows(rows_by_stage)}
