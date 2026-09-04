from __future__ import annotations

import gzip
import hashlib
import json
import math
import numbers
import os
import pickle
import re
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from .types import CoreTransition, EvaluatedTransition, TerminalEvaluation


EVALUATOR_STATE_SCHEMA_VERSION = 2
PAPER_OBJECTIVE_CONTRACT = "dapigen-paper-equation-1-weighted-average-v1"


def _strict_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, numbers.Integral
    ):
        raise TypeError("%s must be an integer." % name)
    return int(value)


def _positive_integer(value: Any, name: str) -> int:
    parsed = _strict_integer(value, name)
    if parsed <= 0:
        raise ValueError("%s must be positive." % name)
    return parsed


def _optional_nonnegative_integer(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    parsed = _strict_integer(value, name)
    if parsed < 0:
        raise ValueError("%s cannot be negative." % name)
    return parsed


class OracleBudgetExceeded(RuntimeError):
    """Raised before a call that would violate the configured evaluator budget."""


class CallableTerminalEvaluator(object):
    def __init__(
        self,
        function: Callable[[Sequence[str]], Sequence[TerminalEvaluation]],
        evaluator_version: str,
    ) -> None:
        self.function = function
        self.evaluator_version = str(evaluator_version)

    def evaluate_batch(self, canonical_smiles: Sequence[str]):
        outputs = list(self.function(canonical_smiles))
        if len(outputs) != len(canonical_smiles):
            raise RuntimeError("Evaluator callable returned an incorrect batch size.")
        return outputs


class BudgetedCachingTerminalEvaluator(object):
    """Terminal-only cache, strict budget enforcement and auditable ledger.

    ``requested_calls`` counts every complete-molecule request, including cache
    hits. ``unique_calls`` counts canonical molecules first encountered in this
    ledger. ``backend_calls`` counts cache-miss molecules submitted across the
    service boundary and can exceed ``unique_calls`` after LRU eviction or retry.
    Stage-0 comparisons should enforce the same requested budget and report all
    three quantities.
    """

    def __init__(
        self,
        evaluator,
        maximum_requested_calls: Optional[int] = None,
        maximum_unique_calls: Optional[int] = None,
        maximum_cache_entries: int = 500000,
        canonicalizer: Optional[Callable[[str], str]] = None,
        validator: Optional[Callable[[str], bool]] = None,
        maximum_audit_events: int = 100000,
        fail_fast: bool = True,
        allowed_sources: Optional[Sequence[str]] = None,
        cache_scope: str = "per_run",
    ) -> None:
        self.evaluator = evaluator
        self.maximum_requested_calls = _optional_nonnegative_integer(
            maximum_requested_calls, "maximum_requested_calls"
        )
        self.maximum_unique_calls = _optional_nonnegative_integer(
            maximum_unique_calls, "maximum_unique_calls"
        )
        self.maximum_cache_entries = _positive_integer(
            maximum_cache_entries, "maximum_cache_entries"
        )
        self.maximum_audit_events = _positive_integer(
            maximum_audit_events, "maximum_audit_events"
        )
        self.canonicalizer = canonicalizer or (lambda value: value)
        self.validator = validator or (lambda value: "*" not in value)
        self.fail_fast = bool(fail_fast)
        self.allowed_sources = (
            None
            if allowed_sources is None
            else frozenset(str(value) for value in allowed_sources)
        )
        self.cache_scope = str(cache_scope)
        if not self.cache_scope:
            raise ValueError("cache_scope must be a non-empty identifier.")
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._events = []
        self.requested_calls = 0
        self.unique_calls = 0
        self.backend_calls = 0
        self.cache_hits = 0
        self.invalid_results = 0
        self.requested_by_source = defaultdict(int)
        self.unique_by_source = defaultdict(int)
        self.backend_by_source = defaultdict(int)
        self._seen_smiles = set()

    @property
    def evaluator_version(self) -> str:
        return str(getattr(self.evaluator, "evaluator_version", "unknown"))

    @property
    def objective_contract(self) -> str:
        return str(getattr(self.evaluator, "objective_contract", "unknown"))

    def _canonicalize_and_validate(self, smiles: str) -> str:
        canonical = str(self.canonicalizer(smiles))
        if not bool(self.validator(canonical)):
            raise ValueError(
                "The terminal evaluator received an incomplete or invalid design: %s"
                % canonical
            )
        return canonical

    def _append_event(self, payload: Mapping[str, Any]) -> None:
        self._events.append(dict(payload))
        if len(self._events) > self.maximum_audit_events:
            del self._events[: len(self._events) - self.maximum_audit_events]

    @staticmethod
    def _smiles_sha256(smiles: str) -> str:
        return hashlib.sha256(smiles.encode("utf-8")).hexdigest()

    def _validate_result(
        self, expected_smiles: str, result: TerminalEvaluation
    ) -> TerminalEvaluation:
        if not isinstance(result, TerminalEvaluation):
            raise TypeError(
                "Wrapped evaluator must return TerminalEvaluation objects."
            )
        if not math.isfinite(float(result.objective)):
            raise ValueError("Terminal evaluator returned a non-finite objective.")
        for name, value in result.properties.items():
            if not math.isfinite(float(value)):
                raise ValueError(
                    "Terminal evaluator returned non-finite property %s." % name
                )
        if result.epistemic_std is not None:
            if not math.isfinite(float(result.epistemic_std)):
                raise ValueError("epistemic_std must be finite when provided.")
            if float(result.epistemic_std) < 0.0:
                raise ValueError("epistemic_std cannot be negative.")
        if result.canonical_smiles is not None:
            returned = self._canonicalize_and_validate(result.canonical_smiles)
            if returned != expected_smiles:
                raise ValueError(
                    "Evaluator output molecule does not match its input: %s != %s"
                    % (returned, expected_smiles)
                )
        if str(result.evaluator_version) != self.evaluator_version:
            raise ValueError(
                "Evaluator result version %s does not match service version %s."
                % (result.evaluator_version, self.evaluator_version)
            )
        return result

    def evaluate_batch(
        self, canonical_smiles: Sequence[str], source: str = "unspecified"
    ) -> Sequence[TerminalEvaluation]:
        source = str(source)
        if not source:
            raise ValueError("source must be a non-empty audit label.")
        if self.allowed_sources is not None and source not in self.allowed_sources:
            raise ValueError("Unregistered evaluator source: %s" % source)
        items = [self._canonicalize_and_validate(item) for item in canonical_smiles]
        if not items:
            return []

        # Holding the lock through the wrapped call prevents duplicate concurrent
        # oracle evaluations and keeps budget accounting transactional.
        with self._lock:
            available = set(self._cache)
            cache_hit_flags = []
            for smiles in items:
                cache_hit_flags.append(smiles in available)
                available.add(smiles)
            missing = []
            missing_seen = set()
            for smiles in items:
                if smiles not in self._cache and smiles not in missing_seen:
                    missing.append(smiles)
                    missing_seen.add(smiles)
            new_distinct = [smiles for smiles in missing if smiles not in self._seen_smiles]
            prospective_requested = self.requested_calls + len(items)
            prospective_unique = self.unique_calls + len(new_distinct)
            if (
                self.maximum_requested_calls is not None
                and prospective_requested > int(self.maximum_requested_calls)
            ):
                raise OracleBudgetExceeded(
                    "Requested evaluator-call budget exhausted: %d + %d > %d"
                    % (
                        self.requested_calls,
                        len(items),
                        int(self.maximum_requested_calls),
                    )
                )
            if (
                self.maximum_unique_calls is not None
                and prospective_unique > int(self.maximum_unique_calls)
            ):
                raise OracleBudgetExceeded(
                    "Unique evaluator-call budget exhausted: %d + %d > %d"
                    % (
                        self.unique_calls,
                        len(new_distinct),
                        int(self.maximum_unique_calls),
                    )
                )

            started = time.time()
            self.requested_calls = prospective_requested
            self.unique_calls = prospective_unique
            self.backend_calls += len(missing)
            self.cache_hits += len(items) - len(missing)
            self.requested_by_source[source] += len(items)
            self.unique_by_source[source] += len(new_distinct)
            self.backend_by_source[source] += len(missing)
            self._seen_smiles.update(missing)

            if missing:
                try:
                    evaluated = list(self.evaluator.evaluate_batch(missing))
                except Exception as exc:
                    if self.fail_fast:
                        self._append_event(
                            {
                                "timestamp": started,
                                "source": source,
                                "requested": len(items),
                                "unique": len(new_distinct),
                                "backend": len(missing),
                                "requested_smiles_sha256": [
                                    self._smiles_sha256(item) for item in items
                                ],
                                "backend_smiles_sha256": [
                                    self._smiles_sha256(item) for item in missing
                                ],
                                "status": "exception",
                                "exception": "%s: %s"
                                % (type(exc).__name__, exc),
                            }
                        )
                        raise
                    evaluated = [
                        TerminalEvaluation(
                            objective=0.0,
                            valid=False,
                            canonical_smiles=smiles,
                            failure_reason="%s: %s" % (type(exc).__name__, exc),
                            evaluator_version=self.evaluator_version,
                        )
                        for smiles in missing
                    ]
                try:
                    if len(evaluated) != len(missing):
                        raise RuntimeError(
                            "Terminal evaluator returned an incorrect batch size."
                        )
                    validated = [
                        self._validate_result(smiles, result)
                        for smiles, result in zip(missing, evaluated)
                    ]
                except Exception as exc:
                    self._append_event(
                        {
                            "timestamp": started,
                            "source": source,
                            "requested": len(items),
                            "unique": len(new_distinct),
                            "backend": len(missing),
                            "requested_smiles_sha256": [
                                self._smiles_sha256(item) for item in items
                            ],
                            "backend_smiles_sha256": [
                                self._smiles_sha256(item) for item in missing
                            ],
                            "status": "invalid_backend_result",
                            "exception": "%s: %s" % (type(exc).__name__, exc),
                        }
                    )
                    raise
                for smiles, result in zip(missing, validated):
                    self.invalid_results += int(not result.valid)
                    self._cache[smiles] = result
                    self._cache.move_to_end(smiles)
                    while len(self._cache) > self.maximum_cache_entries:
                        self._cache.popitem(last=False)

            outputs = []
            for smiles in items:
                value = self._cache.pop(smiles)
                self._cache[smiles] = value
                outputs.append(value)
            self._append_event(
                {
                    "timestamp": started,
                    "source": source,
                    "requested": len(items),
                    "unique": len(new_distinct),
                    "backend": len(missing),
                    "cache_reuse": len(items) - len(missing),
                    "cache_hit_flags": cache_hit_flags,
                    "requested_smiles_sha256": [
                        self._smiles_sha256(item) for item in items
                    ],
                    "backend_smiles_sha256": [
                        self._smiles_sha256(item) for item in missing
                    ],
                    "elapsed_seconds": float(time.time() - started),
                    "status": "ok",
                }
            )
            return outputs

    def evaluate_one(self, canonical_smiles: str, source: str) -> TerminalEvaluation:
        return self.evaluate_batch([canonical_smiles], source=source)[0]

    def ledger(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "budget_protocol_version": "terminal-requested-unique-backend-v2",
                "cache_scope": self.cache_scope,
                "evaluator_version": self.evaluator_version,
                "objective_contract": self.objective_contract,
                "requested_calls": int(self.requested_calls),
                "unique_calls": int(self.unique_calls),
                "backend_calls": int(self.backend_calls),
                "cache_hits": int(self.cache_hits),
                "invalid_results": int(self.invalid_results),
                "cache_entries": len(self._cache),
                "requested_by_source": dict(self.requested_by_source),
                "unique_by_source": dict(self.unique_by_source),
                "backend_by_source": dict(self.backend_by_source),
                "maximum_requested_calls": self.maximum_requested_calls,
                "maximum_unique_calls": self.maximum_unique_calls,
                "remaining_requested_calls": (
                    None
                    if self.maximum_requested_calls is None
                    else int(self.maximum_requested_calls) - int(self.requested_calls)
                ),
                "remaining_unique_calls": (
                    None
                    if self.maximum_unique_calls is None
                    else int(self.maximum_unique_calls) - int(self.unique_calls)
                ),
                "cache_hit_rate": (
                    0.0
                    if self.requested_calls == 0
                    else float(self.cache_hits) / float(self.requested_calls)
                ),
                "allowed_sources": (
                    None
                    if self.allowed_sources is None
                    else sorted(self.allowed_sources)
                ),
            }

    def audit_events(self):
        with self._lock:
            return [dict(event) for event in self._events]

    def write_audit(self, path: str) -> None:
        payload = {"ledger": self.ledger(), "events": self.audit_events()}
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize cache, budgets and audit state for exact run continuation."""

        with self._lock:
            return {
                "schema_version": EVALUATOR_STATE_SCHEMA_VERSION,
                "budget_protocol_version": "terminal-requested-unique-backend-v2",
                "cache_scope": self.cache_scope,
                "evaluator_version": self.evaluator_version,
                "objective_contract": self.objective_contract,
                "maximum_requested_calls": self.maximum_requested_calls,
                "maximum_unique_calls": self.maximum_unique_calls,
                "maximum_cache_entries": self.maximum_cache_entries,
                "maximum_audit_events": self.maximum_audit_events,
                "requested_calls": int(self.requested_calls),
                "unique_calls": int(self.unique_calls),
                "backend_calls": int(self.backend_calls),
                "cache_hits": int(self.cache_hits),
                "invalid_results": int(self.invalid_results),
                "requested_by_source": dict(self.requested_by_source),
                "unique_by_source": dict(self.unique_by_source),
                "backend_by_source": dict(self.backend_by_source),
                "seen_smiles": sorted(self._seen_smiles),
                "cache": [
                    {"canonical_smiles": smiles, "evaluation": value.to_dict()}
                    for smiles, value in self._cache.items()
                ],
                "events": [dict(event) for event in self._events],
            }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore only a state produced by the identical evaluator contract."""

        payload = dict(state)
        with self._lock:
            schema_version = _strict_integer(
                payload.get("schema_version"), "schema_version"
            )
            if schema_version != EVALUATOR_STATE_SCHEMA_VERSION:
                raise ValueError("Unsupported evaluator checkpoint schema.")
            expected = {
                "budget_protocol_version": "terminal-requested-unique-backend-v2",
                "cache_scope": self.cache_scope,
                "evaluator_version": self.evaluator_version,
                "objective_contract": self.objective_contract,
                "maximum_requested_calls": self.maximum_requested_calls,
                "maximum_unique_calls": self.maximum_unique_calls,
                "maximum_cache_entries": self.maximum_cache_entries,
                "maximum_audit_events": self.maximum_audit_events,
            }
            mismatches = {
                name: (expected_value, payload.get(name))
                for name, expected_value in expected.items()
                if payload.get(name) != expected_value
            }
            if mismatches:
                raise ValueError(
                    "Evaluator checkpoint contract mismatch: %s" % mismatches
                )
            cache = OrderedDict()
            for row in payload.get("cache", ()):
                smiles = self._canonicalize_and_validate(row["canonical_smiles"])
                result = self._validate_result(
                    smiles, TerminalEvaluation.from_dict(row["evaluation"])
                )
                if smiles in cache:
                    raise ValueError("Duplicate cache entry in evaluator checkpoint.")
                cache[smiles] = result
            if len(cache) > self.maximum_cache_entries:
                raise ValueError("Evaluator checkpoint cache exceeds configured limit.")
            seen = set(
                self._canonicalize_and_validate(item)
                for item in payload.get("seen_smiles", ())
            )
            if not set(cache).issubset(seen):
                raise ValueError("Evaluator checkpoint cache is not contained in seen_smiles.")
            requested = _strict_integer(
                payload.get("requested_calls"), "requested_calls"
            )
            unique = _strict_integer(
                payload.get("unique_calls"), "unique_calls"
            )
            backend = _strict_integer(
                payload.get("backend_calls"), "backend_calls"
            )
            cache_hits = _strict_integer(
                payload.get("cache_hits"), "cache_hits"
            )
            invalid = _strict_integer(
                payload.get("invalid_results"), "invalid_results"
            )
            if min(requested, unique, backend, cache_hits, invalid) < 0:
                raise ValueError("Evaluator checkpoint contains negative counters.")
            if (
                unique != len(seen)
                or backend < unique
                or requested < backend
                or cache_hits != requested - backend
                or invalid > backend
            ):
                raise ValueError("Evaluator checkpoint counters are inconsistent.")
            if self.maximum_requested_calls is not None and requested > int(
                self.maximum_requested_calls
            ):
                raise ValueError("Evaluator checkpoint exceeds requested-call budget.")
            if self.maximum_unique_calls is not None and unique > int(
                self.maximum_unique_calls
            ):
                raise ValueError("Evaluator checkpoint exceeds unique-call budget.")
            source_counters = {}
            for name, expected_total in (
                ("requested_by_source", requested),
                ("unique_by_source", unique),
                ("backend_by_source", backend),
            ):
                values = dict(payload.get(name, {}))
                parsed_values = {
                    str(key): _strict_integer(value, "%s[%s]" % (name, key))
                    for key, value in values.items()
                }
                if any(value < 0 for value in parsed_values.values()):
                    raise ValueError("Evaluator checkpoint has negative source counters.")
                if sum(parsed_values.values()) != expected_total:
                    raise ValueError(
                        "Evaluator checkpoint %s does not match its total." % name
                    )
                if self.allowed_sources is not None and not set(parsed_values).issubset(
                    self.allowed_sources
                ):
                    raise ValueError("Evaluator checkpoint contains unregistered sources.")
                source_counters[name] = parsed_values
            self._cache = cache
            self._seen_smiles = seen
            self._events = [dict(event) for event in payload.get("events", ())]
            if len(self._events) > self.maximum_audit_events:
                raise ValueError("Evaluator checkpoint contains too many audit events.")
            self.requested_calls = requested
            self.unique_calls = unique
            self.backend_calls = backend
            self.cache_hits = cache_hits
            self.invalid_results = invalid
            self.requested_by_source = defaultdict(
                int, source_counters["requested_by_source"]
            )
            self.unique_by_source = defaultdict(
                int, source_counters["unique_by_source"]
            )
            self.backend_by_source = defaultdict(
                int, source_counters["backend_by_source"]
            )

    def reset_ledger(self, retain_cache: bool = False) -> None:
        with self._lock:
            if retain_cache:
                raise ValueError(
                    "Retaining cache while resetting its ledger breaks per-run budget provenance."
                )
            self._cache.clear()
            self._events = []
            self.requested_calls = 0
            self.unique_calls = 0
            self.backend_calls = 0
            self.cache_hits = 0
            self.invalid_results = 0
            self.requested_by_source.clear()
            self.unique_by_source.clear()
            self.backend_by_source.clear()
            self._seen_smiles.clear()


class LegacyDAPiGenBenchmarkEvaluator(object):
    """Correctness adapter around the released DAPiGen ``Benchmark`` class.

    The released implementation repeatedly reloads models. Keep this adapter as
    a regression oracle; use ``PersistentDAPiGenBenchmarkEvaluator`` for actual
    Stage-0 training after verifying exact output agreement.
    """

    evaluator_version = "dapigen-benchmark-legacy-v1"
    objective_contract = PAPER_OBJECTIVE_CONTRACT

    def evaluate_batch(self, canonical_smiles: Sequence[str]):
        from RL_PPO.GNN.benchmarks import Benchmark

        outputs = []
        for smiles in canonical_smiles:
            try:
                score = Benchmark([smiles])
                outputs.append(
                    TerminalEvaluation(
                        objective=float(score.Score),
                        properties={
                            "transmittance": float(score.transmittance),
                            "cte": float(score.cte),
                            "strength": float(score.strength),
                            "tg": float(score.tg),
                            "sa_score": float(score.ScoreSA),
                        },
                        valid=True,
                        canonical_smiles=smiles,
                        evaluator_version=self.evaluator_version,
                    )
                )
            except Exception as exc:
                outputs.append(
                    TerminalEvaluation(
                        objective=0.0,
                        valid=False,
                        canonical_smiles=smiles,
                        failure_reason="%s: %s" % (type(exc).__name__, exc),
                        evaluator_version=self.evaluator_version,
                    )
                )
        return outputs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class PersistentDAPiGenBenchmarkEvaluator(object):
    """Persistent implementation of the frozen five-property QSPR objective.

    Models, scalers and SA fragment scores are loaded once. Predictions preserve
    the property rounding and the paper/archived-bytecode Equation (1) objective.
    Before primary experiments,
    run ``scripts/compare_evaluators.py`` and require exact equality after the
    original rounding on a fixed validation panel.
    """

    PROPERTY_SPECS = (
        ("transmittance", "transmittance(400)", 43),
        ("cte", "cte", 56),
        ("strength", "strength", 84),
        ("tg", "tg", 64),
    )
    objective_contract = PAPER_OBJECTIVE_CONTRACT

    def __init__(
        self,
        dapigen_root: str,
        device: Optional[str] = None,
        deterministic_torch: bool = True,
    ) -> None:
        import pandas as pd
        import torch

        self.pd = pd
        self.torch = torch
        self.root = Path(dapigen_root).resolve()
        self.gnn_dir = self.root / "RL_PPO" / "GNN"
        self.model_dir = self.gnn_dir / "model"
        if not self.model_dir.exists():
            raise FileNotFoundError("DAPiGen GNN model directory not found: %s" % self.model_dir)
        if str(self.gnn_dir) not in sys.path:
            sys.path.insert(0, str(self.gnn_dir))

        from model.networks.AttentiveFP import AttentiveFPNet as AFP
        from model.src.feature.atom_featurizer import classic_atom_featurizer
        from model.src.feature.bond_featurizer import classic_bond_featurizer
        from model.src.feature.mol_featurizer import classic_mol_featurizer
        from model.utils.mol2graph import smiles_2_bigraph

        self.AFP = AFP
        self.atom_featurizer = classic_atom_featurizer
        self.bond_featurizer = classic_bond_featurizer
        self.mol_featurizer = classic_mol_featurizer
        self.smiles_2_bigraph = smiles_2_bigraph
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        self.models = {}
        self.scalers = {}
        self.model_manifest = {}
        for output_name, dataset_name, model_id in self.PROPERTY_SPECS:
            model, scaler, manifest = self._load_property_model(
                dataset_name, model_id
            )
            self.models[output_name] = model
            self.scalers[output_name] = scaler
            self.model_manifest[output_name] = manifest
        self.fp_scores = self._load_fp_scores()
        self.model_manifest["objective"] = {
            "contract": PAPER_OBJECTIVE_CONTRACT
        }
        manifest_payload = json.dumps(
            self.model_manifest, sort_keys=True, separators=(",", ":")
        )
        self.evaluator_version = "dapigen-persistent-qspr-v2:%s" % hashlib.sha256(
            manifest_payload.encode("utf-8")
        ).hexdigest()[:16]

    def _load_property_model(self, dataset_name: str, model_id: int):
        torch = self.torch
        model_stem = "Ensemble_%s_AFP_%d" % (dataset_name, int(model_id))
        model_path = self.model_dir / (model_stem + ".pt")
        scaler_path = self.model_dir / (dataset_name + "_scaler.pkl")
        settings_stem = model_stem.rsplit("_%d" % int(model_id), 1)[0]
        settings_path = self.model_dir / (settings_stem + "_settings.csv")
        for required in (model_path, scaler_path, settings_path):
            if not required.exists():
                raise FileNotFoundError("Required evaluator file is missing: %s" % required)

        frame = self.pd.read_csv(
            str(settings_path), sep=",", encoding="windows-1250", index_col=-1
        )
        net_params = {}
        for column in frame.columns:
            if ":" not in column:
                continue
            prefix, name = column.split(":", 1)
            if prefix == "net_param":
                net_params[name] = frame[column][int(model_id)]
        if "sigmoid" not in net_params:
            net_params["sigmoid"] = False
        model = self.AFP(net_params).to(device=self.device)
        state = torch.load(str(model_path), map_location=self.device)
        model.load_state_dict(state, strict=False)
        model.eval()
        with scaler_path.open("rb") as handle:
            scaler = pickle.load(handle)
        manifest = {
            "dataset": dataset_name,
            "model_id": int(model_id),
            "model_sha256": _sha256_file(model_path),
            "scaler_sha256": _sha256_file(scaler_path),
            "settings_sha256": _sha256_file(settings_path),
        }
        return model, scaler, manifest

    def _load_fp_scores(self):
        path = self.model_dir / "fpscores.pkl.gz"
        if not path.exists():
            raise FileNotFoundError("SA score fragment file is missing: %s" % path)
        data = pickle.load(gzip.open(str(path)))
        scores = {}
        for row in data:
            for index in range(1, len(row)):
                scores[row[index]] = float(row[0])
        self.model_manifest["sa"] = {"fpscores_sha256": _sha256_file(path)}
        return scores

    def _graph(self, smiles: str):
        return self.smiles_2_bigraph(
            smiles,
            self.atom_featurizer,
            self.bond_featurizer,
            self.mol_featurizer,
        )

    def _predict(self, output_name: str, graphs):
        model = self.models[output_name]
        scores = []
        with self.torch.no_grad():
            for graph in graphs:
                node = graph.ndata["feat"].to(device=self.device)
                edge = graph.edata["feat"].to(device=self.device)
                graph_on_device = graph.to(device=self.device)
                scores.append(model.forward(graph_on_device, node, edge))
        raw = self.torch.cat(scores, dim=0)
        rescaled = self.scalers[output_name].ReScaler(
            raw.detach().to(device="cpu")
        )
        return np.asarray(rescaled).reshape(-1)

    @staticmethod
    def _score_cte(cte: float) -> float:
        value = math.fabs(float(cte))
        if value == 0:
            return 1.0
        if value >= 80:
            return 0.0
        return 1.0 - value / 80.0

    @staticmethod
    def _score_strength(strength: float) -> float:
        value = float(strength)
        if value > 500:
            return 1.0
        if value <= 30:
            return 0.0
        return (value - 30.0) / 440.0

    @staticmethod
    def _score_tg(tg: float) -> float:
        value = float(tg)
        if value > 600:
            return 1.0
        if value <= 100:
            return 0.0
        return (value - 100.0) / 500.0

    @staticmethod
    def _score_sa(sa_score: float) -> float:
        value = float(sa_score)
        if value < 2:
            return 1.0
        if value >= 5:
            return 0.0
        return 1.0 - (value - 2.0) / 3.0

    @classmethod
    def score_objective(
        cls,
        transmittance: float,
        cte: float,
        strength: float,
        tg: float,
        sa_score: float,
    ) -> float:
        coefficient = float(transmittance) / 100.0
        secondary_sum = (
            cls._score_cte(cte)
            + cls._score_strength(strength)
            + cls._score_tg(tg)
            + cls._score_sa(sa_score)
        )
        return float(round(coefficient * (1.0 + secondary_sum) / 5.0, 4))

    def _calculate_sa(self, molecule) -> float:
        from rdkit.Chem import rdMolDescriptors

        fingerprint = rdMolDescriptors.GetMorganFingerprint(molecule, 2)
        fps = fingerprint.GetNonzeroElements()
        score1 = 0.0
        count = 0
        for bit_id, multiplicity in fps.items():
            count += multiplicity
            score1 += self.fp_scores.get(bit_id, -4.0) * multiplicity
        if count <= 0:
            raise ValueError("Cannot compute SA score for an empty fingerprint.")
        score1 /= count

        number_of_atoms = molecule.GetNumAtoms()
        number_of_chiral_centers = len(
            self._chem().FindMolChiralCenters(molecule, includeUnassigned=True)
        )
        ring_info = molecule.GetRingInfo()
        number_of_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(molecule)
        number_of_spiro = rdMolDescriptors.CalcNumSpiroAtoms(molecule)
        number_of_macrocycles = sum(
            1 for ring in ring_info.AtomRings() if len(ring) > 8
        )
        size_penalty = number_of_atoms ** 1.005 - number_of_atoms
        stereo_penalty = math.log10(number_of_chiral_centers + 1)
        spiro_penalty = math.log10(number_of_spiro + 1)
        bridge_penalty = math.log10(number_of_bridgeheads + 1)
        macrocycle_penalty = math.log10(2.0) if number_of_macrocycles > 0 else 0.0
        score2 = -(
            size_penalty
            + stereo_penalty
            + spiro_penalty
            + bridge_penalty
            + macrocycle_penalty
        )
        score3 = 0.0
        if number_of_atoms > len(fps):
            score3 = math.log(float(number_of_atoms) / len(fps)) * 0.5
        raw = score1 + score2 + score3
        minimum = -4.0
        maximum = 2.5
        score = 11.0 - (raw - minimum + 1.0) / (maximum - minimum) * 9.0
        if score > 8.0:
            score = 8.0 + math.log(score + 1.0 - 9.0)
        if score > 10.0:
            score = 10.0
        elif score < 1.0:
            score = 1.0
        return float(score)

    @staticmethod
    def _chem():
        from rdkit import Chem

        return Chem

    def evaluate_batch(self, canonical_smiles: Sequence[str]):
        items = list(canonical_smiles)
        outputs = [None] * len(items)
        for index, smiles in enumerate(items):
            try:
                molecule = self._chem().MolFromSmiles(smiles)
                if molecule is None or "*" in smiles:
                    raise ValueError("Not a complete valid molecule.")
                graph = self._graph(smiles)
                if graph is None:
                    raise ValueError("Graph featurization failed.")
                predictions = {
                    name: float(self._predict(name, [graph])[0])
                    for name, _dataset, _model_id in self.PROPERTY_SPECS
                }
                transmittance = round(predictions["transmittance"], 2)
                cte = round(predictions["cte"], 2)
                strength = round(predictions["strength"], 2)
                tg = round(predictions["tg"], 2)
                sa_score = round(float(self._calculate_sa(molecule)), 2)
                objective = self.score_objective(
                    transmittance, cte, strength, tg, sa_score
                )
                outputs[index] = TerminalEvaluation(
                    objective=float(objective),
                    properties={
                        "transmittance": transmittance,
                        "cte": cte,
                        "strength": strength,
                        "tg": tg,
                        "sa_score": sa_score,
                    },
                    valid=True,
                    canonical_smiles=smiles,
                    evaluator_version=self.evaluator_version,
                )
            except Exception as exc:
                outputs[index] = TerminalEvaluation(
                    objective=0.0,
                    valid=False,
                    canonical_smiles=smiles,
                    failure_reason="%s: %s" % (type(exc).__name__, exc),
                    evaluator_version=self.evaluator_version,
                )
        return outputs


class TerminalRewardAdapter(object):
    """The single terminal reward definition used by all Stage-0 algorithms."""

    adapter_version = "terminal-only-reward-v1"

    def __init__(
        self,
        evaluator: BudgetedCachingTerminalEvaluator,
        failure_reward: float = 0.0,
    ) -> None:
        self.evaluator = evaluator
        self.failure_reward = float(failure_reward)

    def apply(self, transition: CoreTransition, source: str) -> EvaluatedTransition:
        if not (transition.terminated or transition.truncated):
            return EvaluatedTransition(core=transition, reward=0.0, evaluation=None)
        if transition.truncated and transition.terminal_smiles is None:
            # A time-limit truncation is not a completed scientific design and
            # receives no fabricated physical reward; value learning may bootstrap.
            return EvaluatedTransition(core=transition, reward=0.0, evaluation=None)
        if transition.terminal_smiles is None:
            return EvaluatedTransition(
                core=transition, reward=self.failure_reward, evaluation=None
            )
        evaluation = self.evaluator.evaluate_one(
            transition.terminal_smiles, source=str(source)
        )
        reward = float(evaluation.objective) if evaluation.valid else self.failure_reward
        return EvaluatedTransition(core=transition, reward=reward, evaluation=evaluation)
