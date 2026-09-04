from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .chemistry import LegacyDAPiGenChemistryBackend
from .config import DAPiGenEnvConfig
from .core import BranchableDAPiGenCore
from .embedding import MorganFingerprintEncoder, PersistentPolyBERTEncoder
from .evaluator import (
    BudgetedCachingTerminalEvaluator,
    LegacyDAPiGenBenchmarkEvaluator,
    PersistentDAPiGenBenchmarkEvaluator,
    TerminalRewardAdapter,
)


@dataclass(frozen=True)
class CatalogReport:
    path: str
    raw_rows: int
    retained_actions: int
    canonical_duplicates_removed: int
    invalid_rows_removed: int
    retained_source_rows: Tuple[int, ...]
    retained_smiles: Tuple[str, ...]


@dataclass(frozen=True)
class Stage0Components:
    core: Any
    evaluator: Any
    reward_adapter: Any
    catalog_reports: Tuple[CatalogReport, CatalogReport]
    repository_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __iter__(self):
        # Backward-compatible unpacking: core, evaluator, reward_adapter.
        yield self.core
        yield self.evaluator
        yield self.reward_adapter



def _repository_metadata(root: Path) -> Dict[str, Any]:
    """Record the checkout and task-critical DAPiGen sources used by a run."""

    critical_paths = (
        "RL_PPO/moldr/env.py",
        "RL_PPO/moldr/utils.py",
        "RL_PPO/utils/genPI.py",
        "RL_PPO/GNN/benchmarks.py",
        "RL_PPO/utils/polyBERT.py",
        "RL_PPO/outputs/building_blocks/blocks_dianhydride.csv",
        "RL_PPO/outputs/building_blocks/blocks_diamine.csv",
    )
    hashes = {}
    for relative in critical_paths:
        path = root / relative
        hashes[relative] = _sha256_path(path) if path.is_file() else None

    commit = None
    dirty = None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            universal_newlines=True,
        )
        commit = result.stdout.strip() or None
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            universal_newlines=True,
        )
        dirty = bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "dapigen_git_commit": commit,
        "dapigen_git_dirty": dirty,
        "critical_source_sha256": hashes,
        "qspr_code_tree_sha256": _sha256_tree(
            root / "RL_PPO" / "GNN" / "model", suffixes=(".py",)
        ),
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path, suffixes=None) -> Optional[str]:
    if not root.is_dir():
        return None
    files = [item for item in root.rglob("*") if item.is_file()]
    if suffixes is not None:
        suffixes = tuple(str(value).lower() for value in suffixes)
        files = [item for item in files if item.suffix.lower() in suffixes]
    if not files:
        return None
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: str(value.relative_to(root))):
        relative = str(item.relative_to(root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        with item.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def load_block_catalog(
    path: str,
    chemistry=None,
    deduplicate: bool = True,
) -> Tuple[Sequence[str], CatalogReport]:
    values = []
    invalid = 0
    raw_rows = 0
    seen = set()
    duplicate_count = 0
    retained_rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "block" not in reader.fieldnames:
            raise ValueError("Building-block CSV must contain a 'block' column.")
        for row in reader:
            raw_rows += 1
            raw = str(row.get("block", "")).strip()
            if not raw:
                invalid += 1
                continue
            try:
                canonical = chemistry.canonicalize(raw) if chemistry else raw
            except Exception:
                invalid += 1
                continue
            if deduplicate and canonical in seen:
                duplicate_count += 1
                continue
            seen.add(canonical)
            values.append(canonical)
            retained_rows.append(raw_rows)
    if not values:
        raise ValueError("Building-block catalog is empty after validation: %s" % path)
    return values, CatalogReport(
        path=str(path),
        raw_rows=int(raw_rows),
        retained_actions=len(values),
        canonical_duplicates_removed=int(duplicate_count),
        invalid_rows_removed=int(invalid),
        retained_source_rows=tuple(retained_rows),
        retained_smiles=tuple(values),
    )


def load_environment_config(path: str) -> DAPiGenEnvConfig:
    with open(path, "r") as handle:
        payload = json.load(handle)
    return DAPiGenEnvConfig.from_mapping(payload)


def build_stage0_components(
    dapigen_root: str,
    polybert_path: Optional[str] = None,
    config: Optional[DAPiGenEnvConfig] = None,
    device: str = "cpu",
    maximum_requested_calls: Optional[int] = None,
    maximum_unique_calls: Optional[int] = None,
    encoder_mode: str = "polybert",
    evaluator_mode: str = "persistent",
    encoder=None,
    terminal_evaluator=None,
    evaluator_service=None,
    deduplicate_catalogs: bool = True,
    evaluator_fail_fast: bool = True,
    allowed_evaluator_sources: Optional[Sequence[str]] = None,
    cache_scope: str = "per_run",
    polybert_checkpoint_fingerprint: Optional[str] = None,
    allow_rdkit_brics_fallback: bool = False,
) -> Stage0Components:
    """Construct the single environment/evaluator stack used by every method.

    ``encoder_mode='morgan'`` is intended for structural smoke tests only.
    ``evaluator_mode='legacy'`` is intended for output-regression checks only.
    Formal PPO/Policy-CC/MCC-PPO comparisons should all call this same factory
    with one immutable environment config and fresh per-run evaluator ledger.
    """

    root = Path(dapigen_root).resolve()
    if not (root / "RL_PPO" / "moldr" / "env.py").exists():
        raise FileNotFoundError("Not a DAPiGen checkout: %s" % root)
    task_config = config or DAPiGenEnvConfig()
    block_dir = root / "RL_PPO" / "outputs" / "building_blocks"
    chemistry = LegacyDAPiGenChemistryBackend(
        allow_rdkit_brics_fallback=allow_rdkit_brics_fallback
    )
    dianhydride_blocks, d_report = load_block_catalog(
        str(block_dir / "blocks_dianhydride.csv"),
        chemistry=chemistry,
        deduplicate=deduplicate_catalogs,
    )
    diamine_blocks, a_report = load_block_catalog(
        str(block_dir / "blocks_diamine.csv"),
        chemistry=chemistry,
        deduplicate=deduplicate_catalogs,
    )

    if encoder is None:
        if encoder_mode == "polybert":
            if not polybert_path:
                raise ValueError("polybert_path is required for encoder_mode='polybert'.")
            encoder = PersistentPolyBERTEncoder(
                polybert_path,
                device=device,
                checkpoint_fingerprint=polybert_checkpoint_fingerprint,
            )
        elif encoder_mode == "morgan":
            encoder = MorganFingerprintEncoder(radius=2, number_of_bits=2048)
        else:
            raise ValueError("Unsupported encoder_mode: %s" % encoder_mode)

    core = BranchableDAPiGenCore(
        dianhydride_blocks=dianhydride_blocks,
        diamine_blocks=diamine_blocks,
        initial_dianhydride_smiles="[16*]c1ccc2c(c1)C(=O)OC2=O",
        initial_diamine_smiles="[16*]c1ccc(N)cc1",
        chemistry=chemistry,
        encoder=encoder,
        config=task_config,
    )

    if evaluator_service is not None:
        if terminal_evaluator is not None:
            raise ValueError(
                "Pass either evaluator_service or terminal_evaluator, not both."
            )
        evaluator = evaluator_service
        for required_name in (
            "evaluate_one",
            "ledger",
            "evaluator_version",
            "objective_contract",
            "state_dict",
            "load_state_dict",
        ):
            if not hasattr(evaluator, required_name):
                raise TypeError(
                    "evaluator_service is missing required interface: %s"
                    % required_name
                )
    else:
        if terminal_evaluator is None:
            if evaluator_mode == "persistent":
                terminal_evaluator = PersistentDAPiGenBenchmarkEvaluator(
                    str(root), device=device
                )
            elif evaluator_mode == "legacy":
                terminal_evaluator = LegacyDAPiGenBenchmarkEvaluator()
            else:
                raise ValueError("Unsupported evaluator_mode: %s" % evaluator_mode)

        evaluator = BudgetedCachingTerminalEvaluator(
            terminal_evaluator,
            maximum_requested_calls=maximum_requested_calls,
            maximum_unique_calls=maximum_unique_calls,
            canonicalizer=chemistry.canonicalize,
            validator=lambda smiles: "*" not in smiles,
            fail_fast=evaluator_fail_fast,
            allowed_sources=allowed_evaluator_sources,
            cache_scope=cache_scope,
        )
    reward_adapter = TerminalRewardAdapter(
        evaluator, failure_reward=core.config.failure_reward
    )
    return Stage0Components(
        core=core,
        evaluator=evaluator,
        reward_adapter=reward_adapter,
        catalog_reports=(d_report, a_report),
        repository_metadata=_repository_metadata(root),
    )


def _canonical_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _module_sha256(obj: Any) -> Optional[str]:
    try:
        path = Path(inspect.getsourcefile(obj)).resolve()
    except Exception:
        return None
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _package_version(module_name: str) -> Optional[str]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _runtime_contract() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "numpy": _package_version("numpy"),
        "rdkit": _package_version("rdkit"),
        "torch": _package_version("torch"),
        "dgl": _package_version("dgl"),
        "transformers": _package_version("transformers"),
        "gym": _package_version("gym"),
        "ray": _package_version("ray"),
    }


def environment_manifest(components: Stage0Components) -> Dict[str, Any]:
    evaluator = components.evaluator
    ledger = evaluator.ledger()
    environment_spec = components.core.specification()
    implementation_hashes = {
        "core": _module_sha256(type(components.core)),
        "chemistry": _module_sha256(type(components.core.chemistry)),
        "encoder": _module_sha256(type(components.core.encoder)),
        "evaluator_service": _module_sha256(type(evaluator)),
        "terminal_evaluator": _module_sha256(
            type(getattr(evaluator, "evaluator", evaluator))
        ),
        "reward_adapter": _module_sha256(type(components.reward_adapter)),
        "stage0_source_tree": _sha256_tree(
            Path(__file__).resolve().parent, suffixes=(".py",)
        ),
    }
    repository_metadata = dict(components.repository_metadata)
    runtime_contract = _runtime_contract()
    runtime_contract_id = _canonical_hash(runtime_contract)
    source_contract = {
        "dapigen_git_commit": repository_metadata.get("dapigen_git_commit"),
        "dapigen_git_dirty": repository_metadata.get("dapigen_git_dirty"),
        "critical_source_sha256": repository_metadata.get(
            "critical_source_sha256", {}
        ),
        "qspr_code_tree_sha256": repository_metadata.get(
            "qspr_code_tree_sha256"
        ),
    }
    task_contract = {
        "environment": environment_spec,
        "evaluator_version": evaluator.evaluator_version,
        "objective_contract": getattr(evaluator, "objective_contract", "unknown"),
        "reward_adapter_version": components.reward_adapter.adapter_version,
        "failure_reward": float(components.reward_adapter.failure_reward),
        "implementation_sha256": implementation_hashes,
        "dapigen_source_contract": source_contract,
        "runtime_contract": runtime_contract,
    }
    task_contract_id = _canonical_hash(task_contract)
    budget_contract = {
        "task_contract_id": task_contract_id,
        "maximum_requested_calls": ledger.get("maximum_requested_calls"),
        "maximum_unique_calls": ledger.get("maximum_unique_calls"),
        "cache_scope": ledger.get("cache_scope", "unknown"),
        "budget_protocol_version": ledger.get(
            "budget_protocol_version", "terminal-requested-unique-backend-v2"
        ),
    }
    return {
        "manifest_schema_version": 2,
        "environment": environment_spec,
        "catalog_reports": [
            {
                "path": item.path,
                "raw_rows": item.raw_rows,
                "retained_actions": item.retained_actions,
                "canonical_duplicates_removed": item.canonical_duplicates_removed,
                "invalid_rows_removed": item.invalid_rows_removed,
            }
            for item in components.catalog_reports
        ],
        "evaluator_version": evaluator.evaluator_version,
        "objective_contract": task_contract["objective_contract"],
        "reward_adapter_version": components.reward_adapter.adapter_version,
        "task_contract_id": task_contract_id,
        "budget_contract_id": _canonical_hash(budget_contract),
        "runtime_contract_id": runtime_contract_id,
        "oracle_budget": budget_contract,
        "implementation_sha256": implementation_hashes,
        "dapigen_repository": repository_metadata,
        "runtime": dict(
            runtime_contract,
            **{"platform": platform.platform()}
        ),
    }


def write_environment_manifest(
    path: str, components: Stage0Components, extra: Optional[Dict[str, Any]] = None
) -> None:
    payload = environment_manifest(components)
    if extra:
        payload["extra"] = dict(extra)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_action_catalogs(directory: str, components: Stage0Components) -> None:
    """Write the exact post-validation action-id mapping used by the environment."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    side_data = (
        (
            "dianhydride_actions.csv",
            components.catalog_reports[0],
            components.core.dianhydride_metadata,
            components.core.dianhydride_noop_id,
        ),
        (
            "diamine_actions.csv",
            components.catalog_reports[1],
            components.core.diamine_metadata,
            components.core.diamine_noop_id,
        ),
    )
    for name, report, metadata, noop_id in side_data:
        with (target / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "action_id",
                    "source_row",
                    "canonical_smiles",
                    "action_type",
                    "attachment_labels",
                    "attachment_count",
                    "atom_count",
                ]
            )
            for action_id, (source_row, item) in enumerate(
                zip(report.retained_source_rows, metadata)
            ):
                action_type = (
                    "complete_monomer"
                    if item.is_complete_for_side
                    else ("fragment" if item.has_attachment else "unusable")
                )
                writer.writerow(
                    [
                        action_id,
                        source_row,
                        item.canonical_smiles,
                        action_type,
                        ";".join(str(value) for value in sorted(item.attachment_labels)),
                        item.attachment_count,
                        item.atom_count,
                    ]
                )
            writer.writerow([noop_id, "", "", "noop", "", 0, 0])
