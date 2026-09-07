"""Fail-closed AFP evaluator asset routing for SciCF integration-v2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping

from .model_asset import sha256_path


EVALUATOR_ASSET_BINDING_RELATIVE = Path(
    "reproduction/scicf/online/configs/afp_evaluator_asset_binding_v1.json"
)
EVALUATOR_ASSET_BINDING_CLASSIFICATION = (
    "reconstructed_afp_compatibility_evaluator_asset"
)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_repo_file(root: Path, record: Mapping[str, Any], name: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError("invalid AFP evaluator binding record: %s" % name)
    if not _valid_sha256(record["sha256"]):
        raise ValueError("invalid AFP evaluator bound-file digest: %s" % name)
    root = root.resolve(strict=True)
    path = (root / str(record["path"])).resolve(strict=True)
    if root != path and root not in path.parents:
        raise ValueError("AFP evaluator binding escaped the repository root")
    if not path.is_file():
        raise FileNotFoundError("AFP evaluator bound input is not a file: %s" % path)
    observed = sha256_path(path)
    if observed != record["sha256"]:
        raise RuntimeError(
            "AFP evaluator bound input hash mismatch for %s: %s != %s"
            % (record["path"], observed, record["sha256"])
        )
    return path


def _asset_fingerprint(required_hashes: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(required_hashes), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_evaluator_asset_binding(root: Path) -> Mapping[str, Any]:
    root = root.resolve(strict=True)
    binding_path = (root / EVALUATOR_ASSET_BINDING_RELATIVE).resolve(strict=True)
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "asset_binding_id",
        "classification",
        "compatibility_mode",
        "original_author_weights",
        "evaluator_version",
        "required_file_sha256",
        "accepted_asset_gate",
        "accepted_stage0_manifest",
        "fpscores_reference",
        "allowed_stage0_source_delta",
    }
    if set(payload) != required:
        raise ValueError(
            "AFP evaluator binding keys differ; missing=%s extra=%s"
            % (sorted(required - set(payload)), sorted(set(payload) - required))
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported AFP evaluator asset binding schema")
    if not isinstance(payload["asset_binding_id"], str) or not payload[
        "asset_binding_id"
    ]:
        raise ValueError("AFP evaluator asset_binding_id must be non-empty")
    if payload["classification"] != EVALUATOR_ASSET_BINDING_CLASSIFICATION:
        raise ValueError("AFP evaluator binding is not compatibility-only")
    if payload["compatibility_mode"] != "reconstructed-afp-compatibility":
        raise ValueError("AFP evaluator compatibility mode changed")
    if payload["original_author_weights"] is not False:
        raise ValueError("AFP evaluator binding cannot claim original author weights")
    if payload["allowed_stage0_source_delta"] != ["RL_PPO/envs/evaluator.py"]:
        raise ValueError("AFP evaluator source-delta allowlist changed")

    required_hashes = payload["required_file_sha256"]
    if not isinstance(required_hashes, Mapping) or len(required_hashes) != 13:
        raise ValueError("AFP evaluator binding must contain exactly 13 files")
    for name, digest in required_hashes.items():
        if Path(name).name != name or not _valid_sha256(digest):
            raise ValueError("invalid AFP evaluator required file: %s" % name)

    asset_gate_path = _resolve_repo_file(
        root, payload["accepted_asset_gate"], "accepted_asset_gate"
    )
    accepted_manifest_path = _resolve_repo_file(
        root, payload["accepted_stage0_manifest"], "accepted_stage0_manifest"
    )
    fpscores_path = _resolve_repo_file(
        root, payload["fpscores_reference"], "fpscores_reference"
    )
    asset_gate = json.loads(asset_gate_path.read_text(encoding="utf-8"))
    accepted_manifest = json.loads(
        accepted_manifest_path.read_text(encoding="utf-8")
    )
    compatibility = asset_gate.get("reconstructed_compatibility", {})
    if compatibility.get("ready") is not True:
        raise RuntimeError("accepted AFP reconstructed compatibility gate is closed")
    if compatibility.get("mode") != payload["compatibility_mode"]:
        raise RuntimeError("accepted AFP compatibility mode differs from binding")
    if compatibility.get("original_author_weights") is not False:
        raise RuntimeError("accepted AFP gate changed original-weight boundary")
    reward_hashes = dict(required_hashes)
    fpscores_digest = reward_hashes.pop("fpscores.pkl.gz", None)
    if compatibility.get("reward_hashes") != reward_hashes:
        raise RuntimeError("accepted AFP gate and evaluator binding hashes differ")
    if fpscores_digest != payload["fpscores_reference"]["sha256"]:
        raise RuntimeError("AFP fpscores binding differs from repository reference")
    if sha256_path(fpscores_path) != fpscores_digest:
        raise RuntimeError("AFP fpscores repository reference changed")
    if accepted_manifest.get("evaluator_version") != payload["evaluator_version"]:
        raise RuntimeError("accepted evaluator version differs from AFP binding")

    result = dict(payload)
    result["binding_path"] = str(binding_path)
    result["binding_sha256"] = sha256_path(binding_path)
    result["asset_fingerprint"] = _asset_fingerprint(required_hashes)
    result["required_file_sha256"] = dict(required_hashes)
    return result


def validate_evaluator_asset(
    asset_path: Path, binding: Mapping[str, Any]
) -> Mapping[str, Any]:
    asset_path = asset_path.resolve(strict=True)
    if not asset_path.is_dir():
        raise FileNotFoundError("AFP evaluator asset path is not a directory: %s" % asset_path)
    expected_hashes = dict(binding["required_file_sha256"])
    observed_hashes: Dict[str, str] = {}
    for relative, expected_digest in sorted(expected_hashes.items()):
        path = (asset_path / relative).resolve(strict=True)
        if asset_path != path and asset_path not in path.parents:
            raise ValueError("AFP evaluator required file escaped asset root: %s" % relative)
        if not path.is_file():
            raise FileNotFoundError("AFP evaluator required asset is not a file: %s" % relative)
        if path.stat().st_size <= 0:
            raise ValueError("AFP evaluator required asset is empty: %s" % relative)
        observed = sha256_path(path)
        observed_hashes[relative] = observed
        if observed != expected_digest:
            raise RuntimeError(
                "AFP evaluator required asset hash mismatch for %s: %s != %s"
                % (relative, observed, expected_digest)
            )

    asset_suffixes = (".pt", ".pkl", ".csv", ".gz")
    discovered = {
        item.name
        for item in asset_path.iterdir()
        if item.is_file() and item.name.endswith(asset_suffixes)
    }
    if discovered != set(expected_hashes):
        raise RuntimeError(
            "AFP evaluator asset inventory differs; missing=%s extra=%s"
            % (
                sorted(set(expected_hashes) - discovered),
                sorted(discovered - set(expected_hashes)),
            )
        )
    observed_fingerprint = _asset_fingerprint(observed_hashes)
    if observed_fingerprint != binding["asset_fingerprint"]:
        raise RuntimeError("AFP evaluator asset fingerprint mismatch")
    return {
        "asset_binding_id": binding["asset_binding_id"],
        "asset_binding_sha256": binding["binding_sha256"],
        "asset_path": str(asset_path),
        "asset_fingerprint": observed_fingerprint,
        "required_file_count": len(observed_hashes),
        "required_file_sha256": observed_hashes,
        "evaluator_version": binding["evaluator_version"],
        "compatibility_mode": binding["compatibility_mode"],
        "original_author_weights": False,
        "validated_before_credentials": True,
    }


def verify_evaluator_route_source_delta(
    root: Path, accepted_commit: str, binding: Mapping[str, Any]
) -> Mapping[str, Any]:
    root = root.resolve(strict=True)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-only",
            str(accepted_commit),
            "HEAD",
            "--",
            "RL_PPO/envs",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        universal_newlines=True,
    )
    observed = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    expected = sorted(binding["allowed_stage0_source_delta"])
    if observed != expected:
        raise RuntimeError(
            "AFP evaluator routing source delta differs; observed=%s expected=%s"
            % (observed, expected)
        )
    return {
        "accepted_commit": str(accepted_commit),
        "allowed_paths": expected,
        "observed_paths": observed,
        "current_file_sha256": {
            relative: sha256_path(root / relative) for relative in observed
        },
        "exact": True,
    }
