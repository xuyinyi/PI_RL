"""Fail-closed polyBERT asset binding for SciCF integration-v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from RL_PPO.envs.embedding import _checkpoint_fingerprint


MODEL_ASSET_BINDING_RELATIVE = Path(
    "reproduction/scicf/online/configs/polybert_asset_binding_v1.json"
)
MODEL_ASSET_BINDING_CLASSIFICATION = (
    "reconstructed_afp_compatibility_polybert_asset_only"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_file(root: Path, relative: str, expected_sha256: str) -> Path:
    root = root.resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if root != path and root not in path.parents:
        raise ValueError("polyBERT asset binding escaped the repository root")
    if not path.is_file():
        raise FileNotFoundError("polyBERT bound input is not a file: %s" % path)
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(
            "polyBERT bound input hash mismatch for %s: %s != %s"
            % (relative, observed, expected_sha256)
        )
    return path


def _read_sha256_manifest(path: Path) -> Dict[str, str]:
    hashes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        name = relative.strip().replace("\\", "/")
        if name in hashes:
            raise ValueError("duplicate polyBERT manifest path: %s" % name)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid polyBERT manifest digest: %s" % digest)
        hashes[name] = digest
    if not hashes:
        raise ValueError("polyBERT required-file manifest is empty")
    return hashes


def load_polybert_asset_binding(root: Path) -> Mapping[str, Any]:
    root = root.resolve(strict=True)
    binding_path = (root / MODEL_ASSET_BINDING_RELATIVE).resolve(strict=True)
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "asset_binding_id",
        "classification",
        "checkpoint_fingerprint",
        "encoder_version",
        "required_file_manifest",
        "accepted_asset_gate",
        "accepted_fingerprint_report",
        "upstream_polybert_identity_verified",
    }
    if set(payload) != required:
        raise ValueError(
            "polyBERT binding keys differ; missing=%s extra=%s"
            % (sorted(required - set(payload)), sorted(set(payload) - required))
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported polyBERT asset binding schema")
    if not isinstance(payload["asset_binding_id"], str) or not payload[
        "asset_binding_id"
    ]:
        raise ValueError("polyBERT asset_binding_id must be non-empty")
    if payload["classification"] != MODEL_ASSET_BINDING_CLASSIFICATION:
        raise ValueError("polyBERT binding is not compatibility-only")
    fingerprint = payload["checkpoint_fingerprint"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
        char not in "0123456789abcdef" for char in fingerprint
    ):
        raise ValueError("invalid polyBERT checkpoint fingerprint")
    if payload["encoder_version"] != "polybert-sha256:%s" % fingerprint[:24]:
        raise ValueError("polyBERT encoder version does not match fingerprint")
    if payload["upstream_polybert_identity_verified"] is not False:
        raise ValueError("compatibility binding cannot claim verified upstream identity")

    for name in (
        "required_file_manifest",
        "accepted_asset_gate",
        "accepted_fingerprint_report",
    ):
        record = payload[name]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError("invalid polyBERT binding record: %s" % name)

    manifest_path = _resolve_repo_file(
        root,
        payload["required_file_manifest"]["path"],
        payload["required_file_manifest"]["sha256"],
    )
    asset_gate_path = _resolve_repo_file(
        root,
        payload["accepted_asset_gate"]["path"],
        payload["accepted_asset_gate"]["sha256"],
    )
    fingerprint_report_path = _resolve_repo_file(
        root,
        payload["accepted_fingerprint_report"]["path"],
        payload["accepted_fingerprint_report"]["sha256"],
    )
    required_hashes = _read_sha256_manifest(manifest_path)
    asset_gate = json.loads(asset_gate_path.read_text(encoding="utf-8"))
    fingerprint_report = json.loads(
        fingerprint_report_path.read_text(encoding="utf-8")
    )
    if asset_gate.get("polybert", {}).get("ready") is not True:
        raise RuntimeError("accepted asset gate does not mark polyBERT ready")
    compatibility = asset_gate.get("reconstructed_compatibility", {})
    if compatibility.get("ready") is not True:
        raise RuntimeError("accepted reconstructed compatibility asset gate is closed")
    if compatibility.get("polybert_identity_verified") is not False:
        raise RuntimeError("accepted asset gate changed upstream identity boundary")
    if compatibility.get("polybert_hashes") != required_hashes:
        raise RuntimeError("accepted asset gate and required-file manifest differ")
    if fingerprint_report.get("status") != "passed":
        raise RuntimeError("accepted polyBERT fingerprint report is not passed")
    if fingerprint_report.get("checkpoint_fingerprint") != fingerprint:
        raise RuntimeError("accepted fingerprint report and binding differ")
    if fingerprint_report.get("encoder_version") != payload["encoder_version"]:
        raise RuntimeError("accepted encoder version and binding differ")

    result = dict(payload)
    result["binding_path"] = str(binding_path)
    result["binding_sha256"] = sha256_path(binding_path)
    result["required_file_sha256"] = required_hashes
    return result


def validate_polybert_asset(
    model_path: Path, binding: Mapping[str, Any]
) -> Mapping[str, Any]:
    model_path = model_path.resolve(strict=True)
    if not model_path.is_dir():
        raise FileNotFoundError("polyBERT path is not a directory: %s" % model_path)
    required_hashes = binding["required_file_sha256"]
    observed_hashes = {}
    for relative, expected_digest in sorted(required_hashes.items()):
        path = (model_path / relative).resolve(strict=True)
        if model_path != path and model_path not in path.parents:
            raise ValueError("polyBERT required file escaped model root: %s" % relative)
        if not path.is_file():
            raise FileNotFoundError("polyBERT required asset is not a file: %s" % relative)
        if path.stat().st_size <= 0:
            raise ValueError("polyBERT required asset is empty: %s" % relative)
        observed = sha256_path(path)
        observed_hashes[relative] = observed
        if observed != expected_digest:
            raise RuntimeError(
                "polyBERT required asset hash mismatch for %s: %s != %s"
                % (relative, observed, expected_digest)
            )
    observed_fingerprint = _checkpoint_fingerprint(str(model_path))
    expected_fingerprint = binding["checkpoint_fingerprint"]
    if observed_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "polyBERT checkpoint fingerprint mismatch: %s != %s"
            % (observed_fingerprint, expected_fingerprint)
        )
    return {
        "asset_binding_id": binding["asset_binding_id"],
        "asset_binding_sha256": binding["binding_sha256"],
        "model_path": str(model_path),
        "checkpoint_fingerprint": observed_fingerprint,
        "encoder_version": binding["encoder_version"],
        "required_file_count": len(observed_hashes),
        "required_file_sha256": observed_hashes,
        "upstream_polybert_identity_verified": False,
        "validated_before_credentials": True,
    }
