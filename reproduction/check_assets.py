#!/usr/bin/env python3
"""Fail-closed check for assets required by the public DAPiGen PPO code."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


POLYBERT_EXPECTED_REPO = "kuelumbus/polyBERT"
POLYBERT_EXPECTED_REVISION = "3675d021d0938179b6b57dfcca4753ca34048182"
POLYBERT_INSTALLED_REPO = "xushijie/polyBERT"
POLYBERT_INSTALLED_REVISION = "e7dce434fb3eff37905dc114008660e5479ca9a8"
POLYBERT_IDENTITY_VERIFIED = False

ASSET_MODE_ORIGINAL = "original"
ASSET_MODE_RECONSTRUCTED = "reconstructed-afp-compatibility"
ASSET_MODES = (ASSET_MODE_ORIGINAL, ASSET_MODE_RECONSTRUCTED)
AFP_COMPATIBILITY_LABEL = "reconstructed AFP compatibility baseline"
AFP_VALIDATION_REPORT = Path(
    "reproduction/results/afp-full-20260830/validation-report.json"
)
AFP_VALIDATION_REPORT_SHA256 = (
    "7ea185632dbe0e9368e6bd1a435f489b223f86cadc2bf83617460e2f1e4f7bfe"
)

POLYBERT_FILES = (
    "config.json",
    "pytorch_model.bin",
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

REWARD_FILES = (
    "transmittance(400)_scaler.pkl",
    "cte_scaler.pkl",
    "strength_scaler.pkl",
    "tg_scaler.pkl",
    "Ensemble_transmittance(400)_AFP_43.pt",
    "Ensemble_cte_AFP_56.pt",
    "Ensemble_strength_AFP_84.pt",
    "Ensemble_tg_AFP_64.pt",
    "Ensemble_transmittance(400)_AFP_settings.csv",
    "Ensemble_cte_AFP_settings.csv",
    "Ensemble_strength_AFP_settings.csv",
    "Ensemble_tg_AFP_settings.csv",
)

REWARD_FILE_MAP = {
    "transmittance(400)": {
        "scaler": "transmittance(400)_scaler.pkl",
        "weight": "Ensemble_transmittance(400)_AFP_43.pt",
        "settings": "Ensemble_transmittance(400)_AFP_settings.csv",
    },
    "cte": {
        "scaler": "cte_scaler.pkl",
        "weight": "Ensemble_cte_AFP_56.pt",
        "settings": "Ensemble_cte_AFP_settings.csv",
    },
    "strength": {
        "scaler": "strength_scaler.pkl",
        "weight": "Ensemble_strength_AFP_84.pt",
        "settings": "Ensemble_strength_AFP_settings.csv",
    },
    "tg": {
        "scaler": "tg_scaler.pkl",
        "weight": "Ensemble_tg_AFP_64.pt",
        "settings": "Ensemble_tg_AFP_settings.csv",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha256_manifest(path: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        hashes[name.strip()] = digest
    return hashes


def inspect_reconstructed_compatibility(repo_root: Path) -> Dict[str, object]:
    evidence_path = repo_root / AFP_VALIDATION_REPORT
    polybert_hash_path = repo_root / "reproduction/polybert-e7dce434.sha256"
    reward_root = repo_root / "RL_PPO/GNN/model"
    polybert_root = repo_root / "RL_PPO/models"
    errors: List[str] = []
    checked_reward_hashes: Dict[str, str] = {}
    checked_polybert_hashes: Dict[str, str] = {}

    evidence = None
    if not evidence_path.is_file():
        errors.append("missing AFP validation report: " + str(evidence_path))
    else:
        report_hash = sha256_file(evidence_path)
        if report_hash != AFP_VALIDATION_REPORT_SHA256:
            errors.append("AFP validation report SHA-256 mismatch")
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append("invalid AFP validation report: " + str(error))

    if evidence is not None:
        if evidence.get("status") != "passed":
            errors.append("AFP validation report status is not passed")
        if evidence.get("label") != AFP_COMPATIBILITY_LABEL:
            errors.append("AFP compatibility label mismatch")
        if evidence.get("original_author_weights") is not False:
            errors.append("AFP evidence does not preserve reconstructed-weight identity")
        for property_name, file_map in REWARD_FILE_MAP.items():
            property_evidence = evidence.get("properties", {}).get(property_name, {})
            expected_hashes = property_evidence.get("hashes", {})
            for kind, filename in file_map.items():
                path = reward_root / filename
                if not path.is_file():
                    errors.append("missing reconstructed reward asset: " + filename)
                    continue
                digest = sha256_file(path)
                checked_reward_hashes[filename] = digest
                if digest != expected_hashes.get(kind):
                    errors.append("reconstructed reward SHA-256 mismatch: " + filename)

    if not polybert_hash_path.is_file():
        errors.append("missing polyBERT SHA-256 manifest")
    else:
        expected_polybert_hashes = read_sha256_manifest(polybert_hash_path)
        for filename, expected_digest in expected_polybert_hashes.items():
            path = polybert_root / filename
            if not path.is_file():
                errors.append("missing polyBERT mirror asset: " + filename)
                continue
            digest = sha256_file(path)
            checked_polybert_hashes[filename] = digest
            if digest != expected_digest:
                errors.append("polyBERT mirror SHA-256 mismatch: " + filename)

    return {
        "mode": ASSET_MODE_RECONSTRUCTED,
        "label": AFP_COMPATIBILITY_LABEL,
        "original_author_weights": False,
        "polybert_identity_verified": False,
        "evidence_report": str(evidence_path),
        "evidence_report_sha256": (
            sha256_file(evidence_path) if evidence_path.is_file() else None
        ),
        "reward_hashes": checked_reward_hashes,
        "polybert_hashes": checked_polybert_hashes,
        "errors": errors,
        "ready": not errors,
    }


def inspect(root: Path, names: tuple) -> Dict[str, object]:
    missing: List[str] = []
    empty: List[str] = []
    present: List[str] = []
    for name in names:
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size == 0:
            empty.append(name)
        else:
            present.append(name)
    return {
        "root": str(root),
        "present": present,
        "missing": missing,
        "empty": empty,
        "ready": not missing and not empty,
    }


def build_report(repo_root: Path) -> Dict[str, object]:
    repo_root = repo_root.resolve()
    report: Dict[str, object] = {
        "polybert_source": {
            "expected_repo": POLYBERT_EXPECTED_REPO,
            "expected_revision": POLYBERT_EXPECTED_REVISION,
            "installed_repo": POLYBERT_INSTALLED_REPO,
            "installed_revision": POLYBERT_INSTALLED_REVISION,
            "identity_verified": POLYBERT_IDENTITY_VERIFIED,
        },
        "polybert": inspect(repo_root / "RL_PPO" / "models", POLYBERT_FILES),
        "reward_models": inspect(repo_root / "RL_PPO" / "GNN" / "model", REWARD_FILES),
    }
    report["ppo_runtime_files_ready"] = bool(
        report["polybert"]["ready"] and report["reward_models"]["ready"]
    )
    report["ppo_assets_ready"] = bool(
        report["ppo_runtime_files_ready"]
        and report["polybert_source"]["identity_verified"]
    )
    report["reconstructed_compatibility"] = inspect_reconstructed_compatibility(
        repo_root
    )
    report["ppo_reconstructed_compatibility_ready"] = bool(
        report["ppo_runtime_files_ready"]
        and report["reconstructed_compatibility"]["ready"]
    )
    return report


def require_assets(repo_root: Path, asset_mode: str = ASSET_MODE_ORIGINAL) -> None:
    if asset_mode not in ASSET_MODES:
        raise ValueError("unknown DAPiGen asset mode: " + asset_mode)
    report = build_report(repo_root)
    ready_key = (
        "ppo_assets_ready"
        if asset_mode == ASSET_MODE_ORIGINAL
        else "ppo_reconstructed_compatibility_ready"
    )
    if not report[ready_key]:
        raise FileNotFoundError(
            "DAPiGen PPO assets do not satisfy mode {!r}:\n".format(asset_mode)
            + json.dumps(report, indent=2, sort_keys=True)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--asset-mode", choices=ASSET_MODES, default=ASSET_MODE_ORIGINAL
    )
    args = parser.parse_args()
    report = build_report(args.repo_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    ready_key = (
        "ppo_assets_ready"
        if args.asset_mode == ASSET_MODE_ORIGINAL
        else "ppo_reconstructed_compatibility_ready"
    )
    return 0 if report[ready_key] else 1


if __name__ == "__main__":
    raise SystemExit(main())
