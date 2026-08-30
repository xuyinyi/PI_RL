"""Reproducibility manifest construction for SciCF experiments."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from reproduction.framework.io import git_identity, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    repo_root: Path,
    config: Mapping[str, Any],
    run_id: str,
    policy_checkpoint: Optional[str],
    prompt_schema_version: Optional[str],
) -> Dict[str, Any]:
    environment_file = repo_root / "reproduction" / "environment-linux-h100.yml"
    return {
        "schema_version": 1,
        "method": "scicf-ppo",
        "status": "initializing",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": git_identity(repo_root),
        "upstream_dapigen_revision": config["provenance"]["upstream_revision"],
        "frozen_baseline_tag": config["provenance"]["baseline_tag"],
        "specification_sha256": config["provenance"]["specification_sha256"],
        "phase": config["phase"],
        "policy_checkpoint": policy_checkpoint,
        "prompt_schema_version": prompt_schema_version,
        "llm": {
            "enabled": config["llm"]["enabled"],
            "model_id": config["llm"].get("model_id"),
            "decoding": config["llm"].get("decoding", {}),
        },
        "acquisition": config["acquisition"],
        "oracle": config["verification"],
        "seeds": config["gate1"]["seeds"],
        "gate1": config["gate1"],
        "software_environment": {
            "identifier": "environment-linux-h100.yml",
            "sha256": _sha256(environment_file),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "resource_counters": {
            "environment_transitions": 0,
            "atomic_oracle_calls": {
                "factual": 0,
                "counterfactual": 0,
                "evaluation": 0,
                "total": 0,
            },
            "llm_requests": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "wallclock_seconds": 0.0,
        },
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    write_json(path, manifest)
