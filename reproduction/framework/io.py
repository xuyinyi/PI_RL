"""Small, dependency-free helpers for durable experiment records."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return json_value(value.item())
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_value(value), sort_keys=True) + "\n")
        handle.flush()


def git_identity(repo_root: Path) -> Dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=str(repo_root), text=True
        ).strip()

    try:
        status = git("status", "--porcelain")
        tags = git("tag", "--points-at", "HEAD").splitlines()
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "tags": sorted(tag for tag in tags if tag),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": "{}: {}".format(type(error).__name__, error)}
