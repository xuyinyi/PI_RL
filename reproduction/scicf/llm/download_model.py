#!/usr/bin/env python3
"""Download and checksum the pinned local Gate 1 acquisition model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    resolved = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=str(args.output),
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.tiktoken",
                "*.safetensors",
                "*.py",
                "LICENSE",
                "README.md",
                "merges.txt",
                "vocab.json",
            ],
        )
    )
    files = []
    for path in sorted(resolved.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files.append(
                {
                    "path": str(path.relative_to(resolved)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "resolved_path": str(resolved),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    (resolved / "scicf-model-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "model_ready", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
