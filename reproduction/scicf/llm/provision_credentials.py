#!/usr/bin/env python3
"""Provision a private API environment from a raw key inside Slurm."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path


_DEEPSEEK_FLASH_PROFILE = {
    "SCICF_LLM_API_URL": "https://api.deepseek.com/chat/completions",
    "SCICF_LLM_MODEL_ID": "deepseek-v4-flash",
    "SCICF_LLM_PROVIDER_ID": "deepseek-official",
    "SCICF_LLM_API_KEY_HEADER": "Authorization",
    "SCICF_LLM_API_KEY_PREFIX": "Bearer",
    "SCICF_LLM_ALLOW_INSECURE_HTTP": "false",
    "SCICF_LLM_INCLUDE_SEED": "false",
    "SCICF_LLM_JSON_MODE": "true",
    "SCICF_LLM_MAX_TOKENS_FIELD": "max_tokens",
    "SCICF_LLM_THINKING": "disabled",
}
_KEY_PATTERN = re.compile(r"^sk-[A-Za-z0-9_-]{16,256}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=["deepseek-v4-flash"], default="deepseek-v4-flash"
    )
    parser.add_argument("--model-revision", required=True)
    return parser.parse_args()


def _read_private_raw_key(path: Path) -> str:
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("raw API key path must be a regular file")
    if metadata.st_uid != os.getuid():
        raise ValueError("raw API key file must be owned by the Slurm user")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("raw API key file permissions must be 0600 or stricter")
    if metadata.st_size > 4096:
        raise ValueError("raw API key file is unexpectedly large")
    lines = [line.strip() for line in resolved.read_text(encoding="utf-8").splitlines()]
    nonempty = [line for line in lines if line]
    if len(nonempty) != 1 or not _KEY_PATTERN.fullmatch(nonempty[0]):
        raise ValueError("raw API key must contain exactly one valid sk- token")
    return nonempty[0]


def provision(
    raw_key_file: Path,
    output: Path,
    profile: str,
    model_revision: str,
) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("API credential provisioning must run through Slurm")
    if profile != "deepseek-v4-flash":
        raise ValueError("unsupported API credential profile")
    if not model_revision or "\n" in model_revision or "\r" in model_revision:
        raise ValueError("model revision must be a nonempty single-line value")

    api_key = _read_private_raw_key(raw_key_file)
    destination = output.resolve(strict=False)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_mode = stat.S_IMODE(destination.parent.stat().st_mode)
    if parent_mode & 0o077:
        raise ValueError("credential output directory must be 0700 or stricter")

    settings = dict(_DEEPSEEK_FLASH_PROFILE)
    settings["SCICF_LLM_API_KEY"] = api_key
    settings["SCICF_LLM_MODEL_REVISION"] = model_revision
    ordered_keys = [
        "SCICF_LLM_API_URL",
        "SCICF_LLM_API_KEY",
        "SCICF_LLM_MODEL_ID",
        "SCICF_LLM_MODEL_REVISION",
        "SCICF_LLM_PROVIDER_ID",
        "SCICF_LLM_API_KEY_HEADER",
        "SCICF_LLM_API_KEY_PREFIX",
        "SCICF_LLM_ALLOW_INSECURE_HTTP",
        "SCICF_LLM_INCLUDE_SEED",
        "SCICF_LLM_JSON_MODE",
        "SCICF_LLM_MAX_TOKENS_FIELD",
        "SCICF_LLM_THINKING",
    ]
    body = "\n".join("{}={}".format(key, settings[key]) for key in ordered_keys) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(destination), flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(destination), 0o600)
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise

    print(
        json.dumps(
            {
                "event": "api_credentials_provisioned",
                "profile": profile,
                "provider_id": settings["SCICF_LLM_PROVIDER_ID"],
                "model_id": settings["SCICF_LLM_MODEL_ID"],
                "model_revision": model_revision,
                "output_mode": "0600",
                "secret_logged": False,
                "output_path_logged": False,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    provision(args.raw_key_file, args.output, args.profile, args.model_revision)


if __name__ == "__main__":
    main()
