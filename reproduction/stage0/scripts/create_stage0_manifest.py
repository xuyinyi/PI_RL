#!/usr/bin/env python
"""Create the immutable environment/evaluator contract for a Stage-0 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", required=True)
    parser.add_argument("--polybert-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-requested-calls", type=int, required=True)
    parser.add_argument("--maximum-unique-calls", type=int, default=None)
    parser.add_argument("--run-name", default="stage0")
    parser.add_argument(
        "--cache-scope",
        default="per_run",
        help=(
            "Cache lifetime policy recorded in the comparable budget contract. "
            "Keep the default for independent per-run caches; use --run-name "
            "for the instance-specific provenance label."
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development only; formal contracts require a clean checkout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.dapigen_root).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(repository_root))

    from RL_PPO.envs.factory import (
        build_stage0_components,
        load_environment_config,
        write_action_catalogs,
        write_environment_manifest,
    )
    from RL_PPO.envs.sources import ALL_STAGE0_SOURCES

    config = load_environment_config(args.config)
    components = build_stage0_components(
        dapigen_root=str(root),
        polybert_path=args.polybert_path,
        config=config,
        device=args.device,
        maximum_requested_calls=args.maximum_requested_calls,
        maximum_unique_calls=args.maximum_unique_calls,
        encoder_mode="polybert",
        evaluator_mode="persistent",
        allowed_evaluator_sources=ALL_STAGE0_SOURCES,
        cache_scope=args.cache_scope,
    )
    commit = components.repository_metadata.get("dapigen_git_commit")
    dirty = components.repository_metadata.get("dapigen_git_dirty")
    if not args.allow_dirty and (not commit or dirty is not False):
        raise RuntimeError(
            "Formal Stage-0 manifest creation requires a recognized clean "
            "DAPiGen Git checkout."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "environment_manifest.json"
    write_environment_manifest(
        str(manifest_path),
        components,
        extra={"run_name": args.run_name, "cache_scope": args.cache_scope},
    )
    write_action_catalogs(str(output_dir / "action_catalogs"), components)
    manifest = json.loads(manifest_path.read_text())
    print(json.dumps({
        "environment_id": manifest["environment"]["environment_id"],
        "task_contract_id": manifest["task_contract_id"],
        "budget_contract_id": manifest["budget_contract_id"],
        "evaluator_version": manifest["evaluator_version"],
        "objective_contract": manifest["objective_contract"],
        "runtime_contract_id": manifest["runtime_contract_id"],
        "manifest": str(manifest_path.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
