#!/usr/bin/env python3
"""Run any DAPiGen AlgorithmAdapter under one comparison contract."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from reproduction.framework.config import load_config, validate_config
from reproduction.framework.contracts import AlgorithmContext, ContractError
from reproduction.framework.evaluator import CommonEvaluator
from reproduction.framework.io import append_jsonl, git_identity, write_json
from reproduction.framework.registry import load_adapter_class


COMPATIBILITY_LABEL = "reconstructed AFP compatibility baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_execution_provenance(
    repo_root: Path, config: Mapping[str, Any], source: Mapping[str, Any]
) -> None:
    provenance = config["provenance"]
    if provenance["require_slurm"] and not os.environ.get("SLURM_JOB_ID"):
        raise ContractError("this experiment must be launched through Slurm")
    if provenance["require_clean_git"] and source.get("dirty") is not False:
        raise ContractError("formal experiment requires a clean Git worktree")
    baseline_tag = provenance["baseline_tag"]
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline_tag, "HEAD"],
        cwd=str(repo_root),
        check=False,
    )
    if result.returncode != 0:
        raise ContractError(
            "current code is not descended from frozen baseline tag {}".format(
                baseline_tag
            )
        )


def build_task(
    repo_root: Path, config: Mapping[str, Any]
) -> Dict[str, Any]:
    from RL_PPO.GNN.benchmarks import Benchmark
    from RL_PPO.moldr.config import get_default_config
    from RL_PPO.moldr.env import PIEnvValueMax

    block_root = repo_root / "RL_PPO" / "outputs" / "building_blocks"
    dianhydrides = pd.read_csv(block_root / "blocks_dianhydride.csv")["block"].tolist()
    diamines = pd.read_csv(block_root / "blocks_diamine.csv")["block"].tolist()
    resources = config["resources"]
    task = config["task"]
    legacy = get_default_config(
        PIEnvValueMax,
        Benchmark,
        dianhydrides,
        diamines,
        model_path=repo_root / "RL_PPO" / "models",
        num_workers=int(resources["rollout_workers"]),
        num_gpus=float(resources["gpus"]),
        length=int(task["length"]),
        step_length=int(task["step_length"]),
    )
    return {
        "env_class": PIEnvValueMax,
        "env_config": legacy["env_config"],
        "legacy_algorithm_config": legacy,
        "dianhydride_blocks": len(dianhydrides),
        "diamine_blocks": len(diamines),
    }


def evaluation_seed(base_seed: int, requested_environment_steps: int) -> int:
    return base_seed + requested_environment_steps


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    config = load_config(args.config.resolve())
    if args.seed is not None:
        config = copy.deepcopy(config)
        config["training"]["seed"] = args.seed
        config = validate_config(config)

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError("output root already exists: {}".format(output_root))
    output_root.mkdir(parents=True)
    write_json(output_root / "resolved-config.json", config)

    source = git_identity(repo_root)
    algorithm_spec = config["algorithm"]
    task_spec = config["task"]
    seed = int(config["training"]["seed"])
    label = (
        COMPATIBILITY_LABEL
        if task_spec["asset_mode"] == "reconstructed-afp-compatibility"
        else "author-asset algorithm comparison"
    )
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "label": label,
        "original_asset_reproduction": task_spec["asset_mode"] == "original",
        "frozen_baseline_tag": config["provenance"]["baseline_tag"],
        "source": source,
        "algorithm": {
            "name": algorithm_spec["name"],
            "version": algorithm_spec["version"],
            "adapter": algorithm_spec["adapter"],
        },
        "task": task_spec,
        "training": config["training"],
        "evaluation_protocol": config["evaluation"],
        "resources": {
            **config["resources"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "last_completed_iteration": 0,
        "environment_steps": 0,
        "checkpoints": {},
        "evaluations": {},
    }
    write_json(output_root / "run-manifest.json", manifest)

    adapter = None
    try:
        require_execution_provenance(repo_root, config, source)
        import torch
        from reproduction.check_assets import build_report, require_assets

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        require_assets(repo_root, asset_mode=task_spec["asset_mode"])
        manifest["asset_gate"] = build_report(repo_root)
        task = build_task(repo_root, config)
        manifest["task"].update(
            {
                "dianhydride_blocks": task["dianhydride_blocks"],
                "diamine_blocks": task["diamine_blocks"],
            }
        )

        adapter_class = load_adapter_class(algorithm_spec["adapter"])
        adapter = adapter_class()
        slurm_cpus = int(
            os.environ.get("SLURM_CPUS_PER_TASK", config["resources"]["cpus"])
        )
        if slurm_cpus < int(config["resources"]["cpus"]):
            raise ContractError(
                "Slurm exposed {} CPUs but config requires {}".format(
                    slurm_cpus, config["resources"]["cpus"]
                )
            )
        context = AlgorithmContext(
            repo_root=repo_root,
            output_root=output_root,
            env_class=task["env_class"],
            env_config=task["env_config"],
            legacy_algorithm_config=task["legacy_algorithm_config"],
            seed=seed,
            resources=config["resources"],
            slurm_cpus=slurm_cpus,
        )
        adapter.initialize(context, algorithm_spec.get("parameters", {}))
        manifest["algorithm"]["runtime_identity"] = adapter.identity
        manifest["algorithm"]["effective_config"] = adapter.effective_config()
        manifest["status"] = "running"
        write_json(output_root / "run-manifest.json", manifest)

        evaluation = config["evaluation"]
        reference_csv = evaluation.get("reference_csv")
        evaluator = CommonEvaluator(
            reference_csv=(repo_root / reference_csv).resolve()
            if reference_csv
            else None,
            reference_column=evaluation.get("reference_column", "smile"),
            compute_paper_metrics=bool(evaluation["compute_paper_metrics"]),
            protocol=evaluation["protocol"],
        )

        metrics_path = output_root / "training-metrics.jsonl"
        evaluation_metrics_path = output_root / "evaluation-metrics.jsonl"
        current_steps = 0
        iteration = 0
        maximum_iterations = int(config["training"]["max_iterations"])
        for requested_steps in evaluation["checkpoint_environment_steps"]:
            while current_steps < requested_steps:
                if iteration >= maximum_iterations:
                    raise ContractError(
                        "maximum iterations reached before environment-step budget"
                    )
                result = adapter.train_step(
                    target_environment_steps=int(requested_steps)
                )
                next_steps = int(result["environment_steps"])
                if next_steps <= current_steps:
                    raise ContractError("environment_steps did not increase")
                iteration += 1
                current_steps = next_steps
                record = {
                    "iteration": iteration,
                    "checkpoint_target_environment_steps": requested_steps,
                    **result,
                }
                append_jsonl(metrics_path, record)
                print(
                    json.dumps(
                        {"event": "training_iteration", **record},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                manifest["last_completed_iteration"] = iteration
                manifest["environment_steps"] = current_steps
                write_json(output_root / "run-manifest.json", manifest)
            if current_steps != requested_steps:
                raise ContractError(
                    "strict budget violation: checkpoint {} reached at {} steps".format(
                        requested_steps, current_steps
                    )
                )

            checkpoint_dir = (
                output_root / "checkpoints" / "step_{:06d}".format(requested_steps)
            )
            checkpoint_path = adapter.save(checkpoint_dir)
            manifest["checkpoints"][str(requested_steps)] = {
                "requested_environment_steps": requested_steps,
                "actual_environment_steps": current_steps,
                "path": checkpoint_path,
            }
            output_csv = (
                output_root
                / "evaluations"
                / "step_{:06d}".format(requested_steps)
                / "generate.csv"
            )
            summary = evaluator.generate(
                adapter=adapter,
                env_class=task["env_class"],
                env_config=task["env_config"],
                sample_count=int(evaluation["samples"]),
                output_csv=output_csv,
                seed=evaluation_seed(seed, requested_steps),
                explore=False,
                progress_every=int(evaluation["progress_every"]),
                max_episode_steps=int(evaluation["max_episode_steps"]),
            )
            summary.update(
                {
                    "requested_environment_steps": requested_steps,
                    "actual_environment_steps": current_steps,
                }
            )
            manifest["evaluations"][str(requested_steps)] = summary
            append_jsonl(evaluation_metrics_path, summary)
            write_json(output_root / "run-manifest.json", manifest)

        expected_steps = int(config["training"]["max_environment_steps"])
        if current_steps != expected_steps:
            raise ContractError(
                "run ended at {} steps instead of {}".format(
                    current_steps, expected_steps
                )
            )
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        write_json(output_root / "run-manifest.json", manifest)
        print(
            json.dumps(
                {
                    "event": "run_complete",
                    "run_id": args.run_id,
                    "environment_steps": current_steps,
                    "output_root": str(output_root),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["error"] = "{}: {}".format(type(error).__name__, error)
        manifest["traceback"] = traceback.format_exc()
        write_json(output_root / "run-manifest.json", manifest)
        raise
    finally:
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    main()
