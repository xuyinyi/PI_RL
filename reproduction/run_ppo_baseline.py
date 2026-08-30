#!/usr/bin/env python3
"""Run the DAPiGen PPO baseline with explicit asset and execution contracts."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


CODE_COMMIT = "5f692946cbe0d15eede882dfe4cff7fb26eb7d8c"
COMPATIBILITY_LABEL = "reconstructed AFP compatibility baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--asset-mode",
        choices=("original", "reconstructed-afp-compatibility"),
        default="original",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--evaluation-samples", type=int, default=10000)
    parser.add_argument("--evaluation-progress-every", type=int, default=100)
    parser.add_argument("--evaluate-untrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=15)
    parser.add_argument("--num-gpus", type=float, default=4.0)
    parser.add_argument("--evaluation-num-workers", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=500)
    parser.add_argument("--sgd-minibatch-size", type=int, default=128)
    parser.add_argument("--num-sgd-iter", type=int, default=30)
    parser.add_argument("--rollout-fragment-length", type=int, default=200)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARN", "ERROR"), default="WARN"
    )
    parser.add_argument("--length", type=int, default=60)
    parser.add_argument("--step-length", type=int, default=5)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    for name in (
        "iterations",
        "checkpoint_interval",
        "evaluation_samples",
        "evaluation_progress_every",
        "train_batch_size",
        "sgd_minibatch_size",
        "num_sgd_iter",
        "rollout_fragment_length",
        "length",
        "step_length",
    ):
        if getattr(args, name) < 1:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.num_workers < 0 or args.evaluation_num_workers < 0:
        parser.error("worker counts cannot be negative")
    if args.num_gpus < 0:
        parser.error("--num-gpus cannot be negative")
    if args.sgd_minibatch_size > args.train_batch_size:
        parser.error("--sgd-minibatch-size cannot exceed --train-batch-size")
    return args


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def scalar(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def result_summary(result: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "training_iteration",
        "timesteps_total",
        "episodes_total",
        "episode_reward_mean",
        "episode_reward_min",
        "episode_reward_max",
        "episode_len_mean",
        "time_this_iter_s",
        "time_total_s",
        "num_healthy_workers",
    )
    return {key: scalar(result.get(key)) for key in keys if key in result}


def evaluate_policy(
    policy,
    env_config: Dict[str, object],
    sample_count: int,
    output_csv: Path,
    progress_every: int,
    seed: Optional[int],
) -> Dict[str, object]:
    from RL_PPO.moldr.env import PIEnvValueMax

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    environment = PIEnvValueMax(env_config)
    rows: List[Dict[str, object]] = []
    start = time.time()
    for sample_index in range(sample_count):
        action_1: List[int] = []
        action_2: List[int] = []
        observation = environment.reset()
        while True:
            action = policy.compute_single_action(observation)[0]
            observation, reward, done, info = environment.step(action)
            previous_action = info["prev_action"]
            action_1.append(int(previous_action[0]))
            action_2.append(int(previous_action[1]))
            if done:
                rows.append(
                    {
                        "PI": environment.PI,
                        "A_1": ",".join(map(str, action_1)),
                        "A_2": ",".join(map(str, action_2)),
                        "transmittance": environment.transmittance,
                        "cte": environment.cte,
                        "strength": environment.strength,
                        "tg": environment.tg,
                        "SaScore": environment.SaScore,
                        "reward": float(reward),
                    }
                )
                break
        completed = sample_index + 1
        if completed == 1 or completed % progress_every == 0 or completed == sample_count:
            print(
                json.dumps(
                    {
                        "event": "evaluation_progress",
                        "completed": completed,
                        "total": sample_count,
                        "elapsed_seconds": time.time() - start,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    valid = frame["PI"].notna() & (frame["PI"] != "None")
    rewards = pd.to_numeric(frame["reward"], errors="coerce")
    summary = {
        "samples": int(len(frame)),
        "valid_samples": int(valid.sum()),
        "validity": float(valid.mean()),
        "unique_valid_PI": int(frame.loc[valid, "PI"].nunique()),
        "mean_reward": float(rewards.mean()),
        "max_reward": float(rewards.max()),
        "elapsed_seconds": time.time() - start,
        "output_csv": str(output_csv),
    }
    if not all(
        math.isfinite(summary[key])
        for key in ("validity", "mean_reward", "max_reward", "elapsed_seconds")
    ):
        raise RuntimeError("non-finite PPO evaluation summary")
    return summary


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    sys.path.insert(0, str(repo_root))

    import ray
    import torch
    from ray.rllib.agents.ppo import PPOTrainer
    from reproduction.check_assets import build_report, require_assets
    from RL_PPO.GNN.benchmarks import Benchmark
    from RL_PPO.moldr.config import get_default_config
    from RL_PPO.moldr.env import PIEnvValueMax
    from RL_PPO.moldr.utils import custom_log_creator

    if (output_root / "run-manifest.json").exists():
        raise FileExistsError("run manifest already exists: " + str(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    require_assets(repo_root, asset_mode=args.asset_mode)
    asset_report = build_report(repo_root)

    if args.num_gpus > torch.cuda.device_count():
        raise RuntimeError(
            "requested {} GPUs but Slurm exposed {}".format(
                args.num_gpus, torch.cuda.device_count()
            )
        )
    if args.asset_mode == "reconstructed-afp-compatibility":
        run_label = COMPATIBILITY_LABEL
        original_asset_reproduction = False
    else:
        run_label = "author-asset PPO baseline"
        original_asset_reproduction = True

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    rl_root = repo_root / "RL_PPO"
    block_root = rl_root / "outputs" / "building_blocks"
    dianhydrides = pd.read_csv(block_root / "blocks_dianhydride.csv")["block"].tolist()
    diamines = pd.read_csv(block_root / "blocks_diamine.csv")["block"].tolist()
    config = get_default_config(
        PIEnvValueMax,
        Benchmark,
        dianhydrides,
        diamines,
        model_path=rl_root / "models",
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        length=args.length,
        step_length=args.step_length,
    )
    config.update(
        {
            "evaluation_num_workers": args.evaluation_num_workers,
            "train_batch_size": args.train_batch_size,
            "sgd_minibatch_size": args.sgd_minibatch_size,
            "num_sgd_iter": args.num_sgd_iter,
            "rollout_fragment_length": args.rollout_fragment_length,
            "log_level": args.log_level,
        }
    )
    if args.seed is not None:
        config["seed"] = args.seed

    config_summary = {
        "fcnet_hiddens": config["model"]["fcnet_hiddens"],
        "fcnet_activation": config["model"]["fcnet_activation"],
        "framework": config["framework"],
        "num_workers": config["num_workers"],
        "num_gpus": config["num_gpus"],
        "evaluation_num_workers": config["evaluation_num_workers"],
        "train_batch_size": config["train_batch_size"],
        "sgd_minibatch_size": config["sgd_minibatch_size"],
        "num_sgd_iter": config["num_sgd_iter"],
        "rollout_fragment_length": config["rollout_fragment_length"],
        "log_level": config["log_level"],
        "length": config["env_config"]["LENGTH"],
        "step_length": config["env_config"]["STEP_LENGTH"],
        "dianhydride_blocks": len(dianhydrides),
        "diamine_blocks": len(diamines),
        "seed": config.get("seed"),
    }
    with (output_root / "config.pkl").open("wb") as handle:
        pickle.dump(config, handle)
    write_json(output_root / "config-summary.json", config_summary)

    manifest = {
        "status": "initializing",
        "label": run_label,
        "original_asset_reproduction": original_asset_reproduction,
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "https://github.com/xuyinyi/DAPiGen.git",
            "commit": CODE_COMMIT,
            "driver": "reproduction/run_ppo_baseline.py",
        },
        "asset_mode": args.asset_mode,
        "asset_gate": asset_report,
        "execution": {
            "iterations": args.iterations,
            "checkpoint_interval": args.checkpoint_interval,
            "evaluation_samples": args.evaluation_samples,
            "evaluate_untrained": args.evaluate_untrained,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_visible_gpus": torch.cuda.device_count(),
        },
        "config": config_summary,
        "last_completed_iteration": 0,
        "evaluations": {},
        "checkpoints": {},
    }
    write_json(output_root / "run-manifest.json", manifest)

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    ray_temp = Path(
        os.environ.get(
            "DAPIGEN_RAY_TMPDIR",
            "/tmp/dapigen-ray-{}".format(
                os.environ.get("SLURM_JOB_ID", os.getpid())
            ),
        )
    )
    manifest["execution"]["ray_temp"] = str(ray_temp)
    write_json(output_root / "run-manifest.json", manifest)
    trainer = None
    try:
        ray.init(
            include_dashboard=False,
            num_cpus=slurm_cpus,
            num_gpus=torch.cuda.device_count(),
            _temp_dir=str(ray_temp),
        )
        trainer = PPOTrainer(
            env=PIEnvValueMax,
            config=config,
            logger_creator=custom_log_creator(output_root / "ray-results", "PI"),
        )
        manifest["status"] = "running"
        write_json(output_root / "run-manifest.json", manifest)

        policy = trainer.get_policy()
        if args.evaluate_untrained:
            checkpoint_dir = output_root / "checkpoints" / "epoch_0"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = trainer.save(str(checkpoint_dir))
            manifest["checkpoints"]["0"] = checkpoint_path
            evaluation = evaluate_policy(
                policy,
                config["env_config"],
                args.evaluation_samples,
                output_root / "evaluations" / "epoch_0" / "generate_0.csv",
                args.evaluation_progress_every,
                args.seed,
            )
            manifest["evaluations"]["0"] = evaluation
            write_json(output_root / "run-manifest.json", manifest)

        metrics_path = output_root / "training-metrics.jsonl"
        for iteration in range(1, args.iterations + 1):
            result = trainer.train()
            summary = {"iteration": iteration, **result_summary(result)}
            append_jsonl(metrics_path, summary)
            print(json.dumps({"event": "training_iteration", **summary}, sort_keys=True), flush=True)
            manifest["last_completed_iteration"] = iteration

            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                checkpoint_dir = output_root / "checkpoints" / "epoch_{}".format(iteration)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = trainer.save(str(checkpoint_dir))
                manifest["checkpoints"][str(iteration)] = checkpoint_path
                policy = trainer.get_policy()
                evaluation = evaluate_policy(
                    policy,
                    config["env_config"],
                    args.evaluation_samples,
                    output_root
                    / "evaluations"
                    / "epoch_{}".format(iteration)
                    / "generate_{}.csv".format(iteration),
                    args.evaluation_progress_every,
                    None if args.seed is None else args.seed + iteration,
                )
                manifest["evaluations"][str(iteration)] = evaluation
            write_json(output_root / "run-manifest.json", manifest)

        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output_root / "run-manifest.json", manifest)
        print(
            json.dumps(
                {
                    "event": "run_complete",
                    "run_id": args.run_id,
                    "iterations": args.iterations,
                    "output_root": str(output_root),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = "{}: {}".format(type(error).__name__, error)
        manifest["traceback"] = traceback.format_exc()
        write_json(output_root / "run-manifest.json", manifest)
        raise
    finally:
        if trainer is not None:
            trainer.stop()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
