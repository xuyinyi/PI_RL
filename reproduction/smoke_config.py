#!/usr/bin/env python3
"""Construct the repaired RLlib and DAPiGen environment configuration."""

import json
from pathlib import Path

import pandas as pd

from RL_PPO.GNN.benchmarks import Benchmark
from RL_PPO.moldr.config import get_default_config
from RL_PPO.moldr.env import PIEnvValueMax


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    rl_root = repo_root / "RL_PPO"
    dianhydride = pd.read_csv(
        rl_root / "outputs/building_blocks/blocks_dianhydride.csv"
    )["block"].tolist()
    diamine = pd.read_csv(
        rl_root / "outputs/building_blocks/blocks_diamine.csv"
    )["block"].tolist()

    config = get_default_config(
        PIEnvValueMax,
        Benchmark,
        dianhydride,
        diamine,
        model_path=rl_root / "models",
        num_workers=15,
        num_gpus=4,
        length=60,
        step_length=5,
    )
    env_config = config["env_config"]
    env = PIEnvValueMax(env_config)

    assert "ACTION_SPACE_DIANHYDRIDE" not in config
    assert config["num_workers"] == 15
    assert config["num_gpus"] == 4
    assert config["train_batch_size"] == 500
    assert config["model"]["fcnet_hiddens"] == [256, 128, 128]
    assert env.observation_space.shape == (1200,)

    result = {
        "action_space": str(env.action_space),
        "diamine_blocks": len(diamine),
        "dianhydride_blocks": len(dianhydride),
        "fcnet_hiddens": config["model"]["fcnet_hiddens"],
        "num_gpus": config["num_gpus"],
        "num_workers": config["num_workers"],
        "observation_shape": list(env.observation_space.shape),
        "step_length": env.step_length,
        "train_batch_size": config["train_batch_size"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

