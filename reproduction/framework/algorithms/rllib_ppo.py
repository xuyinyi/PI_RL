"""Ray RLlib 1.13 PPO implementation of the common adapter contract."""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from reproduction.framework.contracts import (
    AlgorithmAdapter,
    AlgorithmContext,
    ContractError,
)
from reproduction.framework.io import json_value, write_json


class RLLibPPOAdapter(AlgorithmAdapter):
    def __init__(self) -> None:
        self._context: Optional[AlgorithmContext] = None
        self._trainer = None
        self._ray = None
        self._config: Optional[Dict[str, Any]] = None

    @property
    def identity(self) -> Mapping[str, Any]:
        result: Dict[str, Any] = {"name": "rllib_ppo"}
        if self._ray is not None:
            result["ray_version"] = getattr(self._ray, "__version__", "unknown")
        return result

    def initialize(
        self, context: AlgorithmContext, parameters: Mapping[str, Any]
    ) -> None:
        import ray
        import torch
        from ray.rllib.agents.ppo import PPOTrainer

        from RL_PPO.moldr.utils import custom_log_creator

        requested_gpus = float(context.resources["gpus"])
        visible_gpus = torch.cuda.device_count()
        if requested_gpus > visible_gpus:
            raise ContractError(
                "algorithm requests {} GPUs but only {} are visible".format(
                    requested_gpus, visible_gpus
                )
            )

        config = copy.deepcopy(dict(context.legacy_algorithm_config))
        config.update(
            {
                "num_workers": int(context.resources["rollout_workers"]),
                "num_gpus": requested_gpus,
                "seed": context.seed,
            }
        )
        for key in (
            "train_batch_size",
            "sgd_minibatch_size",
            "num_sgd_iter",
            "rollout_fragment_length",
            "evaluation_num_workers",
            "log_level",
            "framework",
        ):
            if key in parameters:
                config[key] = parameters[key]
        if "model" in parameters:
            config["model"] = copy.deepcopy(parameters["model"])

        ray_temp = Path(
            os.environ.get(
                "DAPIGEN_RAY_TMPDIR",
                "/tmp/dapigen-ray-{}".format(
                    os.environ.get("SLURM_JOB_ID", os.getpid())
                ),
            )
        )
        ray.init(
            include_dashboard=False,
            num_cpus=context.slurm_cpus,
            num_gpus=visible_gpus,
            _temp_dir=str(ray_temp),
        )
        trainer = PPOTrainer(
            env=context.env_class,
            config=config,
            logger_creator=custom_log_creator(
                context.output_root / "native-logs" / "ray", "PI"
            ),
        )
        self._context = context
        self._trainer = trainer
        self._ray = ray
        self._config = config
        with (context.output_root / "algorithm-config.pkl").open("wb") as handle:
            pickle.dump(config, handle)
        write_json(
            context.output_root / "algorithm-config-summary.json",
            self.effective_config(),
        )

    def train_step(self, target_environment_steps: Optional[int] = None) -> Dict[str, Any]:
        if self._trainer is None:
            raise ContractError("adapter is not initialized")
        result = self._trainer.train()
        normalized: Dict[str, Any] = {
            "training_iteration": result.get("training_iteration"),
            "environment_steps": result.get("timesteps_total"),
            "episodes_total": result.get("episodes_total"),
            "episode_reward_mean": result.get("episode_reward_mean"),
            "episode_reward_min": result.get("episode_reward_min"),
            "episode_reward_max": result.get("episode_reward_max"),
            "episode_length_mean": result.get("episode_len_mean"),
            "time_this_iteration_seconds": result.get("time_this_iter_s"),
            "time_total_seconds": result.get("time_total_s"),
            "healthy_workers": result.get("num_healthy_workers"),
        }
        learner = (
            result.get("info", {})
            .get("learner", {})
            .get("default_policy", {})
            .get("learner_stats", {})
        )
        if learner:
            normalized["algorithm_metrics"] = json_value(learner)
        if normalized["environment_steps"] is None:
            raise ContractError("RLlib result does not contain timesteps_total")
        return json_value(normalized)

    def act(self, observation: Any, explore: bool = False) -> Any:
        if self._trainer is None:
            raise ContractError("adapter is not initialized")
        return self._trainer.get_policy().compute_single_action(
            observation, explore=explore
        )[0]

    def save(self, checkpoint_dir: Path) -> str:
        if self._trainer is None:
            raise ContractError("adapter is not initialized")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return str(self._trainer.save(str(checkpoint_dir)))

    def restore(self, checkpoint_path: str) -> None:
        if self._trainer is None:
            raise ContractError("adapter is not initialized")
        self._trainer.restore(checkpoint_path)

    def effective_config(self) -> Mapping[str, Any]:
        if self._config is None:
            return {}
        keys = (
            "framework",
            "num_workers",
            "num_gpus",
            "train_batch_size",
            "sgd_minibatch_size",
            "num_sgd_iter",
            "rollout_fragment_length",
            "evaluation_num_workers",
            "log_level",
            "seed",
        )
        result = {key: self._config.get(key) for key in keys}
        result["model"] = self._config.get("model")
        return json_value(result)

    def close(self) -> None:
        if self._trainer is not None:
            self._trainer.stop()
            self._trainer = None
        if self._ray is not None and self._ray.is_initialized():
            self._ray.shutdown()
        self._ray = None
