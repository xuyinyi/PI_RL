from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from .rng import derive_seed
from .types import CoreTransition, DAPiGenAction, DAPiGenState

try:
    import gym
except Exception:  # pragma: no cover - optional in the standalone test image.
    gym = None


_BaseEnv = gym.Env if gym is not None else object


class DAPiGenGymEnv(_BaseEnv):
    """Gym 0.19/RLlib wrapper over the shared branchable chemistry core.

    The wrapper owns only episode progression and terminal evaluator calls. It
    exposes the exact same core to original PPO, Policy-CC and MCC-PPO.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        core,
        reward_adapter,
        source: str = "on_policy",
        return_masked_dict_observation: bool = True,
        seed: int = 0,
        include_ledger_in_step_info: bool = False,
    ) -> None:
        if gym is None:
            raise ImportError("gym is required to instantiate DAPiGenGymEnv.")
        self.core = core
        self.reward_adapter = reward_adapter
        self.source = str(source)
        self.return_masked_dict_observation = bool(return_masked_dict_observation)
        self.include_ledger_in_step_info = bool(include_ledger_in_step_info)
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(len(core.dianhydride_blocks) + 1),
                gym.spaces.Discrete(len(core.diamine_blocks) + 1),
            )
        )
        initial = core.initial(seed=seed)
        observation_dim = int(initial.observation.size)
        high = np.full(observation_dim, np.finfo(np.float32).max, dtype=np.float32)
        vector_space = gym.spaces.Box(-high, high, dtype=np.float32)
        if self.return_masked_dict_observation:
            total_mask = (
                len(core.dianhydride_blocks)
                + 1
                + len(core.diamine_blocks)
                + 1
            )
            self.observation_space = gym.spaces.Dict(
                {
                    "observations": vector_space,
                    "action_mask": gym.spaces.Box(
                        0, 1, shape=(total_mask,), dtype=np.int8
                    ),
                    "dianhydride_mask": gym.spaces.Box(
                        0,
                        1,
                        shape=(len(core.dianhydride_blocks) + 1,),
                        dtype=np.int8,
                    ),
                    "diamine_mask": gym.spaces.Box(
                        0,
                        1,
                        shape=(len(core.diamine_blocks) + 1,),
                        dtype=np.int8,
                    ),
                }
            )
        else:
            self.observation_space = vector_space
        self._run_seed = int(seed)
        self._episode_index = -1
        self._episode_seed = None
        self._current = initial

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = 0
        self._run_seed = int(seed)
        self._episode_index = -1
        self._episode_seed = None
        return [self._run_seed]

    def _format_observation(self, transition):
        if not self.return_masked_dict_observation:
            return transition.observation.copy()
        d_mask = transition.action_mask.dianhydride.astype(np.int8)
        a_mask = transition.action_mask.diamine.astype(np.int8)
        return {
            "observations": transition.observation.copy(),
            "action_mask": np.concatenate([d_mask, a_mask]).astype(np.int8),
            "dianhydride_mask": d_mask,
            "diamine_mask": a_mask,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        del options
        if seed is not None:
            self.seed(seed)
        self._episode_index += 1
        self._episode_seed = derive_seed(
            self._run_seed, "episode", self._episode_index
        )
        self._current = self.core.initial(seed=self._episode_seed)
        return self._format_observation(self._current)

    def _transition_seed(self, state: DAPiGenState) -> int:
        if self._episode_seed is None:
            self._episode_seed = derive_seed(self._run_seed, "episode", 0)
        return derive_seed(
            self._episode_seed, "transition", int(state.step_index)
        )

    def step(self, action):
        if self._current.state.done:
            raise RuntimeError("step() called after episode completion; call reset().")
        previous_state = self._current.state
        parsed = DAPiGenAction.from_any(action)
        transition_seed = self._transition_seed(previous_state)
        core_transition = self.core.transition(
            previous_state, parsed, seed=transition_seed
        )
        evaluated = self.reward_adapter.apply(core_transition, source=self.source)
        self._current = core_transition
        done = bool(core_transition.terminated or core_transition.truncated)
        info = dict(core_transition.info)
        info.update(
            {
                "previous_state_snapshot": previous_state.to_dict(),
                "state_snapshot": core_transition.state.to_dict(),
                "action_mask": core_transition.action_mask.to_dict(),
                "environment_id": self.core.environment_id,
                "episode_index": int(self._episode_index),
                "episode_seed": int(self._episode_seed),
                "transition_seed": int(transition_seed),
            }
        )
        # A Ray-backed ledger is a synchronous remote call. Do not perform it
        # on every environment step unless an explicit debug run requests it.
        if self.include_ledger_in_step_info:
            info["oracle_ledger"] = self.reward_adapter.evaluator.ledger()
        if evaluated.evaluation is not None:
            info["terminal_evaluation"] = evaluated.evaluation.to_dict()
        return (
            self._format_observation(core_transition),
            float(evaluated.reward),
            done,
            info,
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return all mutable wrapper state needed for exact episode restoration."""

        evaluator = self.reward_adapter.evaluator
        if not hasattr(evaluator, "state_dict"):
            raise TypeError("Evaluator service does not support exact checkpointing.")
        return {
            "checkpoint_schema_version": 2,
            "environment_id": self.core.environment_id,
            "run_seed": int(self._run_seed),
            "episode_index": int(self._episode_index),
            "episode_seed": (
                None if self._episode_seed is None else int(self._episode_seed)
            ),
            "transition": self._current.to_snapshot(),
            "evaluator_state": evaluator.state_dict(),
        }

    def restore(self, snapshot: Mapping[str, Any]):
        if snapshot.get("checkpoint_schema_version") != 2:
            raise ValueError("Unsupported or incomplete Gym checkpoint.")
        if snapshot.get("environment_id") != self.core.environment_id:
            raise ValueError("Gym checkpoint environment_id mismatch.")
        run_seed = int(snapshot["run_seed"])
        episode_index = int(snapshot["episode_index"])
        episode_seed = snapshot.get("episode_seed")
        if episode_seed is not None:
            episode_seed = int(episode_seed)
        transition_snapshot = snapshot.get("transition")
        if transition_snapshot is None:
            raise ValueError(
                "The snapshot predates Stage-0 transition snapshots and cannot "
                "restore terminal-product information exactly."
            )
        evaluator_state = snapshot.get("evaluator_state")
        if evaluator_state is None:
            raise ValueError("Gym checkpoint lacks evaluator state.")
        restored_transition = self.core.restore_transition(transition_snapshot)
        evaluator = self.reward_adapter.evaluator
        if not hasattr(evaluator, "load_state_dict"):
            raise TypeError("Evaluator service does not support exact checkpointing.")
        evaluator.load_state_dict(evaluator_state)
        self._run_seed = run_seed
        self._episode_index = episode_index
        self._episode_seed = episode_seed
        self._current = restored_transition
        return self._format_observation(self._current)

    def transition_from(
        self,
        state: DAPiGenState,
        action,
        transition_seed: int,
        source: str = "counterfactual",
    ):
        """Evaluate one branch without mutating the wrapper's current episode."""

        core_transition = self.core.transition(
            state, DAPiGenAction.from_any(action), seed=int(transition_seed)
        )
        return self.reward_adapter.apply(core_transition, source=str(source))

    def oracle_ledger(self):
        """Read the evaluator ledger explicitly at iteration/checkpoint boundaries."""

        return self.reward_adapter.evaluator.ledger()

    @property
    def current_state(self):
        return self._current.state


class DAPiGenRLlibEnv(DAPiGenGymEnv):
    """RLlib-compatible constructor accepting only an ``EnvContext`` mapping.

    Required keys:

    ``dapigen_root``, ``polybert_path``, and either ``task_config`` (mapping) or
    ``task_config_path``. For ``num_workers > 0``, pass ``evaluator_actor`` so
    all workers share one global evaluator ledger.
    """

    def __init__(self, env_config) -> None:
        from .config import DAPiGenEnvConfig
        from .factory import build_stage0_components, load_environment_config
        from .ray_evaluator import RayTerminalEvaluatorClient

        values = dict(env_config)
        dapigen_root = values.get("dapigen_root")
        polybert_path = values.get("polybert_path")
        if not dapigen_root or not polybert_path:
            raise ValueError("dapigen_root and polybert_path are required.")
        if "task_config" in values and "task_config_path" in values:
            raise ValueError("Pass task_config or task_config_path, not both.")
        if "task_config_path" in values:
            task_config = load_environment_config(values["task_config_path"])
        else:
            task_config = DAPiGenEnvConfig.from_mapping(
                values.get("task_config", {})
            )

        worker_index = int(getattr(env_config, "worker_index", 0))
        vector_index = int(getattr(env_config, "vector_index", 0))
        number_of_workers = int(getattr(env_config, "num_workers", 0))
        base_seed = int(values.get("seed", 0))
        worker_seed = derive_seed(
            base_seed, "rllib_worker", worker_index, "vector", vector_index
        )

        evaluator_actor = values.get("evaluator_actor")
        evaluator_service = None
        if evaluator_actor is not None:
            evaluator_service = RayTerminalEvaluatorClient(evaluator_actor)
        elif number_of_workers > 0 and not bool(
            values.get("allow_worker_local_evaluator", False)
        ):
            raise RuntimeError(
                "A shared evaluator_actor is mandatory when RLlib num_workers > 0; "
                "otherwise each worker receives a separate cache and budget."
            )

        components = build_stage0_components(
            dapigen_root=str(dapigen_root),
            polybert_path=str(polybert_path),
            config=task_config,
            device=str(values.get("encoder_device", "cpu")),
            maximum_requested_calls=values.get("maximum_requested_calls"),
            maximum_unique_calls=values.get("maximum_unique_calls"),
            evaluator_mode=str(values.get("evaluator_mode", "persistent")),
            evaluator_service=evaluator_service,
            allowed_evaluator_sources=values.get("allowed_evaluator_sources"),
            cache_scope=str(values.get("cache_scope", "per_run")),
            polybert_checkpoint_fingerprint=values.get(
                "polybert_checkpoint_fingerprint"
            ),
            allow_rdkit_brics_fallback=bool(
                values.get("allow_rdkit_brics_fallback", False)
            ),
        )
        self.components = components
        self.worker_index = worker_index
        self.vector_index = vector_index
        super(DAPiGenRLlibEnv, self).__init__(
            core=components.core,
            reward_adapter=components.reward_adapter,
            source=str(values.get("source", "ppo/on_policy")),
            return_masked_dict_observation=bool(
                values.get("return_masked_dict_observation", True)
            ),
            seed=worker_seed,
            include_ledger_in_step_info=bool(
                values.get("include_ledger_in_step_info", False)
            ),
        )
