from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .runtime import DAPiGenEpisodeController

try:
    import gymnasium as gymnasium
except Exception:  # pragma: no cover
    gymnasium = None


_BaseEnv = gymnasium.Env if gymnasium is not None else object


class DAPiGenGymnasiumEnv(_BaseEnv):
    """Modern Gymnasium wrapper. The released Ray 1.13 stack can use gym_wrapper."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        core,
        reward_adapter,
        source: str = "on_policy",
        seed: int = 0,
        include_ledger_in_step_info: bool = False,
    ):
        if gymnasium is None:
            raise ImportError("gymnasium is required for DAPiGenGymnasiumEnv.")
        self.controller = DAPiGenEpisodeController(core, reward_adapter, run_seed=seed)
        self.core = core
        self.source = str(source)
        self.include_ledger_in_step_info = bool(include_ledger_in_step_info)
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [len(core.dianhydride_blocks) + 1, len(core.diamine_blocks) + 1]
        )
        initial = core.initial(seed=seed)
        dimension = int(initial.observation.size)
        self.observation_space = gymnasium.spaces.Dict(
            {
                "observations": gymnasium.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(dimension,),
                    dtype=np.float32,
                ),
                "action_mask": gymnasium.spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        len(core.dianhydride_blocks)
                        + 1
                        + len(core.diamine_blocks)
                        + 1,
                    ),
                    dtype=np.int8,
                ),
            }
        )

    @staticmethod
    def _observation(transition):
        return {
            "observations": transition.observation.copy(),
            "action_mask": transition.action_mask.flattened(dtype=np.int8),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        del options
        if seed is not None:
            self.controller.run_seed = int(seed)
            self.controller.episode_index = -1
        transition = self.controller.reset()
        return self._observation(transition), {
            "environment_id": self.core.environment_id,
            "state_snapshot": transition.state.to_dict(),
        }

    def step(self, action):
        evaluated = self.controller.step(action, source=self.source)
        transition = evaluated.core
        info = dict(transition.info)
        info["state_snapshot"] = transition.state.to_dict()
        if self.include_ledger_in_step_info:
            info["oracle_ledger"] = (
                self.controller.reward_adapter.evaluator.ledger()
            )
        if evaluated.evaluation is not None:
            info["terminal_evaluation"] = evaluated.evaluation.to_dict()
        return (
            self._observation(transition),
            float(evaluated.reward),
            bool(transition.terminated),
            bool(transition.truncated),
            info,
        )

    def snapshot(self):
        return self.controller.snapshot()

    def restore(self, snapshot):
        transition = self.controller.restore(snapshot)
        return self._observation(transition)

    def transition_from(
        self,
        state,
        action,
        transition_seed: int,
        source: str = "counterfactual",
    ):
        return self.controller.branch(
            state,
            action,
            transition_seed=int(transition_seed),
            source=str(source),
        )

    def oracle_ledger(self):
        return self.controller.reward_adapter.evaluator.ledger()

    @property
    def current_state(self):
        return self.controller.current.state
