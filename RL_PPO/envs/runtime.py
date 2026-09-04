from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .rng import derive_seed
from .types import CoreTransition, DAPiGenAction, DAPiGenState


class DAPiGenEpisodeController(object):
    """Algorithm-neutral stateful controller over the pure environment core."""

    def __init__(self, core, reward_adapter, run_seed: int = 0) -> None:
        self.core = core
        self.reward_adapter = reward_adapter
        self.run_seed = int(run_seed)
        self.episode_index = -1
        self.episode_seed = None
        self.current = core.initial(seed=0)

    def reset(self, episode_seed: Optional[int] = None):
        self.episode_index += 1
        if episode_seed is None:
            episode_seed = derive_seed(
                self.run_seed, "episode", self.episode_index
            )
        self.episode_seed = int(episode_seed)
        self.current = self.core.initial(seed=self.episode_seed)
        return self.current

    def transition_seed(self, state: Optional[DAPiGenState] = None) -> int:
        state = state or self.current.state
        if self.episode_seed is None:
            self.episode_seed = derive_seed(self.run_seed, "episode", 0)
        return derive_seed(
            self.episode_seed, "transition", int(state.step_index)
        )

    def step(self, action, source: str = "on_policy"):
        if self.current.state.done:
            raise RuntimeError("The episode is finished; call reset().")
        seed = self.transition_seed(self.current.state)
        core_transition = self.core.transition(
            self.current.state, DAPiGenAction.from_any(action), seed=seed
        )
        evaluated = self.reward_adapter.apply(core_transition, source=source)
        self.current = core_transition
        return evaluated

    def branch(
        self,
        state: DAPiGenState,
        action,
        transition_seed: int,
        source: str = "counterfactual",
    ):
        transition = self.core.transition(
            state, DAPiGenAction.from_any(action), seed=int(transition_seed)
        )
        return self.reward_adapter.apply(transition, source=source)

    def snapshot(self) -> Dict[str, Any]:
        evaluator = self.reward_adapter.evaluator
        if not hasattr(evaluator, "state_dict"):
            raise TypeError("Evaluator service does not support exact checkpointing.")
        return {
            "checkpoint_schema_version": 2,
            "environment_id": self.core.environment_id,
            "run_seed": int(self.run_seed),
            "episode_index": int(self.episode_index),
            "episode_seed": (
                None if self.episode_seed is None else int(self.episode_seed)
            ),
            "transition": self.current.to_snapshot(),
            "evaluator_state": evaluator.state_dict(),
        }

    def restore(self, snapshot: Mapping[str, Any]):
        if snapshot.get("checkpoint_schema_version") != 2:
            raise ValueError("Unsupported or incomplete controller checkpoint.")
        if snapshot.get("environment_id") != self.core.environment_id:
            raise ValueError("Controller checkpoint environment_id mismatch.")
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
            raise ValueError("Controller checkpoint lacks evaluator state.")
        restored_transition = self.core.restore_transition(transition_snapshot)
        evaluator = self.reward_adapter.evaluator
        if not hasattr(evaluator, "load_state_dict"):
            raise TypeError("Evaluator service does not support exact checkpointing.")
        evaluator.load_state_dict(evaluator_state)
        self.run_seed = run_seed
        self.episode_index = episode_index
        self.episode_seed = episode_seed
        self.current = restored_transition
        return self.current
