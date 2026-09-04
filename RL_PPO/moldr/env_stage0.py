"""Compatibility entry point for the Stage-0 refactored DAPiGen environment.

The released ``env.py`` is intentionally left untouched for regression. New
RLlib experiments may pass ``DAPiGenRLlibEnv`` directly to ``PPOTrainer``.
"""

from RL_PPO.envs.config import DAPiGenEnvConfig
from RL_PPO.envs.factory import build_stage0_components
from RL_PPO.envs.gym_wrapper import DAPiGenGymEnv, DAPiGenRLlibEnv
from RL_PPO.envs.gymnasium_wrapper import DAPiGenGymnasiumEnv

__all__ = [
    "DAPiGenEnvConfig",
    "DAPiGenGymEnv",
    "DAPiGenGymnasiumEnv",
    "DAPiGenRLlibEnv",
    "build_stage0_components",
]
