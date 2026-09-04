"""RLlib 1.13 action-mask model for the Stage-0 Tuple(Discrete, Discrete) action.

Import this module only inside the released DAPiGen environment, where Ray is
installed. It is intentionally optional so the chemistry core remains usable
without Ray.
"""

from __future__ import annotations

import numpy as np
import torch

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN


class Stage0MaskedFCModel(TorchModelV2, torch.nn.Module):
    """Apply exact per-head masks to the concatenated Tuple-action logits."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)
        original_space = getattr(obs_space, "original_space", obs_space)
        feature_space = original_space["observations"]
        expected_outputs = int(
            action_space.spaces[0].n + action_space.spaces[1].n
        )
        if int(num_outputs) != expected_outputs:
            raise ValueError(
                "RLlib requested %d outputs, expected %d for the two action heads."
                % (int(num_outputs), expected_outputs)
            )
        self.internal_model = TorchFC(
            feature_space,
            action_space,
            num_outputs,
            model_config,
            name + "_unmasked",
        )
        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        observations = input_dict["obs"]["observations"]
        action_mask = input_dict["obs"]["action_mask"].float()
        logits, _ = self.internal_model(
            {"obs": observations}, state, seq_lens
        )
        if tuple(logits.shape) != tuple(action_mask.shape):
            raise ValueError(
                "Action-mask shape %r does not match logits %r."
                % (tuple(action_mask.shape), tuple(logits.shape))
            )
        infinite_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        self._last_value = self.internal_model.value_function()
        return logits + infinite_mask, state

    def value_function(self):
        if self._last_value is None:
            raise RuntimeError("value_function() called before forward().")
        return self._last_value


def register_stage0_masked_model(name: str = "dapigen_stage0_masked_fc") -> str:
    ModelCatalog.register_custom_model(name, Stage0MaskedFCModel)
    return name
