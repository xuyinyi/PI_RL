from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class DAPiGenEnvConfig:
    """Versioned task semantics shared by PPO, Policy-CC, and MCC-PPO.

    The defaults are the strict Stage-0 benchmark semantics. Legacy-like modes
    remain available only for regression diagnostics and must not be mixed into
    the primary comparison.
    """

    max_atoms: int = 60
    max_steps: int = 5

    # closure_exact_cached: scalable BRICS-label prefilter plus exact chemistry
    # only for actions predicted to consume the final two attachments;
    # compatibility: label-only historical diagnostic; exact_cached: enumerate
    # every action while constructing masks; all: no mask.
    mask_mode: str = "closure_exact_cached"

    # pristine_only: a complete no-dummy block can be selected only before that
    # side has grown; legacy_replace reproduces the released silent replacement;
    # never removes direct complete molecules from the sequential task.
    complete_block_policy: str = "pristine_only"

    # Optional controlled long-horizon variants. A completion before the stated
    # number of side actions is masked in exact mode and rejected at transition.
    minimum_dianhydride_growth_steps: int = 0
    minimum_diamine_growth_steps: int = 0

    # seeded_uniform is reward-independent and uses common-quantile named RNG.
    product_selection: str = "seeded_uniform"

    # augmented_v3 adds attachment-label histograms to the v2 metadata. The
    # immutable DAPiGenState is the exact Markov state; the learned embedding is
    # an explicitly lossy policy observation.
    observation_mode: str = "augmented_v3"

    # The construction horizon is normally part of the task and therefore a
    # true terminal failure. time_limit_truncation is available for diagnostics.
    horizon_semantics: str = "failure_terminal"

    # raise is recommended during development; terminate supports robust large
    # sweeps while still auditing invalid actions as failures.
    invalid_action_handling: str = "raise"
    failure_reward: float = 0.0

    attachment_label_max: int = 16
    include_terminal_candidates_in_info: bool = False
    environment_version: str = "dapigen-stage0-v2.3"

    def __post_init__(self) -> None:
        for name in (
            "max_atoms",
            "max_steps",
            "minimum_dianhydride_growth_steps",
            "minimum_diamine_growth_steps",
            "attachment_label_max",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError("%s must be an integer." % name)
            object.__setattr__(self, name, int(value))
        if self.max_atoms <= 0:
            raise ValueError("max_atoms must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.mask_mode not in (
            "closure_exact_cached",
            "compatibility",
            "exact_cached",
            "all",
        ):
            raise ValueError("Unsupported mask_mode: %s" % self.mask_mode)
        if self.complete_block_policy not in (
            "pristine_only",
            "legacy_replace",
            "never",
        ):
            raise ValueError(
                "Unsupported complete_block_policy: %s"
                % self.complete_block_policy
            )
        if self.minimum_dianhydride_growth_steps < 0:
            raise ValueError("minimum_dianhydride_growth_steps cannot be negative.")
        if self.minimum_diamine_growth_steps < 0:
            raise ValueError("minimum_diamine_growth_steps cannot be negative.")
        if self.product_selection not in ("seeded_uniform", "canonical_first"):
            raise ValueError("Unsupported product_selection: %s" % self.product_selection)
        if self.observation_mode not in (
            "augmented_v3",
            "markov_v2",
            "markov_v1",
            "legacy_polybert",
        ):
            raise ValueError("Unsupported observation_mode: %s" % self.observation_mode)
        if self.horizon_semantics not in (
            "failure_terminal",
            "time_limit_truncation",
        ):
            raise ValueError("Unsupported horizon_semantics: %s" % self.horizon_semantics)
        if self.invalid_action_handling not in ("terminate", "raise"):
            raise ValueError(
                "Unsupported invalid_action_handling: %s"
                % self.invalid_action_handling
            )
        if self.attachment_label_max <= 0:
            raise ValueError("attachment_label_max must be positive.")
        if isinstance(self.failure_reward, bool) or not isinstance(
            self.failure_reward, numbers.Real
        ):
            raise TypeError("failure_reward must be numeric.")
        failure_reward = float(self.failure_reward)
        if not math.isfinite(failure_reward):
            raise ValueError("failure_reward must be finite.")
        object.__setattr__(self, "failure_reward", failure_reward)
        if not isinstance(self.include_terminal_candidates_in_info, bool):
            raise TypeError("include_terminal_candidates_in_info must be Boolean.")
        if not isinstance(self.environment_version, str) or not self.environment_version:
            raise ValueError("environment_version must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DAPiGenEnvConfig":
        return cls(**dict(data))

    @classmethod
    def legacy_regression(cls, max_atoms: int = 60, max_steps: int = 6):
        """Approximate the released task for regression, excluding oracle argmax.

        Reward-dependent terminal-product selection is deliberately not
        restored because it makes the transition kernel depend on the evaluator.
        """

        return cls(
            max_atoms=max_atoms,
            max_steps=max_steps,
            mask_mode="all",
            complete_block_policy="legacy_replace",
            minimum_dianhydride_growth_steps=0,
            minimum_diamine_growth_steps=0,
            product_selection="seeded_uniform",
            observation_mode="legacy_polybert",
            horizon_semantics="failure_terminal",
            invalid_action_handling="terminate",
            environment_version="dapigen-stage0-legacy-regression-v2.3",
        )
