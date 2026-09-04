from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


STATE_SCHEMA_VERSION = 3
TRANSITION_SNAPSHOT_SCHEMA_VERSION = 2

TERMINATION_SUCCESS = "success"
TERMINATION_INVALID_ACTION = "invalid_action"
TERMINATION_NO_PRODUCT = "no_reaction_product"
TERMINATION_NO_CONTINUATION = "no_valid_continuation"
TERMINATION_ATOM_LIMIT = "atom_limit_exceeded"
TERMINATION_HORIZON = "design_horizon_exhausted"
TERMINATION_PI_REACTION_FAILED = "polyimide_reaction_failed"
TERMINATION_PREMATURE_COMPLETION = "minimum_growth_not_reached"

TERMINATION_REASONS = frozenset(
    (
        TERMINATION_SUCCESS,
        TERMINATION_INVALID_ACTION,
        TERMINATION_NO_PRODUCT,
        TERMINATION_NO_CONTINUATION,
        TERMINATION_ATOM_LIMIT,
        TERMINATION_HORIZON,
        TERMINATION_PI_REACTION_FAILED,
        TERMINATION_PREMATURE_COMPLETION,
    )
)


def _coerce_state_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError("%s must be an integer." % name)
    return int(value)


def _coerce_state_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("%s must be Boolean." % name)
    return bool(value)


def _coerce_action_id(value: Any, name: str) -> int:
    """Convert a scalar action id without silently truncating a float."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError("%s cannot be Boolean." % name)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise TypeError("%s must be an integer-valued scalar." % name)
        return int(numeric)
    raise TypeError("%s must be an integer scalar." % name)


@dataclass(frozen=True)
class DAPiGenAction:
    """Factorized action for the dianhydride and diamine construction heads."""

    dianhydride_id: int
    diamine_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dianhydride_id",
            _coerce_action_id(self.dianhydride_id, "dianhydride_id"),
        )
        object.__setattr__(
            self,
            "diamine_id",
            _coerce_action_id(self.diamine_id, "diamine_id"),
        )

    def as_tuple(self) -> Tuple[int, int]:
        return int(self.dianhydride_id), int(self.diamine_id)

    @classmethod
    def from_any(cls, action: Any) -> "DAPiGenAction":
        if isinstance(action, cls):
            return action
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                raise TypeError("A DAPiGen action requires two action ids.")
            action = action.tolist()
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            raise TypeError("A DAPiGen action must contain exactly two integer ids.")
        return cls(action[0], action[1])


@dataclass(frozen=True)
class DAPiGenState:
    """Immutable state snapshot used for exact restoration and branching.

    ``*_growth_steps`` are part of the Markov state. They make direct-complete
    building-block semantics explicit and support controlled long-horizon task
    variants without consulting the historical trajectory.
    """

    dianhydride_smiles: str
    diamine_smiles: str
    environment_id: str
    dianhydride_complete: bool
    diamine_complete: bool
    dianhydride_growth_steps: int
    diamine_growth_steps: int
    step_index: int
    max_steps: int
    terminated: bool = False
    truncated: bool = False
    termination_reason: Optional[str] = None
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = _coerce_state_integer(self.schema_version, "schema_version")
        if schema_version != STATE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported state schema version: %s" % self.schema_version
            )
        object.__setattr__(self, "schema_version", schema_version)
        if not isinstance(self.dianhydride_smiles, str) or not self.dianhydride_smiles:
            raise ValueError("dianhydride_smiles must be a non-empty string.")
        if not isinstance(self.diamine_smiles, str) or not self.diamine_smiles:
            raise ValueError("diamine_smiles must be a non-empty string.")
        if not isinstance(self.environment_id, str) or not self.environment_id:
            raise ValueError("environment_id must be a non-empty string.")
        for name in (
            "dianhydride_complete",
            "diamine_complete",
            "terminated",
            "truncated",
        ):
            object.__setattr__(
                self, name, _coerce_state_boolean(getattr(self, name), name)
            )
        for name in (
            "dianhydride_growth_steps",
            "diamine_growth_steps",
            "step_index",
            "max_steps",
        ):
            object.__setattr__(
                self, name, _coerce_state_integer(getattr(self, name), name)
            )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.step_index < 0 or self.step_index > self.max_steps:
            raise ValueError("step_index must lie in [0, max_steps].")
        if self.dianhydride_growth_steps < 0:
            raise ValueError("dianhydride_growth_steps cannot be negative.")
        if self.diamine_growth_steps < 0:
            raise ValueError("diamine_growth_steps cannot be negative.")
        if self.dianhydride_growth_steps > self.step_index:
            raise ValueError("dianhydride_growth_steps cannot exceed step_index.")
        if self.diamine_growth_steps > self.step_index:
            raise ValueError("diamine_growth_steps cannot exceed step_index.")
        if self.terminated and self.truncated:
            raise ValueError("A state cannot be terminated and truncated together.")
        if self.done and self.termination_reason is None:
            raise ValueError("A finished state must record a termination_reason.")
        if not self.done and self.termination_reason is not None:
            raise ValueError("An unfinished state cannot record a termination_reason.")
        if (
            self.termination_reason is not None
            and self.termination_reason not in TERMINATION_REASONS
        ):
            raise ValueError(
                "Unsupported termination_reason: %s" % self.termination_reason
            )
        if self.truncated and self.termination_reason != TERMINATION_HORIZON:
            raise ValueError("Only a design horizon may produce truncation.")
        if self.step_index == self.max_steps and not self.done:
            raise ValueError("A state at max_steps must be finished.")
        if (
            self.dianhydride_complete
            and self.diamine_complete
            and not self.done
        ):
            raise ValueError("A state with two completed monomers must be finished.")
        if (
            self.termination_reason == TERMINATION_HORIZON
            and self.step_index != self.max_steps
        ):
            raise ValueError("A horizon termination must occur at max_steps.")
        if (
            self.termination_reason == TERMINATION_SUCCESS
            and not (self.dianhydride_complete and self.diamine_complete)
        ):
            raise ValueError("A successful state must contain two completed monomers.")

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)

    @property
    def remaining_steps(self) -> int:
        return max(0, int(self.max_steps) - int(self.step_index))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "dianhydride_smiles": self.dianhydride_smiles,
            "diamine_smiles": self.diamine_smiles,
            "environment_id": self.environment_id,
            "dianhydride_complete": bool(self.dianhydride_complete),
            "diamine_complete": bool(self.diamine_complete),
            "dianhydride_growth_steps": int(self.dianhydride_growth_steps),
            "diamine_growth_steps": int(self.diamine_growth_steps),
            "step_index": int(self.step_index),
            "max_steps": int(self.max_steps),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "termination_reason": self.termination_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DAPiGenState":
        payload = dict(data)
        schema = payload.get("schema_version")
        if schema is None:
            raise ValueError(
                "Legacy state snapshots cannot be restored exactly because they "
                "do not record per-side growth counts. Regenerate the trajectory "
                "with the Stage-0 environment."
            )
        if int(schema) != STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported state schema version: %s" % schema)
        required = (
            "dianhydride_smiles",
            "diamine_smiles",
            "environment_id",
            "dianhydride_complete",
            "diamine_complete",
            "dianhydride_growth_steps",
            "diamine_growth_steps",
            "step_index",
            "max_steps",
            "terminated",
            "truncated",
            "termination_reason",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                "State snapshot is incomplete; missing fields: %s"
                % ", ".join(sorted(missing))
            )
        return cls(**payload)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def state_id(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:24]


@dataclass
class ActionMask:
    """Boolean action masks for two factorized heads, including explicit NOOP."""

    dianhydride: np.ndarray
    diamine: np.ndarray

    def __post_init__(self) -> None:
        self.dianhydride = np.asarray(self.dianhydride, dtype=np.bool_).reshape(-1)
        self.diamine = np.asarray(self.diamine, dtype=np.bool_).reshape(-1)
        if self.dianhydride.size == 0 or not bool(self.dianhydride.any()):
            raise ValueError("The dianhydride action mask cannot be empty.")
        if self.diamine.size == 0 or not bool(self.diamine.any()):
            raise ValueError("The diamine action mask cannot be empty.")

    def copy(self) -> "ActionMask":
        return ActionMask(self.dianhydride.copy(), self.diamine.copy())

    def to_dict(self) -> Dict[str, Sequence[int]]:
        return {
            "dianhydride": self.dianhydride.astype(np.int8).tolist(),
            "diamine": self.diamine.astype(np.int8).tolist(),
        }

    def flattened(self, dtype=np.float32) -> np.ndarray:
        return np.concatenate([self.dianhydride, self.diamine]).astype(dtype)

    def validate(self, action: DAPiGenAction) -> None:
        action = DAPiGenAction.from_any(action)
        d_id, a_id = action.as_tuple()
        if not 0 <= d_id < self.dianhydride.size:
            raise ValueError("Dianhydride action id is outside the action space.")
        if not 0 <= a_id < self.diamine.size:
            raise ValueError("Diamine action id is outside the action space.")
        if not bool(self.dianhydride[d_id]):
            raise ValueError("Dianhydride action %d is masked out." % d_id)
        if not bool(self.diamine[a_id]):
            raise ValueError("Diamine action %d is masked out." % a_id)


@dataclass
class CoreTransition:
    """Transition emitted by the chemistry core before terminal scoring."""

    state: DAPiGenState
    observation: np.ndarray
    action_mask: ActionMask
    terminated: bool
    truncated: bool
    terminal_smiles: Optional[str] = None
    terminal_candidates: Tuple[str, ...] = field(default_factory=tuple)
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, DAPiGenState):
            raise TypeError("Transition state must be a DAPiGenState.")
        if not isinstance(self.action_mask, ActionMask):
            raise TypeError("Transition action_mask must be an ActionMask.")
        self.observation = np.asarray(self.observation, dtype=np.float32).reshape(-1)
        if not bool(np.isfinite(self.observation).all()):
            raise ValueError("Transition observation must contain only finite values.")
        self.terminal_candidates = tuple(str(item) for item in self.terminal_candidates)
        self.info = dict(self.info)
        self.terminated = _coerce_state_boolean(self.terminated, "terminated")
        self.truncated = _coerce_state_boolean(self.truncated, "truncated")
        if self.terminated != self.state.terminated:
            raise ValueError("Transition termination flag disagrees with state.")
        if self.truncated != self.state.truncated:
            raise ValueError("Transition truncation flag disagrees with state.")
        if self.terminal_smiles is not None and not self.terminated:
            raise ValueError("A terminal molecule may only appear on termination.")
        if self.state.termination_reason == TERMINATION_SUCCESS:
            if self.terminal_smiles is None:
                raise ValueError("A successful transition must contain terminal_smiles.")
            if "*" in self.terminal_smiles:
                raise ValueError("terminal_smiles must be a complete molecule.")
        elif self.terminal_smiles is not None:
            raise ValueError(
                "terminal_smiles is only valid for a successful transition."
            )
        if self.terminal_smiles is not None and self.terminal_candidates:
            if self.terminal_smiles not in self.terminal_candidates:
                raise ValueError("terminal_smiles must belong to terminal_candidates.")
        if any("*" in item for item in self.terminal_candidates):
            raise ValueError("terminal_candidates must contain complete molecules.")
        if (
            self.terminal_candidates
            and self.state.termination_reason != TERMINATION_SUCCESS
        ):
            raise ValueError(
                "terminal_candidates are only valid for a successful transition."
            )

    def to_snapshot(self) -> Dict[str, Any]:
        """Serialize the branch-relevant transition state without observations."""

        environment_id = self.info.get("environment_id")
        if not isinstance(environment_id, str) or not environment_id:
            raise ValueError(
                "A transition snapshot requires the producing environment_id."
            )
        if environment_id != self.state.environment_id:
            raise ValueError(
                "Transition environment_id disagrees with the state environment_id."
            )
        return {
            "snapshot_schema_version": TRANSITION_SNAPSHOT_SCHEMA_VERSION,
            "environment_id": environment_id,
            "state": self.state.to_dict(),
            "terminal_smiles": self.terminal_smiles,
            "terminal_candidates": list(self.terminal_candidates),
        }


@dataclass(frozen=True)
class TerminalEvaluation:
    """Evaluation of one complete polyimide; never of a partial structure."""

    objective: float
    properties: Mapping[str, float] = field(default_factory=dict)
    valid: bool = True
    canonical_smiles: Optional[str] = None
    epistemic_std: Optional[float] = None
    failure_reason: Optional[str] = None
    evaluator_version: str = "unknown"

    def __post_init__(self) -> None:
        objective = float(self.objective)
        if not math.isfinite(objective):
            raise ValueError("objective must be finite.")
        object.__setattr__(self, "objective", objective)
        properties = {}
        for key, value in self.properties.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("Property %s must be finite." % key)
            properties[str(key)] = numeric
        object.__setattr__(self, "properties", properties)
        object.__setattr__(self, "valid", _coerce_state_boolean(self.valid, "valid"))
        if self.canonical_smiles is not None and (
            not isinstance(self.canonical_smiles, str)
            or not self.canonical_smiles
        ):
            raise ValueError("canonical_smiles must be a non-empty string when set.")
        if self.failure_reason is not None and not isinstance(
            self.failure_reason, str
        ):
            raise TypeError("failure_reason must be a string when set.")
        if self.epistemic_std is not None:
            epistemic_std = float(self.epistemic_std)
            if not math.isfinite(epistemic_std) or epistemic_std < 0.0:
                raise ValueError("epistemic_std must be finite and non-negative.")
            object.__setattr__(self, "epistemic_std", epistemic_std)
        if not isinstance(self.evaluator_version, str) or not self.evaluator_version:
            raise ValueError("evaluator_version must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": float(self.objective),
            "properties": {
                str(key): float(value) for key, value in self.properties.items()
            },
            "valid": bool(self.valid),
            "canonical_smiles": self.canonical_smiles,
            "epistemic_std": (
                None if self.epistemic_std is None else float(self.epistemic_std)
            ),
            "failure_reason": self.failure_reason,
            "evaluator_version": str(self.evaluator_version),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TerminalEvaluation":
        payload = dict(data)
        required = ("objective", "properties", "valid", "evaluator_version")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                "Terminal evaluation is incomplete; missing fields: %s"
                % ", ".join(sorted(missing))
            )
        return cls(**payload)


@dataclass
class EvaluatedTransition:
    """Core transition plus the reward exposed to PPO."""

    core: CoreTransition
    reward: float
    evaluation: Optional[TerminalEvaluation] = None

    def __post_init__(self) -> None:
        if not isinstance(self.core, CoreTransition):
            raise TypeError("core must be a CoreTransition.")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite.")
        self.reward = reward
        if self.evaluation is not None:
            if not isinstance(self.evaluation, TerminalEvaluation):
                raise TypeError("evaluation must be a TerminalEvaluation when set.")
            if self.core.terminal_smiles is None:
                raise ValueError(
                    "A terminal evaluation requires core.terminal_smiles."
                )

    @property
    def state(self) -> DAPiGenState:
        return self.core.state

    @property
    def observation(self) -> np.ndarray:
        return self.core.observation

    @property
    def action_mask(self) -> ActionMask:
        return self.core.action_mask

    @property
    def terminated(self) -> bool:
        return self.core.terminated

    @property
    def truncated(self) -> bool:
        return self.core.truncated
