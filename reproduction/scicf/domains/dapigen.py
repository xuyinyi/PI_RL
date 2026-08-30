"""Minimal DAPiGen wrapper for trajectory capture and matched replay."""

from __future__ import annotations

import copy
import hashlib
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from reproduction.scicf.core.domain import (
    ContinuationOutcome,
    PolicyCallable,
    ScientificDomainAdapter,
)
from reproduction.scicf.core.oracle import CountingOracle, OracleLedger
from reproduction.scicf.core.records import (
    ActionTuple,
    Intervention,
    TrajectoryRecord,
    TrajectoryStep,
)


_MUTABLE_ENV_FIELDS = (
    "base_smiles_dianhydride",
    "base_smiles_diamine",
    "env_step",
    "flag_dianhydride",
    "flag_diamine",
    "PI",
    "transmittance",
    "cte",
    "strength",
    "tg",
    "SaScore",
    "prev_smiles_dianhydride",
    "prev_smiles_diamine",
    "prev_action",
    "base_mol_dianhydride",
    "base_mol_diamine",
)


@dataclass(frozen=True)
class DAPiGenSnapshot:
    """Trusted internal snapshot of one pre-action DAPiGen state."""

    snapshot_id: str
    observation: np.ndarray
    environment_state: Mapping[str, Any]
    python_random_state: object
    numpy_random_state: tuple
    environment_rng_kind: Optional[str]
    environment_rng_state: Optional[object]


def _action_tuple(action: Any) -> ActionTuple:
    if len(action) != 2:
        raise ValueError("DAPiGen action must have two components")
    return int(action[0]), int(action[1])


def _observation_tuple(observation: Any) -> Tuple[float, ...]:
    return tuple(float(item) for item in np.asarray(observation).reshape(-1))


def _environment_rng_state(environment: Any) -> Tuple[Optional[str], Optional[object]]:
    generator = getattr(environment, "np_random", None)
    if generator is None:
        return None, None
    if hasattr(generator, "get_state"):
        return "random-state", copy.deepcopy(generator.get_state())
    if hasattr(generator, "bit_generator"):
        return "generator", copy.deepcopy(generator.bit_generator.state)
    raise TypeError("unsupported environment RNG type: {}".format(type(generator).__name__))


def _restore_environment_rng(
    environment: Any, kind: Optional[str], state: Optional[object]
) -> None:
    if kind is None:
        return
    generator = getattr(environment, "np_random", None)
    if generator is None:
        raise RuntimeError("snapshot contains environment RNG state but environment has no RNG")
    if kind == "random-state" and hasattr(generator, "set_state"):
        generator.set_state(copy.deepcopy(state))
        return
    if kind == "generator" and hasattr(generator, "bit_generator"):
        generator.bit_generator.state = copy.deepcopy(state)
        return
    raise TypeError("environment RNG implementation changed since snapshot capture")


class DAPiGenDomainAdapter(ScientificDomainAdapter):
    """Wrap DAPiGen without changing its PPO observation, action, or reward."""

    protocol = "dapigen-domain-adapter-v1"

    def __init__(self, environment: Any, ledger: Optional[OracleLedger] = None) -> None:
        self.environment = environment
        self.ledger = ledger or OracleLedger()
        scoring_function = environment.scoring_function
        if isinstance(scoring_function, CountingOracle):
            if scoring_function.ledger is not self.ledger:
                raise ValueError("environment is already attached to another oracle ledger")
        else:
            environment.scoring_function = CountingOracle(scoring_function, self.ledger)

    def reset(self, seed: int) -> Any:
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(self.environment, "seed"):
            self.environment.seed(seed)
        return self.environment.reset()

    def serialize_state(self) -> Mapping[str, Any]:
        return {
            "protocol": self.protocol,
            "env_step": int(getattr(self.environment, "env_step", 0)),
            "dianhydride_structure": getattr(
                self.environment, "base_smiles_dianhydride", None
            ),
            "diamine_structure": getattr(self.environment, "base_smiles_diamine", None),
            "dianhydride_complete": bool(
                getattr(self.environment, "flag_dianhydride", False)
            ),
            "diamine_complete": bool(getattr(self.environment, "flag_diamine", False)),
            "terminal_polymer": getattr(self.environment, "PI", None),
            "properties": self._terminal_properties(),
            "legal_action_counts": {
                "dianhydride": int(self.environment.action_space_dianhydride.n),
                "diamine": int(self.environment.action_space_diamine.n),
            },
        }

    def capture_snapshot(self, observation: Any) -> DAPiGenSnapshot:
        state: Dict[str, Any] = {}
        for field_name in _MUTABLE_ENV_FIELDS:
            if hasattr(self.environment, field_name):
                state[field_name] = copy.deepcopy(getattr(self.environment, field_name))
        rng_kind, rng_state = _environment_rng_state(self.environment)
        payload = (
            state,
            np.asarray(observation),
            random.getstate(),
            np.random.get_state(),
            rng_kind,
            rng_state,
        )
        snapshot_id = hashlib.sha256(
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        ).hexdigest()
        return DAPiGenSnapshot(
            snapshot_id=snapshot_id,
            observation=np.asarray(observation).copy(),
            environment_state=state,
            python_random_state=copy.deepcopy(random.getstate()),
            numpy_random_state=copy.deepcopy(np.random.get_state()),
            environment_rng_kind=rng_kind,
            environment_rng_state=rng_state,
        )

    def restore_snapshot(self, snapshot: DAPiGenSnapshot) -> Any:
        for field_name in _MUTABLE_ENV_FIELDS:
            if field_name in snapshot.environment_state:
                setattr(
                    self.environment,
                    field_name,
                    copy.deepcopy(snapshot.environment_state[field_name]),
                )
            elif hasattr(self.environment, field_name):
                delattr(self.environment, field_name)
        random.setstate(copy.deepcopy(snapshot.python_random_state))
        np.random.set_state(copy.deepcopy(snapshot.numpy_random_state))
        _restore_environment_rng(
            self.environment,
            snapshot.environment_rng_kind,
            snapshot.environment_rng_state,
        )
        return snapshot.observation.copy()

    def enumerate_interventions(
        self,
        trajectory_id: str,
        timestep: int,
        factual_action: ActionTuple,
    ) -> Sequence[Intervention]:
        factual = _action_tuple(factual_action)
        components = (
            (
                "dianhydride",
                0,
                int(self.environment.action_space_dianhydride.n),
                self.environment.building_blocks_dianhydride,
            ),
            (
                "diamine",
                1,
                int(self.environment.action_space_diamine.n),
                self.environment.building_blocks_diamine,
            ),
        )
        interventions = []
        for component, component_index, size, structures in components:
            completion_flag = (
                "flag_dianhydride" if component == "dianhydride" else "flag_diamine"
            )
            if bool(getattr(self.environment, completion_flag, False)):
                continue
            if factual[component_index] < 0 or factual[component_index] >= size:
                raise ValueError("factual action is outside the legal action space")
            seen_structures = {str(structures[factual[component_index]])}
            for alternative in range(size):
                if alternative == factual[component_index]:
                    continue
                alternative_structure = str(structures[alternative])
                if alternative_structure in seen_structures:
                    continue
                seen_structures.add(alternative_structure)
                action = list(factual)
                action[component_index] = alternative
                intervention_id = "{}:t{:03d}:{}:{:04d}-{:04d}".format(
                    trajectory_id,
                    timestep,
                    component,
                    factual[component_index],
                    alternative,
                )
                interventions.append(
                    Intervention(
                        intervention_id=intervention_id,
                        trajectory_id=trajectory_id,
                        timestep=timestep,
                        component=component,
                        factual_action=factual,
                        alternative_action=_action_tuple(action),
                        factual_component_value=factual[component_index],
                        alternative_component_value=alternative,
                        alternative_structure=alternative_structure,
                        metadata={
                            "factual_structure": str(
                                structures[factual[component_index]]
                            )
                        },
                    )
                )
        return tuple(interventions)

    def apply_intervention(
        self, factual_action: ActionTuple, intervention: Intervention
    ) -> ActionTuple:
        factual = _action_tuple(factual_action)
        if factual != intervention.factual_action:
            raise ValueError("intervention does not belong to the supplied factual action")
        result = intervention.alternative_action
        unchanged_index = 1 if intervention.component == "dianhydride" else 0
        if result[unchanged_index] != factual[unchanged_index]:
            raise ValueError("intervention changed an unrelated action component")
        return result

    def continue_from_snapshot(
        self,
        snapshot: DAPiGenSnapshot,
        first_action: ActionTuple,
        continuation_policy: PolicyCallable,
        policy_version: str,
        continuation_seed: int,
        oracle_scope: str,
        max_steps: int,
    ) -> ContinuationOutcome:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        observation = self.restore_snapshot(snapshot)
        random.seed(continuation_seed)
        np.random.seed(continuation_seed)
        if hasattr(self.environment, "seed"):
            self.environment.seed(continuation_seed)
        policy_rng = np.random.RandomState(continuation_seed)
        actions = []
        total_return = 0.0
        before = self.ledger.snapshot()
        done = False
        for local_step in range(max_steps):
            action = (
                _action_tuple(first_action)
                if local_step == 0
                else _action_tuple(continuation_policy(observation, policy_rng))
            )
            with self.ledger.scope(oracle_scope):
                observation, reward, done, _ = self.environment.step(action)
            actions.append(action)
            total_return += float(reward)
            if done:
                break
        if not done:
            raise RuntimeError("counterfactual continuation exceeded max_steps")
        return ContinuationOutcome(
            terminal_return=total_return,
            terminal_scientific_object=self._terminal_object(),
            terminal_properties=self._terminal_properties(),
            actions=tuple(actions),
            environment_transitions=len(actions),
            atomic_oracle_calls=self.ledger.delta(before, oracle_scope),
            policy_version=policy_version,
            continuation_seed=continuation_seed,
        )

    def record_episode(
        self,
        policy: PolicyCallable,
        policy_version: str,
        seed: int,
        max_steps: int,
        trajectory_id: Optional[str] = None,
    ) -> Tuple[TrajectoryRecord, Mapping[str, DAPiGenSnapshot]]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        trajectory_id = trajectory_id or "dapigen-{}-seed-{}".format(
            hashlib.sha256(policy_version.encode("utf-8")).hexdigest()[:12], seed
        )
        observation = self.reset(seed)
        policy_rng = np.random.RandomState(seed)
        steps = []
        snapshots: Dict[str, DAPiGenSnapshot] = {}
        total_return = 0.0
        before = self.ledger.snapshot()
        for timestep in range(max_steps):
            snapshot = self.capture_snapshot(observation)
            snapshots[snapshot.snapshot_id] = snapshot
            pre_state = self.serialize_state()
            action = _action_tuple(policy(observation, policy_rng))
            with self.ledger.scope("factual"):
                next_observation, reward, done, _ = self.environment.step(action)
            total_return += float(reward)
            properties = self._terminal_properties() if done else {}
            steps.append(
                TrajectoryStep(
                    timestep=timestep,
                    observation=_observation_tuple(observation),
                    factual_action=action,
                    pre_state=pre_state,
                    post_state=self.serialize_state(),
                    terminated=bool(done),
                    environment_reward=float(reward),
                    snapshot_id=snapshot.snapshot_id,
                    oracle_outputs=properties,
                )
            )
            observation = next_observation
            if done:
                break
        else:
            raise RuntimeError("factual trajectory exceeded max_steps")

        record = TrajectoryRecord(
            trajectory_id=trajectory_id,
            policy_version=policy_version,
            seed=seed,
            steps=tuple(steps),
            episode_return=total_return,
            terminal_scientific_object=self._terminal_object(),
            terminal_properties=self._terminal_properties(),
            environment_transitions=len(steps),
            atomic_oracle_calls=self.ledger.delta(before, "factual"),
        )
        return record, snapshots

    def _terminal_object(self) -> Optional[str]:
        value = getattr(self.environment, "PI", None)
        return None if value in (None, "None") else str(value)

    def _terminal_properties(self) -> Mapping[str, Any]:
        return {
            "transmittance": float(getattr(self.environment, "transmittance", 0.0)),
            "cte": float(getattr(self.environment, "cte", 0.0)),
            "strength": float(getattr(self.environment, "strength", 0.0)),
            "tg": float(getattr(self.environment, "tg", 0.0)),
            "sa_score": float(getattr(self.environment, "SaScore", 0.0)),
        }
