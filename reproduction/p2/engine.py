"""One native PyTorch PPO path shared by every P2 credit provider.

The implementation is intentionally small, but it is not a toy optimizer: it
collects masked factorized actions from the accepted Stage-0 Gymnasium API,
computes environment-return GAE, performs clipped PPO updates and checkpoints
the policy, optimizer, environment/evaluator ledger and RNG state together.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from reproduction.p2.budget import (
    RequestedCallBudgetManager,
    evaluator_ledger_delta,
)
from reproduction.p2.contracts import (
    CREDIT_ESTIMATOR_CONTRACT_ID,
    METHOD_EVALUATOR_SOURCES,
    METHOD_ON_POLICY_SOURCE,
    PPO,
    PPO_ENGINE_CONTRACT_ID,
    ContractViolation,
    CreditEstimate,
    CreditRequest,
    EvaluatorLedgerDelta,
    FrozenPolicyHandle,
    MethodRunContract,
    PPOUpdateReceipt,
    PendingLabelBatch,
    RolloutBatch,
    array_sha256,
    canonical_sha256,
    validate_credit_estimate,
    validate_pending_label_commit,
    validate_update_receipt,
)
from reproduction.p2.gae import compute_gae


PPO_ENGINE_STATE_SCHEMA_VERSION = 1


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer." % name)
    return int(value)


def _probability(value: float, name: str, include_zero: bool = True) -> float:
    parsed = float(value)
    lower_ok = parsed >= 0.0 if include_zero else parsed > 0.0
    if not math.isfinite(parsed) or not lower_ok or parsed > 1.0:
        bracket = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError("%s must lie in %s." % (name, bracket))
    return parsed


@dataclass(frozen=True)
class PPOEngineConfig:
    rollout_steps: int = 512
    gamma: float = 1.0
    gae_lambda: float = 0.95
    update_epochs: int = 10
    minibatch_size: int = 128
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    maximum_gradient_norm: float = 0.5
    target_kl: Optional[float] = 0.02
    normalize_actor_advantages: bool = True
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-8
    hidden_sizes: Tuple[int, ...] = (256, 256)
    seed: int = 20260904
    deterministic_torch: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("rollout_steps", "update_epochs", "minibatch_size"):
            _positive_integer(getattr(self, name), name)
        _probability(self.gamma, "gamma")
        _probability(self.gae_lambda, "gae_lambda")
        _probability(self.clip_ratio, "clip_ratio")
        _probability(self.value_clip, "value_clip")
        if not math.isfinite(float(self.value_loss_coefficient)) or self.value_loss_coefficient < 0:
            raise ValueError("value_loss_coefficient must be finite and non-negative.")
        if not math.isfinite(float(self.entropy_coefficient)) or self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be finite and non-negative.")
        if not math.isfinite(float(self.maximum_gradient_norm)) or self.maximum_gradient_norm <= 0:
            raise ValueError("maximum_gradient_norm must be finite and positive.")
        if self.target_kl is not None and (
            not math.isfinite(float(self.target_kl)) or self.target_kl <= 0
        ):
            raise ValueError("target_kl must be finite and positive when set.")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not math.isfinite(float(self.adam_epsilon)) or self.adam_epsilon <= 0:
            raise ValueError("adam_epsilon must be finite and positive.")
        hidden = tuple(
            _positive_integer(value, "hidden_size") for value in self.hidden_sizes
        )
        if not hidden:
            raise ValueError("hidden_sizes cannot be empty.")
        object.__setattr__(self, "hidden_sizes", hidden)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not isinstance(self.normalize_actor_advantages, bool):
            raise TypeError("normalize_actor_advantages must be Boolean.")
        if not isinstance(self.deterministic_torch, bool):
            raise TypeError("deterministic_torch must be Boolean.")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty identifier.")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


class FactorizedActorCritic(nn.Module):
    """Shared MLP with separate dianhydride, diamine and value heads."""

    def __init__(
        self,
        observation_dimension: int,
        number_of_dianhydride_actions: int,
        number_of_diamine_actions: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.observation_dimension = _positive_integer(
            int(observation_dimension), "observation_dimension"
        )
        self.number_of_dianhydride_actions = _positive_integer(
            int(number_of_dianhydride_actions), "number_of_dianhydride_actions"
        )
        self.number_of_diamine_actions = _positive_integer(
            int(number_of_diamine_actions), "number_of_diamine_actions"
        )
        layers = []
        previous = self.observation_dimension
        for width in hidden_sizes:
            width = _positive_integer(int(width), "hidden_size")
            layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.dianhydride_head = nn.Linear(
            previous, self.number_of_dianhydride_actions
        )
        self.diamine_head = nn.Linear(previous, self.number_of_diamine_actions)
        self.value_head = nn.Linear(previous, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.dianhydride_head.weight, gain=0.01)
        nn.init.orthogonal_(self.diamine_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, observations):
        features = self.trunk(observations)
        return (
            self.dianhydride_head(features),
            self.diamine_head(features),
            self.value_head(features).squeeze(-1),
        )


def torch_state_sha256(state: Mapping[str, Any]) -> str:
    """Stable digest over tensor names, dtypes, shapes and raw CPU bytes."""

    import hashlib
    import json

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(str(name).encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _readonly_array(values: Any, dtype, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    if not bool(np.isfinite(array).all()):
        raise ValueError("%s contains a non-finite value." % name)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PPOTransition:
    transition_id: str
    episode_id: int
    timestep: int
    observation: np.ndarray
    dianhydride_mask: np.ndarray
    diamine_mask: np.ndarray
    dianhydride_action: int
    diamine_action: int
    old_log_probability: float
    old_value: float
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    state_snapshot_before: Optional[Mapping[str, Any]] = None
    state_snapshot_after: Optional[Mapping[str, Any]] = None
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.transition_id, str) or not self.transition_id:
            raise ValueError("transition_id must be non-empty.")
        if self.episode_id < 0 or self.timestep < 0:
            raise ValueError("episode_id and timestep must be non-negative.")
        object.__setattr__(
            self,
            "observation",
            _readonly_array(self.observation, np.float32, "observation").reshape(-1),
        )
        object.__setattr__(
            self,
            "next_observation",
            _readonly_array(
                self.next_observation, np.float32, "next_observation"
            ).reshape(-1),
        )
        for name in ("dianhydride_mask", "diamine_mask"):
            mask = np.asarray(getattr(self, name), dtype=bool).reshape(-1).copy()
            if not bool(mask.any()):
                raise ValueError("%s must contain a legal action." % name)
            mask.setflags(write=False)
            object.__setattr__(self, name, mask)
        if not 0 <= int(self.dianhydride_action) < self.dianhydride_mask.size:
            raise ValueError("The recorded dianhydride action is out of range.")
        if not 0 <= int(self.diamine_action) < self.diamine_mask.size:
            raise ValueError("The recorded diamine action is out of range.")
        if not self.dianhydride_mask[int(self.dianhydride_action)]:
            raise ValueError("The recorded dianhydride action is masked.")
        if not self.diamine_mask[int(self.diamine_action)]:
            raise ValueError("The recorded diamine action is masked.")
        for name in ("old_log_probability", "old_value", "reward"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError("%s must be finite." % name)
        if bool(self.terminated) and bool(self.truncated):
            raise ValueError("A transition cannot be both terminated and truncated.")
        object.__setattr__(self, "info", dict(self.info))


@dataclass(frozen=True)
class PPOIterationResult:
    rollout: RolloutBatch
    credit: CreditEstimate
    receipt: PPOUpdateReceipt
    update_metrics: Mapping[str, float]
    rollout_evaluator_delta: EvaluatorLedgerDelta
    pending_label_commit: Optional[Mapping[str, Any]] = None


class PPOEngine:
    """Single masked factorized PPO implementation for the frozen P2 seam."""

    engine_contract_id = PPO_ENGINE_CONTRACT_ID

    def __init__(
        self,
        *,
        environment,
        run_contract: MethodRunContract,
        credit_estimator,
        observation_dimension: int,
        number_of_dianhydride_actions: int,
        number_of_diamine_actions: int,
        config: Optional[PPOEngineConfig] = None,
    ) -> None:
        self.environment = environment
        self.run_contract = run_contract
        self.credit_estimator = credit_estimator
        self.config = config or PPOEngineConfig()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA PPO device requested but unavailable.")
        if run_contract.ppo_engine_contract_id != self.engine_contract_id:
            raise ContractViolation("Run contract names another PPO engine.")
        if run_contract.credit_estimator_contract_id != CREDIT_ESTIMATOR_CONTRACT_ID:
            raise ContractViolation("Run contract names another credit estimator.")
        if run_contract.ppo_hyperparameters_sha256 != self.config.sha256:
            raise ContractViolation("PPO hyperparameter hash differs from run contract.")
        if credit_estimator.contract_id != CREDIT_ESTIMATOR_CONTRACT_ID:
            raise ContractViolation("Credit estimator contract identity mismatch.")
        if credit_estimator.method != run_contract.method:
            raise ContractViolation("Credit estimator method differs from run contract.")
        if frozenset(run_contract.allowed_evaluator_sources) != METHOD_EVALUATOR_SOURCES[
            run_contract.method
        ]:
            raise ContractViolation("Run evaluator-source allowlist is incorrect.")
        expected_source = METHOD_ON_POLICY_SOURCE[run_contract.method]
        if getattr(environment, "source", None) != expected_source:
            raise ContractViolation("Environment on-policy source tag is incorrect.")
        observed_environment_id = getattr(
            getattr(environment, "core", None), "environment_id", None
        )
        if observed_environment_id != run_contract.environment_id:
            raise ContractViolation("Environment identity differs from run contract.")
        initial_ledger = dict(environment.oracle_ledger())
        if initial_ledger.get("evaluator_version") not in (
            None,
            run_contract.evaluator_version,
        ):
            raise ContractViolation("Evaluator identity differs from run contract.")
        if initial_ledger.get("objective_contract") not in (
            None,
            run_contract.objective_contract,
        ):
            raise ContractViolation("Objective identity differs from run contract.")
        ledger_sources = initial_ledger.get("allowed_sources")
        if ledger_sources is not None and frozenset(ledger_sources) != frozenset(
            run_contract.allowed_evaluator_sources
        ):
            raise ContractViolation("Evaluator service source allowlist is incorrect.")

        torch.manual_seed(int(self.config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.seed))
        if self.config.deterministic_torch:
            torch.use_deterministic_algorithms(True)
        self.model = FactorizedActorCritic(
            observation_dimension=observation_dimension,
            number_of_dianhydride_actions=number_of_dianhydride_actions,
            number_of_diamine_actions=number_of_diamine_actions,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            eps=float(self.config.adam_epsilon),
        )
        self.observation_dimension = int(observation_dimension)
        self.number_of_dianhydride_actions = int(number_of_dianhydride_actions)
        self.number_of_diamine_actions = int(number_of_diamine_actions)
        self.policy_version = 0
        self._rng = np.random.default_rng(int(self.config.seed))
        random.seed(int(self.config.seed))
        self._budget = RequestedCallBudgetManager(environment.oracle_ledger)
        self._frozen_model = None
        self._frozen_handle = None
        self._frozen_rollout_collected = False
        self._approved_credit: Dict[str, str] = {}
        self._consumed_batches = set()
        self._rollout_sequence = 0
        self._transition_sequence = 0
        self._episode_sequence = 0
        self._episode_timestep = 0
        self._current_observation = None
        self._last_update_metrics: Dict[str, float] = {}

    @property
    def policy_state_sha256(self) -> str:
        return torch_state_sha256(self.model.state_dict())

    @property
    def last_update_metrics(self) -> Mapping[str, float]:
        return dict(self._last_update_metrics)

    def freeze_policy(self) -> FrozenPolicyHandle:
        if self._frozen_handle is not None:
            raise ContractViolation("A frozen policy is already active.")
        self._frozen_model = copy.deepcopy(self.model).to(self.device).eval()
        for parameter in self._frozen_model.parameters():
            parameter.requires_grad_(False)
        handle = FrozenPolicyHandle(
            policy_version=self.policy_version,
            state_sha256=torch_state_sha256(self._frozen_model.state_dict()),
            environment_id=self.run_contract.environment_id,
            task_contract_id=self.run_contract.task_contract_id,
        )
        self._frozen_handle = handle
        self._frozen_rollout_collected = False
        return handle

    def _assert_frozen(self, handle: FrozenPolicyHandle) -> None:
        if self._frozen_handle != handle or self._frozen_model is None:
            raise ContractViolation("Unknown or stale frozen-policy handle.")
        if handle.policy_version != self.policy_version:
            raise ContractViolation("Frozen policy does not match current policy version.")
        if torch_state_sha256(self._frozen_model.state_dict()) != handle.state_sha256:
            raise ContractViolation("Frozen policy state changed after freezing.")

    def _split_observation(self, observation) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(observation, Mapping):
            raise ContractViolation("P2 requires the masked dictionary observation API.")
        vector = np.asarray(observation.get("observations"), dtype=np.float32).reshape(-1)
        if vector.size != self.observation_dimension or not bool(np.isfinite(vector).all()):
            raise ContractViolation("Observation vector violates the PPO model contract.")
        if "dianhydride_mask" in observation and "diamine_mask" in observation:
            d_mask = np.asarray(observation["dianhydride_mask"], dtype=bool).reshape(-1)
            a_mask = np.asarray(observation["diamine_mask"], dtype=bool).reshape(-1)
        else:
            flat = np.asarray(observation.get("action_mask"), dtype=bool).reshape(-1)
            split = self.number_of_dianhydride_actions
            d_mask, a_mask = flat[:split], flat[split:]
        if d_mask.size != self.number_of_dianhydride_actions:
            raise ContractViolation("Dianhydride mask dimension mismatch.")
        if a_mask.size != self.number_of_diamine_actions:
            raise ContractViolation("Diamine mask dimension mismatch.")
        if not bool(d_mask.any()) or not bool(a_mask.any()):
            raise ContractViolation("Every factor must retain at least one legal action.")
        return vector.copy(), d_mask.copy(), a_mask.copy()

    @staticmethod
    def _masked_logits(logits, masks):
        if logits.shape != masks.shape:
            raise ContractViolation("Policy logits and masks do not align.")
        if not bool(torch.all(masks.any(dim=-1)).item()):
            raise ContractViolation("A policy row has no legal action.")
        return logits.masked_fill(~masks, torch.finfo(logits.dtype).min)

    def _policy_outputs(self, model, observation, d_mask, a_mask):
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        d_mask_tensor = torch.as_tensor(d_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        a_mask_tensor = torch.as_tensor(a_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            d_logits, a_logits, value = model(observation_tensor)
            d_logits = self._masked_logits(d_logits, d_mask_tensor)
            a_logits = self._masked_logits(a_logits, a_mask_tensor)
            d_probabilities = torch.softmax(d_logits, dim=-1)[0].cpu().numpy()
            a_probabilities = torch.softmax(a_logits, dim=-1)[0].cpu().numpy()
        return d_probabilities, a_probabilities, float(value.item())

    def _sample_action(self, observation, d_mask, a_mask):
        d_probabilities, a_probabilities, value = self._policy_outputs(
            self._frozen_model, observation, d_mask, a_mask
        )
        d_action = int(self._rng.choice(d_probabilities.size, p=d_probabilities))
        a_action = int(self._rng.choice(a_probabilities.size, p=a_probabilities))
        log_probability = float(
            math.log(max(float(d_probabilities[d_action]), 1e-30))
            + math.log(max(float(a_probabilities[a_action]), 1e-30))
        )
        return (d_action, a_action), log_probability, value

    def _value(self, model, observation) -> float:
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, _, value = model(observation_tensor)
        return float(value.item())

    def _reset_environment(self):
        episode_seed = int(self._rng.integers(0, np.iinfo(np.int64).max))
        reset_result = self.environment.reset(seed=episode_seed)
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        self._episode_sequence += 1
        self._episode_timestep = 0
        self._current_observation = copy.deepcopy(observation)

    def collect_rollout(self, frozen_policy: FrozenPolicyHandle) -> RolloutBatch:
        self._assert_frozen(frozen_policy)
        if self._frozen_rollout_collected:
            raise ContractViolation("A frozen policy can collect only one rollout batch.")
        reservation = self._budget.reserve(int(self.config.rollout_steps))
        ledger_before = dict(self.environment.oracle_ledger())
        transitions = []
        next_values = []
        try:
            for _ in range(int(self.config.rollout_steps)):
                if self._current_observation is None:
                    self._reset_environment()
                observation, d_mask, a_mask = self._split_observation(
                    self._current_observation
                )
                state_before = None
                current_state = getattr(self.environment, "current_state", None)
                if current_state is not None and hasattr(current_state, "to_dict"):
                    state_before = current_state.to_dict()
                action, old_log_probability, old_value = self._sample_action(
                    observation, d_mask, a_mask
                )
                step_result = self.environment.step(action)
                if not isinstance(step_result, tuple) or len(step_result) not in (4, 5):
                    raise ContractViolation("Environment step does not follow Gym/Gymnasium API.")
                if len(step_result) == 5:
                    next_raw, reward, terminated, truncated, info = step_result
                else:
                    next_raw, reward, done, info = step_result
                    terminated, truncated = bool(done), False
                next_observation, _, _ = self._split_observation(next_raw)
                next_value = (
                    0.0
                    if bool(terminated)
                    else self._value(self._frozen_model, next_observation)
                )
                transition_id = "p2t-%012d" % self._transition_sequence
                self._transition_sequence += 1
                state_after = None
                next_state = getattr(self.environment, "current_state", None)
                if next_state is not None and hasattr(next_state, "to_dict"):
                    state_after = next_state.to_dict()
                transitions.append(
                    PPOTransition(
                        transition_id=transition_id,
                        episode_id=self._episode_sequence - 1,
                        timestep=self._episode_timestep,
                        observation=observation,
                        dianhydride_mask=d_mask,
                        diamine_mask=a_mask,
                        dianhydride_action=action[0],
                        diamine_action=action[1],
                        old_log_probability=old_log_probability,
                        old_value=old_value,
                        reward=float(reward),
                        next_observation=next_observation,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        state_snapshot_before=state_before,
                        state_snapshot_after=state_after,
                        info=dict(info),
                    )
                )
                next_values.append(next_value)
                self._episode_timestep += 1
                if bool(terminated) or bool(truncated):
                    self._current_observation = None
                    self._episode_timestep = 0
                else:
                    self._current_observation = copy.deepcopy(next_raw)
            ledger_after = dict(self.environment.oracle_ledger())
            evaluator_delta = evaluator_ledger_delta(ledger_before, ledger_after)
            self._budget.reconcile(reservation, evaluator_delta.requested_calls)
        except Exception:
            try:
                self._budget.release(reservation)
            except ContractViolation:
                pass
            raise

        rewards = np.asarray([item.reward for item in transitions], dtype=np.float64)
        values = np.asarray([item.old_value for item in transitions], dtype=np.float64)
        advantages, critic_returns = compute_gae(
            rewards=rewards,
            values=values,
            next_values=np.asarray(next_values, dtype=np.float64),
            terminated=np.asarray([item.terminated for item in transitions]),
            truncated=np.asarray([item.truncated for item in transitions]),
            episode_ids=np.asarray([item.episode_id for item in transitions]),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        batch_id = "p2b-%06d-%s" % (
            self._rollout_sequence,
            canonical_sha256(
                {
                    "policy": frozen_policy.state_sha256,
                    "transition_ids": [item.transition_id for item in transitions],
                }
            )[:16],
        )
        self._rollout_sequence += 1
        rollout = RolloutBatch(
            batch_id=batch_id,
            environment_id=self.run_contract.environment_id,
            task_contract_id=self.run_contract.task_contract_id,
            budget_contract_id=self.run_contract.budget_contract_id,
            frozen_policy=frozen_policy,
            transition_ids=tuple(item.transition_id for item in transitions),
            transitions=tuple(transitions),
            gae_advantages=advantages,
            critic_returns=critic_returns,
        )
        self._frozen_rollout_collected = True
        self._rollout_evaluator_delta = evaluator_delta
        return rollout

    def _tensor_batch(self, rollout: RolloutBatch, credit: CreditEstimate):
        transitions = tuple(rollout.transitions)
        return {
            "observations": torch.as_tensor(
                np.stack([item.observation for item in transitions]),
                dtype=torch.float32,
                device=self.device,
            ),
            "dianhydride_masks": torch.as_tensor(
                np.stack([item.dianhydride_mask for item in transitions]),
                dtype=torch.bool,
                device=self.device,
            ),
            "diamine_masks": torch.as_tensor(
                np.stack([item.diamine_mask for item in transitions]),
                dtype=torch.bool,
                device=self.device,
            ),
            "dianhydride_actions": torch.as_tensor(
                [item.dianhydride_action for item in transitions],
                dtype=torch.long,
                device=self.device,
            ),
            "diamine_actions": torch.as_tensor(
                [item.diamine_action for item in transitions],
                dtype=torch.long,
                device=self.device,
            ),
            "old_log_probabilities": torch.as_tensor(
                [item.old_log_probability for item in transitions],
                dtype=torch.float32,
                device=self.device,
            ),
            "old_values": torch.as_tensor(
                [item.old_value for item in transitions],
                dtype=torch.float32,
                device=self.device,
            ),
            "returns": torch.as_tensor(
                np.asarray(rollout.critic_returns, dtype=np.float32).copy(),
                dtype=torch.float32,
                device=self.device,
            ),
            "advantages": torch.as_tensor(
                np.asarray(credit.actor_advantages, dtype=np.float32).copy(),
                dtype=torch.float32,
                device=self.device,
            ),
        }

    def _authorize_credit(self, request: CreditRequest, estimate: CreditEstimate) -> None:
        validate_credit_estimate(request, estimate)
        self._approved_credit[request.rollout.batch_id] = estimate.actor_advantages_sha256

    def update(self, rollout: RolloutBatch, credit: CreditEstimate) -> PPOUpdateReceipt:
        self._assert_frozen(rollout.frozen_policy)
        if self.policy_state_sha256 != rollout.frozen_policy.state_sha256:
            raise ContractViolation("Live policy changed after behavior-policy freeze.")
        if rollout.batch_id in self._consumed_batches:
            raise ContractViolation("A rollout batch cannot be updated twice.")
        if self._approved_credit.get(rollout.batch_id) != credit.actor_advantages_sha256:
            raise ContractViolation("Credit was not validated and staged for this batch.")
        tensors = self._tensor_batch(rollout, credit)
        advantages = tensors["advantages"]
        if self.config.normalize_actor_advantages:
            standard_deviation = advantages.std(unbiased=False)
            if float(standard_deviation.item()) > 1e-8:
                advantages = (advantages - advantages.mean()) / (
                    standard_deviation + 1e-8
                )
            else:
                advantages = advantages - advantages.mean()
        total = int(advantages.shape[0])
        update_count = 0
        actor_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        approximate_kl = 0.0
        clip_fraction_total = 0.0
        stopped_for_kl = False
        self.model.train()
        for _ in range(int(self.config.update_epochs)):
            permutation = self._rng.permutation(total)
            for start in range(0, total, int(self.config.minibatch_size)):
                indices = torch.as_tensor(
                    permutation[start : start + int(self.config.minibatch_size)],
                    dtype=torch.long,
                    device=self.device,
                )
                d_logits, a_logits, values = self.model(tensors["observations"][indices])
                d_logits = self._masked_logits(
                    d_logits, tensors["dianhydride_masks"][indices]
                )
                a_logits = self._masked_logits(
                    a_logits, tensors["diamine_masks"][indices]
                )
                d_distribution = torch.distributions.Categorical(logits=d_logits)
                a_distribution = torch.distributions.Categorical(logits=a_logits)
                new_log_probability = d_distribution.log_prob(
                    tensors["dianhydride_actions"][indices]
                ) + a_distribution.log_prob(tensors["diamine_actions"][indices])
                old_log_probability = tensors["old_log_probabilities"][indices]
                log_ratio = new_log_probability - old_log_probability
                ratio = torch.exp(log_ratio)
                selected_advantages = advantages[indices]
                unclipped = ratio * selected_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(self.config.clip_ratio),
                    1.0 + float(self.config.clip_ratio),
                ) * selected_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean()

                old_values = tensors["old_values"][indices]
                selected_returns = tensors["returns"][indices]
                clipped_values = old_values + torch.clamp(
                    values - old_values,
                    -float(self.config.value_clip),
                    float(self.config.value_clip),
                )
                value_loss = 0.5 * torch.maximum(
                    torch.square(values - selected_returns),
                    torch.square(clipped_values - selected_returns),
                ).mean()
                entropy = (d_distribution.entropy() + a_distribution.entropy()).mean()
                loss = (
                    actor_loss
                    + float(self.config.value_loss_coefficient) * value_loss
                    - float(self.config.entropy_coefficient) * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.config.maximum_gradient_norm)
                )
                if not bool(torch.isfinite(gradient_norm).item()):
                    raise ContractViolation("PPO produced a non-finite gradient norm.")
                self.optimizer.step()
                with torch.no_grad():
                    approximate_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                    clip_fraction = float(
                        (torch.abs(ratio - 1.0) > float(self.config.clip_ratio))
                        .float()
                        .mean()
                        .item()
                    )
                update_count += 1
                actor_loss_total += float(actor_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy.item())
                clip_fraction_total += clip_fraction
                if (
                    self.config.target_kl is not None
                    and approximate_kl > float(self.config.target_kl)
                ):
                    stopped_for_kl = True
                    break
            if stopped_for_kl:
                break
        if update_count == 0:
            raise ContractViolation("PPO completed no optimizer step.")

        policy_version_before = self.policy_version
        self.policy_version += 1
        receipt = PPOUpdateReceipt(
            engine_contract_id=self.engine_contract_id,
            source_batch_id=rollout.batch_id,
            policy_version_before=policy_version_before,
            policy_version_after=self.policy_version,
            actor_advantages_sha256=credit.actor_advantages_sha256,
            critic_returns_sha256=rollout.critic_returns_sha256,
            optimizer_step_completed=True,
        )
        validate_update_receipt(rollout, credit, receipt)
        self._last_update_metrics = {
            "optimizer_steps": float(update_count),
            "actor_loss": actor_loss_total / update_count,
            "value_loss": value_loss_total / update_count,
            "entropy": entropy_total / update_count,
            "approximate_kl": approximate_kl,
            "clip_fraction": clip_fraction_total / update_count,
            "stopped_for_target_kl": float(stopped_for_kl),
        }
        self._consumed_batches.add(rollout.batch_id)
        self._approved_credit.pop(rollout.batch_id, None)
        self._frozen_model = None
        self._frozen_handle = None
        self._frozen_rollout_collected = False
        return receipt

    def run_iteration(
        self,
        *,
        query_requested_calls: int = 0,
        query_seed: Optional[int] = None,
    ) -> PPOIterationResult:
        frozen = self.freeze_policy()
        rollout = self.collect_rollout(frozen)
        credit_model_version = int(getattr(self.credit_estimator, "model_version", 0))
        if query_seed is None:
            query_seed = int(self._rng.integers(0, np.iinfo(np.int64).max))
        query_reservation = self._budget.reserve(int(query_requested_calls))
        ledger_before = dict(self.environment.oracle_ledger())
        request = CreditRequest(
            method=self.run_contract.method,
            rollout=rollout,
            credit_model_version=credit_model_version,
            reserved_query_requested_calls=int(query_requested_calls),
            query_seed=int(query_seed),
        )
        try:
            credit = self.credit_estimator.estimate(request)
            ledger_after = dict(self.environment.oracle_ledger())
            observed_delta = evaluator_ledger_delta(ledger_before, ledger_after)
            if observed_delta != credit.evaluator_delta:
                raise ContractViolation(
                    "Credit-reported evaluator delta differs from the global ledger."
                )
            self._budget.reconcile(query_reservation, observed_delta.requested_calls)
        except Exception:
            try:
                self._budget.release(query_reservation)
            except ContractViolation:
                pass
            raise
        self._authorize_credit(request, credit)
        receipt = self.update(rollout, credit)
        committed = None
        if credit.pending_labels is not None:
            validate_pending_label_commit(credit.pending_labels, receipt)
            committed = self.credit_estimator.commit_after_update(
                credit.pending_labels, receipt
            )
        return PPOIterationResult(
            rollout=rollout,
            credit=credit,
            receipt=receipt,
            update_metrics=self.last_update_metrics,
            rollout_evaluator_delta=self._rollout_evaluator_delta,
            pending_label_commit=committed,
        )

    def _checkpoint_ready(self) -> None:
        if self._frozen_handle is not None or self._approved_credit:
            raise ContractViolation("Checkpoint only at an atomic iteration boundary.")

    def state_dict(self) -> Mapping[str, Any]:
        self._checkpoint_ready()
        cuda_rng = []
        if torch.cuda.is_available():
            cuda_rng = [value.cpu() for value in torch.cuda.get_rng_state_all()]
        return {
            "schema_version": PPO_ENGINE_STATE_SCHEMA_VERSION,
            "engine_contract_id": self.engine_contract_id,
            "credit_estimator_contract_id": self.credit_estimator.contract_id,
            "run_contract": asdict(self.run_contract),
            "ppo_config": self.config.to_dict(),
            "ppo_config_sha256": self.config.sha256,
            "observation_dimension": self.observation_dimension,
            "number_of_dianhydride_actions": self.number_of_dianhydride_actions,
            "number_of_diamine_actions": self.number_of_diamine_actions,
            "policy_version": self.policy_version,
            "policy_state": copy.deepcopy(self.model.state_dict()),
            "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
            "credit_estimator_state": copy.deepcopy(
                self.credit_estimator.state_dict()
            ),
            "environment_state": copy.deepcopy(self.environment.snapshot()),
            "budget_manager_state": copy.deepcopy(self._budget.state_dict()),
            "numpy_rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "python_rng_state": random.getstate(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": cuda_rng,
            "rollout_sequence": self._rollout_sequence,
            "transition_sequence": self._transition_sequence,
            "episode_sequence": self._episode_sequence,
            "episode_timestep": self._episode_timestep,
            "current_observation": copy.deepcopy(self._current_observation),
            "consumed_batch_ids": sorted(self._consumed_batches),
            "last_update_metrics": dict(self._last_update_metrics),
            "pending_labels": [],
            "committed_credit_replay": [],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._checkpoint_ready()
        payload = dict(state)
        expected = {
            "schema_version": PPO_ENGINE_STATE_SCHEMA_VERSION,
            "engine_contract_id": self.engine_contract_id,
            "credit_estimator_contract_id": self.credit_estimator.contract_id,
            "run_contract": asdict(self.run_contract),
            "ppo_config": self.config.to_dict(),
            "ppo_config_sha256": self.config.sha256,
            "observation_dimension": self.observation_dimension,
            "number_of_dianhydride_actions": self.number_of_dianhydride_actions,
            "number_of_diamine_actions": self.number_of_diamine_actions,
        }
        mismatches = {
            name: (expected_value, payload.get(name))
            for name, expected_value in expected.items()
            if payload.get(name) != expected_value
        }
        if mismatches:
            raise ContractViolation("PPO checkpoint contract mismatch: %s" % mismatches)
        self.model.load_state_dict(payload["policy_state"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.credit_estimator.load_state_dict(payload["credit_estimator_state"])
        restored_observation = self.environment.restore(payload["environment_state"])
        self._budget.load_state_dict(payload["budget_manager_state"])
        self.policy_version = int(payload["policy_version"])
        self._rng.bit_generator.state = copy.deepcopy(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        cuda_states = payload.get("torch_cuda_rng_states", [])
        if cuda_states:
            if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
                raise ContractViolation("PPO checkpoint CUDA RNG topology mismatch.")
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
        self._rollout_sequence = int(payload["rollout_sequence"])
        self._transition_sequence = int(payload["transition_sequence"])
        self._episode_sequence = int(payload["episode_sequence"])
        self._episode_timestep = int(payload["episode_timestep"])
        self._current_observation = copy.deepcopy(payload["current_observation"])
        if self._current_observation is not None:
            self._split_observation(self._current_observation)
        elif restored_observation is not None:
            # A terminal boundary intentionally resets on the next collection.
            pass
        self._consumed_batches = set(payload.get("consumed_batch_ids", ()))
        self._last_update_metrics = dict(payload.get("last_update_metrics", {}))
        if payload.get("pending_labels") != []:
            raise ContractViolation("PPO GAE checkpoint cannot contain pending labels.")
        if payload.get("committed_credit_replay") != []:
            raise ContractViolation("PPO GAE checkpoint cannot contain credit replay.")
        self._frozen_model = None
        self._frozen_handle = None
        self._frozen_rollout_collected = False
        self._approved_credit = {}

    def save_checkpoint(self, path: Path) -> str:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(target))
        return str(target)

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(str(Path(path).resolve()), map_location=self.device)
        self.load_state_dict(payload)


def rollout_digest(rollout: RolloutBatch) -> str:
    """Deterministic digest used by checkpoint/resume acceptance tests."""

    payload = []
    for item in rollout.transitions:
        payload.append(
            {
                "transition_id": item.transition_id,
                "episode_id": item.episode_id,
                "timestep": item.timestep,
                "observation": array_sha256(item.observation),
                "dianhydride_mask": array_sha256(item.dianhydride_mask),
                "diamine_mask": array_sha256(item.diamine_mask),
                "action": [item.dianhydride_action, item.diamine_action],
                "old_log_probability": item.old_log_probability,
                "old_value": item.old_value,
                "reward": item.reward,
                "next_observation": array_sha256(item.next_observation),
                "terminated": item.terminated,
                "truncated": item.truncated,
            }
        )
    return canonical_sha256(payload)
