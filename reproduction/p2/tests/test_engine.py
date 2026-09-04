import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reproduction.p2.contracts import (  # noqa: E402
    CREDIT_ESTIMATOR_CONTRACT_ID,
    MCC_PPO,
    METHOD_EVALUATOR_SOURCES,
    METHOD_ON_POLICY_SOURCE,
    POLICY_CC,
    PPO,
    PPO_ENGINE_CONTRACT_ID,
    CreditEstimate,
    EvaluatorLedgerDelta,
    MethodRunContract,
)
from reproduction.p2.engine import (  # noqa: E402
    PPOEngine,
    PPOEngineConfig,
    rollout_digest,
)
from reproduction.p2.gae import GAECreditEstimator  # noqa: E402


class _FakeState:
    def __init__(self, episode_seed, step, done, history):
        self.episode_seed = int(episode_seed)
        self.step = int(step)
        self.done = bool(done)
        self.history = tuple(tuple(item) for item in history)

    def to_dict(self):
        return {
            "episode_seed": self.episode_seed,
            "step": self.step,
            "done": self.done,
            "history": [list(item) for item in self.history],
        }


class _FakeMaskedEnvironment:
    """Deterministic Gymnasium-shaped environment with a restorable ledger."""

    def __init__(self, source, maximum_requested_calls=128):
        self.source = source
        self.core = SimpleNamespace(environment_id="fake-stage0-env")
        self.maximum_requested_calls = int(maximum_requested_calls)
        self.requested_calls = 0
        self.unique_calls = 0
        self.backend_calls = 0
        self.cache_hits = 0
        self.requested_by_source = {}
        self.seen = set()
        self._lock = threading.Lock()
        self._state = _FakeState(0, 0, False, ())

    @property
    def current_state(self):
        return self._state

    def _observation(self):
        history_sum = sum(sum(item) for item in self._state.history)
        vector = np.asarray(
            [
                self._state.step / 3.0,
                (self._state.episode_seed % 101) / 101.0,
                history_sum / 10.0,
                float(self._state.done),
            ],
            dtype=np.float32,
        )
        if self._state.done:
            d_mask = np.asarray([False, False, True])
            a_mask = np.asarray([False, True])
        elif self._state.step % 2 == 0:
            d_mask = np.asarray([True, False, True])
            a_mask = np.asarray([True, True])
        else:
            d_mask = np.asarray([False, True, True])
            a_mask = np.asarray([True, False])
        return {
            "observations": vector,
            "action_mask": np.concatenate([d_mask, a_mask]).astype(np.int8),
        }

    def reset(self, *, seed=None, options=None):
        del options
        self._state = _FakeState(int(seed or 0), 0, False, ())
        return self._observation(), {"environment_id": self.core.environment_id}

    def step(self, action):
        if self._state.done:
            raise RuntimeError("reset required")
        d_action, a_action = (int(action[0]), int(action[1]))
        history = self._state.history + ((d_action, a_action),)
        next_step = self._state.step + 1
        terminated = next_step == 3
        self._state = _FakeState(
            self._state.episode_seed, next_step, terminated, history
        )
        reward = 0.0
        info = {"environment_id": self.core.environment_id}
        if terminated:
            reward = 1.0 + 0.1 * sum(sum(item) for item in history)
            key = (self._state.episode_seed, history)
            with self._lock:
                if self.requested_calls >= self.maximum_requested_calls:
                    raise RuntimeError("requested-call budget exceeded")
                self.requested_calls += 1
                self.requested_by_source[self.source] = (
                    self.requested_by_source.get(self.source, 0) + 1
                )
                if key in self.seen:
                    self.cache_hits += 1
                else:
                    self.seen.add(key)
                    self.unique_calls += 1
                    self.backend_calls += 1
            info["terminal_evaluation"] = {"objective": reward}
        return self._observation(), reward, terminated, False, info

    def oracle_ledger(self):
        with self._lock:
            return {
                "requested_calls": self.requested_calls,
                "unique_calls": self.unique_calls,
                "backend_calls": self.backend_calls,
                "cache_hits": self.cache_hits,
                "requested_by_source": dict(self.requested_by_source),
                "remaining_requested_calls": self.maximum_requested_calls
                - self.requested_calls,
            }

    def snapshot(self):
        return {
            "state": self._state.to_dict(),
            "maximum_requested_calls": self.maximum_requested_calls,
            "requested_calls": self.requested_calls,
            "unique_calls": self.unique_calls,
            "backend_calls": self.backend_calls,
            "cache_hits": self.cache_hits,
            "requested_by_source": dict(self.requested_by_source),
            "seen": list(self.seen),
        }

    def restore(self, snapshot):
        if snapshot["maximum_requested_calls"] != self.maximum_requested_calls:
            raise ValueError("budget mismatch")
        state = snapshot["state"]
        self._state = _FakeState(
            state["episode_seed"], state["step"], state["done"], state["history"]
        )
        self.requested_calls = int(snapshot["requested_calls"])
        self.unique_calls = int(snapshot["unique_calls"])
        self.backend_calls = int(snapshot["backend_calls"])
        self.cache_hits = int(snapshot["cache_hits"])
        self.requested_by_source = dict(snapshot["requested_by_source"])
        self.seen = set(tuple((item[0], tuple(tuple(x) for x in item[1]))) for item in snapshot["seen"])
        return self._observation()


class _PassThroughCredit:
    contract_id = CREDIT_ESTIMATOR_CONTRACT_ID
    model_version = 0

    def __init__(self, method):
        self.method = method

    def estimate(self, request):
        return CreditEstimate(
            method=self.method,
            source_batch_id=request.rollout.batch_id,
            source_policy_version=request.rollout.frozen_policy.policy_version,
            actor_advantages=request.rollout.gae_advantages,
            credit_model_version_before=0,
            credit_model_version_after=0,
            evaluator_delta=EvaluatorLedgerDelta(),
        )

    def commit_after_update(self, pending, receipt):
        raise AssertionError("no pending labels expected")

    def state_dict(self):
        return {"method": self.method, "model_version": 0}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("credit checkpoint mismatch")


def _config(seed=19):
    return PPOEngineConfig(
        rollout_steps=9,
        gamma=0.99,
        gae_lambda=0.95,
        update_epochs=2,
        minibatch_size=3,
        clip_ratio=0.2,
        value_clip=0.2,
        value_loss_coefficient=0.5,
        entropy_coefficient=0.01,
        maximum_gradient_norm=0.5,
        target_kl=None,
        learning_rate=3e-4,
        hidden_sizes=(16, 16),
        seed=seed,
        deterministic_torch=True,
        device="cpu",
    )


def _contract(method, config):
    return MethodRunContract(
        method=method,
        environment_id="fake-stage0-env",
        task_contract_id="fake-task",
        budget_contract_id="fake-budget",
        evaluator_version="fake-evaluator",
        objective_contract="fake-objective",
        ppo_engine_contract_id=PPO_ENGINE_CONTRACT_ID,
        ppo_hyperparameters_sha256=config.sha256,
        credit_estimator_contract_id=CREDIT_ESTIMATOR_CONTRACT_ID,
        allowed_evaluator_sources=tuple(METHOD_EVALUATOR_SOURCES[method]),
    )


def _engine(method=PPO, seed=19):
    config = _config(seed)
    environment = _FakeMaskedEnvironment(METHOD_ON_POLICY_SOURCE[method])
    provider = GAECreditEstimator() if method == PPO else _PassThroughCredit(method)
    return PPOEngine(
        environment=environment,
        run_contract=_contract(method, config),
        credit_estimator=provider,
        observation_dimension=4,
        number_of_dianhydride_actions=3,
        number_of_diamine_actions=2,
        config=config,
    )


def test_ppo_iteration_uses_exact_gae_and_environment_critic_returns():
    engine = _engine()
    initial_policy = engine.policy_state_sha256
    result = engine.run_iteration()
    np.testing.assert_array_equal(
        result.credit.actor_advantages, result.rollout.gae_advantages
    )
    assert result.receipt.critic_returns_sha256 == result.rollout.critic_returns_sha256
    assert result.receipt.policy_version_before == 0
    assert result.receipt.policy_version_after == 1
    assert engine.policy_version == 1
    assert engine.policy_state_sha256 != initial_policy
    assert result.credit.evaluator_delta.requested_calls == 0
    assert result.rollout_evaluator_delta.requested_calls == 3
    assert result.rollout_evaluator_delta.requested_by_source == {"ppo/on_policy": 3}
    for transition in result.rollout.transitions:
        assert transition.dianhydride_mask[transition.dianhydride_action]
        assert transition.diamine_mask[transition.diamine_action]


def test_deterministic_action_is_legal_and_does_not_mutate_engine_state():
    engine = _engine(seed=21)
    observation, _info = engine.environment.reset(seed=20260910)
    policy_before = engine.policy_state_sha256
    rng_before = copy.deepcopy(engine._rng.bit_generator.state)
    first = engine.deterministic_action(observation)
    second = engine.deterministic_action(observation)
    _vector, dianhydride_mask, diamine_mask = engine._split_observation(observation)
    assert first == second
    assert dianhydride_mask[first[0]]
    assert diamine_mask[first[1]]
    assert engine.policy_state_sha256 == policy_before
    assert engine._rng.bit_generator.state == rng_before


def test_identical_engine_path_is_method_invariant_when_credit_is_identical():
    results = {}
    for method in (PPO, POLICY_CC, MCC_PPO):
        engine = _engine(method=method, seed=23)
        result = engine.run_iteration()
        results[method] = (
            rollout_digest(result.rollout),
            result.rollout.gae_sha256,
            result.receipt.actor_advantages_sha256,
            result.receipt.critic_returns_sha256,
            engine.policy_state_sha256,
            result.update_metrics,
        )
    assert results[PPO] == results[POLICY_CC] == results[MCC_PPO]


def test_checkpoint_resume_preserves_rollout_optimizer_ledger_and_rng_identity():
    uninterrupted = _engine(seed=29)
    uninterrupted.run_iteration()
    checkpoint = copy.deepcopy(uninterrupted.state_dict())
    expected = uninterrupted.run_iteration()
    expected_policy = uninterrupted.policy_state_sha256
    expected_ledger = uninterrupted.environment.oracle_ledger()

    resumed = _engine(seed=29)
    resumed.load_state_dict(checkpoint)
    observed = resumed.run_iteration()
    assert rollout_digest(observed.rollout) == rollout_digest(expected.rollout)
    assert observed.receipt == expected.receipt
    assert observed.update_metrics == expected.update_metrics
    assert resumed.policy_state_sha256 == expected_policy
    assert resumed.environment.oracle_ledger() == expected_ledger


def test_checkpoint_rejects_contract_drift():
    engine = _engine(seed=31)
    engine.run_iteration()
    checkpoint = copy.deepcopy(engine.state_dict())
    checkpoint["ppo_config_sha256"] = "tampered"
    with pytest.raises(Exception, match="checkpoint contract mismatch"):
        _engine(seed=31).load_state_dict(checkpoint)
