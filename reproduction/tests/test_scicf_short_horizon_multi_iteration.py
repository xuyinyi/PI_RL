from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproduction.p2.contracts import ContractViolation
from reproduction.scicf.llm.api_client import APITransportError, ChatCompletion
from reproduction.scicf.online.contracts import (
    SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID,
)
from reproduction.scicf.online.pipeline import numerically_nonnegative_kl
from reproduction.scicf.online.resilience import (
    DeadlineBoundClient,
    LLMAuxiliaryCircuitBreaker,
    LLMAuxiliaryResilienceConfig,
    LLMWallClockBudget,
    LLMWallClockBudgetConfig,
    LLMWallTimeBudgetExhausted,
)
from reproduction.scicf.online.run_short_horizon_multi_iteration import (
    AUTHORIZATION_OPERATIONS_SHORT_HORIZON,
    load_execution_authorization_short_horizon,
)
from reproduction.scicf.online.short_horizon import (
    chained_history_digest,
    load_short_horizon_checkpoint,
    load_short_horizon_protocol,
    save_short_horizon_checkpoint,
    sha256_path,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "reproduction/scicf/online/configs/"
    / "short_horizon_multi_iteration_v1.json"
)


class _Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class _RetryClient:
    def __init__(self, clock):
        self.clock = clock
        self.timeouts = []

    def complete(self, **kwargs):
        self.timeouts.append(float(kwargs["timeout_seconds"]))
        assert kwargs["transport_retries"] == 0
        if len(self.timeouts) == 1:
            self.clock.advance(30.0)
            raise APITransportError("synthetic retry", retryable=True)
        return ChatCompletion(
            content='{"ranked_intervention_ids": [], "abstain": true}',
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            provider_response_id="mock",
            system_fingerprint="mock",
            response_sha256="a" * 64,
            transport_retries_used=0,
        )


class _FakeEngine:
    def __init__(self, payload=b"fake-engine-state"):
        self.payload = payload
        self.loaded = None

    def save_checkpoint(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payload)

    def load_checkpoint(self, path):
        self.loaded = Path(path).read_bytes()


def test_frozen_short_horizon_protocol_binds_successful_v3_evidence():
    protocol = load_short_horizon_protocol(PROTOCOL, ROOT)
    assert protocol["protocol_id"] == SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID
    assert protocol["horizon"]["maximum_iterations"] == 6
    assert protocol["llm_wall_clock_budget"][
        "maximum_seconds_per_iteration"
    ] == 60.0
    assert protocol["llm_wall_clock_budget"]["maximum_seconds_total"] == 180.0
    assert (
        protocol["authorization_state"][
            "real_multi_iteration_execution_authorized"
        ]
        is False
    )


def test_tiny_negative_kl_is_clamped_but_material_negative_is_fatal():
    assert numerically_nonnegative_kl(-2.9e-7) == 0.0
    assert numerically_nonnegative_kl(3.0e-7) == 3.0e-7
    with pytest.raises(ContractViolation, match="materially negative"):
        numerically_nonnegative_kl(-1.1e-6)


def test_deadline_bounds_retry_and_persists_consumed_time():
    clock = _Clock()
    config = LLMWallClockBudgetConfig()
    budget = LLMWallClockBudget(config, clock=clock)
    lease = budget.begin_iteration(1)
    client = _RetryClient(clock)
    bounded = DeadlineBoundClient(client, lease, sleeper=clock.advance)
    result = bounded.complete(
        messages=[{"role": "user", "content": "mock"}],
        max_tokens=10,
        seed=1,
        timeout_seconds=120.0,
        transport_retries=1,
    )
    receipt = lease.close()
    assert result.total_tokens == 2
    assert client.timeouts[0] == 60.0
    assert 28.9 < client.timeouts[1] < 29.1
    assert receipt["elapsed_seconds"] == 31.0
    assert budget.state_dict()["consumed_seconds"] == 31.0

    restored = LLMWallClockBudget(config, clock=clock)
    restored.load_state_dict(budget.state_dict())
    assert restored.remaining_total_seconds == 149.0
    assert restored.completed_attempt_iterations == [1]


def test_persisted_total_budget_cannot_be_reset_by_new_iteration():
    clock = _Clock()
    config = LLMWallClockBudgetConfig(
        maximum_seconds_per_iteration=60.0,
        maximum_seconds_total=60.0,
        minimum_transport_timeout_seconds=0.01,
    )
    budget = LLMWallClockBudget(config, clock=clock)
    lease = budget.begin_iteration(1)
    clock.advance(59.995)
    lease.close()
    with pytest.raises(LLMWallTimeBudgetExhausted):
        budget.begin_iteration(2)


def test_circuit_and_wall_budget_restore_with_engine_checkpoint(tmp_path):
    clock = _Clock()
    breaker = LLMAuxiliaryCircuitBreaker(LLMAuxiliaryResilienceConfig())
    breaker.record_failure(1)
    breaker.record_failure(2)
    wall = LLMWallClockBudget(LLMWallClockBudgetConfig(), clock=clock)
    lease = wall.begin_iteration(1)
    clock.advance(12.0)
    lease.close()
    record = {"iteration": 2, "status": "ppo_only_degraded"}
    digest = chained_history_digest(None, record)
    saved = save_short_horizon_checkpoint(
        engine=_FakeEngine(),
        breaker=breaker,
        wall_budget=wall,
        engine_checkpoint_path=tmp_path / "engine.pt",
        control_checkpoint_path=tmp_path / "control.json",
        completed_iteration=2,
        protocol_sha256="b" * 64,
        history_digest=digest,
        previous_control_sha256=None,
    )

    restored_engine = _FakeEngine()
    restored_breaker = LLMAuxiliaryCircuitBreaker(
        LLMAuxiliaryResilienceConfig()
    )
    restored_wall = LLMWallClockBudget(
        LLMWallClockBudgetConfig(), clock=clock
    )
    receipt = load_short_horizon_checkpoint(
        engine=restored_engine,
        breaker=restored_breaker,
        wall_budget=restored_wall,
        control_checkpoint_path=tmp_path / "control.json",
        expected_protocol_sha256="b" * 64,
        expected_control_sha256=saved["control_checkpoint_sha256"],
    )
    assert restored_engine.loaded == b"fake-engine-state"
    assert restored_breaker.open_until_iteration == 6
    assert restored_breaker.allow_request(5) is False
    assert restored_breaker.allow_request(6) is True
    assert restored_wall.consumed_seconds == 12.0
    assert receipt["completed_iteration"] == 2
    assert receipt["history_digest"] == digest


def test_control_restore_rejects_invalid_chain_before_loading_engine(tmp_path):
    clock = _Clock()
    breaker = LLMAuxiliaryCircuitBreaker(LLMAuxiliaryResilienceConfig())
    wall = LLMWallClockBudget(LLMWallClockBudgetConfig(), clock=clock)
    saved = save_short_horizon_checkpoint(
        engine=_FakeEngine(),
        breaker=breaker,
        wall_budget=wall,
        engine_checkpoint_path=tmp_path / "engine.pt",
        control_checkpoint_path=tmp_path / "control.json",
        completed_iteration=1,
        protocol_sha256="b" * 64,
        history_digest="c" * 64,
        previous_control_sha256=None,
    )
    payload = json.loads((tmp_path / "control.json").read_text(encoding="utf-8"))
    payload["previous_control_sha256"] = "invalid"
    (tmp_path / "control.json").write_text(json.dumps(payload), encoding="utf-8")
    restored_engine = _FakeEngine()
    with pytest.raises(ContractViolation, match="previous control"):
        load_short_horizon_checkpoint(
            engine=restored_engine,
            breaker=LLMAuxiliaryCircuitBreaker(LLMAuxiliaryResilienceConfig()),
            wall_budget=LLMWallClockBudget(
                LLMWallClockBudgetConfig(), clock=clock
            ),
            control_checkpoint_path=tmp_path / "control.json",
            expected_protocol_sha256="b" * 64,
            expected_control_sha256=sha256_path(tmp_path / "control.json"),
        )
    assert restored_engine.loaded is None
    assert saved["completed_iteration"] == 1


def _authorization(tmp_path):
    output = tmp_path / "output"
    polybert = tmp_path / "polybert"
    evaluator = tmp_path / "evaluator"
    polybert.mkdir()
    evaluator.mkdir()
    payload = {
        "schema_version": 5,
        "authorization_id": "short-horizon-unit-test",
        "protocol_id": SHORT_HORIZON_MULTI_ITERATION_PROTOCOL_ID,
        "protocol_sha256": sha256_path(PROTOCOL),
        "implementation_commit": "test-commit",
        "authorized_output_directory": str(output),
        "authorized_polybert_path": str(polybert),
        "polybert_asset_binding_sha256": "a" * 64,
        "polybert_checkpoint_fingerprint": "b" * 64,
        "authorized_evaluator_asset_path": str(evaluator),
        "evaluator_asset_binding_sha256": "c" * 64,
        "evaluator_asset_fingerprint": "d" * 64,
        "authorized_start_iteration": 1,
        "authorized_end_iteration": 6,
        "authorized_resume_control_checkpoint": None,
        "resume_control_checkpoint_sha256": None,
        "maximum_slurm_runs": 1,
        "authorized_operations": dict(
            AUTHORIZATION_OPERATIONS_SHORT_HORIZON
        ),
    }
    return payload, output, polybert, evaluator


def _load(path, payload, output, polybert, evaluator):
    return load_execution_authorization_short_horizon(
        path,
        protocol_sha256=payload["protocol_sha256"],
        implementation_commit=payload["implementation_commit"],
        output_dir=output,
        polybert_path=polybert,
        polybert_asset_binding_sha256=payload[
            "polybert_asset_binding_sha256"
        ],
        polybert_checkpoint_fingerprint=payload[
            "polybert_checkpoint_fingerprint"
        ],
        evaluator_asset_path=evaluator,
        evaluator_asset_binding_sha256=payload[
            "evaluator_asset_binding_sha256"
        ],
        evaluator_asset_fingerprint=payload["evaluator_asset_fingerprint"],
        resume_control_checkpoint=None,
        resume_control_checkpoint_sha256=None,
    )


def test_schema5_authorization_is_exact_and_does_not_allow_rerun(tmp_path):
    payload, output, polybert, evaluator = _authorization(tmp_path)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = _load(path, payload, output, polybert, evaluator)
    assert loaded["authorized_end_iteration"] == 6

    payload["authorized_operations"]["automatic_rerun_authorized"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="operation scope mismatch"):
        _load(path, payload, output, polybert, evaluator)


def test_runner_orders_every_primary_checkpoint_before_llm_and_saves_control():
    source = (
        ROOT
        / "reproduction/scicf/online/run_short_horizon_multi_iteration.py"
    ).read_text(encoding="utf-8")
    ppo = source.index("ppo_result = engine.run_iteration(")
    primary = source.index("engine.save_checkpoint(primary_checkpoint)")
    credentials = source.index("settings = APISettings.from_private_file(")
    bounded_client = source.index("client=DeadlineBoundClient(")
    control = source.index("checkpoint = save_short_horizon_checkpoint(")
    assert ppo < primary < credentials < bounded_client < control
    assert "load_short_horizon_checkpoint(" in source
    assert "automatic_rerun_authorized\": False" in source
