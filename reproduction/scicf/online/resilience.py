"""Fail-open availability boundary for the optional LLM auxiliary stage."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from reproduction.p2.contracts import ContractViolation
from reproduction.scicf.llm.api_client import APITransportError


RECOVERABLE_LLM_FAILURE_CODES = frozenset(
    {
        "provider_timeout",
        "provider_rate_limited",
        "provider_unavailable",
        "transport_exhausted",
        "schema_exhausted",
    }
)
LLM_WALL_TIME_EXHAUSTED_CODE = "llm_wall_time_exhausted"
RECOVERABLE_LLM_FAILURE_CODES_V2 = frozenset(
    set(RECOVERABLE_LLM_FAILURE_CODES) | {LLM_WALL_TIME_EXHAUSTED_CODE}
)


@dataclass(frozen=True)
class LLMAuxiliaryResilienceConfig:
    maximum_consecutive_llm_failures: int = 2
    circuit_cooldown_iterations: int = 3
    primary_ppo_checkpoint_required: bool = True
    abstention_is_recoverable: bool = True
    no_pair_mass_is_recoverable: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_consecutive_llm_failures",
            "circuit_cooldown_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("%s must be a positive integer" % name)
        for name in (
            "primary_ppo_checkpoint_required",
            "abstention_is_recoverable",
            "no_pair_mass_is_recoverable",
        ):
            if getattr(self, name) is not True:
                raise ValueError("resilient auxiliary policy must enable %s" % name)


class RecoverableLLMError(RuntimeError):
    """Expected provider/schema failure that may degrade one PPO iteration."""

    def __init__(self, code: str, message: str = "") -> None:
        if code not in RECOVERABLE_LLM_FAILURE_CODES_V2:
            raise ValueError("unsupported recoverable LLM failure code: %s" % code)
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class LLMWallClockBudgetConfig:
    maximum_seconds_per_iteration: float = 60.0
    maximum_seconds_total: float = 180.0
    minimum_transport_timeout_seconds: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "maximum_seconds_per_iteration",
            "maximum_seconds_total",
            "minimum_transport_timeout_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("%s must be finite and positive" % name)
        if self.maximum_seconds_total < self.maximum_seconds_per_iteration:
            raise ValueError("total LLM wall time cannot be below one iteration cap")
        if (
            self.minimum_transport_timeout_seconds
            >= self.maximum_seconds_per_iteration
        ):
            raise ValueError("minimum transport timeout exceeds iteration cap")


class LLMWallTimeBudgetExhausted(RuntimeError):
    """Raised before a provider attempt when the persisted budget is empty."""


class LLMIterationDeadline:
    def __init__(
        self,
        *,
        owner: "LLMWallClockBudget",
        iteration: int,
        started: float,
    ) -> None:
        self.owner = owner
        self.iteration = int(iteration)
        self.started = float(started)
        self.closed = False
        self.exhausted = False
        self.transmission_attempts = 0

    def elapsed_seconds(self) -> float:
        elapsed = float(self.owner.clock()) - self.started
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ContractViolation("LLM wall clock moved backwards or became invalid")
        return elapsed

    def remaining_seconds(self) -> float:
        elapsed = self.elapsed_seconds()
        remaining = min(
            float(self.owner.config.maximum_seconds_per_iteration) - elapsed,
            float(self.owner.config.maximum_seconds_total)
            - float(self.owner.consumed_seconds)
            - elapsed,
        )
        if remaining <= float(
            self.owner.config.minimum_transport_timeout_seconds
        ):
            self.exhausted = True
            return 0.0
        return float(remaining)

    def bounded_timeout(self, requested: float) -> float:
        requested = float(requested)
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("requested LLM timeout must be finite and positive")
        remaining = self.remaining_seconds()
        if remaining <= 0.0:
            raise LLMWallTimeBudgetExhausted(
                "bounded LLM wall-time budget is exhausted"
            )
        self.transmission_attempts += 1
        return min(requested, remaining)

    def close(self) -> Mapping[str, Any]:
        if self.closed:
            raise ContractViolation("LLM iteration deadline was closed twice")
        elapsed = self.elapsed_seconds()
        if (
            elapsed >= float(self.owner.config.maximum_seconds_per_iteration)
            or float(self.owner.consumed_seconds) + elapsed
            >= float(self.owner.config.maximum_seconds_total)
        ):
            self.exhausted = True
        self.closed = True
        self.owner._close_iteration(self, elapsed)
        return {
            "iteration": self.iteration,
            "elapsed_seconds": elapsed,
            "transmission_attempts": int(self.transmission_attempts),
            "budget_exhausted": bool(self.exhausted),
            "remaining_total_seconds": self.owner.remaining_total_seconds,
            "config": asdict(self.owner.config),
        }


class LLMWallClockBudget:
    """Persisted aggregate and per-iteration LLM wall-clock budget."""

    def __init__(
        self,
        config: LLMWallClockBudgetConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.clock = clock
        self.consumed_seconds = 0.0
        self.completed_attempt_iterations = []
        self._active: Optional[LLMIterationDeadline] = None

    @property
    def remaining_total_seconds(self) -> float:
        return max(
            0.0,
            float(self.config.maximum_seconds_total) - self.consumed_seconds,
        )

    def begin_iteration(self, iteration: int) -> LLMIterationDeadline:
        if self._active is not None:
            raise ContractViolation("LLM wall-time budget already has an active lease")
        if isinstance(iteration, bool) or int(iteration) < 1:
            raise ValueError("iteration must be a positive integer")
        if self.remaining_total_seconds <= float(
            self.config.minimum_transport_timeout_seconds
        ):
            raise LLMWallTimeBudgetExhausted(
                "persisted total LLM wall-time budget is exhausted"
            )
        lease = LLMIterationDeadline(
            owner=self,
            iteration=int(iteration),
            started=float(self.clock()),
        )
        self._active = lease
        return lease

    def _close_iteration(
        self, lease: LLMIterationDeadline, elapsed_seconds: float
    ) -> None:
        if self._active is not lease:
            raise ContractViolation("closing an unknown LLM wall-time lease")
        self.consumed_seconds += float(elapsed_seconds)
        self.completed_attempt_iterations.append(int(lease.iteration))
        self._active = None

    def state_dict(self) -> Mapping[str, Any]:
        if self._active is not None:
            raise ContractViolation("cannot checkpoint an active LLM wall-time lease")
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "consumed_seconds": float(self.consumed_seconds),
            "remaining_total_seconds": self.remaining_total_seconds,
            "completed_attempt_iterations": list(
                self.completed_attempt_iterations
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._active is not None:
            raise ContractViolation("cannot restore an active LLM wall-time lease")
        required = {
            "schema_version",
            "config",
            "consumed_seconds",
            "remaining_total_seconds",
            "completed_attempt_iterations",
        }
        if set(state) != required:
            raise ContractViolation("invalid LLM wall-time checkpoint")
        if state["schema_version"] != 1 or state["config"] != asdict(self.config):
            raise ContractViolation("LLM wall-time checkpoint identity mismatch")
        consumed = float(state["consumed_seconds"])
        if not math.isfinite(consumed) or consumed < 0.0:
            raise ContractViolation("invalid consumed LLM wall time")
        expected_remaining = max(
            0.0, float(self.config.maximum_seconds_total) - consumed
        )
        if not math.isclose(
            float(state["remaining_total_seconds"]),
            expected_remaining,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ContractViolation("LLM wall-time remaining value mismatch")
        iterations = list(state["completed_attempt_iterations"])
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in iterations
        ):
            raise ContractViolation("invalid LLM attempted-iteration history")
        self.consumed_seconds = consumed
        self.completed_attempt_iterations = iterations


class DeadlineBoundClient:
    """Apply one shared iteration deadline across pools, repairs, and retries."""

    def __init__(
        self,
        client,
        deadline: LLMIterationDeadline,
        *,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.deadline = deadline
        self.sleeper = sleeper

    def complete(
        self,
        *,
        messages,
        max_tokens,
        seed,
        timeout_seconds,
        transport_retries,
    ):
        for attempt in range(int(transport_retries) + 1):
            try:
                bounded = self.deadline.bounded_timeout(timeout_seconds)
                return self.client.complete(
                    messages=messages,
                    max_tokens=max_tokens,
                    seed=seed,
                    timeout_seconds=bounded,
                    transport_retries=0,
                )
            except LLMWallTimeBudgetExhausted as error:
                raise APITransportError(str(error), retryable=False)
            except APITransportError as error:
                if not error.retryable or attempt >= int(transport_retries):
                    raise
                remaining = self.deadline.remaining_seconds()
                if remaining <= 0.0:
                    raise APITransportError(
                        "bounded LLM wall-time budget is exhausted",
                        retryable=False,
                    )
                self.sleeper(min(float(2**attempt), remaining))
        raise AssertionError("unreachable bounded-client retry state")


class LLMAuxiliaryCircuitBreaker:
    """Bound repeated provider failures while keeping PPO iterations alive."""

    def __init__(self, config: LLMAuxiliaryResilienceConfig) -> None:
        self.config = config
        self.consecutive_llm_failures = 0
        self.open_until_iteration = 0

    def allow_request(self, iteration: int) -> bool:
        if int(iteration) < 1:
            raise ValueError("iteration must be positive")
        return int(iteration) >= int(self.open_until_iteration)

    def record_success(self) -> None:
        self.consecutive_llm_failures = 0
        self.open_until_iteration = 0

    def record_failure(self, iteration: int) -> None:
        self.consecutive_llm_failures += 1
        if self.consecutive_llm_failures >= int(
            self.config.maximum_consecutive_llm_failures
        ):
            self.open_until_iteration = int(iteration) + int(
                self.config.circuit_cooldown_iterations
            ) + 1

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "consecutive_llm_failures": int(
                self.consecutive_llm_failures
            ),
            "open_until_iteration": int(self.open_until_iteration),
            "config": asdict(self.config),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {
            "schema_version",
            "consecutive_llm_failures",
            "open_until_iteration",
            "config",
        }:
            raise ContractViolation("invalid LLM circuit-breaker checkpoint")
        if state["schema_version"] != 1 or state["config"] != asdict(self.config):
            raise ContractViolation("LLM circuit-breaker identity mismatch")
        failures = state["consecutive_llm_failures"]
        open_until = state["open_until_iteration"]
        if (
            isinstance(failures, bool)
            or not isinstance(failures, int)
            or isinstance(open_until, bool)
            or not isinstance(open_until, int)
        ):
            raise ContractViolation("invalid LLM circuit-breaker state type")
        if failures < 0 or open_until < 0:
            raise ContractViolation("invalid LLM circuit-breaker state")
        self.consecutive_llm_failures = failures
        self.open_until_iteration = open_until


def _degraded_receipt(
    *,
    iteration: int,
    status: str,
    reason: str,
    breaker: LLMAuxiliaryCircuitBreaker,
) -> Mapping[str, Any]:
    return {
        "iteration": int(iteration),
        "primary_ppo_status": "committed_before_llm",
        "auxiliary_status": status,
        "reason": reason,
        "continue_training": True,
        "method_observed": "ppo_only_degraded",
        "scicf_applied": False,
        "silent_fallback_used": False,
        "circuit_breaker": breaker.state_dict(),
    }


def run_optional_llm_acquisition(
    *,
    iteration: int,
    primary_ppo_checkpoint_path: Path,
    primary_ppo_checkpoint_sha256: str,
    breaker: LLMAuxiliaryCircuitBreaker,
    acquire: Callable[[], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Run one bounded acquisition without allowing provider failure to kill PPO."""

    checkpoint_path = Path(primary_ppo_checkpoint_path).resolve()
    if breaker.config.primary_ppo_checkpoint_required:
        if not checkpoint_path.is_file():
            raise ContractViolation("primary PPO checkpoint must exist before LLM")
        if (
            not isinstance(primary_ppo_checkpoint_sha256, str)
            or len(primary_ppo_checkpoint_sha256) != 64
        ):
            raise ContractViolation("primary PPO checkpoint hash is invalid")
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != primary_ppo_checkpoint_sha256:
            raise ContractViolation("primary PPO checkpoint hash mismatch")
    if not breaker.allow_request(iteration):
        return _degraded_receipt(
            iteration=iteration,
            status="skipped_circuit_open",
            reason="bounded_provider_cooldown",
            breaker=breaker,
        )
    try:
        outcome = dict(acquire())
    except RecoverableLLMError as error:
        breaker.record_failure(iteration)
        return _degraded_receipt(
            iteration=iteration,
            status="skipped_recoverable_llm_failure",
            reason=error.code,
            breaker=breaker,
        )
    if set(outcome) != {"status", "payload"}:
        raise ContractViolation("LLM acquisition outcome schema mismatch")
    if outcome["status"] not in {"validated", "abstained"}:
        raise ContractViolation("unvalidated LLM outcome cannot continue")
    breaker.record_success()
    if outcome["status"] == "abstained":
        return _degraded_receipt(
            iteration=iteration,
            status="skipped_llm_abstained",
            reason="valid_schema_abstention",
            breaker=breaker,
        )
    return {
        "iteration": int(iteration),
        "primary_ppo_status": "committed_before_llm",
        "auxiliary_status": "ready_for_oracle_verification",
        "continue_training": True,
        "method_observed": "scicf_auxiliary_pending",
        "scicf_applied": False,
        "silent_fallback_used": False,
        "acquisition_payload": outcome["payload"],
        "circuit_breaker": breaker.state_dict(),
    }


def finalize_optional_auxiliary(
    acquisition_receipt: Mapping[str, Any],
    refinement_receipt: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Normalize applied/skipped auxiliary state without hiding degradation."""

    receipt: Dict[str, Any] = dict(acquisition_receipt)
    if receipt.get("auxiliary_status") != "ready_for_oracle_verification":
        if refinement_receipt is not None:
            raise ContractViolation("skipped acquisition cannot refine")
        return receipt
    if refinement_receipt is None:
        raise ContractViolation("validated acquisition requires refinement receipt")
    refinement = dict(refinement_receipt)
    if refinement.get("continue_primary_training") is not True:
        raise ContractViolation("auxiliary receipt cannot stop committed PPO")
    receipt["refinement"] = refinement
    if refinement.get("status") == "applied":
        receipt.update(
            {
                "auxiliary_status": "applied",
                "method_observed": "ppo_plus_scicf_soft_pair",
                "scicf_applied": True,
            }
        )
    else:
        receipt.update(
            {
                "auxiliary_status": "skipped_after_verification",
                "reason": str(refinement.get("status")),
                "method_observed": "ppo_only_degraded",
                "scicf_applied": False,
            }
        )
    receipt["continue_training"] = True
    receipt["silent_fallback_used"] = False
    return receipt
