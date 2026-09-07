"""Fail-open availability boundary for the optional LLM auxiliary stage."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from reproduction.p2.contracts import ContractViolation


RECOVERABLE_LLM_FAILURE_CODES = frozenset(
    {
        "provider_timeout",
        "provider_rate_limited",
        "provider_unavailable",
        "transport_exhausted",
        "schema_exhausted",
    }
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
        if code not in RECOVERABLE_LLM_FAILURE_CODES:
            raise ValueError("unsupported recoverable LLM failure code: %s" % code)
        super().__init__(message or code)
        self.code = code


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
        failures = int(state["consecutive_llm_failures"])
        open_until = int(state["open_until_iteration"])
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
