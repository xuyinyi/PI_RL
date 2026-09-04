"""Requested-call reservation and exact evaluator-ledger differencing."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from reproduction.p2.contracts import ContractViolation, EvaluatorLedgerDelta


@dataclass(frozen=True)
class BudgetReservation:
    token_id: int
    requested_calls: int


class RequestedCallBudgetManager:
    """Thread-safe pre-reservation over one global evaluator ledger."""

    state_schema_version = 1

    def __init__(self, ledger_reader: Callable[[], Mapping[str, Any]]) -> None:
        self._ledger_reader = ledger_reader
        self._lock = threading.RLock()
        self._next_token_id = 0
        self._reservations: Dict[int, int] = {}

    def reserve(self, requested_calls: int) -> BudgetReservation:
        if isinstance(requested_calls, bool) or not isinstance(requested_calls, int):
            raise TypeError("requested_calls reservation must be an integer.")
        if requested_calls < 0:
            raise ValueError("requested_calls reservation cannot be negative.")
        with self._lock:
            ledger = dict(self._ledger_reader())
            remaining = ledger.get("remaining_requested_calls")
            outstanding = sum(self._reservations.values())
            if remaining is not None and requested_calls > int(remaining) - outstanding:
                raise ContractViolation(
                    "Requested-call capacity is unavailable before execution."
                )
            token = BudgetReservation(self._next_token_id, requested_calls)
            self._next_token_id += 1
            self._reservations[token.token_id] = requested_calls
            return token

    def reconcile(self, token: BudgetReservation, consumed_calls: int) -> None:
        if isinstance(consumed_calls, bool) or not isinstance(consumed_calls, int):
            raise TypeError("consumed_calls must be an integer.")
        with self._lock:
            reserved = self._reservations.pop(token.token_id, None)
            if reserved is None or reserved != token.requested_calls:
                raise ContractViolation("Unknown or already released budget reservation.")
            if consumed_calls < 0 or consumed_calls > reserved:
                raise ContractViolation("Execution exceeded its pre-reserved calls.")

    def release(self, token: BudgetReservation) -> None:
        with self._lock:
            if self._reservations.pop(token.token_id, None) is None:
                raise ContractViolation("Unknown or already released budget reservation.")

    def state_dict(self) -> Mapping[str, Any]:
        with self._lock:
            if self._reservations:
                raise ContractViolation(
                    "Checkpointing with outstanding budget reservations is forbidden."
                )
            return {
                "schema_version": self.state_schema_version,
                "next_token_id": int(self._next_token_id),
                "outstanding_requested_calls": 0,
            }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        if payload.get("schema_version") != self.state_schema_version:
            raise ContractViolation("Budget-manager checkpoint schema mismatch.")
        if payload.get("outstanding_requested_calls") != 0:
            raise ContractViolation("Cannot restore outstanding budget reservations.")
        next_token_id = payload.get("next_token_id")
        if isinstance(next_token_id, bool) or not isinstance(next_token_id, int):
            raise ContractViolation("Budget-manager token counter is invalid.")
        with self._lock:
            self._next_token_id = int(next_token_id)
            self._reservations = {}


def evaluator_ledger_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> EvaluatorLedgerDelta:
    requested_by_source = {}
    before_sources = dict(before.get("requested_by_source", {}))
    after_sources = dict(after.get("requested_by_source", {}))
    for source in sorted(set(before_sources).union(after_sources)):
        delta = int(after_sources.get(source, 0)) - int(before_sources.get(source, 0))
        if delta:
            requested_by_source[source] = delta
    counters = {}
    for name in ("requested_calls", "unique_calls", "backend_calls", "cache_hits"):
        counters[name] = int(after.get(name, 0)) - int(before.get(name, 0))
        if counters[name] < 0:
            raise ContractViolation("Evaluator ledger moved backwards: %s" % name)
    return EvaluatorLedgerDelta(
        requested_calls=counters["requested_calls"],
        unique_calls=counters["unique_calls"],
        backend_calls=counters["backend_calls"],
        cache_hits=counters["cache_hits"],
        requested_by_source=requested_by_source,
    )
