"""Atomic scientific-oracle accounting independent of environment steps."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional


ORACLE_SCOPES = ("factual", "counterfactual", "evaluation")


class OracleBudgetExceeded(RuntimeError):
    """Raised before an oracle call would exceed the declared atomic budget."""


@dataclass(frozen=True)
class OracleCounterSnapshot:
    factual: int
    counterfactual: int
    evaluation: int

    @property
    def total(self) -> int:
        return self.factual + self.counterfactual + self.evaluation

    def as_dict(self) -> Dict[str, int]:
        return {
            "factual": self.factual,
            "counterfactual": self.counterfactual,
            "evaluation": self.evaluation,
            "total": self.total,
        }


class OracleLedger:
    """Fail-closed counter for every atomic scientific object scored."""

    def __init__(self, max_total: Optional[int] = None) -> None:
        if max_total is not None and max_total < 1:
            raise ValueError("max_total must be positive when supplied")
        self._counts = {scope: 0 for scope in ORACLE_SCOPES}
        self.max_total = max_total
        self._scope: ContextVar[Optional[str]] = ContextVar(
            "scicf_oracle_scope", default=None
        )

    @contextmanager
    def scope(self, name: str) -> Iterator[None]:
        if name not in self._counts:
            raise ValueError("unsupported oracle scope: {}".format(name))
        token = self._scope.set(name)
        try:
            yield
        finally:
            self._scope.reset(token)

    def record(self, atomic_objects: int) -> None:
        if isinstance(atomic_objects, bool) or atomic_objects < 1:
            raise ValueError("atomic_objects must be a positive integer")
        scope = self._scope.get()
        if scope is None:
            raise RuntimeError("scientific oracle call has no declared accounting scope")
        if self.max_total is not None:
            current_total = sum(self._counts.values())
            if current_total + int(atomic_objects) > self.max_total:
                raise OracleBudgetExceeded(
                    "atomic oracle budget {} would be exceeded".format(self.max_total)
                )
        self._counts[scope] += int(atomic_objects)

    def snapshot(self) -> OracleCounterSnapshot:
        return OracleCounterSnapshot(**self._counts)

    def delta(self, before: OracleCounterSnapshot, scope: str) -> int:
        if scope not in self._counts:
            raise ValueError("unsupported oracle scope: {}".format(scope))
        return self._counts[scope] - getattr(before, scope)


class CountingOracle:
    """Transparent callable wrapper that counts each item in an oracle batch."""

    def __init__(self, delegate: Any, ledger: OracleLedger) -> None:
        self.delegate = delegate
        self.ledger = ledger

    def __call__(self, scientific_objects: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            count = len(scientific_objects)
        except TypeError:
            count = 1
        if count < 1:
            raise ValueError("scientific oracle received an empty batch")
        self.ledger.record(int(count))
        return self.delegate(scientific_objects, *args, **kwargs)

    @property
    def identity(self) -> Mapping[str, str]:
        return {
            "wrapper": "scicf-counting-oracle-v1",
            "delegate": getattr(self.delegate, "__name__", type(self.delegate).__name__),
        }
