from __future__ import annotations

"""Centralized evaluator service for Ray RLlib workers.

A Python evaluator object is copied when each rollout worker creates its own
Gym environment. That would multiply the cache and oracle budget. Stage-0 uses
one Ray actor and gives every worker a lightweight client to that actor.
"""

from typing import Optional, Sequence

from .chemistry import LegacyDAPiGenChemistryBackend
from .evaluator import (
    BudgetedCachingTerminalEvaluator,
    LegacyDAPiGenBenchmarkEvaluator,
    PersistentDAPiGenBenchmarkEvaluator,
)


class RayTerminalEvaluatorClient(object):
    def __init__(self, actor_handle) -> None:
        self.actor_handle = actor_handle
        self._evaluator_version = None
        self._objective_contract = None

    @property
    def evaluator_version(self) -> str:
        import ray

        if self._evaluator_version is None:
            self._evaluator_version = str(
                ray.get(self.actor_handle.evaluator_version.remote())
            )
        return self._evaluator_version

    @property
    def objective_contract(self) -> str:
        import ray

        if self._objective_contract is None:
            self._objective_contract = str(
                ray.get(self.actor_handle.objective_contract.remote())
            )
        return self._objective_contract

    def evaluate_batch(
        self, canonical_smiles: Sequence[str], source: str = "unspecified"
    ):
        import ray

        return ray.get(
            self.actor_handle.evaluate_batch.remote(
                list(canonical_smiles), str(source)
            )
        )

    def evaluate_one(self, canonical_smiles: str, source: str):
        return self.evaluate_batch([canonical_smiles], source=source)[0]

    def ledger(self):
        import ray

        return ray.get(self.actor_handle.ledger.remote())

    def audit_events(self):
        import ray

        return ray.get(self.actor_handle.audit_events.remote())

    def state_dict(self):
        import ray

        return ray.get(self.actor_handle.state_dict.remote())

    def load_state_dict(self, state):
        import ray

        return ray.get(self.actor_handle.load_state_dict.remote(state))

    def reset_ledger(self, retain_cache: bool = False):
        import ray

        return ray.get(
            self.actor_handle.reset_ledger.remote(bool(retain_cache))
        )


def create_stage0_evaluator_actor(
    dapigen_root: str,
    evaluator_mode: str = "persistent",
    device: Optional[str] = None,
    maximum_requested_calls: Optional[int] = None,
    maximum_unique_calls: Optional[int] = None,
    maximum_cache_entries: int = 500000,
    allowed_sources: Optional[Sequence[str]] = None,
    cache_scope: str = "per_run",
    fail_fast: bool = True,
    num_cpus: float = 1.0,
    num_gpus: float = 0.0,
):
    """Create the single terminal-evaluator actor for a formal Stage-0 run."""

    import ray

    actor_options = {"num_cpus": float(num_cpus), "num_gpus": float(num_gpus)}

    @ray.remote(**actor_options)
    class _Stage0EvaluatorActor(object):
        def __init__(self):
            if evaluator_mode == "persistent":
                base = PersistentDAPiGenBenchmarkEvaluator(
                    dapigen_root=dapigen_root,
                    device=device,
                )
            elif evaluator_mode == "legacy":
                base = LegacyDAPiGenBenchmarkEvaluator()
            else:
                raise ValueError("Unsupported evaluator_mode: %s" % evaluator_mode)
            chemistry = LegacyDAPiGenChemistryBackend()
            self.service = BudgetedCachingTerminalEvaluator(
                base,
                maximum_requested_calls=maximum_requested_calls,
                maximum_unique_calls=maximum_unique_calls,
                maximum_cache_entries=maximum_cache_entries,
                canonicalizer=chemistry.canonicalize,
                validator=lambda smiles: "*" not in smiles,
                allowed_sources=allowed_sources,
                cache_scope=cache_scope,
                fail_fast=fail_fast,
            )

        def evaluator_version(self):
            return self.service.evaluator_version

        def objective_contract(self):
            return self.service.objective_contract

        def evaluate_batch(self, canonical_smiles, source):
            return self.service.evaluate_batch(canonical_smiles, source=source)

        def ledger(self):
            return self.service.ledger()

        def audit_events(self):
            return self.service.audit_events()

        def state_dict(self):
            return self.service.state_dict()

        def load_state_dict(self, state):
            self.service.load_state_dict(state)
            return self.service.ledger()

        def reset_ledger(self, retain_cache=False):
            self.service.reset_ledger(retain_cache=bool(retain_cache))
            return self.service.ledger()

    return _Stage0EvaluatorActor.remote()


def create_legacy_evaluator_actor(
    dapigen_root: str = ".",
    maximum_requested_calls: Optional[int] = None,
    maximum_unique_calls: Optional[int] = None,
    maximum_cache_entries: int = 500000,
    num_cpus: float = 1.0,
    num_gpus: float = 0.0,
):
    """Backward-compatible legacy regression actor constructor."""

    return create_stage0_evaluator_actor(
        dapigen_root=dapigen_root,
        evaluator_mode="legacy",
        maximum_requested_calls=maximum_requested_calls,
        maximum_unique_calls=maximum_unique_calls,
        maximum_cache_entries=maximum_cache_entries,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )
