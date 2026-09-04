import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from reproduction.p2.budget import (
    RequestedCallBudgetManager,
    evaluator_ledger_delta,
)
from reproduction.p2.contracts import ContractViolation


class _ConcurrentLedger:
    def __init__(self, maximum_requested_calls):
        self.maximum_requested_calls = int(maximum_requested_calls)
        self.requested_calls = 0
        self._lock = threading.Lock()

    def ledger(self):
        with self._lock:
            return {
                "requested_calls": self.requested_calls,
                "unique_calls": self.requested_calls,
                "backend_calls": self.requested_calls,
                "cache_hits": 0,
                "requested_by_source": (
                    {} if self.requested_calls == 0 else {"ppo/on_policy": self.requested_calls}
                ),
                "remaining_requested_calls": self.maximum_requested_calls
                - self.requested_calls,
            }

    def consume(self, count):
        with self._lock:
            if self.requested_calls + count > self.maximum_requested_calls:
                raise RuntimeError("backend budget exceeded")
            self.requested_calls += count


class RequestedBudgetTests(unittest.TestCase):
    def test_concurrent_pre_reservation_closes_exactly_at_exhaustion(self):
        ledger = _ConcurrentLedger(4)
        manager = RequestedCallBudgetManager(ledger.ledger)
        barrier = threading.Barrier(6)

        def worker():
            barrier.wait()
            try:
                token = manager.reserve(2)
            except ContractViolation:
                token = None
            barrier.wait()
            if token is None:
                return False
            ledger.consume(2)
            manager.reconcile(token, 2)
            return True

        with ThreadPoolExecutor(max_workers=6) as executor:
            accepted = list(executor.map(lambda _: worker(), range(6)))
        self.assertEqual(sum(accepted), 2)
        self.assertEqual(ledger.ledger()["requested_calls"], 4)
        self.assertEqual(ledger.ledger()["remaining_requested_calls"], 0)
        with self.assertRaisesRegex(ContractViolation, "unavailable"):
            manager.reserve(1)
        self.assertEqual(manager.state_dict()["outstanding_requested_calls"], 0)

    def test_ledger_delta_is_source_exact_and_rejects_rollback(self):
        before = {
            "requested_calls": 1,
            "unique_calls": 1,
            "backend_calls": 1,
            "cache_hits": 0,
            "requested_by_source": {"ppo/on_policy": 1},
        }
        after = {
            "requested_calls": 3,
            "unique_calls": 2,
            "backend_calls": 2,
            "cache_hits": 1,
            "requested_by_source": {"ppo/on_policy": 3},
        }
        delta = evaluator_ledger_delta(before, after)
        self.assertEqual(delta.requested_calls, 2)
        self.assertEqual(delta.requested_by_source, {"ppo/on_policy": 2})
        with self.assertRaisesRegex(ContractViolation, "moved backwards"):
            evaluator_ledger_delta(after, before)


if __name__ == "__main__":
    unittest.main()
