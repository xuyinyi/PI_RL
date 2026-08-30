from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from reproduction.framework.config import load_config, validate_config
from reproduction.framework.contracts import ContractError
from reproduction.framework.io import write_json
from reproduction.framework.registry import load_adapter_class


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]


class FrameworkContractTests(unittest.TestCase):
    def test_full_config_is_valid(self):
        config = load_config(
            REPRODUCTION_ROOT / "configs" / "ppo-compat-common-v1.json"
        )
        self.assertEqual(config["training"]["max_environment_steps"], 99000)
        self.assertEqual(config["evaluation"]["checkpoint_environment_steps"][-1], 99000)

    def test_smoke_config_is_valid(self):
        config = load_config(
            REPRODUCTION_ROOT / "configs" / "ppo-compat-framework-smoke-v1.json"
        )
        self.assertEqual(config["evaluation"]["checkpoint_environment_steps"], [0, 20])

    def test_missing_seed_fails_closed(self):
        config = load_config(
            REPRODUCTION_ROOT / "configs" / "ppo-compat-framework-smoke-v1.json"
        )
        config["training"]["seed"] = None
        with self.assertRaises(ContractError):
            validate_config(config)

    def test_checkpoint_budget_must_close(self):
        config = load_config(
            REPRODUCTION_ROOT / "configs" / "ppo-compat-framework-smoke-v1.json"
        )
        config["evaluation"]["checkpoint_environment_steps"] = [0, 10]
        with self.assertRaises(ContractError):
            validate_config(config)

    def test_adapter_locator(self):
        adapter = load_adapter_class(
            "reproduction.framework.algorithms.rllib_ppo:RLLibPPOAdapter"
        )
        self.assertEqual(adapter.__name__, "RLLibPPOAdapter")

    def test_atomic_json_converts_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            write_json(path, {"finite": 1.5, "infinite": math.inf})
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["finite"], 1.5)
            self.assertIsNone(value["infinite"])
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
