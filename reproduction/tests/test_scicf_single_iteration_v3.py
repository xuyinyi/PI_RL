from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reproduction.scicf.online.contracts import (
    SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID,
)
from reproduction.scicf.online.run_single_iteration_integration_v3 import (
    AUTHORIZATION_OPERATIONS_V3,
    load_execution_authorization_v3,
    load_v3_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "reproduction/scicf/online/configs/"
    / "single_iteration_integration_v3_protocol.json"
)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorization(tmp_path: Path):
    output = tmp_path / "authorized-output"
    polybert = tmp_path / "polybert"
    evaluator = tmp_path / "evaluator"
    polybert.mkdir()
    evaluator.mkdir()
    payload = {
        "schema_version": 4,
        "authorization_id": "v3-unit-test-only",
        "protocol_id": SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID,
        "protocol_sha256": _sha256_path(PROTOCOL),
        "soft_pair_config_sha256": "a" * 64,
        "implementation_commit": "v3-test-commit",
        "authorized_output_directory": str(output),
        "authorized_polybert_path": str(polybert),
        "polybert_asset_binding_sha256": "b" * 64,
        "polybert_checkpoint_fingerprint": "c" * 64,
        "authorized_evaluator_asset_path": str(evaluator),
        "evaluator_asset_binding_sha256": "d" * 64,
        "evaluator_asset_fingerprint": "e" * 64,
        "maximum_slurm_runs": 1,
        "authorized_operations": dict(AUTHORIZATION_OPERATIONS_V3),
    }
    return payload, output, polybert, evaluator


def _load_authorization(path, payload, output, polybert, evaluator):
    return load_execution_authorization_v3(
        path,
        protocol_sha256=payload["protocol_sha256"],
        soft_pair_config_sha256=payload["soft_pair_config_sha256"],
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
    )


def test_v3_protocol_binds_v2_and_passed_soft_pair_component_evidence():
    protocol, base, soft = load_v3_protocol(PROTOCOL, ROOT)
    assert protocol["protocol_id"] == SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL_ID
    assert protocol["primary_transaction"]["checkpoint_before_llm"] is True
    assert protocol["verification"]["replicates"] == 5
    assert base["verification"]["replicates"] == 2
    assert soft["soft_pair_aggregation"]["replicates"] == 5
    assert (
        protocol["authorization_state"]["real_single_iteration_authorized"]
        is False
    )


def test_schema4_authorization_requires_exact_single_run_scope(tmp_path):
    payload, output, polybert, evaluator = _authorization(tmp_path)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = _load_authorization(
        path, payload, output, polybert, evaluator
    )
    assert loaded["authorization_id"] == "v3-unit-test-only"

    payload["authorized_operations"]["multi_iteration_training_authorized"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="operation scope mismatch"):
        _load_authorization(path, payload, output, polybert, evaluator)


def test_schema4_authorization_binds_soft_config_and_output(tmp_path):
    payload, output, polybert, evaluator = _authorization(tmp_path)
    path = tmp_path / "authorization.json"
    payload["soft_pair_config_sha256"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="soft-pair config hash mismatch"):
        load_execution_authorization_v3(
            path,
            protocol_sha256=payload["protocol_sha256"],
            soft_pair_config_sha256="a" * 64,
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
            evaluator_asset_fingerprint=payload[
                "evaluator_asset_fingerprint"
            ],
        )


def test_runner_orders_primary_checkpoint_before_credentials_and_llm():
    source = (
        ROOT
        / "reproduction/scicf/online/run_single_iteration_integration_v3.py"
    ).read_text(encoding="utf-8")
    authorization = source.index("authorization = load_execution_authorization_v3(")
    model_asset = source.index("model_asset = validate_polybert_asset(")
    evaluator_asset = source.index("evaluator_asset = validate_evaluator_asset(")
    ppo = source.index("ppo_result = engine.run_iteration(")
    primary_checkpoint = source.index("engine.save_checkpoint(primary_checkpoint)")
    credentials = source.index("settings = APISettings.from_private_file(")
    optional_acquisition = source.index("acquisition_receipt = run_optional_llm_acquisition(")
    assert model_asset < authorization < ppo
    assert evaluator_asset < authorization < ppo
    assert ppo < primary_checkpoint < credentials < optional_acquisition
    assert "replicates=int(protocol[\"verification\"][\"replicates\"])" in source
    assert "finalize_optional_auxiliary(" in source

