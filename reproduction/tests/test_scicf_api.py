from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from reproduction.scicf.core.config import load_scicf_config
from reproduction.scicf.llm.api_client import APISettings
from reproduction.scicf.llm.provision_credentials import provision
from reproduction.scicf.llm.rank_requests_api import run


REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCTION_ROOT = REPO_ROOT / "reproduction"


def _write_credentials(path: Path, endpoint: str, mode: int = 0o600) -> None:
    path.write_text(
        "\n".join(
            [
                "SCICF_LLM_API_URL={}".format(endpoint),
                "SCICF_LLM_API_KEY=unit-test-secret",
                "SCICF_LLM_MODEL_ID=mock-science-model",
                "SCICF_LLM_MODEL_REVISION=mock-deployment-001",
                "SCICF_LLM_PROVIDER_ID=slurm-local-mock",
                "SCICF_LLM_ALLOW_INSECURE_HTTP=true",
                "SCICF_LLM_INCLUDE_SEED=false",
                "SCICF_LLM_JSON_MODE=true",
                "SCICF_LLM_THINKING=disabled",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(mode)


class _MockHandler(BaseHTTPRequestHandler):
    calls = 0
    last_payload = None

    def do_POST(self):
        type(self).calls += 1
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).last_payload = payload
        if self.headers.get("Authorization") != "Bearer unit-test-secret":
            self.send_error(401)
            return
        if payload.get("model") != "mock-science-model":
            self.send_error(400)
            return
        selected_id = "candidate-invented" if type(self).calls == 1 else "candidate-b"
        body = json.dumps(
            {
                "id": "mock-response-{}".format(type(self).calls),
                "system_fingerprint": "mock-fingerprint",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"ranked_intervention_ids": [selected_id]}
                            ),
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 17,
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class SciCFAPITests(unittest.TestCase):
    def test_api_gate_config_is_pending_and_fail_closed(self):
        config = load_scicf_config(
            REPRODUCTION_ROOT / "configs" / "scicf-gate1-api-v1.json"
        )
        self.assertEqual(config["llm"]["provider"], "openai-compatible-api")
        self.assertEqual(config["gate1"]["status"], "pending")
        self.assertFalse(config["pairwise_refinement"]["enabled"])

    def test_private_credentials_reject_group_readable_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.env"
            _write_credentials(path, "http://127.0.0.1/chat/completions", 0o640)
            with self.assertRaises(ValueError):
                APISettings.from_private_file(path)

    def test_deepseek_flash_provisioning_is_slurm_only_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            raw_key = root / "raw-key.txt"
            raw_key.write_text(
                "sk-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8"
            )
            raw_key.chmod(0o600)
            output = root / "deepseek.env"
            previous_job = os.environ.get("SLURM_JOB_ID")
            try:
                os.environ.pop("SLURM_JOB_ID", None)
                with self.assertRaises(RuntimeError):
                    provision(
                        raw_key,
                        output,
                        "deepseek-v4-flash",
                        "DeepSeek-V4-Flash-0731-api-snapshot-2026-09-03",
                    )
                os.environ["SLURM_JOB_ID"] = previous_job or "unit-test-slurm"
                provision(
                    raw_key,
                    output,
                    "deepseek-v4-flash",
                    "DeepSeek-V4-Flash-0731-api-snapshot-2026-09-03",
                )
            finally:
                if previous_job is None:
                    os.environ.pop("SLURM_JOB_ID", None)
                else:
                    os.environ["SLURM_JOB_ID"] = previous_job
            settings = APISettings.from_private_file(output)
            self.assertEqual(settings.model_id, "deepseek-v4-flash")
            self.assertEqual(settings.provider_id, "deepseek-official")
            self.assertFalse(settings.include_seed)
            self.assertTrue(settings.json_mode)
            self.assertEqual(settings.thinking, "disabled")
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertNotIn(settings.api_key, json.dumps(settings.public_identity()))

    def test_api_runner_uses_strict_identity_and_cache_without_logging_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _MockHandler.calls = 0
            _MockHandler.last_payload = None
            server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                credentials = root / "api.env"
                endpoint = "http://127.0.0.1:{}/chat/completions".format(
                    server.server_port
                )
                _write_credentials(credentials, endpoint)
                request_path = root / "requests.jsonl"
                request_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "request_id": "api-test-request",
                            "pool_id": "api-test-pool",
                            "prompt_sha256": "a" * 64,
                            "prompt_version": "scicf-dapigen-acquisition-blinded-v2",
                            "candidate_presentation": "sha256-shuffle-v1",
                            "candidate_ids": ["candidate-a", "candidate-b"],
                            "presented_candidate_ids": [
                                "candidate-b",
                                "candidate-a",
                            ],
                            "budget": 1,
                            "messages": [
                                {"role": "system", "content": "Return JSON."},
                                {"role": "user", "content": "Rank the candidates."},
                            ],
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                cache_root = root / "cache"
                previous_job = os.environ.get("SLURM_JOB_ID")
                os.environ["SLURM_JOB_ID"] = previous_job or "unit-test-slurm"
                try:
                    for index in (1, 2):
                        run(
                            argparse.Namespace(
                                repo_root=REPO_ROOT,
                                credentials_file=credentials,
                                requests=[request_path],
                                output_root=root / "output-{}".format(index),
                                cache_root=cache_root,
                                max_output_tokens=64,
                                validation_retries=1,
                                transport_retries=1,
                                timeout_seconds=5.0,
                                seed=2023,
                                expected_request_count=1,
                                expected_budget=1,
                                expected_pool_size=2,
                                expected_prompt_version="scicf-dapigen-acquisition-blinded-v2",
                                expected_candidate_presentation="sha256-shuffle-v1",
                                limit=None,
                            )
                        )
                finally:
                    if previous_job is None:
                        os.environ.pop("SLURM_JOB_ID", None)
                    else:
                        os.environ["SLURM_JOB_ID"] = previous_job

                self.assertEqual(_MockHandler.calls, 2)
                first_manifest = json.loads(
                    (root / "output-1" / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                second_manifest = json.loads(
                    (root / "output-2" / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(first_manifest["cache_hits"], 0)
                self.assertEqual(second_manifest["cache_hits"], 1)
                self.assertEqual(first_manifest["api_calls_this_run"], 2)
                self.assertEqual(second_manifest["api_calls_this_run"], 0)
                self.assertEqual(first_manifest["origin_reported_total_tokens"], 34)
                response_record = json.loads(
                    (root / "output-1" / "responses.jsonl").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(response_record["attempt"], 1)
                self.assertEqual(response_record["api_calls"], 2)
                self.assertEqual(len(response_record["validation_errors"]), 1)
                serialized = json.dumps(first_manifest, sort_keys=True)
                self.assertNotIn("unit-test-secret", serialized)
                self.assertNotIn(str(credentials), serialized)
                self.assertEqual(
                    first_manifest["provider"]["model_revision"],
                    "mock-deployment-001",
                )
                self.assertNotIn("seed", _MockHandler.last_payload)
                self.assertEqual(
                    _MockHandler.last_payload["response_format"],
                    {"type": "json_object"},
                )
                self.assertEqual(
                    _MockHandler.last_payload["thinking"], {"type": "disabled"}
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
