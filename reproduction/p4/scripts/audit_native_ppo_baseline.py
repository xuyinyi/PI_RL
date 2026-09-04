#!/usr/bin/env python
"""Fail closed unless a P4-A run report satisfies the frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduction.p4.protocol import (
    evaluate_acceptance,
    file_sha256,
    load_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    report_path = Path(args.report).resolve()
    protocol = load_protocol(protocol_path)
    report = json.loads(report_path.read_text())
    checks, accepted = evaluate_acceptance(report, protocol)
    audit = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(protocol_path),
        "report": str(report_path),
        "report_status": report.get("status"),
        "mode": report.get("mode"),
        "seed": report.get("seed"),
        "checks": checks,
        "accepted": bool(accepted),
    }
    print(json.dumps(audit, indent=2, sort_keys=True))
    if report.get("protocol_sha256") != audit["protocol_sha256"]:
        raise SystemExit("Protocol hash differs from the run report.")
    if report.get("mode") != args.mode:
        raise SystemExit("Run mode differs from the audit request.")
    if report.get("status") != "passed" or not accepted:
        raise SystemExit("P4-A run report did not pass the frozen gate.")


if __name__ == "__main__":
    main()

