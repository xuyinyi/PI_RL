#!/usr/bin/env python
"""Numerically compare the released polyBERT helper with the persistent encoder.

Run this inside the complete DAPiGen environment after applying the Stage-0
overlay. The comparison uses exactly the same attachment-label normalization as
the released environment and checks repeatability as well as numerical parity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from RL_PPO.envs.embedding import PersistentPolyBERTEncoder
from RL_PPO.utils.polyBERT import Embedding_smiles


def normalize_attachment_labels(smiles: str) -> str:
    value = str(smiles)
    for label in range(1, 17):
        value = value.replace("([%d*])" % label, "([*])")
        value = value.replace("[%d*]" % label, "[*]")
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polybert-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--output", default="stage0_audit/polybert_parity.json")
    parser.add_argument(
        "--smiles",
        action="append",
        default=None,
        help="Optional partial SMILES; may be supplied repeatedly.",
    )
    return parser.parse_args()


def default_panel() -> List[str]:
    return [
        "[16*]c1ccc2c(c1)C(=O)OC2=O",
        "[16*]c1ccc(N)cc1",
        "[16*]c1ccc([16*])cc1",
        "[8*]C([8*])(C)C",
        "[3*]O[3*]",
    ]


def main() -> None:
    args = parse_args()
    panel = args.smiles or default_panel()
    normalized = [normalize_attachment_labels(item) for item in panel]

    persistent = PersistentPolyBERTEncoder(
        args.polybert_path,
        device=args.device,
        maximum_entries=max(32, len(normalized) * 4),
    )

    rows = []
    all_passed = True
    for raw, clean in zip(panel, normalized):
        released_first = np.asarray(
            Embedding_smiles(args.polybert_path, clean), dtype=np.float32
        ).reshape(-1)
        released_second = np.asarray(
            Embedding_smiles(args.polybert_path, clean), dtype=np.float32
        ).reshape(-1)
        persistent_first = np.asarray(persistent(clean), dtype=np.float32).reshape(-1)
        persistent_second = np.asarray(persistent(clean), dtype=np.float32).reshape(-1)

        released_repeatable = bool(
            np.allclose(
                released_first,
                released_second,
                atol=args.atol,
                rtol=args.rtol,
            )
        )
        persistent_repeatable = bool(
            np.array_equal(persistent_first, persistent_second)
        )
        parity = bool(
            np.allclose(
                released_first,
                persistent_first,
                atol=args.atol,
                rtol=args.rtol,
            )
        )
        max_abs = float(np.max(np.abs(released_first - persistent_first)))
        row_passed = released_repeatable and persistent_repeatable and parity
        all_passed = all_passed and row_passed
        rows.append(
            {
                "raw_smiles": raw,
                "normalized_smiles": clean,
                "dimension": int(released_first.size),
                "released_repeatable": released_repeatable,
                "persistent_repeatable": persistent_repeatable,
                "released_vs_persistent": parity,
                "maximum_absolute_difference": max_abs,
                "passed": row_passed,
            }
        )

    payload = {
        "status": "passed" if all_passed else "failed",
        "polybert_path": str(Path(args.polybert_path).resolve()),
        "encoder_version": persistent.encoder_version,
        "checkpoint_fingerprint": persistent.checkpoint_fingerprint,
        "absolute_tolerance": float(args.atol),
        "relative_tolerance": float(args.rtol),
        "panel_size": len(rows),
        "rows": rows,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
