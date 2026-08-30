#!/usr/bin/env python3
"""Audit public AFP inputs without creating scalers, caches, or weights."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


PROPERTIES = (
    ("transmittance(400)", True, 825, 43),
    ("cte", False, 854, 56),
    ("strength", False, 331, 84),
    ("tg", False, 525, 64),
)
ATOM_REMOVE = ("Sn", "As", "Ti", "Ca", "Fe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    random.seed(2023)
    initialization_seeds = [random.randint(0, 1000) for _ in range(100)]
    report = {"properties": {}}

    for name, sigmoid, split_seed, model_id in PROPERTIES:
        path = args.repo_root / "raw_data" / f"{name}.csv"
        frame = pd.read_csv(path)
        raw_rows = len(frame)
        for symbol in ATOM_REMOVE:
            frame = frame.loc[~frame["smile"].str.contains(symbol, na=False)].copy()
        frame.reset_index(drop=True, inplace=True)

        canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(value)) if Chem.MolFromSmiles(value) else None for value in frame["smile"]]
        values = frame["value_mean"].astype(float).to_numpy()
        train_rows = int(0.8 * len(frame))
        validation_rows = int(0.1 * len(frame))

        scaler = {"type": "zero_one", "range": 100.0}
        if not sigmoid:
            scaler = {
                "type": "population_standardization",
                "mu": float(np.mean(values)),
                "dev": float(np.sqrt(np.mean((values - np.mean(values)) ** 2))),
            }

        report["properties"][name] = {
            "raw_rows": raw_rows,
            "filtered_rows": len(frame),
            "removed_rows": raw_rows - len(frame),
            "invalid_smiles": sum(value is None for value in canonical),
            "duplicate_canonical_smiles": len(canonical)
            - len({value for value in canonical if value is not None}),
            "split_seed": split_seed,
            "split_rows": {
                "train": train_rows,
                "validation": validation_rows,
                "test": len(frame) - train_rows - validation_rows,
            },
            "sigmoid": sigmoid,
            "target_model_id": model_id,
            "target_initialization_seed": initialization_seeds[model_id],
            "target_min": float(np.min(values)),
            "target_max": float(np.max(values)),
            "scaler": scaler,
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
