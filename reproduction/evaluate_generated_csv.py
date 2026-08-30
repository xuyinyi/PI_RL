#!/usr/bin/env python3
"""Apply the common metric protocol to an existing generation CSV."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from reproduction.framework.evaluator import CommonEvaluator
from reproduction.framework.io import git_identity, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--reference-column", default="smile")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    evaluator = CommonEvaluator(
        reference_csv=args.reference_csv.resolve(),
        reference_column=args.reference_column,
        compute_paper_metrics=True,
    )
    summary = evaluator.evaluate_csv(args.input_csv.resolve())
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": git_identity(args.repo_root.resolve()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "input_csv": str(args.input_csv.resolve()),
        }
    )
    write_json(args.output_json.resolve(), summary)


if __name__ == "__main__":
    main()
