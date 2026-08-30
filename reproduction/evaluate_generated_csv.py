#!/usr/bin/env python3
"""Apply the common metric protocol to an existing generation CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from reproduction.framework.evaluator import CommonEvaluator
from reproduction.framework.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--reference-column", default="smile")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    evaluator = CommonEvaluator(
        reference_csv=args.reference_csv.resolve(),
        reference_column=args.reference_column,
        compute_paper_metrics=True,
    )
    summary = evaluator.evaluate_csv(args.input_csv.resolve())
    write_json(args.output_json.resolve(), summary)


if __name__ == "__main__":
    main()
