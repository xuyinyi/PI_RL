#!/usr/bin/env python3
"""Verify that the compatibility branch implements main-text Equation (1)."""

import json

from RL_PPO.GNN.benchmarks import Benchmark


class FormulaProbe:
    score = Benchmark.score
    score_cte = staticmethod(Benchmark.score_cte)
    score_strength = staticmethod(Benchmark.score_strength)
    score_tg = staticmethod(Benchmark.score_tg)
    score_SA = staticmethod(Benchmark.score_SA)

    @staticmethod
    def pred_transmittance():
        return [80.0]

    @staticmethod
    def pred_cte():
        return [20.0]

    @staticmethod
    def pred_strength():
        return [250.0]

    @staticmethod
    def pred_tg():
        return [350.0]

    @staticmethod
    def pred_SA():
        return [3.5]


def main() -> int:
    probe = FormulaProbe()
    probe.score()
    coefficient = probe.transmittance / 100
    expected = round(
        coefficient
        * (
            1
            + probe.Score_cte
            + probe.Score_strength
            + probe.Score_tg
            + probe.Score_SA
        )
        / 5,
        4,
    )
    assert probe.Score == expected
    print(
        json.dumps(
            {
                "equation": "R_T400 * (1 + R_CLTE + R_strength + R_Tg + R_SA) / 5",
                "expected": expected,
                "observed": probe.Score,
                "passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

