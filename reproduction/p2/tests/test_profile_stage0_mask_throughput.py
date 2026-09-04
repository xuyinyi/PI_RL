import json
from pathlib import Path

from reproduction.p2.scripts.profile_stage0_mask_throughput import (
    build_profile_core,
    classify_profile,
    determinism_violations,
    load_profile_config,
    numeric_deltas,
    summarize_scaling,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _worker(index, state_digest="state", mask_digest="mask"):
    return {
        "worker_index": index,
        "workload": {
            "state_digest": "%s-%d" % (state_digest, index),
            "termination_reasons": {"success": 1},
            "no_product_count": 0,
        },
        "mask_digest": "%s-%d" % (mask_digest, index),
        "allowed_dianhydride_total": 10 + index,
        "allowed_diamine_total": 20 + index,
    }


def _group(count, repetition, throughput):
    return {
        "worker_count": count,
        "repetition": repetition,
        "status": "passed",
        "no_product_count": 0,
        "aggregate_masks_per_second": throughput,
        "workers": [_worker(index) for index in range(count)],
    }


def test_frozen_profile_config_is_valid():
    config = load_profile_config(
        Path("reproduction/p2/configs/stage0_mask_throughput_v1.json")
    )
    assert config["worker_counts"] == [1, 2, 4, 8]
    assert config["repetitions"] == 3
    assert config["mask_mode"] == "closure_exact_cached"


def test_scaling_summary_and_admission_pass():
    groups = []
    for repetition in range(3):
        groups.extend(
            [
                _group(1, repetition, 10.0),
                _group(2, repetition, 18.0),
                _group(4, repetition, 30.0),
                _group(8, repetition, 40.0),
            ]
        )
    scaling = summarize_scaling(groups, [1, 2, 4, 8])
    result = classify_profile(
        groups,
        scaling,
        [1, 2, 4, 8],
        {
            "minimum_max_worker_speedup": 1.5,
            "minimum_max_worker_efficiency": 0.25,
            "minimum_adjacent_throughput_ratio": 0.85,
        },
        {
            "profile_git_dirty": False,
            "stage0_source_changed_from_accepted": False,
        },
    )
    assert result["status"] == "passed"
    assert result["maximum_worker_median_speedup"] == 4.0
    assert result["maximum_worker_median_efficiency"] == 0.5


def test_throughput_and_determinism_fail_closed():
    groups = [_group(1, 0, 10.0), _group(8, 0, 12.0)]
    groups[1]["workers"][0]["mask_digest"] = "changed"
    scaling = summarize_scaling(groups, [1, 8])
    result = classify_profile(
        groups,
        scaling,
        [1, 8],
        {
            "minimum_max_worker_speedup": 1.5,
            "minimum_max_worker_efficiency": 0.25,
            "minimum_adjacent_throughput_ratio": 0.85,
        },
        {
            "profile_git_dirty": False,
            "stage0_source_changed_from_accepted": False,
        },
    )
    assert determinism_violations(groups)
    assert result["status"] == "failed_functional_gate"
    assert "nondeterministic_worker_digest" in result["functional_failures"]
    assert "maximum_worker_speedup_below_threshold" in result["throughput_failures"]
    assert "maximum_worker_efficiency_below_threshold" in result["throughput_failures"]


def test_numeric_deltas_keep_nested_counters_and_ignore_booleans():
    assert numeric_deltas(
        {"calls": 2, "ready": False, "nested": {"hits": 4}},
        {"calls": 7, "ready": True, "nested": {"hits": 10}},
    ) == {"calls": 5, "nested": {"hits": 6}}


def test_profile_core_builder_returns_accepted_mask_core():
    config = json.loads(
        (
            REPOSITORY_ROOT
            / "reproduction/stage0/configs/stage0_environment.json"
        ).read_text()
    )
    core = build_profile_core(REPOSITORY_ROOT, config)
    assert core is not None
    assert core.config.mask_mode == "closure_exact_cached"
    assert "dapigen_custom" in core.specification()["chemistry_backend"]
