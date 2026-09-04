#!/usr/bin/env python
"""Audit the released DAPiGen action catalogs under Stage-0 semantics."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapigen-root", required=True)
    parser.add_argument("--output", default="stage0_catalog_audit.json")
    parser.add_argument(
        "--mask-mode",
        choices=(
            "closure_exact_cached",
            "compatibility",
            "exact_cached",
            "all",
        ),
        default="closure_exact_cached",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--reachable-states", type=int, default=100)
    parser.add_argument("--rollout-seed", type=int, default=20260904)
    return parser.parse_args()


def audit_side(blocks, source_rows, side, chemistry):
    labels = collections.Counter()
    complete = []
    fragments = []
    unusable = []
    attachment_counts = collections.Counter()
    atom_counts = []
    records = []
    for action_id, (smiles, source_row) in enumerate(zip(blocks, source_rows)):
        metadata = chemistry.block_metadata(action_id, smiles, side)
        labels.update(metadata.attachment_labels)
        attachment_counts[metadata.attachment_count] += 1
        atom_counts.append(metadata.atom_count)
        if metadata.is_complete_for_side:
            category = "complete_monomer"
            complete.append(action_id)
        elif metadata.has_attachment:
            category = "fragment"
            fragments.append(action_id)
        else:
            category = "unusable"
            unusable.append(action_id)
        records.append(
            {
                "action_id": action_id,
                "source_row": source_row,
                "canonical_smiles": metadata.canonical_smiles,
                "category": category,
                "attachment_labels": sorted(metadata.attachment_labels),
                "attachment_count": metadata.attachment_count,
                "atom_count": metadata.atom_count,
            }
        )
    return {
        "side": side,
        "retained_action_count": len(blocks),
        "complete_monomer_count": len(complete),
        "fragment_count": len(fragments),
        "unusable_count": len(unusable),
        "complete_monomer_ids": complete,
        "unusable_ids": unusable,
        "attachment_label_frequency": dict(sorted(labels.items())),
        "attachment_count_frequency": dict(sorted(attachment_counts.items())),
        "atom_count_min": min(atom_counts),
        "atom_count_max": max(atom_counts),
        "records": records,
    }


def audit_reachable_masks(core, maximum_states, rollout_seed):
    from RL_PPO.envs.rng import derive_seed, named_index
    from RL_PPO.envs.types import DAPiGenAction

    maximum_states = int(maximum_states)
    if maximum_states <= 0:
        raise ValueError("reachable-states must be positive.")

    def empty_totals():
        return {
            side: {
                "allowed": 0,
                "exact_allowed": 0,
                "false_positive": 0,
                "false_negative": 0,
            }
            for side in ("dianhydride", "diamine")
        }

    totals = empty_totals()
    label_compatibility_totals = empty_totals()
    mask_build_seconds = {
        "compatibility": 0.0,
        "closure_exact_cached": 0.0,
        "exact_cached_incremental_after_refined": 0.0,
    }
    examples = []
    visited_state_ids = set()
    step_index_frequency = collections.Counter()
    productive_transitions = 0
    overlap_dead_end_states = 0
    current = core.initial(seed=derive_seed(rollout_seed, "episode", 0))
    episode_index = 0
    visited = 0
    while visited < maximum_states:
        if current.state.done:
            episode_index += 1
            current = core.initial(
                seed=derive_seed(rollout_seed, "episode", episode_index)
            )
        visited_state_ids.add(current.state.state_id)
        step_index_frequency[int(current.state.step_index)] += 1
        comparison = {}
        for output_name, mask_mode in (
            ("compatibility", "compatibility"),
            ("refined", "closure_exact_cached"),
            ("exact", "exact_cached"),
        ):
            mask_started = time.perf_counter()
            mask = core.raw_action_mask_for_mode(current.state, mask_mode)
            elapsed = time.perf_counter() - mask_started
            timing_name = (
                "exact_cached_incremental_after_refined"
                if output_name == "exact"
                else mask_mode
            )
            mask_build_seconds[timing_name] += elapsed
            comparison[output_name] = {
                "dianhydride": mask["dianhydride"],
                "diamine": mask["diamine"],
            }

        for audit_name, output_totals in (
            ("compatibility", label_compatibility_totals),
            ("refined", totals),
        ):
            for side in ("dianhydride", "diamine"):
                proposed = comparison[audit_name][side]
                exact = comparison["exact"][side]
                false_positive = [
                    index
                    for index, (left, right) in enumerate(zip(proposed, exact))
                    if bool(left) and not bool(right)
                ]
                false_negative = [
                    index
                    for index, (left, right) in enumerate(zip(proposed, exact))
                    if not bool(left) and bool(right)
                ]
                output_totals[side]["allowed"] += int(proposed.sum())
                output_totals[side]["exact_allowed"] += int(exact.sum())
                output_totals[side]["false_positive"] += len(false_positive)
                output_totals[side]["false_negative"] += len(false_negative)
                if (false_positive or false_negative) and len(examples) < 100:
                    examples.append(
                        {
                            "mask": audit_name,
                            "state_id": current.state.state_id,
                            "episode_index": episode_index,
                            "step_index": current.state.step_index,
                            "side": side,
                            "false_positive_action_ids": false_positive[:20],
                            "false_negative_action_ids": false_negative[:20],
                        }
                    )
        # Traverse using actions accepted by both the proposed refined mask and
        # the independent exact reference. With zero false negatives this is
        # the same chemistry-state sequence used by the earlier exact audit.
        exact = comparison["exact"]
        refined = comparison["refined"]
        d_ids = [
            index for index, allowed in enumerate(exact["dianhydride"])
            if allowed and refined["dianhydride"][index]
        ]
        a_ids = [
            index for index, allowed in enumerate(exact["diamine"])
            if allowed and refined["diamine"][index]
        ]
        if not d_ids or not a_ids:
            overlap_dead_end_states += 1
            visited += 1
            episode_index += 1
            current = core.initial(
                seed=derive_seed(rollout_seed, "episode", episode_index)
            )
            continue
        action_seed = derive_seed(rollout_seed, "state", visited)
        action = DAPiGenAction(
            d_ids[named_index(action_seed, "dianhydride_action", len(d_ids))],
            a_ids[named_index(action_seed, "diamine_action", len(a_ids))],
        )
        current = core.transition(current.state, action, seed=action_seed)
        productive_transitions += int(
            current.state.termination_reason != "no_reaction_product"
        )
        visited += 1

    def finalize(values_by_side):
        false_positive_total = 0
        false_negative_total = 0
        for values in values_by_side.values():
            false_positive_total += values["false_positive"]
            false_negative_total += values["false_negative"]
            denominator = values["allowed"]
            values["false_positive_rate_among_allowed"] = (
                0.0
                if denominator == 0
                else float(values["false_positive"]) / float(denominator)
            )
            exact_denominator = values["exact_allowed"]
            values["false_negative_rate_among_exact"] = (
                0.0
                if exact_denominator == 0
                else float(values["false_negative"]) / float(exact_denominator)
            )
        return false_positive_total, false_negative_total

    false_positive_total, false_negative_total = finalize(totals)
    finalize(label_compatibility_totals)
    false_positive_reduction = {}
    for side in ("dianhydride", "diamine"):
        baseline = label_compatibility_totals[side]["false_positive"]
        remaining = totals[side]["false_positive"]
        false_positive_reduction[side] = {
            "removed": int(baseline - remaining),
            "fraction_removed": (
                0.0 if baseline == 0 else float(baseline - remaining) / baseline
            ),
        }
    status = "passed"
    if false_negative_total:
        status = "failed_false_negative"
    elif false_positive_total:
        status = "review_false_positive_rate"
    return {
        "status": status,
        "visited_states": visited,
        "distinct_state_count": len(visited_state_ids),
        "step_index_frequency": dict(sorted(step_index_frequency.items())),
        "productive_transitions": productive_transitions,
        "overlap_dead_end_states": overlap_dead_end_states,
        "episodes_started": episode_index + 1,
        "rollout_seed": int(rollout_seed),
        "totals": totals,
        "label_compatibility_totals": label_compatibility_totals,
        "false_positive_reduction": false_positive_reduction,
        "mask_build_seconds": dict(mask_build_seconds),
        "exact_reference_equivalent_seconds": float(
            mask_build_seconds["closure_exact_cached"]
            + mask_build_seconds["exact_cached_incremental_after_refined"]
        ),
        "examples": examples,
    }


def main():
    args = parse_args()
    root = Path(args.dapigen_root).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(repository_root))

    from RL_PPO.envs.chemistry import LegacyDAPiGenChemistryBackend
    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.core import BranchableDAPiGenCore
    from RL_PPO.envs.embedding import MorganFingerprintEncoder
    from RL_PPO.envs.factory import load_block_catalog

    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=False)
    block_dir = root / "RL_PPO" / "outputs" / "building_blocks"
    d_blocks, d_report = load_block_catalog(
        str(block_dir / "blocks_dianhydride.csv"), chemistry=chemistry
    )
    a_blocks, a_report = load_block_catalog(
        str(block_dir / "blocks_diamine.csv"), chemistry=chemistry
    )

    started = time.time()
    core = BranchableDAPiGenCore(
        dianhydride_blocks=d_blocks,
        diamine_blocks=a_blocks,
        initial_dianhydride_smiles="[16*]c1ccc2c(c1)C(=O)OC2=O",
        initial_diamine_smiles="[16*]c1ccc(N)cc1",
        chemistry=chemistry,
        encoder=MorganFingerprintEncoder(radius=2, number_of_bits=256),
        config=DAPiGenEnvConfig(
            max_atoms=60,
            max_steps=args.max_steps,
            mask_mode=args.mask_mode,
            complete_block_policy="pristine_only",
            observation_mode="augmented_v3",
            invalid_action_handling="raise",
        ),
    )
    initial = core.initial(seed=0)
    mask_audit = audit_reachable_masks(
        core, args.reachable_states, args.rollout_seed
    )
    elapsed = time.time() - started

    report = {
        "status": (
            "failed"
            if mask_audit["status"] == "failed_false_negative"
            else "needs_review"
            if mask_audit["status"] == "review_false_positive_rate"
            else "passed"
        ),
        "dapigen_root": str(root),
        "environment": core.specification(),
        "mask_build_seconds": elapsed,
        "initial_valid_actions": {
            "dianhydride": int(initial.action_mask.dianhydride.sum()),
            "diamine": int(initial.action_mask.diamine.sum()),
        },
        "reachable_mask_audit": mask_audit,
        "catalog_loader": {
            "dianhydride": {
                "raw_rows": d_report.raw_rows,
                "retained_actions": d_report.retained_actions,
                "canonical_duplicates_removed": d_report.canonical_duplicates_removed,
                "invalid_rows_removed": d_report.invalid_rows_removed,
            },
            "diamine": {
                "raw_rows": a_report.raw_rows,
                "retained_actions": a_report.retained_actions,
                "canonical_duplicates_removed": a_report.canonical_duplicates_removed,
                "invalid_rows_removed": a_report.invalid_rows_removed,
            },
        },
        "dianhydride": audit_side(
            d_blocks, d_report.retained_source_rows, "dianhydride", chemistry
        ),
        "diamine": audit_side(
            a_blocks, a_report.retained_source_rows, "diamine", chemistry
        ),
        "diagnostics": core.diagnostics(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "status": report["status"],
        "output": str(output.resolve()),
        "environment": report["environment"],
        "initial_valid_actions": report["initial_valid_actions"],
        "reachable_mask_audit": report["reachable_mask_audit"],
        "mask_build_seconds": report["mask_build_seconds"],
        "catalog_loader": report["catalog_loader"],
    }, indent=2, sort_keys=True))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
