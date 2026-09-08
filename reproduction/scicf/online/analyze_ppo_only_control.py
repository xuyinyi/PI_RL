"""Descriptive one-seed comparison under the frozen matched-control contract."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

from reproduction.framework.io import write_json
from .ppo_only_control import load_contract, require, sha256_path


def compare(reference, control):
    complete = (reference.get('execution_status') == control.get('execution_status') == 'passed'
                and control.get('first_iteration_equivalence_passed') is True
                and len(reference['iterations']) == len(control['iterations']) == 6
                and all(r['standard_ppo']['transition_count'] == 128
                        for arm in (reference, control) for r in arm['iterations']))
    rows = []
    for left, right in zip(reference['iterations'], control['iterations']):
        require(left['iteration'] == right['iteration'], 'iteration alignment differs')
        item = {'iteration': left['iteration'], 'cumulative_transitions': left['iteration'] * 128}
        for name, rec in [('scicf', left), ('ppo_only', right)]:
            p = rec['standard_ppo']
            item[name] = {'successful_terminal_count': p['successful_terminal_count'],
                          'transition_count': p['transition_count'],
                          'terminal_yield': p['successful_terminal_count'] / p['transition_count'],
                          'ppo': p['update_metrics'], 'ledger': rec['evaluator_ledger_after_iteration']}
        item['terminal_count_difference_scicf_minus_ppo'] = (
            item['scicf']['successful_terminal_count'] - item['ppo_only']['successful_terminal_count'])
        item['ppo_metric_differences_scicf_minus_ppo'] = {
            key: item['scicf']['ppo'][key] - item['ppo_only']['ppo'][key]
            for key in item['scicf']['ppo'].keys() & item['ppo_only']['ppo'].keys()}
        rows.append(item)
    endpoint = None
    secondary = None
    if complete:
        a = sum(x['scicf']['successful_terminal_count'] for x in rows)
        b = sum(x['ppo_only']['successful_terminal_count'] for x in rows)
        endpoint = {'denominator_per_arm': 768, 'scicf_count': a, 'ppo_only_count': b,
                    'scicf_yield': a / 768, 'ppo_only_yield': b / 768,
                    'count_difference_scicf_minus_ppo': a - b,
                    'yield_difference_scicf_minus_ppo': (a - b) / 768}
        a2 = sum(x['scicf']['successful_terminal_count'] for x in rows[1:])
        b2 = sum(x['ppo_only']['successful_terminal_count'] for x in rows[1:])
        secondary = {'denominator_per_arm': 640, 'scicf_count': a2, 'ppo_only_count': b2,
                     'yield_difference_scicf_minus_ppo': (a2 - b2) / 640}
    return {'classification': 'single_seed_exploratory_after_scicf_results_known',
            'eligible_complete_comparison': complete, 'primary_endpoint': endpoint,
            'iteration_2_to_6_secondary': secondary, 'iterations': rows,
            'missing_outcomes': {key: 'not_available_in_matched_evidence' for key in
                 ('reward_return', 'validity_per_completed_episode', 'uniqueness', 'diversity', 'held_out_quality')},
            'actual_total_cost_matched': False, 'independent_runs_per_arm': 1,
            'rng_coupling': 'same_initial_seed_later_stream_coupling_not_isolated',
            'effectiveness_established': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--control-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID') and socket.gethostname() == 'yanlih100n1', 'n001 Slurm required')
    c, _, reference, source = load_contract(args.repo_root.resolve())
    control = json.loads(args.control_report.read_text())
    require(control['protocol_id'] == c['protocol_id'], 'wrong control report')
    previous = None
    for index, record in enumerate(control['iterations'], 1):
        path = args.control_report.parent / 'iterations' / ('iteration-%02d' % index) / 'iteration-report.json'
        require(json.loads(path.read_text()) == record, 'iteration report differs from terminal report')
        require(record['previous_iteration_report_sha256'] == previous, 'broken checkpoint chain')
        require(sha256_path(Path(record['checkpoint']['path'])) == record['checkpoint']['sha256'], 'checkpoint hash mismatch')
        previous = sha256_path(path)
    if control['execution_status'] == 'passed':
        require(previous == control['final_iteration_report_sha256'], 'final report chain differs')
    result = compare(reference, control)
    result['provenance'] = {'reference_report_sha256': c['reference']['report_sha256'],
                            'control_report_path': str(args.control_report),
                            'control_report_sha256': sha256_path(args.control_report),
                            'analysis_source': source, 'analysis_slurm_job': os.environ['SLURM_JOB_ID']}
    result['final_evaluator_ledgers'] = {'scicf': reference['final_evaluator_ledger'],
                                       'ppo_only': control.get('final_evaluator_ledger')}
    result['evaluator_count_differences_scicf_minus_ppo'] = {
        key: reference['final_evaluator_ledger'][key] - control['final_evaluator_ledger'][key]
        for key in ('requested_calls', 'unique_calls', 'backend_calls', 'cache_hits')}
    result['elapsed_seconds'] = {'scicf_runner': reference['elapsed_seconds'],
                                'ppo_only_runner': control['elapsed_seconds']}
    result['auxiliary_optimizer_steps'] = {'scicf': 6, 'ppo_only': control['auxiliary_optimizer_steps']}
    result['llm_cost'] = {'scicf': {'seconds': reference['final_llm_wall_clock_budget']['consumed_seconds'],
                                  'pool_decisions': 12, 'http_transmissions': 13,
                                  'prompt_tokens': 102845, 'completion_tokens': 1695},
                          'ppo_only': {'seconds': 0, 'pool_decisions': 0, 'http_transmissions': 0,
                                       'prompt_tokens': 0, 'completion_tokens': 0}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'comparison.json', result)
    print(json.dumps({'eligible_complete_comparison': result['eligible_complete_comparison'],
                      'primary_endpoint': result['primary_endpoint']}))


if __name__ == '__main__':
    main()
