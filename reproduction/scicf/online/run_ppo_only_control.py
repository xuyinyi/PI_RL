"""Slurm-only preflight and one-run matched PPO-only control. No provider path."""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path

from reproduction.framework.io import write_json
from .ppo_only_control import (
    PROTOCOL_SHA, ANALYSIS_SHA, require, sha256_path, load_contract, validate_assets,
    build_control_runtime, check_iteration, check_ledger, summarize, rng_digests,
    expected_authorization, consume_authorization,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'real'), required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--polybert-path', type=Path, required=True)
    parser.add_argument('--evaluator-asset-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--preflight-report', type=Path)
    args = parser.parse_args()
    require(os.environ.get('SLURM_JOB_ID'), 'Slurm required')
    require(not os.environ.get('SLURM_ARRAY_JOB_ID'), 'job arrays forbidden')
    require(os.environ.get('SLURM_RESTART_COUNT', '0') == '0', 'job restart forbidden')
    require(socket.gethostname() == 'yanlih100n1', 'n001 required')
    root, output = args.repo_root.resolve(strict=True), args.output_dir.resolve()
    require(not output.exists(), 'output already exists')
    require(args.mode == 'real' or (args.authorization is None and args.preflight_report is None),
            'preflight accepts no execution authorization')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'schema_version': 1, 'mode': args.mode, 'execution_status': 'failed',
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
              'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
              'protocol_sha256': PROTOCOL_SHA, 'analysis_sha256': ANALYSIS_SHA,
              'credentials_loaded': False, 'external_api_requests': 0,
              'auxiliary_optimizer_steps': 0, 'sealed_test_accessed': False,
              'automatic_rerun': False, 'automatic_resume': False, 'iterations': []}
    engine = None
    try:
        c, base, reference, source = load_contract(root)
        report.update({'protocol_id': c['protocol_id'], 'source': source,
                       'reference_report_sha256': c['reference']['report_sha256']})
        manifest, binding, assets = validate_assets(root, c, base, args.polybert_path, args.evaluator_asset_path)
        report['assets'] = assets
        if args.mode == 'real':
            require(str(output) == c['slurm']['output_directory'], 'unique output mismatch')
            require(args.authorization is not None and args.preflight_report is not None, 'authorization and preflight required')
            preflight = json.loads(args.preflight_report.read_text())
            require(preflight['execution_status'] == 'passed' and preflight['mode'] == 'preflight', 'preflight not passed')
            require(preflight['source']['commit'] == source['commit'], 'preflight source differs')
            require(preflight['protocol_sha256'] == PROTOCOL_SHA and preflight['analysis_sha256'] == ANALYSIS_SHA, 'preflight protocol differs')
            require(preflight['assets'] == assets, 'preflight assets differ')
            require(preflight['iteration_count'] == 0 and preflight['final_evaluator_ledger']['requested_calls'] == 0, 'preflight exceeded scope')
            expected = expected_authorization(c, source, assets, output, args.preflight_report)
            report['authorization'] = consume_authorization(args.authorization, expected, os.environ['SLURM_JOB_ID'])
        write_json(output / 'run-intent.json', report)
        components, engine, spec, contract = build_control_runtime(root, c, base, manifest, binding, assets)
        report['stack_specification'] = spec
        report['run_contract'] = asdict(contract)
        report['initial_policy_sha256'] = engine.policy_state_sha256
        report['initial_rng_digests'] = rng_digests(engine)
        report['initial_evaluator_ledger'] = dict(engine.environment.oracle_ledger())
        require(engine.policy_state_sha256 == c['initialization_checks']['initial_policy_sha256'],
                'comparison_ineligible_initial_equivalence_failed: initial policy')
        report['initial_equivalence_passed'] = True
        if args.mode == 'preflight':
            require(engine.policy_version == 0, 'preflight PPO execution')
            require(engine.environment.oracle_ledger()['requested_calls'] == 0, 'preflight AFP calls')
            report['runtime_constructed'] = True
            report['encoder_diagnostics'] = components.core.encoder.diagnostics()
            report['module_decision'] = 'preflight_passed_for_authorized_one_run_control'
        else:
            previous_hash = None
            for iteration in range(1, 7):
                directory = output / 'iterations' / ('iteration-%02d' % iteration)
                directory.mkdir(parents=True)
                record = {'iteration': iteration, 'initial_policy_sha256': engine.policy_state_sha256,
                          'rng_before': rng_digests(engine), 'previous_iteration_report_sha256': previous_hash}
                result = engine.run_iteration(query_requested_calls=0)
                summary = summarize(result)
                record.update({'standard_ppo': summary, 'final_policy_sha256': engine.policy_state_sha256,
                               'rng_after': rng_digests(engine),
                               'evaluator_ledger_after_iteration': dict(engine.environment.oracle_ledger())})
                checkpoint = directory / 'primary-ppo.pt'
                engine.save_checkpoint(checkpoint)
                record['checkpoint'] = {'path': str(checkpoint), 'sha256': sha256_path(checkpoint)}
                record['integrity_passed'] = False
                report['iterations'].append(record)
                write_json(directory / 'iteration-report.json', record)
                check_iteration(summary, c, iteration, engine.policy_state_sha256)
                check_ledger(engine.environment.oracle_ledger(), c)
                if iteration == 1:
                    report['first_iteration_equivalence_passed'] = True
                record['integrity_passed'] = True
                write_json(directory / 'iteration-report.json', record)
                previous_hash = sha256_path(directory / 'iteration-report.json')
                write_json(output / 'progress.json', {'completed_iterations': iteration,
                           'latest_iteration_report_sha256': previous_hash})
                print(json.dumps({'completed_iteration': iteration, 'summary': summary}), flush=True)
            report['final_iteration_report_sha256'] = previous_hash
            report['module_decision'] = 'matched_ppo_only_six_iteration_engineering_complete'
        report['execution_status'] = 'passed'
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        report['module_decision'] = ('comparison_ineligible_initial_equivalence_failed'
            if 'comparison_ineligible_initial_equivalence_failed' in str(error) else 'failed_no_automatic_rerun')
        raise
    finally:
        report['iteration_count'] = len(report['iterations'])
        report['elapsed_seconds'] = time.perf_counter() - started
        if engine is not None:
            report['final_evaluator_ledger'] = dict(engine.environment.oracle_ledger())
        write_json(output / ('preflight-report.json' if args.mode == 'preflight' else 'control-report.json'), report)
        print(json.dumps({'execution_status': report['execution_status'], 'iteration_count': report['iteration_count'],
                          'output_directory': str(output)}), flush=True)


if __name__ == '__main__':
    main()
