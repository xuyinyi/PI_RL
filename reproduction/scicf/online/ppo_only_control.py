"""Frozen matched-control contracts and a provider-free native PPO runtime."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from reproduction.framework.io import git_identity
from reproduction.p2.contracts import (
    PPO, PPO_ON_POLICY, PPO_ENGINE_CONTRACT_ID, CREDIT_ESTIMATOR_CONTRACT_ID,
    MethodRunContract, canonical_sha256, METHOD_EVALUATOR_SOURCES,
)
from reproduction.p2.engine import PPOEngine, PPOEngineConfig, rollout_digest
from reproduction.p2.gae import GAECreditEstimator
from reproduction.p2.scripts.profile_stage0_mask_throughput import verify_accepted_binding
from .model_asset import load_polybert_asset_binding, validate_polybert_asset, sha256_path
from .evaluator_asset import (
    load_evaluator_asset_binding, validate_evaluator_asset,
    verify_evaluator_route_source_delta,
)

PROTOCOL = 'reproduction/scicf/online/configs/ppo_only_matched_short_horizon_v1.json'
ANALYSIS = 'reproduction/scicf/PPO_ONLY_MATCHED_SHORT_HORIZON_V1.md'
PROTOCOL_SHA = '0a0f138f51862652ae4033700a25099b76b1f441d11d4acec0759d3ae473cc80'
ANALYSIS_SHA = '66c53241055aab8bf6a649c5271404731eece2e68fa099933f90c7498cc4029d'


def require(condition, message):
    if not condition:
        raise ValueError(message)


class OnPolicyOnlyEvaluator:
    """Restrict calls while retaining the shared engine's declared PPO contract."""
    def __init__(self, evaluator):
        self._service = evaluator

    def __getattr__(self, name):
        return getattr(self._service, name)

    def evaluate_batch(self, canonical_smiles, source='unspecified'):
        require(source == PPO_ON_POLICY, 'control forbids non-on-policy evaluator call')
        return self._service.evaluate_batch(canonical_smiles, source=source)

    def evaluate_one(self, canonical_smiles, source):
        return self.evaluate_batch([canonical_smiles], source=source)[0]


def bound_json(root, path, expected):
    target = (root / path).resolve(strict=True)
    require(root in target.parents, 'bound path escaped repository')
    require(sha256_path(target) == expected, 'bound-file hash mismatch: ' + path)
    return json.loads(target.read_text())


def load_contract(root):
    c = bound_json(root, PROTOCOL, PROTOCOL_SHA)
    require(sha256_path(root / ANALYSIS) == ANALYSIS_SHA, 'analysis protocol changed')
    ref = c['reference']
    base = bound_json(root, ref['base_protocol_path'], ref['base_protocol_sha256'])
    bound_json(root, ref['short_protocol_path'], ref['short_protocol_sha256'])
    reference = bound_json(root, ref['report_path'], ref['report_sha256'])
    require(c['ppo'] == base['ppo'], 'PPO hyperparameters differ')
    require(c['base_seed'] == base['base_seed'], 'base seed differs')
    source = git_identity(root)
    require(source.get('dirty') is False, 'clean Git source required')
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code',
                    ref['implementation_commit'], '--', 'RL_PPO', 'QSPR', 'raw_data',
                    'reproduction/p2', 'reproduction/framework', 'reproduction/stage0'],
                   check=True, stdout=subprocess.DEVNULL)
    return c, base, reference, source


def validate_assets(root, c, base, polybert_path, evaluator_path):
    a = c['asset_binding']
    manifest = bound_json(root, a['accepted_manifest_path'], a['accepted_manifest_sha256'])
    binding = verify_accepted_binding(root, manifest, base['accepted_binding']['environment_id'])
    model = validate_polybert_asset(polybert_path.resolve(strict=True), load_polybert_asset_binding(root))
    evaluator_binding = load_evaluator_asset_binding(root)
    evaluator = validate_evaluator_asset(evaluator_path.resolve(strict=True), evaluator_binding)
    require(model['asset_binding_sha256'] == a['polybert_binding_sha256'], 'polyBERT binding')
    require(model['checkpoint_fingerprint'] == a['polybert_fingerprint'], 'polyBERT fingerprint')
    require(evaluator['asset_binding_sha256'] == a['afp_binding_sha256'], 'AFP binding')
    require(evaluator['asset_fingerprint'] == a['afp_fingerprint'], 'AFP fingerprint')
    require(manifest['environment']['encoder_version'] == model['encoder_version'], 'encoder identity')
    require(manifest['evaluator_version'] == evaluator['evaluator_version'], 'evaluator identity')
    delta = verify_evaluator_route_source_delta(root, binding['accepted_git_commit'], evaluator_binding)
    return manifest, binding, {'polybert': model, 'afp': evaluator, 'source_delta': delta}


def build_control_runtime(root, c, base, manifest, binding, assets):
    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.evaluator import PersistentDAPiGenBenchmarkEvaluator, TerminalRewardAdapter
    from RL_PPO.envs.factory import build_stage0_components
    from RL_PPO.envs.gymnasium_wrapper import DAPiGenGymnasiumEnv

    config = PPOEngineConfig(**c['ppo'])
    budget = c['budget']
    sources = tuple(sorted(METHOD_EVALUATOR_SOURCES[PPO]))
    terminal = PersistentDAPiGenBenchmarkEvaluator(
        str(root), device=config.device, model_dir=assets['afp']['asset_path'])
    components = build_stage0_components(
        dapigen_root=str(root), polybert_path=assets['polybert']['model_path'],
        config=DAPiGenEnvConfig.from_mapping(binding['accepted_task_config']),
        device=config.device, maximum_requested_calls=budget['maximum_requested_evaluator_calls'],
        maximum_unique_calls=budget['maximum_unique_evaluator_calls'],
        encoder_mode='polybert', evaluator_mode='persistent', terminal_evaluator=terminal,
        evaluator_fail_fast=True, allowed_evaluator_sources=sources,
        cache_scope=budget['cache_scope'],
        polybert_checkpoint_fingerprint=assets['polybert']['checkpoint_fingerprint'],
        allow_rdkit_brics_fallback=False)
    guarded = OnPolicyOnlyEvaluator(components.evaluator)
    components = replace(components, evaluator=guarded,
        reward_adapter=TerminalRewardAdapter(guarded, failure_reward=components.reward_adapter.failure_reward))
    spec = components.core.specification()
    for name in ('environment_id', 'environment_version', 'state_schema_version', 'config',
                 'observation_dimension', 'number_of_dianhydride_actions',
                 'number_of_diamine_actions', 'dianhydride_noop_id', 'diamine_noop_id',
                 'dianhydride_catalog_sha256', 'diamine_catalog_sha256',
                 'chemistry_backend', 'encoder_version'):
        require(spec[name] == manifest['environment'][name], 'runtime mismatch: ' + name)
    require(components.evaluator.evaluator_version == base['accepted_binding']['evaluator_version'], 'AFP runtime version')
    require(components.evaluator.objective_contract == base['accepted_binding']['objective_contract'], 'objective runtime')
    require(str(terminal.model_dir.resolve()) == assets['afp']['asset_path'], 'AFP runtime path')
    runtime_hashes = {}
    for name, dataset, model_id in terminal.PROPERTY_SPECS:
        record = terminal.model_manifest[name]
        stem = 'Ensemble_%s_AFP' % dataset
        runtime_hashes[stem + '_%d.pt' % model_id] = record['model_sha256']
        runtime_hashes[stem + '_settings.csv'] = record['settings_sha256']
        runtime_hashes[dataset + '_scaler.pkl'] = record['scaler_sha256']
    runtime_hashes['fpscores.pkl.gz'] = terminal.model_manifest['sa']['fpscores_sha256']
    require(runtime_hashes == assets['afp']['required_file_sha256'], 'AFP runtime hashes')
    environment = DAPiGenGymnasiumEnv(components.core, components.reward_adapter,
        source=PPO_ON_POLICY, seed=c['base_seed'], include_ledger_in_step_info=False)
    contract = MethodRunContract(
        method=PPO, environment_id=spec['environment_id'], task_contract_id=manifest['task_contract_id'],
        budget_contract_id=canonical_sha256({'budget': budget, 'allowed_sources': sources}),
        evaluator_version=components.evaluator.evaluator_version,
        objective_contract=components.evaluator.objective_contract,
        ppo_engine_contract_id=PPO_ENGINE_CONTRACT_ID, ppo_hyperparameters_sha256=config.sha256,
        credit_estimator_contract_id=CREDIT_ESTIMATOR_CONTRACT_ID, allowed_evaluator_sources=sources)
    engine = PPOEngine(environment=environment, run_contract=contract, credit_estimator=GAECreditEstimator(),
        observation_dimension=spec['observation_dimension'],
        number_of_dianhydride_actions=spec['number_of_dianhydride_actions'],
        number_of_diamine_actions=spec['number_of_diamine_actions'], config=config)
    check_ledger(environment.oracle_ledger(), c)
    require(environment.oracle_ledger()['requested_calls'] == 0, 'initial ledger not empty')
    return components, engine, spec, contract


def check_ledger(ledger, c):
    require(set(ledger['allowed_sources']) == set(METHOD_EVALUATOR_SOURCES[PPO]), 'declared engine source allowlist mismatch')
    require(set(ledger['requested_by_source']) <= {PPO_ON_POLICY}, 'forbidden evaluator source')
    for field, limit in [('requested_calls', 'maximum_requested_evaluator_calls'),
                         ('unique_calls', 'maximum_unique_evaluator_calls')]:
        require(0 <= ledger[field] <= c['budget'][limit], 'evaluator budget exceeded')
    require(ledger.get('invalid_results', 0) == 0, 'invalid evaluator results')


def summarize(result):
    return {
        'batch_id': result.rollout.batch_id, 'rollout_digest': rollout_digest(result.rollout),
        'transition_count': len(result.rollout.transitions),
        'successful_terminal_count': sum('terminal_evaluation' in dict(t.info) for t in result.rollout.transitions),
        'gae_sha256': result.rollout.gae_sha256, 'critic_returns_sha256': result.rollout.critic_returns_sha256,
        'actor_advantages_sha256': result.credit.actor_advantages_sha256,
        'rollout_evaluator_delta': asdict(result.rollout_evaluator_delta),
        'credit_evaluator_delta': asdict(result.credit.evaluator_delta),
        'update_receipt': asdict(result.receipt), 'update_metrics': dict(result.update_metrics),
    }


def check_iteration(summary, c, iteration, policy_hash):
    require(summary['transition_count'] == 128, 'transition budget mismatch')
    steps = summary['update_metrics']['optimizer_steps']
    require(0 < steps <= 8, 'optimizer budget exceeded')
    require(all(math.isfinite(float(x)) for x in summary['update_metrics'].values()), 'nonfinite PPO metrics')
    require(summary['actor_advantages_sha256'] == summary['gae_sha256'], 'actor credit differs from GAE')
    require(summary['update_receipt']['critic_returns_sha256'] == summary['critic_returns_sha256'], 'critic target differs from rollout returns')
    require(summary['credit_evaluator_delta']['requested_calls'] == 0, 'credit issued Oracle query')
    if iteration == 1:
        check = c['initialization_checks']
        require(policy_hash == check['iteration_1_post_ppo_policy_sha256'], 'comparison_ineligible_initial_equivalence_failed: policy')
        require(summary['gae_sha256'] == check['iteration_1_gae_sha256'], 'comparison_ineligible_initial_equivalence_failed: gae')
        require(summary['critic_returns_sha256'] == check['iteration_1_critic_returns_sha256'], 'comparison_ineligible_initial_equivalence_failed: returns')


def rng_digests(engine):
    import torch
    return {
        'engine_numpy': canonical_sha256(engine._rng.bit_generator.state),
        'python': canonical_sha256(random.getstate()),
        'torch_cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        'torch_cuda': [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
                       for x in torch.cuda.get_rng_state_all()],
    }


def expected_authorization(c, source, assets, output, preflight_path):
    return {
        'schema_version': 1, 'authorization_id': 'ppo-only-matched-v1-single-run-20260908',
        'protocol_id': c['protocol_id'], 'protocol_sha256': PROTOCOL_SHA,
        'analysis_sha256': ANALYSIS_SHA, 'implementation_commit': source['commit'],
        'preflight_path': str(preflight_path.resolve()), 'preflight_sha256': sha256_path(preflight_path),
        'polybert_path': assets['polybert']['model_path'],
        'polybert_fingerprint': c['asset_binding']['polybert_fingerprint'],
        'polybert_binding_sha256': c['asset_binding']['polybert_binding_sha256'],
        'afp_path': assets['afp']['asset_path'], 'afp_fingerprint': c['asset_binding']['afp_fingerprint'],
        'afp_binding_sha256': c['asset_binding']['afp_binding_sha256'],
        'output_directory': str(output), 'seed': c['base_seed'], 'iterations': 6,
        'maximum_real_submissions': 1, 'real_execution_authorized': True,
        'external_api_authorized': False, 'automatic_rerun': False, 'automatic_resume': False,
        'sealed_test_access': False, 'formal_training': False,
    }


def consume_authorization(path, expected, job_id):
    require(path.stat().st_mode & 0o777 == 0o600, 'authorization permissions must be 600')
    require(path.stat().st_uid == os.getuid(), 'authorization owner mismatch')
    require(json.loads(path.read_text()) == expected, 'execution authorization mismatch')
    marker = path.with_suffix(path.suffix + '.consumed')
    fd = os.open(str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump({'slurm_job_id': job_id, 'authorization_sha256': sha256_path(path)}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return {'path': str(path), 'sha256': sha256_path(path), 'consumption_path': str(marker)}
