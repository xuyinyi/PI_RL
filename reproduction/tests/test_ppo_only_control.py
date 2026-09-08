import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reproduction.scicf.online.ppo_only_control import (
    PROTOCOL, PROTOCOL_SHA, ANALYSIS, ANALYSIS_SHA, sha256_path,
    OnPolicyOnlyEvaluator, check_iteration, check_ledger, consume_authorization,
)
from reproduction.scicf.online.analyze_ppo_only_control import compare
from reproduction.p2.contracts import PPO, PPO_ON_POLICY, METHOD_EVALUATOR_SOURCES
from reproduction.p2.tests.test_engine import _engine
from reproduction.scicf.online.ppo_only_control import summarize, rng_digests

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / PROTOCOL).read_text())
REFERENCE = json.loads((ROOT / CONFIG['reference']['report_path']).read_text())


def test_frozen_files():
    assert sha256_path(ROOT / PROTOCOL) == PROTOCOL_SHA
    assert sha256_path(ROOT / ANALYSIS) == ANALYSIS_SHA
    base = json.loads((ROOT / CONFIG['reference']['base_protocol_path']).read_text())
    assert CONFIG['ppo'] == base['ppo']


@pytest.mark.parametrize('source', ['evaluation', 'scicf_ppo/factual', 'scicf_ppo/counterfactual', 'unspecified'])
def test_guard_prevents_backend_and_ledger_access(source):
    guard = OnPolicyOnlyEvaluator(SimpleNamespace())
    with pytest.raises(ValueError, match='non-on-policy'):
        guard.evaluate_one('C', source)


def test_guard_preserves_on_policy():
    guard = OnPolicyOnlyEvaluator(SimpleNamespace(evaluate_batch=lambda values, source: [source, values]))
    assert guard.evaluate_batch(['C'], PPO_ON_POLICY) == [PPO_ON_POLICY, ['C']]


def test_once_only_authorization(tmp_path):
    path = tmp_path / 'authorization.json'
    expected = {'run': 1, 'external_api': False}
    path.write_text(json.dumps(expected))
    path.chmod(0o600)
    with pytest.raises(ValueError):
        consume_authorization(path, {'run': 2}, 'mock')
    assert not path.with_suffix('.json.consumed').exists()
    consume_authorization(path, expected, 'mock')
    with pytest.raises(FileExistsError):
        consume_authorization(path, expected, 'mock2')


def test_first_iteration_and_budget_rejections():
    first = REFERENCE['iterations'][0]
    summary = copy.deepcopy(first['standard_ppo'])
    policy = first['policy_hashes']['after_standard_ppo']
    check_iteration(summary, CONFIG, 1, policy)
    with pytest.raises(ValueError, match='equivalence'):
        check_iteration(summary, CONFIG, 1, 'different')
    for field in ('gae_sha256', 'critic_returns_sha256'):
        changed = copy.deepcopy(summary)
        changed[field] = 'different'
        with pytest.raises(ValueError):
            check_iteration(changed, CONFIG, 1, policy)
    for field, value in [('optimizer_steps', 9), ('actor_loss', float('nan'))]:
        changed = copy.deepcopy(summary)
        changed['update_metrics'][field] = value
        with pytest.raises(ValueError):
            check_iteration(changed, CONFIG, 2, policy)


def test_ledger_ceiling_and_source():
    ledger = {'allowed_sources': list(METHOD_EVALUATOR_SOURCES[PPO]),
              'requested_by_source': {PPO_ON_POLICY: 5}, 'requested_calls': 5, 'unique_calls': 5}
    check_ledger(ledger, CONFIG)
    with pytest.raises(ValueError):
        check_ledger(dict(ledger, requested_calls=1249), CONFIG)
    with pytest.raises(ValueError):
        check_ledger(dict(ledger, requested_by_source={'evaluation': 1}), CONFIG)


def test_analysis_denominators_and_partial_failure():
    control = copy.deepcopy(REFERENCE)
    control['first_iteration_equivalence_passed'] = True
    result = compare(REFERENCE, control)
    assert result['primary_endpoint']['denominator_per_arm'] == 768
    assert result['primary_endpoint']['count_difference_scicf_minus_ppo'] == 0
    assert result['iteration_2_to_6_secondary']['denominator_per_arm'] == 640
    control['iterations'][5]['standard_ppo']['successful_terminal_count'] -= 1
    assert compare(REFERENCE, control)['primary_endpoint']['count_difference_scicf_minus_ppo'] == 1
    control['iterations'].pop()
    assert compare(REFERENCE, control)['primary_endpoint'] is None
    control['execution_status'] = 'failed'
    assert not compare(REFERENCE, control)['eligible_complete_comparison']


def test_fake_ppo_checkpoint_rng_and_gae(tmp_path):
    engine = _engine(method=PPO, seed=20260907)
    # All execution here is synthetic and runs only in the server Slurm suite.
    first = summarize(engine.run_iteration(query_requested_calls=0))
    assert first['actor_advantages_sha256'] == first['gae_sha256']
    assert first['credit_evaluator_delta']['requested_calls'] == 0
    digest = rng_digests(engine)
    path = tmp_path / 'engine.pt'
    engine.save_checkpoint(path)
    restored = _engine(method=PPO, seed=20260907)
    restored.load_checkpoint(path)
    assert rng_digests(restored) == digest
    assert restored.policy_state_sha256 == engine.policy_state_sha256
