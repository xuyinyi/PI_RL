from RL_PPO.envs.chemistry import LegacyDAPiGenChemistryBackend
from RL_PPO.envs.factory import load_block_catalog


def test_catalog_loader_removes_canonical_duplicates_and_invalid_rows(tmp_path):
    path = tmp_path / "blocks.csv"
    path.write_text("block,count\nCC,1\nCC,2\n,3\nCCC,1\n")
    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=True)
    values, report = load_block_catalog(
        str(path), chemistry=chemistry, deduplicate=True
    )
    assert values == ["CC", "CCC"]
    assert report.raw_rows == 4
    assert report.canonical_duplicates_removed == 1
    assert report.invalid_rows_removed == 1


def test_action_catalog_writer_includes_explicit_noop(tmp_path):
    import numpy as np

    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.core import BranchableDAPiGenCore
    from RL_PPO.envs.factory import CatalogReport, Stage0Components, write_action_catalogs

    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=True)

    class Encoder(object):
        encoder_version = "factory-test"

        def __call__(self, smiles):
            return np.asarray([len(smiles)], dtype=np.float32)

    d = "[16*]c1ccc2c(c1)C(=O)OC2=O"
    a = "[16*]c1ccc(N)cc1"
    core = BranchableDAPiGenCore(
        (d,),
        (a,),
        d,
        a,
        chemistry,
        Encoder(),
        DAPiGenEnvConfig(max_steps=2, mask_mode="compatibility"),
    )
    reports = (
        CatalogReport("d.csv", 1, 1, 0, 0, (1,), (core.dianhydride_blocks[0],)),
        CatalogReport("a.csv", 1, 1, 0, 0, (1,), (core.diamine_blocks[0],)),
    )
    components = Stage0Components(core, object(), object(), reports)
    write_action_catalogs(str(tmp_path), components)
    d_text = (tmp_path / "dianhydride_actions.csv").read_text()
    a_text = (tmp_path / "diamine_actions.csv").read_text()
    assert ",noop," in d_text
    assert ",noop," in a_text


def test_manifest_separates_task_identity_from_budget_identity():
    import numpy as np

    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.core import BranchableDAPiGenCore
    from RL_PPO.envs.evaluator import (
        BudgetedCachingTerminalEvaluator,
        TerminalRewardAdapter,
    )
    from RL_PPO.envs.factory import CatalogReport, Stage0Components, environment_manifest
    from RL_PPO.envs.types import TerminalEvaluation

    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=True)

    class Encoder(object):
        encoder_version = "manifest-test-encoder"

        def __call__(self, smiles):
            return np.asarray([len(smiles)], dtype=np.float32)

    class Evaluator(object):
        evaluator_version = "manifest-test-evaluator"

        def evaluate_batch(self, items):
            return [
                TerminalEvaluation(
                    objective=0.1,
                    canonical_smiles=item,
                    evaluator_version=self.evaluator_version,
                )
                for item in items
            ]

    d = "[16*]c1ccc2c(c1)C(=O)OC2=O"
    a = "[16*]c1ccc(N)cc1"
    core = BranchableDAPiGenCore(
        (d,),
        (a,),
        d,
        a,
        chemistry,
        Encoder(),
        DAPiGenEnvConfig(max_steps=2, mask_mode="compatibility"),
    )
    reports = (
        CatalogReport("d.csv", 1, 1, 0, 0, (1,), (core.dianhydride_blocks[0],)),
        CatalogReport("a.csv", 1, 1, 0, 0, (1,), (core.diamine_blocks[0],)),
    )

    def build(budget):
        evaluator = BudgetedCachingTerminalEvaluator(
            Evaluator(), maximum_requested_calls=budget, cache_scope="unit"
        )
        adapter = TerminalRewardAdapter(evaluator)
        return environment_manifest(Stage0Components(core, evaluator, adapter, reports))

    small = build(10)
    large = build(20)
    assert small["task_contract_id"] == large["task_contract_id"]
    assert small["budget_contract_id"] != large["budget_contract_id"]


def test_task_contract_changes_when_critical_dapigen_source_changes():
    import numpy as np

    from RL_PPO.envs.config import DAPiGenEnvConfig
    from RL_PPO.envs.core import BranchableDAPiGenCore
    from RL_PPO.envs.evaluator import (
        BudgetedCachingTerminalEvaluator,
        TerminalRewardAdapter,
    )
    from RL_PPO.envs.factory import CatalogReport, Stage0Components, environment_manifest
    from RL_PPO.envs.types import TerminalEvaluation

    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=True)

    class Encoder(object):
        encoder_version = "source-contract-test"

        def __call__(self, smiles):
            return np.asarray([len(smiles)], dtype=np.float32)

    class Evaluator(object):
        evaluator_version = "source-contract-evaluator"

        def evaluate_batch(self, items):
            return [
                TerminalEvaluation(
                    objective=0.1,
                    canonical_smiles=item,
                    evaluator_version=self.evaluator_version,
                )
                for item in items
            ]

    d = "[16*]c1ccc2c(c1)C(=O)OC2=O"
    a = "[16*]c1ccc(N)cc1"
    core = BranchableDAPiGenCore(
        (d,),
        (a,),
        d,
        a,
        chemistry,
        Encoder(),
        DAPiGenEnvConfig(max_steps=2, mask_mode="compatibility"),
    )
    reports = (
        CatalogReport("d.csv", 1, 1, 0, 0, (1,), (core.dianhydride_blocks[0],)),
        CatalogReport("a.csv", 1, 1, 0, 0, (1,), (core.diamine_blocks[0],)),
    )

    def manifest(source_hash):
        evaluator = BudgetedCachingTerminalEvaluator(Evaluator())
        adapter = TerminalRewardAdapter(evaluator)
        components = Stage0Components(
            core,
            evaluator,
            adapter,
            reports,
            repository_metadata={
                "dapigen_git_commit": "same-commit",
                "critical_source_sha256": {"RL_PPO/moldr/utils.py": source_hash},
            },
        )
        return environment_manifest(components)

    first = manifest("aaa")
    second = manifest("bbb")
    assert first["task_contract_id"] != second["task_contract_id"]
