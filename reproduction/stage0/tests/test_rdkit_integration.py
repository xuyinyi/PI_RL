import numpy as np

from RL_PPO.envs.chemistry import LegacyDAPiGenChemistryBackend
from RL_PPO.envs.config import DAPiGenEnvConfig
from RL_PPO.envs.core import BranchableDAPiGenCore
from RL_PPO.envs.types import DAPiGenAction


class TinyEncoder(object):
    encoder_version = "tiny-rdkit-test"

    def __call__(self, smiles):
        return np.asarray([len(smiles), smiles.count("*")], dtype=np.float32)


def test_real_rdkit_brics_and_pi_reaction_form_a_terminal_design():
    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=True)
    d_fragment = "[16*]c1ccc2c(c1)C(=O)OC2=O"
    a_fragment = "[16*]c1ccc(N)cc1"
    core = BranchableDAPiGenCore(
        dianhydride_blocks=(d_fragment,),
        diamine_blocks=(a_fragment,),
        initial_dianhydride_smiles=d_fragment,
        initial_diamine_smiles=a_fragment,
        chemistry=chemistry,
        encoder=TinyEncoder(),
        config=DAPiGenEnvConfig(
            max_atoms=100,
            max_steps=2,
            mask_mode="exact_cached",
            complete_block_policy="never",
            invalid_action_handling="raise",
        ),
    )
    initial = core.initial(0)
    terminal = core.transition(initial.state, DAPiGenAction(0, 0), seed=20260903)
    assert terminal.terminated
    assert terminal.state.termination_reason == "success"
    assert terminal.terminal_smiles is not None
    assert "*" not in terminal.terminal_smiles
    assert chemistry.is_complete_dianhydride(terminal.state.dianhydride_smiles)
    assert chemistry.is_complete_diamine(terminal.state.diamine_smiles)


def test_targeted_single_attachment_closure_matches_full_legacy_brics():
    chemistry = LegacyDAPiGenChemistryBackend(allow_rdkit_brics_fallback=True)
    panels = (
        "[16*]c1ccc2c(c1)C(=O)OC2=O",
        "[16*]c1ccc(N)cc1",
    )
    for fragment in panels:
        targeted = chemistry.assemble_closure_candidates(fragment, fragment)
        full = chemistry.assemble_candidates(fragment, fragment)
        assert targeted == full
        assert len(targeted) == 1
