import os
import copy
import random
import gym
import pandas as pd
from pathlib import Path
from gym.utils import seeding
from numpy import ndarray
from rdkit.Chem import Draw
from config import get_default_config
from RL_PPO.moldr.utils import BRICSBuild
from RL_PPO.utils.polyBERT import Embedding_smiles
from RL_PPO.utils.genPI import generate_PI
from RL_PPO.utils.chemutils import *
from RL_PPO.GNN.benchmarks import Benchmark


class PIEnvValueMax(gym.Env):
    def __init__(self, env_config):
        self.action_space_dianhydride = env_config["ACTION_SPACE_DIANHYDRIDE"]
        self.action_space_diamine = env_config["ACTION_SPACE_DIAMINE"]
        self.action_space = gym.spaces.Tuple([self.action_space_dianhydride, self.action_space_diamine])
        self.observation_space = env_config["OBS_SPACE"]
        self.building_blocks_dianhydride = env_config["BUILDING_BLOCKS_DIANHYDRIDE"]
        self.building_blocks_diamine = env_config["BUILDING_BLOCKS_DIAMINE"]
        self.scoring_function = env_config["SCORE_FUNCTION"]
        self.length = env_config["LENGTH"]
        self.step_length = env_config["STEP_LENGTH"]
        self._base_smiles_dianhydride = env_config["BASE_SMILES_DIANHYDRIDE"]
        self._base_smiles_diamine = env_config["BASE_SMILES_DIAMINE"]
        self.model = env_config["MODEL_PATH"]
        self.dianhydride_pattern = Chem.MolFromSmarts("[#8]=[#6]1[#6][#6][#6](=[#8])[#8]1")
        self.diamine_pattern = Chem.MolFromSmarts("[#7H2]")

        self.env_step = 0
        # self.action_mask = np.ones(len(self.building_blocks))
        self.flag_dianhydride = False
        self.flag_diamine = False
        self.PI = 'None'
        self.transmittance = 0.0
        self.cte = 0.0
        self.strength = 0.0
        self.tg = 0.0
        self.SaScore = 0.0

    def reset(self):
        self.base_smiles_dianhydride = copy.deepcopy(self._base_smiles_dianhydride)
        self.base_smiles_diamine = copy.deepcopy(self._base_smiles_diamine)
        self.env_step = 0
        self.flag_dianhydride = False
        self.flag_diamine = False
        self.PI = 'None'
        self.transmittance = 0.0
        self.cte = 0.0
        self.strength = 0.0
        self.tg = 0.0
        self.SaScore = 0.0
        vec_dianhydride = Embedding_smiles(self.model, clean_smiles(self.base_smiles_dianhydride))
        vec_diamine = Embedding_smiles(self.model, clean_smiles(self.base_smiles_diamine))
        vec = np.concatenate((vec_dianhydride, vec_diamine))
        return vec

    def render(self, mode="human"):
        mol = self.base_mol
        smiles = get_smiles(mol)
        reward = float(self.compute_score([smiles]))
        mol.SetProp("score", str(reward))
        return Draw.MolsToGridImage([mol], subImgSize=(300, 300), legends=[mol.GetProp("score")])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reassemble(self, action):
        """
        It will be executed only when self.flag_dianhydride = False or self.flag_diamine = False
        """
        action_dianhydride, action_diamine = action

        if self.flag_dianhydride:
            smiles_dianhydride = [self.base_smiles_dianhydride]
        else:
            attached_block_dianhydride = self.building_blocks_dianhydride[action_dianhydride]
            if "*" not in attached_block_dianhydride:
                smiles_dianhydride = [attached_block_dianhydride]
            else:
                gen_mols_dianhydride = list(
                    BRICSBuild([get_mol(self.base_smiles_dianhydride), get_mol(attached_block_dianhydride)],
                               onlyCompleteMols=False, scrambleReagents=False, maxDepth=0))
                smiles_dianhydride = [get_smiles(mol) for mol in gen_mols_dianhydride]

        if self.flag_diamine:
            smiles_diamine = [self.base_smiles_diamine]
        else:
            attached_block_diamine = self.building_blocks_diamine[action_diamine]
            if "*" not in attached_block_diamine:
                smiles_diamine = [attached_block_diamine]
            else:
                gen_mols_diamine = list(BRICSBuild([get_mol(self.base_smiles_diamine), get_mol(attached_block_diamine)],
                                                   onlyCompleteMols=False, scrambleReagents=False, maxDepth=0))
                smiles_diamine = [get_smiles(mol) for mol in gen_mols_diamine]

        return smiles_dianhydride, smiles_diamine

    def step(self, action: Optional[Tuple[List[int], List[int]]] = None) -> Tuple[ndarray, float, bool, dict]:
        self.env_step += 1
        self.prev_smiles_dianhydride = self.base_smiles_dianhydride
        self.prev_smiles_diamine = self.base_smiles_diamine
        self.prev_action = action

        smiles_dianhydride, smiles_diamine = self.reassemble(action)
        infos = {
            "gen_smiles_dianhydride": smiles_dianhydride,
            "gen_smiles_diamine": smiles_diamine,
            "prev_action": self.prev_action
        }
        if len(smiles_dianhydride) == 0 or len(smiles_diamine) == 0:
            return np.zeros(1200), 0.0, True, infos

        dianhydride, diamine, gen_PI = [], [], []
        for smiles in smiles_dianhydride:
            if "*" not in smiles and len(get_mol(smiles).GetSubstructMatches(self.dianhydride_pattern)) == 2:
                self.flag_dianhydride = True
                dianhydride.append(smiles)
        if not self.flag_dianhydride:
            _smiles_dianhydride = []
            for smiles in smiles_dianhydride:
                if "*" in smiles:
                    _smiles_dianhydride.append(smiles)
            if not _smiles_dianhydride:
                return np.zeros(1200), 0.0, True, infos
            smiles_dianhydride = _smiles_dianhydride

        for smiles in smiles_diamine:
            if "*" not in smiles and len(get_mol(smiles).GetSubstructMatches(self.diamine_pattern)) == 2:
                self.flag_diamine = True
                diamine.append(smiles)
        if not self.flag_diamine:
            _smiles_diamine = []
            for smiles in smiles_diamine:
                if "*" in smiles:
                    _smiles_diamine.append(smiles)
            if not _smiles_diamine:
                return np.zeros(1200), 0.0, True, infos
            smiles_diamine = _smiles_diamine

        if self.flag_dianhydride and self.flag_diamine:
            for idx1, reac1 in enumerate(dianhydride):
                for idx2, reac2 in enumerate(diamine):
                    try:
                        gen_PI.append((generate_PI(reac1, reac2), idx1, idx2))
                    except:
                        pass
            if not gen_PI:
                return np.zeros(1200), 0.0, True, infos

            print(gen_PI)
            transmittances, ctes, strengths, tgs, SaScores, rewards = self.compute_score([PI[0] for PI in gen_PI])

            # TODO: SELECT NEW NODE
            idx = np.argmax(rewards)
            self.PI = gen_PI[idx][0]
            self.transmittance = transmittances[idx]
            self.cte = ctes[idx]
            self.strength = strengths[idx]
            self.tg = tgs[idx]
            self.SaScore = SaScores[idx]
            dianhydride_max = dianhydride[gen_PI[idx][1]]
            diamine_max = diamine[gen_PI[idx][2]]
            obs_dianhydride = Embedding_smiles(self.model, dianhydride_max)
            obs_diamine = Embedding_smiles(self.model, diamine_max)
            obs = np.concatenate((obs_dianhydride, obs_diamine))
            reward = float(rewards[idx])

            return obs, reward, True, infos

        # If dianhydride or diamine has not been generated.
        else:
            if self.flag_dianhydride and not self.flag_diamine:
                self.base_smiles_dianhydride = np.random.choice(dianhydride)
                self.base_smiles_diamine = np.random.choice(smiles_diamine)
                obs_dianhydride = Embedding_smiles(self.model, self.base_smiles_dianhydride)
                obs_diamine = Embedding_smiles(self.model, clean_smiles(self.base_smiles_diamine))
                obs = np.concatenate((obs_dianhydride, obs_diamine))
            elif not self.flag_dianhydride and self.flag_diamine:
                self.base_smiles_dianhydride = np.random.choice(smiles_dianhydride)
                self.base_smiles_diamine = np.random.choice(diamine)
                obs_dianhydride = Embedding_smiles(self.model, clean_smiles(self.base_smiles_dianhydride))
                obs_diamine = Embedding_smiles(self.model, self.base_smiles_diamine)
                obs = np.concatenate((obs_dianhydride, obs_diamine))
            elif not self.flag_dianhydride and not self.flag_diamine:
                self.base_smiles_dianhydride = np.random.choice(smiles_dianhydride)
                self.base_smiles_diamine = np.random.choice(smiles_diamine)
                obs_dianhydride = Embedding_smiles(self.model, clean_smiles(self.base_smiles_dianhydride))
                obs_diamine = Embedding_smiles(self.model, clean_smiles(self.base_smiles_diamine))
                obs = np.concatenate((obs_dianhydride, obs_diamine))
            reward = 0.0
            done = self.is_done()

            return obs, reward, done, infos

    def compute_score(self, smiles) -> ndarray:
        calculate = [self.scoring_function(s) for s in smiles]
        transmittances = np.array([c.transmittance for c in calculate])
        ctes = np.array([c.cte for c in calculate])
        strengths = np.array([c.strength for c in calculate])
        tgs = np.array([c.tg for c in calculate])
        SaScores = np.array([c.ScoreSA for c in calculate])
        scores = np.array([c.Score for c in calculate])
        return transmittances, ctes, strengths, tgs, SaScores, scores

    def is_done(self):
        self.base_mol_dianhydride = get_mol(self.base_smiles_dianhydride)
        self.base_mol_diamine = get_mol(self.base_smiles_diamine)
        if len(self.base_mol_dianhydride.GetAtoms()) + len(self.base_mol_diamine.GetAtoms()) > self.length:
            return True
        if self.env_step > self.step_length:
            return True
        else:
            return False

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = copy.deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return obs.flatten()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    block_dianhydride_path = os.getcwd().strip('moldr') + f"outputs/building_blocks/blocks_dianhydride.csv"
    block_diamine_path = os.getcwd().strip('moldr') + f"outputs/building_blocks/blocks_diamine.csv"
    building_blocks_dianhydride = pd.read_csv(block_dianhydride_path)["block"].values.tolist()
    building_blocks_diamine = pd.read_csv(block_diamine_path)["block"].values.tolist()

    config = get_default_config(
        PIEnvValueMax,
        Benchmark,
        building_blocks_dianhydride,
        building_blocks_diamine,
        model_path=Path(__file__).resolve().parent.parent / "models",
        num_workers=15,
        num_gpus=4,
        length=60,
        step_length=5,
    )
    env = PIEnvValueMax(config)

    save_path = os.getcwd().strip('moldr') + f"outputs/PPO/random/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    PI, A_1, A_2, transmittance, cte, strength, tg, SaScore, reward_l = \
        list(), list(), list(), list(), list(), list(), list(), list(), list()
    for j in range(10000):
        action_1, action_2 = list(), list()
        obs = env.reset()
        while True:
            action = (
                random.choice(range(len(building_blocks_dianhydride))),
                random.choice(range(len(building_blocks_diamine))))
            obs, reward, done, info = env.step(action)
            action_1.append(info["prev_action"][0])
            action_2.append(info["prev_action"][1])
            if done:
                PI.append(env.PI)
                A_1.append(','.join(map(str, action_1)))
                A_2.append(','.join(map(str, action_2)))
                transmittance.append(env.transmittance)
                cte.append(env.cte)
                strength.append(env.strength)
                tg.append(env.tg)
                SaScore.append(env.SaScore)
                reward_l.append(reward)
                break

    pd.DataFrame({
        'PI': PI,
        'A_1': A_1,
        'A_2': A_2,
        'transmittance': transmittance,
        'cte': cte,
        'strength': strength,
        'tg': tg,
        'SaScore': SaScore,
        'reward': reward_l
    }).to_csv(os.path.join(save_path, f'generate_dark_random.csv'), index=False)
