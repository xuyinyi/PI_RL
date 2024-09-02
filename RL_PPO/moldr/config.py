import gym
import numpy as np
from RL_PPO.utils.chemutils import get_mol
from ray.rllib.agents.ppo import ppo


def get_default_config(
        env,
        sc,
        building_blocks_dianhydride,
        building_blocks_diamine,
        model_path,
        base_smiles_dianhydride="[16*]c1ccc2c(c1)C(=O)OC2=O",
        base_smiles_diamine="[16*]c1ccc(N)cc1",
        num_workers=4,
        # num_gpus=1,
        length=50,
        step_length=10,
):
    def check_valid_smiles(smiles):
        mol = get_mol(smiles)
        if mol is None:
            raise ValueError("INVALID SMILES.")
        else:
            return smiles

    base_smiles_dianhydride = check_valid_smiles(base_smiles_dianhydride)
    base_smiles_diamine = check_valid_smiles(base_smiles_diamine)
    high = np.array([np.finfo(np.float32).max for _ in range(1200)])
    observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
    scoring_function = sc

    config = ppo.DEFAULT_CONFIG
    config.update(
        {
            "env": env,
            "ACTION_SPACE_DIANHYDRIDE": gym.spaces.Discrete(len(building_blocks_dianhydride)),
            "ACTION_SPACE_DIAMINE": gym.spaces.Discrete(len(building_blocks_diamine)),
            "OBS_SPACE": observation_space,
            "BUILDING_BLOCKS_DIANHYDRIDE": building_blocks_dianhydride,
            "BUILDING_BLOCKS_DIAMINE": building_blocks_diamine,
            "SCORE_FUNCTION": scoring_function,
            "BASE_SMILES_DIANHYDRIDE": base_smiles_dianhydride,  # Starting point of dianhydride
            "BASE_SMILES_DIAMINE": base_smiles_diamine,  # Starting point of dianhydride
            "MODEL_PATH": model_path,  # polyBERT PATH
            "LENGTH": length,  # Max nodes of molecules
            "STEP_LENGTH": step_length,  # Step size
            "model": {
                "fcnet_hiddens": [256, 128, 128],  # [256, 128] # old version
                "fcnet_activation": "relu",
                "max_seq_len": 100,
            },
            "framework": "torch",
            # Set up a separate evaluation worker set for the
            # `trainer.evaluate()` call after training (see below).
            "num_workers": num_workers,
            # "num_gpus": num_gpus,
            # "num_gpus_per_worker": 0.25,
            "train_batch_size": 500,
            # "sgd_minibatch_size": 20,
            #              "use_lstm": True,
            #             # Max seq len for training the LSTM, defaults to 20.
            #             "max_seq_len": 20,
            #             # Size of the LSTM cell.
            #             "lstm_cell_size": 256,
            #             # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            #             "lstm_use_prev_action": True,
            #             # Whether to feed r_{t-1} to LSTM.
            #             "lstm_use_prev_reward": False,
            #             # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
            #             "_time_major": False,
            # "callbacks": MolEnvCallbacks,
            "evaluation_num_workers": 1,
            # Only for evaluation runs, render the env.
            "evaluation_config": {
                "render_env": False,
            },
        }
    )
    return config
