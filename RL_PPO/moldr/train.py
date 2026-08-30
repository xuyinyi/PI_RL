import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from datetime import datetime
from ray.rllib.agents.ppo import PPOTrainer
from tqdm import tqdm
from RL_PPO.moldr.env import PIEnvValueMax
from RL_PPO.moldr.config import get_default_config
from RL_PPO.moldr.utils import save, custom_log_creator
from RL_PPO.GNN.benchmarks import Benchmark
from reproduction.check_assets import require_assets


def main():
    repo_root = Path(__file__).resolve().parents[2]
    rl_root = Path(__file__).resolve().parents[1]
    require_assets(repo_root)

    model_path = rl_root / "models"
    block_dianhydride_path = rl_root / "outputs/building_blocks/blocks_dianhydride.csv"
    block_diamine_path = rl_root / "outputs/building_blocks/blocks_diamine.csv"
    building_blocks_dianhydride = pd.read_csv(block_dianhydride_path)["block"].values.tolist()
    building_blocks_diamine = pd.read_csv(block_diamine_path)["block"].values.tolist()

    timeLog = pd.DataFrame(data=None, columns=['Epoch', 'RunTime'])
    _time = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    config = get_default_config(
        PIEnvValueMax,
        Benchmark,
        building_blocks_dianhydride,
        building_blocks_diamine,
        model_path=model_path,
        num_workers=15,
        num_gpus=4,
        length=60,
        step_length=5,
    )
    save_path = rl_root / f"outputs/PPO/models/{_time}/config.pkl"
    save(save_path, config)
    custom_path = rl_root / "ray_results/PPO_PIEnvValueMax"
    trainer = PPOTrainer(
        env=PIEnvValueMax,
        config=config,
        logger_creator=custom_log_creator(custom_path, "PI"),
    )

    for i in tqdm(range(1, 101)):
        startTime = time.time()
        trainer.train()

        print(f'epoch_{i} has completed.')
        if i % 10 == 0:
            save_path = rl_root / f"outputs/PPO/models/{_time}/epoch_{i}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            policy = trainer.get_policy()
            trainer.save(save_path)
            env = PIEnvValueMax(config["env_config"])
            PI, A_1, A_2, transmittance, cte, strength, tg, SaScore, reward_l = \
                list(), list(), list(), list(), list(), list(), list(), list(), list()
            for j in range(10000):
                action_1, action_2 = list(), list()
                obs = env.reset()
                while True:
                    action = policy.compute_single_action(obs)[0]
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
            }).to_csv(os.path.join(save_path, f'generate_{i}.csv'), index=False)

        endTime = time.time()
        runTime = endTime - startTime
        log_temp = pd.DataFrame(data=np.array([i, runTime]).reshape(1, -1), columns=['Epoch', 'RunTime'])
        timeLog = timeLog.append(log_temp, ignore_index=True)
        timeLog.to_csv(rl_root / f"outputs/PPO/models/{_time}/log.csv", index=False)

    ray.shutdown()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    main()
