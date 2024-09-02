import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.dataloading import import_dataset
from data.csv_dataset import MoleculeCSVDataset
from data.model_library import load_model, load_optimal_model
from utils.splitter import Splitter
from utils.count_parameters import count_parameters
from utils.smile2vec import Embedding_smiles
from utils.piplines import evaluate, PreFetch
from Network import MLPModel


def collate_vectors(samples):
    vectors, targets, smiles = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)
    return vectors, targets, smiles


def main(params):
    for j in range(len(splitting_seed)):
        params.update({'Dataset': dataset_list[j]})
        if dataset_list[j] == 'transmittance(400)':
            params.update({'sigmoid': True})
        else:
            params.update({'sigmoid': False})
        df, scaling = import_dataset(params)
        cache_file_dir = os.path.realpath('./cache')
        if not os.path.exists(cache_file_dir):
            os.mkdir(cache_file_dir)
        cache_file_path = os.path.join(cache_file_dir, params['Dataset'])

        error_path = os.path.realpath('./error_log')
        if not os.path.exists(error_path):
            os.mkdir(error_path)
        error_log_path = os.path.join(error_path, f'{params["Dataset"]}_{time.strftime("%Y-%m-%d-%H-%M")}' + '.csv')
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polybert')
        dataset = MoleculeCSVDataset(df, model_path, Embedding_smiles, cache_file_path, load=True,
                                     error_log=error_log_path)

        splitter = Splitter(dataset)
        seed = splitting_seed[j]

        train_set, val_set, test_set, raw_set = splitter.Random_Splitter(seed=seed, frac_train=0.8, frac_val=0.1)

        train_loader = DataLoader(train_set, collate_fn=collate_vectors, batch_size=len(train_set), shuffle=False,
                                  num_workers=4, worker_init_fn=np.random.seed(2023))
        val_loader = DataLoader(val_set, collate_fn=collate_vectors, batch_size=len(val_set), shuffle=False,
                                num_workers=4, worker_init_fn=np.random.seed(2023))
        test_loader = DataLoader(test_set, collate_fn=collate_vectors, batch_size=len(test_set), shuffle=False,
                                 num_workers=4, worker_init_fn=np.random.seed(2023))
        raw_loader = DataLoader(raw_set, collate_fn=collate_vectors, batch_size=len(raw_set), shuffle=False,
                                num_workers=4, worker_init_fn=np.random.seed(2023))

        fetched_data = PreFetch(train_loader, val_loader, test_loader, raw_loader)

        path = '/Ensembles/' + f'{params["Dataset"]}_polyBERT'
        name = '{}_{}_{}'.format('Ensemble', params['Dataset'], 'polyBERT')

        for i in tqdm(range(100)):
            name_idx = name + '_' + str(i)
            params, net_params, model = load_model(path, name_idx)
            n_param = count_parameters(model)
            _, _, train_predict, train_target, train_smiles = evaluate(model, scaling, fetched_data.train_iter,
                                                                       fetched_data.train_vector_list,
                                                                       fetched_data.train_targets_list,
                                                                       fetched_data.train_smiles_list, flag=True)
            _, _, val_predict, val_target, val_smiles = evaluate(model, scaling, fetched_data.val_iter,
                                                                 fetched_data.val_vector_list,
                                                                 fetched_data.val_targets_list,
                                                                 fetched_data.val_smiles_list, flag=True)
            _, _, test_predict, test_target, test_smiles = evaluate(model, scaling, fetched_data.test_iter,
                                                                    fetched_data.test_vector_list,
                                                                    fetched_data.test_targets_list,
                                                                    fetched_data.test_smiles_list, flag=True)
            if i == 0:
                df_train = pd.DataFrame(
                    {'SMILES': train_smiles[0], 'Tag': 'Train', 'Target': train_target.numpy().flatten().tolist(),
                     'Predict_' + str(i): train_predict.numpy().flatten().tolist()})
                df_val = pd.DataFrame(
                    {'SMILES': val_smiles[0], 'Tag': 'Val', 'Target': val_target.numpy().flatten().tolist(),
                     'Predict_' + str(i): val_predict.numpy().flatten().tolist()})
                df_test = pd.DataFrame(
                    {'SMILES': test_smiles[0], 'Tag': 'Test', 'Target': test_target.numpy().flatten().tolist(),
                     'Predict_' + str(i): test_predict.numpy().flatten().tolist()})
            else:
                df_train['Predict_' + str(i)] = train_predict.numpy().flatten().tolist()
                df_val['Predict_' + str(i)] = val_predict.numpy().flatten().tolist()
                df_test['Predict_' + str(i)] = test_predict.numpy().flatten().tolist()

        df_results = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True, sort=False)

        op_idx, init_seed, seed, params, net_params, model = load_optimal_model(path, name)

        save_file_path = os.path.join('./library/' + path,
                                      f'{name}_PI_descriptors_{seed}_OP{op_idx}_{time.strftime("%Y-%m-%d-%H-%M")}.csv')
        df_results.to_csv(save_file_path, index=False)


if __name__ == "__main__":
    params = {}
    dataset_list = ['transmittance(400)', 'cte', 'strength', 'tg']
    splitting_seed = [825, 854, 331, 525]  # AFP
    main(params)
