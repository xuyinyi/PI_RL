import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.mol2graph import smiles_2_bigraph
from utils.splitter import Splitter
from data.csv_dataset import MoleculeCSVDataset
from src.dgltools import collate_molgraphs
from data.dataloading import import_dataset
from utils.count_parameters import count_parameters
from networks.AttentiveFP import AttentiveFPNet
from utils.piplines import evaluate, PreFetch, evaluate_descriptors
from data.model_library import load_model, load_optimal_model


def main(params):
    for j in range(len(splitting_seed)):
        params.update({'Dataset': dataset_list[j]})
        df, scaling = import_dataset(params)
        cache_file_dir = os.path.realpath('./cache')
        if not os.path.exists(cache_file_dir):
            os.mkdir(cache_file_dir)
        cache_file_path = os.path.join(cache_file_dir, params['Dataset'])

        error_path = os.path.realpath('./error_log')
        if not os.path.exists(error_path):
            os.mkdir(error_path)
        error_log_path = os.path.join(error_path, f'{params["Dataset"]}_{time.strftime("%Y-%m-%d-%H-%M")}' + '.csv')

        dataset = MoleculeCSVDataset(df, smiles_2_bigraph, classic_atom_featurizer, classic_bond_featurizer,
                                     classic_mol_featurizer, cache_file_path, load=True, error_log=error_log_path)

        splitter = Splitter(dataset)
        seed = splitting_seed[j]

        train_set, val_set, test_set, raw_set = splitter.Random_Splitter(seed=seed, frac_train=0.8, frac_val=0.1)

        train_loader = DataLoader(train_set, collate_fn=collate_molgraphs, batch_size=len(train_set), shuffle=False,
                                  num_workers=0, worker_init_fn=np.random.seed(2023))
        val_loader = DataLoader(val_set, collate_fn=collate_molgraphs, batch_size=len(val_set), shuffle=False,
                                num_workers=0, worker_init_fn=np.random.seed(2023))
        test_loader = DataLoader(test_set, collate_fn=collate_molgraphs, batch_size=len(test_set), shuffle=False,
                                 num_workers=0, worker_init_fn=np.random.seed(2023))
        raw_loader = DataLoader(raw_set, collate_fn=collate_molgraphs, batch_size=len(raw_set), shuffle=False,
                                num_workers=0, worker_init_fn=np.random.seed(2023))

        fetched_data = PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=False)

        path = '/Ensembles/' + f'{params["Dataset"]}_AFP'
        name = '{}_{}_{}'.format('Ensemble', params['Dataset'], 'AFP')

        for i in tqdm(range(100)):
            name_idx = name + '_' + str(i)
            params, net_params, model = load_model(path, name_idx)
            n_param = count_parameters(model)
            _, _, train_predict, train_target, train_smiles = evaluate(model, scaling, fetched_data.train_iter,
                                                                       fetched_data.train_batched_origin_graph_list,
                                                                       fetched_data.train_targets_list,
                                                                       fetched_data.train_smiles_list, n_param,
                                                                       flag=True)
            _, _, val_predict, val_target, val_smiles = evaluate(model, scaling, fetched_data.val_iter,
                                                                 fetched_data.val_batched_origin_graph_list,
                                                                 fetched_data.val_targets_list,
                                                                 fetched_data.val_smiles_list, n_param, flag=True)
            _, _, test_predict, test_target, test_smiles = evaluate(model, scaling, fetched_data.test_iter,
                                                                    fetched_data.test_batched_origin_graph_list,
                                                                    fetched_data.test_targets_list,
                                                                    fetched_data.test_smiles_list, n_param, flag=True)
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
        all_smiles, all_descriptors = evaluate_descriptors(model, scaling, fetched_data.all_iter,
                                                           fetched_data.all_batched_origin_graph_list,
                                                           fetched_data.all_targets_list, fetched_data.all_smiles_list,
                                                           n_param)

        df_descriptors = pd.DataFrame(all_descriptors.detach().to(device='cpu').numpy())
        df_descriptors['SMILES'] = all_smiles[0]

        df_merge = pd.merge(df_results, df_descriptors, how='outer', on='SMILES')

        save_file_path = os.path.join('./library/' + path,
                                      f'{name}_PI_descriptors_{seed}_OP{op_idx}_{time.strftime("%Y-%m-%d-%H-%M")}.csv')
        df_merge.to_csv(save_file_path, index=False)


if __name__ == "__main__":
    params, net_params = {"sigmoid": True}, {}
    dataset_list = ['transmittance(400)']
    splitting_seed = [825]  # AFP
    main(params)
