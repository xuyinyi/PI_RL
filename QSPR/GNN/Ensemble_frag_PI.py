import os
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.mol2graph import smiles_2_bigraph
from utils.junctiontree_encoder import JT_SubGraph
from utils.splitter import Splitter
from data.csv_dataset import MoleculeCSVDataset
from data.model_library import load_model, load_optimal_model
from src.dgltools import collate_fraggraphs
from data.dataloading import import_dataset
from utils.count_parameters import count_parameters
from networks.FraGAT import NewFraGATNet
from utils.piplines import PreFetch, evaluate_frag, evaluate_frag_descriptors, evaluate_frag_attention
from utils.Set_Seed_Reproducibility import set_seed


def main(params):
    set_seed(seed=2023)
    for j in range(4):
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

        fragmentation = JT_SubGraph(scheme='MG_plus_reference')
        dataset = MoleculeCSVDataset(df, smiles_2_bigraph, classic_atom_featurizer, classic_bond_featurizer,
                                     classic_mol_featurizer, cache_file_path, load=True
                                     , error_log=error_log_path, fragmentation=fragmentation)

        splitter = Splitter(dataset)
        seed = splitting_seed[j]

        train_set, val_set, test_set, raw_set = splitter.Random_Splitter(seed=seed, frac_train=0.8,
                                                                         frac_val=0.1)

        train_loader = DataLoader(train_set, collate_fn=collate_fraggraphs, batch_size=len(train_set), shuffle=False,
                                  num_workers=0, worker_init_fn=np.random.seed(2023))
        val_loader = DataLoader(val_set, collate_fn=collate_fraggraphs, batch_size=len(val_set), shuffle=False,
                                num_workers=0, worker_init_fn=np.random.seed(2023))
        test_loader = DataLoader(test_set, collate_fn=collate_fraggraphs, batch_size=len(test_set), shuffle=False,
                                 num_workers=0, worker_init_fn=np.random.seed(2023))
        raw_loader = DataLoader(raw_set, collate_fn=collate_fraggraphs, batch_size=len(raw_set), shuffle=False,
                                num_workers=0, worker_init_fn=np.random.seed(2023))

        fetched_data = PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=1)

        path = '/Ensembles/' + f'{params["Dataset"]}_FraGAT'
        name = '{}_{}_{}'.format('Ensemble', params['Dataset'], 'FraGAT')

        for i in range(100):
            name_idx = name + '_' + str(i)
            params, net_params, model = load_model(path, name_idx)
            n_param = count_parameters(model)
            _, _, train_predict, train_target, train_smiles = evaluate_frag(model, scaling, fetched_data.train_iter,
                                                                            fetched_data.train_batched_origin_graph_list,
                                                                            fetched_data.train_batched_frag_graph_list,
                                                                            fetched_data.train_batched_motif_graph_list,
                                                                            fetched_data.train_targets_list,
                                                                            fetched_data.train_smiles_list, n_param,
                                                                            flag=True)
            _, _, val_predict, val_target, val_smiles = evaluate_frag(model, scaling, fetched_data.val_iter,
                                                                      fetched_data.val_batched_origin_graph_list,
                                                                      fetched_data.val_batched_frag_graph_list,
                                                                      fetched_data.val_batched_motif_graph_list,
                                                                      fetched_data.val_targets_list,
                                                                      fetched_data.val_smiles_list, n_param, flag=True)
            _, _, test_predict, test_target, test_smiles = evaluate_frag(model, scaling, fetched_data.test_iter,
                                                                         fetched_data.test_batched_origin_graph_list,
                                                                         fetched_data.test_batched_frag_graph_list,
                                                                         fetched_data.test_batched_motif_graph_list,
                                                                         fetched_data.test_targets_list,
                                                                         fetched_data.test_smiles_list, n_param,
                                                                         flag=True)
            _, _, raw_predict, raw_target, raw_smiles = evaluate_frag(model, scaling, fetched_data.all_iter,
                                                                      fetched_data.all_batched_origin_graph_list,
                                                                      fetched_data.all_batched_frag_graph_list,
                                                                      fetched_data.all_batched_motif_graph_list,
                                                                      fetched_data.all_targets_list,
                                                                      fetched_data.all_smiles_list, n_param,
                                                                      flag=True)
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
                df_raw = pd.DataFrame(
                    {'SMILES': raw_smiles[0], 'Target': raw_target.numpy().flatten().tolist(),
                     'Predict_' + str(i): raw_predict.numpy().flatten().tolist()})
            else:
                df_train['Predict_' + str(i)] = train_predict.numpy().flatten().tolist()
                df_val['Predict_' + str(i)] = val_predict.numpy().flatten().tolist()
                df_test['Predict_' + str(i)] = test_predict.numpy().flatten().tolist()
                df_raw['Predict_' + str(i)] = raw_predict.numpy().flatten().tolist()

        df_results = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True, sort=False)

        op_idx, init_seed, seed, params, net_params, model = load_optimal_model(path, name)
        _, all_descriptors = evaluate_frag_descriptors(model, fetched_data.all_iter,
                                                       fetched_data.all_batched_origin_graph_list,
                                                       fetched_data.all_batched_frag_graph_list,
                                                       fetched_data.all_batched_motif_graph_list,
                                                       fetched_data.all_smiles_list)
        _, all_attentions = evaluate_frag_attention(model, scaling, fetched_data.all_iter,
                                                    fetched_data.all_batched_origin_graph_list,
                                                    fetched_data.all_batched_frag_graph_list,
                                                    fetched_data.all_batched_motif_graph_list)
        all_attentions = np.array(all_attentions, dtype=object)
        df_descriptors = pd.DataFrame(all_attentions.detach().to(device='cpu').numpy())

        df_merge = pd.concat([df_raw, df_descriptors], axis=1)

        save_file_path = os.path.join('./library/' + path,
                                      f'{name}_PI_{seed}_OP{op_idx}_{time.strftime("%Y-%m-%d-%H-%M")}.csv')
        _save_file_path = os.path.join('./library/' + path, f'{name}_PI_attentions_OP{op_idx}.csv')
        _attention_cache_file_path = cache_file_path + '_attentions.npy'
        df_results.to_csv(save_file_path, index=False)
        df_merge.to_csv(_save_file_path, index=False)
        np.save(_attention_cache_file_path, all_attentions)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    params, net_params = {}, {}
    dataset_list = ['transmittance(400)', 'cte', 'strength', 'tg']
    splitting_seed = [825, 854, 331, 525]  # AFP
    main(params)
