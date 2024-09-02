import os
import re
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from QSPR.GNN.src.feature.atom_featurizer import classic_atom_featurizer
from QSPR.GNN.src.feature.bond_featurizer import classic_bond_featurizer
from QSPR.GNN.src.feature.mol_featurizer import classic_mol_featurizer
from QSPR.GNN.utils.mol2graph import smiles_2_bigraph
from QSPR.GNN.utils.junctiontree_encoder import JT_SubGraph
from load_data import MoleculeCSVDataset
from QSPR.GNN.src.dgltools import collate_fraggraphs_pre
from QSPR.GNN.utils.count_parameters import count_parameters
from QSPR.GNN.networks.FraGAT import NewFraGATNet as FraGAT
from QSPR.GNN.utils.piplines import PreFetch_frag, evaluate_frag_pre, evaluate_frag_descriptors, evaluate_frag_attention
from QSPR.GNN.utils.Set_Seed_Reproducibility import set_seed


def load_model(path, name_idx):
    lib_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/library" + path
    model_file_path = os.path.join(lib_file_path, name_idx + '.pt')

    directory, filename = os.path.split(model_file_path)
    idx = int(re.findall('_(\d+).pt', filename)[0])
    name = filename.split(f"_{idx}")[0]
    setting_file_path = os.path.join(lib_file_path, name + '_settings' + '.csv')
    df = pd.read_csv(setting_file_path, sep=',', encoding='windows-1250', index_col=-1)
    params = {}
    net_params = {}
    for _, item in enumerate(df.columns):
        if ':' in item:
            if item.split(':')[0] == 'param':
                params[item.split(':')[1]] = df[item][idx]
            if item.split(':')[0] == 'net_param':
                net_params[item.split(':')[1]] = df[item][idx]

    network = re.findall('\w*_(\w*)_\d*.pt', filename)[0]
    model = eval(network)(net_params).to(device='cuda')

    model.load_state_dict(torch.load(model_file_path), strict=False)
    return params, net_params, model


def load_optimal_model(path, name, idx):
    lib_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/library" + path
    setting_file_path = os.path.join(lib_file_path, name + '_settings' + '.csv')
    df = pd.read_csv(setting_file_path, sep=',', encoding='windows-1250', index_col=-1)

    # idx = df[['test_RMSE']].idxmin()['test_RMSE']
    name_idx = name + '_' + str(idx)
    model_file_path = os.path.join(lib_file_path, name_idx + '.pt')
    params = {}
    net_params = {}

    for _, item in enumerate(df.columns):
        if ':' in item:
            if item.split(':')[0] == 'param':
                params[item.split(':')[1]] = df[item][idx]
            if item.split(':')[0] == 'net_param':
                net_params[item.split(':')[1]] = df[item][idx]

    init_seed = df['init_seed'][idx]
    seed = df['seed'][idx]
    network = re.findall('\w*_(\w*)_\d*', name_idx)[0]
    model = eval(network)(net_params).to(device='cuda')
    model.load_state_dict(torch.load(model_file_path), strict=False)
    return idx, init_seed, seed, params, net_params, model


def main(params):
    set_seed(seed=2023)
    for j in range(4):
        params.update({'Dataset': dataset_list[j]})
        if dataset_list[j] == 'transmittance(400)':
            params.update({'sigmoid': True})
        else:
            params.update({'sigmoid': False})

        scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'model/{dataset_list[j]}_scaler.pkl')
        with open(scaler_path, 'rb') as fw:
            scaling = pickle.load(fw)

        cache_file_dir = os.path.realpath('./cache/predict/')
        if not os.path.exists(cache_file_dir):
            os.makedirs(cache_file_dir, exist_ok=True)
        cache_file_path = os.path.join(cache_file_dir, params['Dataset'])

        fragmentation = JT_SubGraph(scheme='MG_plus_reference')
        dataset = MoleculeCSVDataset(df, smiles_2_bigraph, classic_atom_featurizer, classic_bond_featurizer,
                                     classic_mol_featurizer, cache_file_path, load=False, fragmentation=fragmentation)

        raw_loader = DataLoader(dataset, collate_fn=collate_fraggraphs_pre, batch_size=len(dataset), shuffle=False,
                                num_workers=0, worker_init_fn=np.random.seed(2023))

        fetched_data = PreFetch_frag(raw_loader, frag=1)

        path = '/Ensembles/' + f'{params["Dataset"]}_FraGAT'
        name = '{}_{}_{}'.format('Ensemble', params['Dataset'], 'FraGAT')

        for i in range(100):
            name_idx = name + '_' + str(i)
            params, net_params, model = load_model(path, name_idx)
            n_param = count_parameters(model)
            raw_predict, raw_smiles = evaluate_frag_pre(model, scaling, fetched_data.all_iter,
                                                                      fetched_data.all_batched_origin_graph_list,
                                                                      fetched_data.all_batched_frag_graph_list,
                                                                      fetched_data.all_batched_motif_graph_list,
                                                                      fetched_data.all_smiles_list)
            if i == 0:
                df_raw = pd.DataFrame(
                    {'SMILES': raw_smiles[0], 'Predict_' + str(i): raw_predict.numpy().flatten().tolist()})
            else:
                df_raw['Predict_' + str(i)] = raw_predict.numpy().flatten().tolist()

        op_idx, init_seed, seed, params, net_params, model = load_optimal_model(path, name, op_list[j])
        all_smiles, all_descriptors = evaluate_frag_descriptors(model, fetched_data.all_iter,
                                                                fetched_data.all_batched_origin_graph_list,
                                                                fetched_data.all_batched_frag_graph_list,
                                                                fetched_data.all_batched_motif_graph_list,
                                                                fetched_data.all_smiles_list)

        _, all_attentions = evaluate_frag_attention(model, scaling, fetched_data.all_iter,
                                                    fetched_data.all_batched_origin_graph_list,
                                                    fetched_data.all_batched_frag_graph_list,
                                                    fetched_data.all_batched_motif_graph_list)
        all_attentions = np.array(all_attentions, dtype=object)
        df_descriptors = pd.DataFrame(all_descriptors.detach().to(device='cpu').numpy())

        df_merge = pd.concat([df_raw, df_descriptors], axis=1)

        _save_file_path = os.path.join('result', f'{params["Dataset"]}_descriptors_OP{op_idx}.csv')
        _attention_cache_file_path = cache_file_path + '_attentions.npy'
        df_merge.to_csv(_save_file_path, index=False)
        np.save(_attention_cache_file_path, all_attentions)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    params = {}
    dataset_list = ['transmittance(400)', 'cte', 'strength', 'tg']
    splitting_seed = [825, 854, 331, 525]  # AFP
    op_list = [95, 98, 86, 96]

    CurrentPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(CurrentPath, 'raw_data/predict.csv')
    df = pd.read_csv(path)
    main(params)
