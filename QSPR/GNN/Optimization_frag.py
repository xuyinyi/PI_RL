import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.mol2graph import smiles_2_bigraph
from utils.junctiontree_encoder import JT_SubGraph
from utils.splitter import Splitter
from utils.Earlystopping import EarlyStopping
from data.csv_dataset import MoleculeCSVDataset
from src.dgltools import collate_fraggraphs
from data.dataloading import import_dataset
from utils.count_parameters import count_parameters
from networks.FraGAT import NewFraGATNet
from utils.Set_Seed_Reproducibility import set_seed
from utils.piplines import train_epoch_frag, evaluate_frag, PreFetch
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer


def main(params, net_params):
    model = NewFraGATNet(net_params).to(device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                           patience=params['lr_schedule_patience'], verbose=False)

    early_stopping = EarlyStopping(patience=params['earlystopping_patience'],
                                   path='checkpoint_' + params['Dataset'] + '_FraGAT' + '.pt')

    n_param = count_parameters(model)
    for epoch in range(params['max_epoch']):
        model, epoch_train_loss, epoch_train_metrics = train_epoch_frag(model, optimizer, scaling,
                                                                        fetched_data.train_iter,
                                                                        fetched_data.train_batched_origin_graph_list,
                                                                        fetched_data.train_batched_frag_graph_list,
                                                                        fetched_data.train_batched_motif_graph_list,
                                                                        fetched_data.train_targets_list,
                                                                        fetched_data.train_smiles_list, n_param)
        epoch_val_loss, epoch_val_metrics = evaluate_frag(model, scaling, fetched_data.val_iter,
                                                          fetched_data.val_batched_origin_graph_list,
                                                          fetched_data.val_batched_frag_graph_list,
                                                          fetched_data.val_batched_motif_graph_list,
                                                          fetched_data.val_targets_list, fetched_data.val_smiles_list,
                                                          n_param)

        scheduler.step(epoch_val_loss)
        if optimizer.param_groups[0]['lr'] < params['min_lr']:
            print('\n! LR equal to min LR set.')
            break

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            break

    model = early_stopping.load_checkpoint(model)
    _, epoch_train_metrics = evaluate_frag(model, scaling, fetched_data.train_iter,
                                           fetched_data.train_batched_origin_graph_list,
                                           fetched_data.train_batched_frag_graph_list,
                                           fetched_data.train_batched_motif_graph_list,
                                           fetched_data.train_targets_list, fetched_data.train_smiles_list, n_param)
    _, epoch_val_metrics = evaluate_frag(model, scaling, fetched_data.val_iter,
                                         fetched_data.val_batched_origin_graph_list,
                                         fetched_data.val_batched_frag_graph_list,
                                         fetched_data.val_batched_motif_graph_list,
                                         fetched_data.val_targets_list, fetched_data.val_smiles_list, n_param)
    _, epoch_test_metrics = evaluate_frag(model, scaling, fetched_data.test_iter,
                                          fetched_data.test_batched_origin_graph_list,
                                          fetched_data.test_batched_frag_graph_list,
                                          fetched_data.test_batched_motif_graph_list,
                                          fetched_data.test_targets_list, fetched_data.test_smiles_list, n_param)
    _, epoch_raw_metrics = evaluate_frag(model, scaling, fetched_data.all_iter,
                                         fetched_data.all_batched_origin_graph_list,
                                         fetched_data.all_batched_frag_graph_list,
                                         fetched_data.all_batched_motif_graph_list,
                                         fetched_data.all_targets_list, fetched_data.all_smiles_list, n_param)

    return epoch_train_metrics, epoch_val_metrics, epoch_test_metrics, epoch_raw_metrics


def Splitting_Main_MO(params, net_params):
    """Built-in function. Use the basic block of implementation in Multi-objective Bayesian Optimization.
    Parameters
    ----------
    params : dict
        Set of parameters for workflow.
    net_params : dict
        Set of parameters for architectures of models.

    Returns
    ----------
    -train_metrics.RMSE : float
        Optimization Objective-1, negative number of RMSE metric in training
    -val_metrics.RMSE : float
        Optimization Objective-2, negative number of RMSE metric in validation
    """
    train_metrics, val_metrics, test_metrics, all_metrics = main(params, net_params)
    return -val_metrics.RMSE


def func_to_be_opt_MO(hidden_dim, depth, layers, decay, dropout, init_lr, lr_reduce_factor):
    """Built-in function. Objective function in Single-objective Bayesian Optimization.
    Parameters
    ----------
    Returns
    ----------
    """
    net_params['hidden_dim'] = int(hidden_dim)
    net_params['layers'] = int(layers)
    net_params['depth'] = int(depth)
    params['weight_decay'] = 0.05 * int(decay)
    params['init_lr'] = 10 ** init_lr
    params['lr_reduce_factor'] = 0.4 + 0.05 * int(lr_reduce_factor)
    net_params['dropout'] = 0.05 * int(dropout)

    return Splitting_Main_MO(params, net_params)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    params, net_params = {}, {}
    dataset_list = ['transmittance(400)', 'cte', 'strength', 'tg']
    params.update({
        'init_lr': 2e-3,
        'min_lr': 1e-6,
        'weight_decay': 0,
        'lr_reduce_factor': 0.8,
        'lr_schedule_patience': 30,
        'earlystopping_patience': 150,
        'max_epoch': 1000
    })

    net_params.update({
        'num_atom_type': 36,
        'num_bond_type': 12,
        'hidden_dim': 64,
        'num_heads': 1,
        'dropout': 0,
        'depth': 3,
        'layers': 3,
        'residual': False,
        'batch_norm': False,
        'layer_norm': False,
        'device': 'cuda'
    })

    sigmoid = [True, False, False, False]
    splitting_list = [825, 854, 331, 525]  # AFP

    set_seed(seed=2023)
    for i in range(4):
        params['Dataset'] = dataset_list[i]
        params["sigmoid"] = sigmoid[i]
        net_params["sigmoid"] = sigmoid[i]
        split_seed = splitting_list[i]
        df, scaling = import_dataset(params)
        cache_file_path = os.path.realpath('./cache')
        if not os.path.exists(cache_file_path):
            os.mkdir(cache_file_path)
        cache_file = os.path.join(cache_file_path, params['Dataset'])

        error_path = os.path.realpath('./error_log')
        if not os.path.exists(error_path):
            os.mkdir(error_path)
        error_log_path = os.path.join(error_path, f'{params["Dataset"]}_{time.strftime("%Y-%m-%d-%H-%M")}' + '.csv')

        fragmentation = JT_SubGraph(scheme='MG_plus_reference')
        net_params['frag_dim'] = fragmentation.frag_dim
        dataset = MoleculeCSVDataset(df, smiles_2_bigraph, classic_atom_featurizer, classic_bond_featurizer,
                                     classic_mol_featurizer, cache_file, load=True
                                     , error_log=error_log_path, fragmentation=fragmentation)

        splitter = Splitter(dataset)

        file_path = os.path.realpath('./output')
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        save_file_path = os.path.join(file_path,
                                      f'{params["Dataset"]}_FraGAT_{split_seed}_{time.strftime("%Y-%m-%d-%H-%M")}.csv')

        train_set, val_set, test_set, raw_set = splitter.Random_Splitter(seed=split_seed, frac_train=0.8, frac_val=0.1)

        train_loader = DataLoader(train_set, collate_fn=collate_fraggraphs, batch_size=16, shuffle=False,
                                  num_workers=4, worker_init_fn=np.random.seed(2023))
        val_loader = DataLoader(val_set, collate_fn=collate_fraggraphs, batch_size=len(val_set), shuffle=False,
                                num_workers=4, worker_init_fn=np.random.seed(2023))
        test_loader = DataLoader(test_set, collate_fn=collate_fraggraphs, batch_size=len(test_set), shuffle=False,
                                 num_workers=4, worker_init_fn=np.random.seed(2023))
        raw_loader = DataLoader(raw_set, collate_fn=collate_fraggraphs, batch_size=len(raw_set), shuffle=False,
                                num_workers=4, worker_init_fn=np.random.seed(2023))

        fetched_data = PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=1)

        hpbounds = {'hidden_dim': (16, 256.99), 'depth': (1, 4.99), 'layers': (1, 4.99), 'decay': (0, 8.99),
                    'dropout': (0, 8.99), 'init_lr': (-3, -1), 'lr_reduce_factor': (0, 10.99)}

        bounds_transformer = SequentialDomainReductionTransformer()
        mutating_optimizer = BayesianOptimization(f=func_to_be_opt_MO, pbounds=hpbounds, verbose=0, random_state=2023)
        mutating_optimizer.maximize(init_points=50, n_iter=300, acq='ucb', kappa=5)
        lst = mutating_optimizer.space.keys
        lst.append('target')
        df = pd.DataFrame(columns=lst)
        for i, res in enumerate(mutating_optimizer.res):
            _dict = res['params']
            _dict['target'] = res['target']
            row = pd.DataFrame(_dict, index=[0])
            df = pd.concat([df, row], axis=0, ignore_index=True, sort=False)

        df.to_csv(save_file_path, index=False)
