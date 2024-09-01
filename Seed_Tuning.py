from src.feature.atom_featurizer import classic_atom_featurizer
from src.feature.bond_featurizer import classic_bond_featurizer
from src.feature.mol_featurizer import classic_mol_featurizer
from utils.mol2graph import smiles_2_bigraph
from utils.splitter import Splitter
from utils.Earlystopping import EarlyStopping
from data.csv_dataset import MoleculeCSVDataset
from src.dgltools import collate_molgraphs
from data.dataloading import import_dataset
import torch
from torch.utils.data import DataLoader
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.count_parameters import count_parameters

from networks.DMPNN import DMPNNNet
from networks.MPNN import MPNNNet
from networks.AttentiveFP import AttentiveFPNet
from utils.piplines import PreFetch, train_epoch, evaluate
from utils.Set_Seed_Reproducibility import set_seed


def main():
    for i in range(len(dataset_list)):
        params['Dataset'] = dataset_list[i]
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

        file_path = os.path.realpath('./output')
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        save_file_path = os.path.join(file_path, f'{params["Dataset"]}_AFP_{time.strftime("%Y-%m-%d-%H-%M")}.csv')

        df = pd.DataFrame(
            columns=['seed', 'train_R2', 'val_R2', 'test_R2', 'all_R2', 'train_MAE', 'val_MAE', 'test_MAE', 'all_MAE',
                     'train_RMSE', 'val_RMSE', 'test_RMSE', 'all_RMSE'])
        for i in range(0, 100):
            seed = np.random.randint(1, 1000)
            set_seed(seed=2023)
            train_set, val_set, test_set, raw_set = splitter.Random_Splitter(seed=seed, frac_train=0.8, frac_val=0.1)

            train_loader = DataLoader(train_set, collate_fn=collate_molgraphs, batch_size=len(train_set), shuffle=False)
            val_loader = DataLoader(val_set, collate_fn=collate_molgraphs, batch_size=len(val_set), shuffle=False)
            test_loader = DataLoader(test_set, collate_fn=collate_molgraphs, batch_size=len(test_set), shuffle=False)
            raw_loader = DataLoader(raw_set, collate_fn=collate_molgraphs, batch_size=len(raw_set), shuffle=False)

            fetched_data = PreFetch(train_loader, val_loader, test_loader, raw_loader, frag=False)
            model = AttentiveFPNet(net_params).to(device='cuda')
            optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=params['lr_reduce_factor'],
                                                                   patience=params['lr_schedule_patience'],
                                                                   verbose=False)
            t0 = time.time(),
            per_epoch_time = []
            early_stopping = EarlyStopping(patience=params['earlystopping_patience'])

            with tqdm(range(params['max_epoch'])) as t:
                n_param = count_parameters(model)
                print('The number of parameter: %d' % n_param)
                for epoch in t:
                    t.set_description('Epoch %d' % epoch)
                    start = time.time()
                    model, epoch_train_loss, epoch_train_metrics = train_epoch(model, optimizer, scaling,
                                                                               fetched_data.train_iter,
                                                                               fetched_data.train_batched_origin_graph_list,
                                                                               fetched_data.train_targets_list,
                                                                               fetched_data.train_smiles_list, n_param)

                    epoch_val_loss, epoch_val_metrics = evaluate(model, scaling, fetched_data.val_iter,
                                                                 fetched_data.val_batched_origin_graph_list,
                                                                 fetched_data.val_targets_list,
                                                                 fetched_data.val_smiles_list, n_param)

                    epoch_test_loss, epoch_test_metrics = evaluate(model, scaling, fetched_data.test_iter,
                                                                   fetched_data.test_batched_origin_graph_list,
                                                                   fetched_data.test_targets_list,
                                                                   fetched_data.test_smiles_list, n_param)

                    t.set_postfix({'time': time.time() - start, 'lr': optimizer.param_groups[0]['lr'],
                                   'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss,
                                   'test_loss': epoch_test_loss,
                                   'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2,
                                   'test_R2': epoch_test_metrics.R2})
                    per_epoch_time.append(time.time() - start)

                    scheduler.step(epoch_val_loss)
                    if optimizer.param_groups[0]['lr'] < params['min_lr']:
                        print('\n! LR equal to min LR set.')
                        break

                    early_stopping(epoch_val_loss, model)
                    if early_stopping.early_stop:
                        break

            _, epoch_raw_metrics = evaluate(model, scaling, fetched_data.all_iter,
                                            fetched_data.all_batched_origin_graph_list, fetched_data.all_targets_list,
                                            fetched_data.all_smiles_list, n_param)

            row = pd.Series({'seed': seed, 'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2,
                             'test_R2': epoch_test_metrics.R2, 'all_R2': epoch_raw_metrics.R2,
                             'train_MAE': epoch_train_metrics.MAE, 'val_MAE': epoch_val_metrics.MAE,
                             'test_MAE': epoch_test_metrics.MAE, 'all_MAE': epoch_raw_metrics.MAE,
                             'train_RMSE': epoch_train_metrics.RMSE, 'val_RMSE': epoch_val_metrics.RMSE,
                             'test_RMSE': epoch_test_metrics.RMSE, 'all_RMSE': epoch_raw_metrics.RMSE})
            df = df.append(row, ignore_index=True)

        df.to_csv(save_file_path)


if __name__ == '__main__':
    params, net_params = {}, {}
    dataset_list = ['transmittance(400)']
    params.update({
        'init_lr': 2e-3,
        'sigmoid': True,
        'min_lr': 1e-6,
        'weight_decay': 0,
        'lr_reduce_factor': 0.5,
        'lr_schedule_patience': 30,
        'earlystopping_patience': 150,
        'max_epoch': 1000
    })

    net_params.update({
        'num_atom_type': 36,
        'num_bond_type': 12,
        'hidden_dim': 64,
        'dropout': 0,
        'depth': 3,
        'layers': 3,
        'sigmoid': True,
        'residual': False,
        'batch_norm': False,
        'layer_norm': False,
        'device': 'cuda'
    })

    main()
