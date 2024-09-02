import os
import time
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.dataloading import import_dataset
from data.csv_dataset import MoleculeCSVDataset
from data.model_library import save_model
from Network import MLPModel
from utils.splitter import Splitter
from utils.Earlystopping import EarlyStopping
from utils.count_parameters import count_parameters
from utils.smile2vec import Embedding_smiles
from utils.Set_Seed_Reproducibility import set_seed
from utils.piplines import train_epoch, evaluate, PreFetch


def collate_vectors(samples):
    vectors, targets, smiles = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)
    return vectors, targets, smiles


def main():
    model = MLPModel(net_params).to(device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                           patience=params['lr_schedule_patience'], verbose=False)
    per_epoch_time = []
    early_stopping = EarlyStopping(patience=params['earlystopping_patience'],
                                   path='checkpoint_ensemble_' + params['Dataset'] + '_polyBERT' + '.pt')

    with tqdm(range(params['max_epoch'])) as t:
        n_param = count_parameters(model)
        for epoch in t:
            t.set_description('Epoch %d' % epoch)
            start = time.time()
            model, epoch_train_loss, epoch_train_metrics = train_epoch(model, optimizer, scaling,
                                                                       fetched_data.train_iter,
                                                                       fetched_data.train_vector_list,
                                                                       fetched_data.train_targets_list)

            epoch_val_loss, epoch_val_metrics = evaluate(model, scaling, fetched_data.val_iter,
                                                         fetched_data.val_vector_list,
                                                         fetched_data.val_targets_list,
                                                         fetched_data.val_smiles_list)

            epoch_test_loss, epoch_test_metrics = evaluate(model, scaling, fetched_data.test_iter,
                                                           fetched_data.test_vector_list,
                                                           fetched_data.test_targets_list,
                                                           fetched_data.test_smiles_list)

            t.set_postfix({'time': time.time() - start, 'lr': optimizer.param_groups[0]['lr'],
                           'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss,
                           'test_loss': epoch_test_loss, 'train_R2': epoch_train_metrics.R2,
                           'val_R2': epoch_val_metrics.R2, 'test_R2': epoch_test_metrics.R2})

            per_epoch_time.append(time.time() - start)

            scheduler.step(epoch_val_loss)
            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print('\n! LR equal to min LR set.')
                break

            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                break

    model = early_stopping.load_checkpoint(model)
    _, epoch_train_metrics = evaluate(model, scaling, fetched_data.train_iter, fetched_data.train_vector_list,
                                      fetched_data.train_targets_list, fetched_data.train_smiles_list)
    _, epoch_val_metrics = evaluate(model, scaling, fetched_data.val_iter, fetched_data.val_vector_list,
                                    fetched_data.val_targets_list, fetched_data.val_smiles_list)
    _, epoch_test_metrics = evaluate(model, scaling, fetched_data.test_iter, fetched_data.test_vector_list,
                                     fetched_data.test_targets_list, fetched_data.test_smiles_list)
    _, epoch_raw_metrics = evaluate(model, scaling, fetched_data.all_iter, fetched_data.all_vector_list,
                                    fetched_data.all_targets_list, fetched_data.all_smiles_list)

    return epoch_train_metrics, epoch_val_metrics, epoch_test_metrics, epoch_raw_metrics, model


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset_list = ['transmittance(400)', 'cte', 'strength', 'tg']
    splitting_seed = [825, 854, 331, 525]  # AFP

    set_seed(seed=2023)
    init_seed_list = [random.randint(0, 1000) for i in range(100)]
    active_function_l = [0, 1, 0, 0]
    num_hidden_l = [5, 5, 3, 5]
    hidden_dim_l = [197, 237, 184, 198]
    dropout_l = [0.35, 0, 0.25, 0.40]
    init_lr_l = [-1.9, -3, -3, -3]
    decay_l = [0, 0, 0, 0]
    lr_reduce_l = [0.70, 0.90, 0.85, 0.40]
    sigmoid = [True, False, False, False]

    for i in range(4):
        params, net_params = {}, {}
        params.update({
            'Dataset': dataset_list[i],
            'init_lr': 10 ** init_lr_l[i],
            'min_lr': 1e-6,
            'sigmoid': sigmoid[i],
            'weight_decay': decay_l[i],
            'lr_reduce_factor': lr_reduce_l[i],
            'lr_schedule_patience': 30,
            'earlystopping_patience': 150,
            'max_epoch': 1000
        })

        net_params.update({
            'active_function': active_function_l[i],
            'num_hidden': num_hidden_l[i],
            'hidden_dim': hidden_dim_l[i],
            'dropout': dropout_l[i],
            'sigmoid': sigmoid[i],
            'input_dim': 600,
            'device': 'cuda'
        })

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
        seed = splitting_seed[i]

        for j in range(100):
            torch.manual_seed(init_seed_list[j])
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

            epoch_train_metrics, epoch_val_metrics, epoch_test_metrics, epoch_raw_metrics, model = main()

            path = '/Ensembles/' + f'{params["Dataset"]}_polyBERT'
            name = '{}_{}_{}'.format('Ensemble', params['Dataset'], 'polyBERT')
            results = pd.Series({'init_seed': init_seed_list[j], 'seed': seed,
                                 'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2,
                                 'test_R2': epoch_test_metrics.R2, 'all_R2': epoch_raw_metrics.R2,
                                 'train_MAE': epoch_train_metrics.MAE, 'val_MAE': epoch_val_metrics.MAE,
                                 'test_MAE': epoch_test_metrics.MAE, 'all_MAE': epoch_raw_metrics.MAE,
                                 'train_RMSE': epoch_train_metrics.RMSE, 'val_RMSE': epoch_val_metrics.RMSE,
                                 'test_RMSE': epoch_test_metrics.RMSE, 'all_RMSE': epoch_raw_metrics.RMSE,
                                 'train_SSE': epoch_train_metrics.SSE, 'val_SSE': epoch_val_metrics.SSE,
                                 'test_SSE': epoch_test_metrics.SSE, 'all_SSE': epoch_raw_metrics.SSE,
                                 'train_MAPE': epoch_train_metrics.MAPE, 'val_MAPE': epoch_val_metrics.MAPE,
                                 'test_MAPE': epoch_test_metrics.MAPE, 'all_MAPE': epoch_raw_metrics.MAPE})
            comments = ''
            save_model(model, path, name, params, net_params, results, comments)
