import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.dataloading import import_dataset
from data.csv_dataset import MoleculeCSVDataset
from Network import MLPModel
from utils.splitter import Splitter
from utils.Earlystopping import EarlyStopping
from utils.count_parameters import count_parameters
from utils.Set_Seed_Reproducibility import set_seed

from utils.piplines import train_epoch, evaluate, PreFetch
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer


def collate_vectors(samples):
    vectors, targets, smiles = map(list, zip(*samples))
    targets = torch.tensor(np.array(targets)).unsqueeze(1)
    return vectors, targets, smiles


def main(params, net_params):
    model = MLPModel(net_params).to(device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                           patience=params['lr_schedule_patience'], verbose=False)
    per_epoch_time = []
    early_stopping = EarlyStopping(patience=params['earlystopping_patience'],
                                   path='checkpoint_' + params['Dataset'] + '_ANN' + '.pt')

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

            # t.set_postfix({'time': time.time() - start, 'lr': optimizer.param_groups[0]['lr'],
            #                'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss,
            #                'train_R2': epoch_train_metrics.R2, 'val_R2': epoch_val_metrics.R2})

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


def func_to_be_opt_MO(active_function, num_hidden, hidden_dim, dropout, init_lr, decay, lr_reduce_factor):
    """Built-in function. Objective function in Multi-objective Bayesian Optimization.
    Parameters
    ----------
    x : list
        Set of hyper-parameters in search domain.
    Returns
    ----------
    """
    net_params['active_function'] = int(active_function)
    net_params['num_hidden'] = int(num_hidden)
    net_params['hidden_dim'] = int(hidden_dim)
    net_params['dropout'] = 0.05 * int(dropout)
    params['init_lr'] = 10 ** init_lr
    params['weight_decay'] = 0.05 * int(decay)
    params['lr_reduce_factor'] = 0.4 + 0.05 * int(lr_reduce_factor)

    return Splitting_Main_MO(params, net_params)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        'active_function': 0,
        'num_hidden': 2,
        'hidden_dim': 64,
        'dropout': 0,
        'device': 'cuda'
    })
    sigmoid = [True, False, False, False]
    splitting_list = [825, 854, 331, 525]  # AFP

    set_seed(seed=2023)
    for i in range(3, 4):
        params['Dataset'] = dataset_list[i]
        params["sigmoid"] = sigmoid[i]
        net_params["sigmoid"] = sigmoid[i]
        split_seed = splitting_list[i]
        df, scaling, norm, num_des = import_dataset(params)
        net_params["input_dim"] = num_des
        cache_file_dir = os.path.realpath('./cache')
        if not os.path.exists(cache_file_dir):
            os.mkdir(cache_file_dir)
        cache_file_path = os.path.join(cache_file_dir, params['Dataset'] + '.npy')

        dataset = MoleculeCSVDataset(df, cache_file_path, load=True)
        splitter = Splitter(dataset)

        file_path = os.path.realpath('./output')
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        save_file_path = os.path.join(file_path,
                                      f'{params["Dataset"]}_ANN_{split_seed}_{time.strftime("%Y-%m-%d-%H-%M")}.csv')

        train_set, val_set, test_set, raw_set = splitter.Random_Splitter(seed=split_seed, frac_train=0.8, frac_val=0.1)

        train_loader = DataLoader(train_set, collate_fn=collate_vectors, batch_size=len(train_set), shuffle=False,
                                  num_workers=0, worker_init_fn=np.random.seed(2023))
        val_loader = DataLoader(val_set, collate_fn=collate_vectors, batch_size=len(val_set), shuffle=False,
                                num_workers=0, worker_init_fn=np.random.seed(2023))
        test_loader = DataLoader(test_set, collate_fn=collate_vectors, batch_size=len(test_set), shuffle=False,
                                 num_workers=0, worker_init_fn=np.random.seed(2023))
        raw_loader = DataLoader(raw_set, collate_fn=collate_vectors, batch_size=len(raw_set), shuffle=False,
                                num_workers=0, worker_init_fn=np.random.seed(2023))

        fetched_data = PreFetch(train_loader, val_loader, test_loader, raw_loader)

        hpbounds = {'active_function': (0, 1.99), 'num_hidden': (1, 5.99), 'hidden_dim': (16, 128.99),
                    'dropout': (0, 8.99), 'init_lr': (-3, -1), 'decay': (0, 8.99), 'lr_reduce_factor': (0, 10.99)}

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
