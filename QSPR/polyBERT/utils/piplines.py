import torch
from .metrics import Metrics


class PreFetch(object):
    def __init__(self, train_loader, val_loader, test_loader, raw_loader):
        self.train_vector_list, self.train_targets_list, self.train_smiles_list, self.train_iter = [], [], [], []
        for iter, batch in enumerate(train_loader):
            _vector, _targets, _smiles = batch
            self.train_vector_list.append(_vector)
            self.train_targets_list.append(_targets)
            self.train_smiles_list.append(_smiles)
            self.train_iter.append(iter)

        self.val_vector_list, self.val_targets_list, self.val_smiles_list, self.val_iter = [], [], [], []
        for iter, batch in enumerate(val_loader):
            _vector, _targets, _smiles = batch
            self.val_vector_list.append(_vector)
            self.val_targets_list.append(_targets)
            self.val_smiles_list.append(_smiles)
            self.val_iter.append(iter)

        self.test_vector_list, self.test_targets_list, self.test_smiles_list, self.test_iter = [], [], [], []
        for iter, batch in enumerate(test_loader):
            _vector, _targets, _smiles = batch
            self.test_vector_list.append(_vector)
            self.test_targets_list.append(_targets)
            self.test_smiles_list.append(_smiles)
            self.test_iter.append(iter)

        self.all_vector_list, self.all_targets_list, self.all_smiles_list, self.all_iter = [], [], [], []
        for iter, batch in enumerate(raw_loader):
            _vector, _targets, _smiles = batch
            self.all_vector_list.append(_vector)
            self.all_targets_list.append(_targets)
            self.all_smiles_list.append(_smiles)
            self.all_iter.append(iter)


def train_epoch(model, optimizer, scaling, iter, vectors, targets):
    model.train()
    score_list, target_list = [], []
    epoch_loss = 0
    for i in iter:
        vector = torch.tensor(vectors[i], dtype=torch.float32).to(device='cuda')
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)
        score = model.forward(vector)
        target = targets[i].float().to(device='cuda')
        loss = model.loss(score, target)
        loss.backward()
        optimizer.step()
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    epoch_train_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                  scaling.ReScaler(score_list.detach().to(device='cpu')))
    return model, epoch_loss, epoch_train_metrics


def evaluate(model, scaling, iter, vectors, targets, smiles, flag=False):
    model.eval()
    score_list, target_list = [], []
    epoch_loss = 0
    for i in iter:
        vector = torch.tensor(vectors[i], dtype=torch.float32).to(device='cuda')
        torch.autograd.set_detect_anomaly(False)
        score = model.forward(vector)
        target = targets[i].float().to(device='cuda')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    true = scaling.ReScaler(target_list.detach().to(device='cpu'))

    epoch_eval_metrics = Metrics(true, predict)

    if flag:
        return epoch_loss, epoch_eval_metrics, predict, true, smiles
    else:
        return epoch_loss, epoch_eval_metrics
