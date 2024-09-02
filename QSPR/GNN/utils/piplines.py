import torch
from .metrics import Metrics
from .Set_Seed_Reproducibility import set_seed


class PreFetch(object):
    def __init__(self, train_loader, val_loader, test_loader, raw_loader, frag):
        if frag == 2:
            self.train_batched_origin_graph_list, self.train_batched_frag_graph_list, self.train_batched_motif_graph_list, \
            self.train_batched_channel_graph_list, self.train_batched_index_list, self.train_targets_list, self.train_smiles_list, \
            self.train_iter = [], [], [], [], [], [], [], []
            for iter, batch in enumerate(train_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _batched_channel_graph, _batched_index_list, _targets, _smiles = batch
                self.train_batched_origin_graph_list.append(_batched_origin_graph)
                self.train_batched_frag_graph_list.append(_batched_frag_graph)
                self.train_batched_motif_graph_list.append(_batched_motif_graph)
                self.train_batched_channel_graph_list.append(_batched_channel_graph)
                self.train_batched_index_list.append(_batched_index_list)
                self.train_targets_list.append(_targets)
                self.train_smiles_list.append(_smiles)
                self.train_iter.append(iter)

            self.val_batched_origin_graph_list, self.val_batched_frag_graph_list, self.val_batched_motif_graph_list, \
            self.val_batched_channel_graph_list, self.val_batched_index_list, self.val_targets_list, self.val_smiles_list, \
            self.val_iter = [], [], [], [], [], [], [], []
            for iter, batch in enumerate(val_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _batched_channel_graph, _batched_index_list, _targets, _smiles = batch
                self.val_batched_origin_graph_list.append(_batched_origin_graph)
                self.val_batched_frag_graph_list.append(_batched_frag_graph)
                self.val_batched_motif_graph_list.append(_batched_motif_graph)
                self.val_batched_channel_graph_list.append(_batched_channel_graph)
                self.val_batched_index_list.append(_batched_index_list)
                self.val_targets_list.append(_targets)
                self.val_smiles_list.append(_smiles)
                self.val_iter.append(iter)

            self.test_batched_origin_graph_list, self.test_batched_frag_graph_list, self.test_batched_motif_graph_list, \
            self.test_batched_channel_graph_list, self.test_batched_index_list, self.test_targets_list, self.test_smiles_list, \
            self.test_iter = [], [], [], [], [], [], [], []
            for iter, batch in enumerate(test_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _batched_channel_graph, _batched_index_list, _targets, _smiles = batch
                self.test_batched_origin_graph_list.append(_batched_origin_graph)
                self.test_batched_frag_graph_list.append(_batched_frag_graph)
                self.test_batched_motif_graph_list.append(_batched_motif_graph)
                self.test_batched_channel_graph_list.append(_batched_channel_graph)
                self.test_batched_index_list.append(_batched_index_list)
                self.test_targets_list.append(_targets)
                self.test_smiles_list.append(_smiles)
                self.test_iter.append(iter)

            self.all_batched_origin_graph_list, self.all_batched_frag_graph_list, self.all_batched_motif_graph_list, \
            self.all_batched_channel_graph_list, self.all_batched_index_list, self.all_targets_list, self.all_smiles_list, \
            self.all_iter = [], [], [], [], [], [], [], []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _batched_channel_graph, _batched_index_list, _targets, _smiles = batch
                self.all_batched_origin_graph_list.append(_batched_origin_graph)
                self.all_batched_frag_graph_list.append(_batched_frag_graph)
                self.all_batched_motif_graph_list.append(_batched_motif_graph)
                self.all_batched_channel_graph_list.append(_batched_channel_graph)
                self.all_batched_index_list.append(_batched_index_list)
                self.all_targets_list.append(_targets)
                self.all_smiles_list.append(_smiles)
                self.all_iter.append(iter)

        elif frag == 1:
            self.train_batched_origin_graph_list, self.train_batched_frag_graph_list, self.train_batched_motif_graph_list, \
            self.train_targets_list, self.train_smiles_list, self.train_iter = [], [], [], [], [], []
            for iter, batch in enumerate(train_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _targets, _smiles = batch
                self.train_batched_origin_graph_list.append(_batched_origin_graph)
                self.train_batched_frag_graph_list.append(_batched_frag_graph)
                self.train_batched_motif_graph_list.append(_batched_motif_graph)
                self.train_targets_list.append(_targets)
                self.train_smiles_list.append(_smiles)
                self.train_iter.append(iter)

            self.val_batched_origin_graph_list, self.val_batched_frag_graph_list, self.val_batched_motif_graph_list, \
            self.val_targets_list, self.val_smiles_list, self.val_iter = [], [], [], [], [], []
            for iter, batch in enumerate(val_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _targets, _smiles = batch
                self.val_batched_origin_graph_list.append(_batched_origin_graph)
                self.val_batched_frag_graph_list.append(_batched_frag_graph)
                self.val_batched_motif_graph_list.append(_batched_motif_graph)
                self.val_targets_list.append(_targets)
                self.val_smiles_list.append(_smiles)
                self.val_iter.append(iter)

            self.test_batched_origin_graph_list, self.test_batched_frag_graph_list, self.test_batched_motif_graph_list, \
            self.test_targets_list, self.test_smiles_list, self.test_iter = [], [], [], [], [], []
            for iter, batch in enumerate(test_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _targets, _smiles = batch
                self.test_batched_origin_graph_list.append(_batched_origin_graph)
                self.test_batched_frag_graph_list.append(_batched_frag_graph)
                self.test_batched_motif_graph_list.append(_batched_motif_graph)
                self.test_targets_list.append(_targets)
                self.test_smiles_list.append(_smiles)
                self.test_iter.append(iter)

            self.all_batched_origin_graph_list, self.all_batched_frag_graph_list, self.all_batched_motif_graph_list, \
            self.all_targets_list, self.all_smiles_list, self.all_iter = [], [], [], [], [], []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _targets, _smiles = batch
                self.all_batched_origin_graph_list.append(_batched_origin_graph)
                self.all_batched_frag_graph_list.append(_batched_frag_graph)
                self.all_batched_motif_graph_list.append(_batched_motif_graph)
                self.all_targets_list.append(_targets)
                self.all_smiles_list.append(_smiles)
                self.all_iter.append(iter)

        else:
            self.train_batched_origin_graph_list, self.train_targets_list, self.train_smiles_list, self.train_iter = [], [], [], []
            for iter, batch in enumerate(train_loader):
                _batched_origin_graph, _targets, _smiles = batch
                self.train_batched_origin_graph_list.append(_batched_origin_graph)
                self.train_targets_list.append(_targets)
                self.train_smiles_list.append(_smiles)
                self.train_iter.append(iter)

            self.val_batched_origin_graph_list, self.val_targets_list, self.val_smiles_list, self.val_iter = [], [], [], []
            for iter, batch in enumerate(val_loader):
                _batched_origin_graph, _targets, _smiles = batch
                self.val_batched_origin_graph_list.append(_batched_origin_graph)
                self.val_targets_list.append(_targets)
                self.val_smiles_list.append(_smiles)
                self.val_iter.append(iter)

            self.test_batched_origin_graph_list, self.test_targets_list, self.test_smiles_list, self.test_iter = [], [], [], []
            for iter, batch in enumerate(test_loader):
                _batched_origin_graph, _targets, _smiles = batch
                self.test_batched_origin_graph_list.append(_batched_origin_graph)
                self.test_targets_list.append(_targets)
                self.test_smiles_list.append(_smiles)
                self.test_iter.append(iter)

            self.all_batched_origin_graph_list, self.all_targets_list, self.all_smiles_list, self.all_iter = [], [], [], []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph, _targets, _smiles = batch
                self.all_batched_origin_graph_list.append(_batched_origin_graph)
                self.all_targets_list.append(_targets)
                self.all_smiles_list.append(_smiles)
                self.all_iter.append(iter)


class PreFetch_frag(object):
    def __init__(self, raw_loader, frag):
        if frag == 1:
            self.all_batched_origin_graph_list, self.all_batched_frag_graph_list, self.all_batched_motif_graph_list, self.all_smiles_list, self.all_iter = [], [], [], [], []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph, _batched_frag_graph, _batched_motif_graph, _smiles = batch
                self.all_batched_origin_graph_list.append(_batched_origin_graph)
                self.all_batched_frag_graph_list.append(_batched_frag_graph)
                self.all_batched_motif_graph_list.append(_batched_motif_graph)
                self.all_smiles_list.append(_smiles)
                self.all_iter.append(iter)
        else:
            self.all_batched_origin_graph_list, self.all_targets_list, self.all_smiles_list, self.all_iter = [], [], [], []
            for iter, batch in enumerate(raw_loader):
                _batched_origin_graph, _targets, _smiles = batch
                self.all_batched_origin_graph_list.append(_batched_origin_graph)
                self.all_targets_list.append(_targets)
                self.all_smiles_list.append(_smiles)
                self.all_iter.append(iter)


def train_epoch(model, optimizer, scaling, iter, batched_origin_graph, targets, smiles, n_param=None):
    model.train()
    score_list, target_list = [], []
    epoch_loss = 0
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)
        score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge)
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
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)
    return model, epoch_loss, epoch_train_metrics


def evaluate(model, scaling, iter, batched_origin_graph, targets, smiles, n_param=None, flag=False):
    model.eval()
    score_list, target_list = [], []
    epoch_loss = 0
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge)
        target = targets[i].float().to(device='cuda')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    true = scaling.ReScaler(target_list.detach().to(device='cpu'))

    epoch_eval_metrics = Metrics(true, predict, n_param)

    if flag:
        return epoch_loss, epoch_eval_metrics, predict, true, smiles
    else:
        return epoch_loss, epoch_eval_metrics


def evaluate_frag(model, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph, targets, smiles,
                  n_param=None, flag=False):
    model.eval()
    score_list = []
    target_list = []
    epoch_loss = 0
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        if True:
            score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge)
        target = targets[i].float().to(device='cuda')
        loss = model.loss(score, target)
        score_list.append(score)
        target_list.append(target)
        epoch_loss += loss.detach().item()
    score_list = torch.cat(score_list, dim=0)
    target_list = torch.cat(target_list, dim=0)
    epoch_eval_metrics = Metrics(scaling.ReScaler(target_list.detach().to(device='cpu')),
                                 scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    true = scaling.ReScaler(target_list.detach().to(device='cpu'))
    if flag:
        return epoch_loss, epoch_eval_metrics, predict, true, smiles
    else:
        return epoch_loss, epoch_eval_metrics


def evaluate_frag_pre(model, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph, smiles):
    model.eval()
    score_list = []
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        if True:
            score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge)
        score_list.append(score)
    score_list = torch.cat(score_list, dim=0)

    predict = scaling.ReScaler(score_list.detach().to(device='cpu'))
    return predict, smiles


def evaluate_descriptors(model, scaling, iter, batched_origin_graph, targets, smiles, n_param=None):
    model.eval()
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        _, descriptors = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge, get_descriptors=True)
    return smiles, descriptors


def evaluate_attention(model, scaling, iter, batched_origin_graph):
    model.eval()
    score_list = []
    attentions_list = []
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        score, attention = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge, get_attention=True)
        score_list.append(score)
        attentions_list.extend(attention)
    score_list = torch.cat(score_list, dim=0)

    predictions_list = scaling.ReScaler(score_list.detach().to(device='cpu').numpy())
    return predictions_list, attentions_list


def train_epoch_frag(model, optimizer, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph,
                     targets, smiles, n_param=None):
    model.train()
    score_list = []
    target_list = []
    epoch_loss = 0
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(False)
        if True:
            score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                  batch_frag_graph, batch_frag_node, batch_frag_edge,
                                  batch_motif_graph, batch_motif_node, batch_motif_edge)
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
                                  scaling.ReScaler(score_list.detach().to(device='cpu')), n_param)
    return model, epoch_loss, epoch_train_metrics


def evaluate_frag_descriptors(model, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph, smiles):
    model.eval()
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        if True:
            _, descriptors = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                           batch_frag_graph, batch_frag_node, batch_frag_edge,
                                           batch_motif_graph, batch_motif_node, batch_motif_edge, get_descriptors=True)

    return smiles, descriptors


def evaluate_frag_attention(model, scaling, iter, batched_origin_graph, batched_frag_graph, batched_motif_graph):
    model.eval()
    score_list = []
    attentions_list = []
    for i in iter:
        batch_origin_node = batched_origin_graph[i].ndata['feat'].to(device='cuda')
        batch_origin_edge = batched_origin_graph[i].edata['feat'].to(device='cuda')
        batch_origin_global = batched_origin_graph[i].ndata['global_feat'].to(device='cuda')
        batch_origin_graph = batched_origin_graph[i].to(device='cuda')

        batch_frag_node = batched_frag_graph[i].ndata['feat'].to(device='cuda')
        batch_frag_edge = batched_frag_graph[i].edata['feat'].to(device='cuda')
        batch_frag_graph = batched_frag_graph[i].to(device='cuda')

        batch_motif_node = batched_motif_graph[i].ndata['feat'].to(device='cuda')
        batch_motif_edge = batched_motif_graph[i].edata['feat'].to(device='cuda')
        batch_motif_graph = batched_motif_graph[i].to(device='cuda')

        torch.autograd.set_detect_anomaly(False)
        if True:
            score, attention = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge,
                                             batch_frag_graph, batch_frag_node, batch_frag_edge,
                                             batch_motif_graph, batch_motif_node, batch_motif_edge,
                                             get_descriptors=True, get_attention=True)
        score_list.append(score)
        attentions_list.extend(attention)
    score_list = torch.cat(score_list, dim=0)

    predictions_list = scaling.ReScaler(score_list.detach().to(device='cpu').numpy())
    return predictions_list, attentions_list
