import os
import dgl
import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AllChem as Chem
from dgl.data.utils import save_graphs, load_graphs
from QSPR.GNN.utils.mol2graph import graph_2_frag


class MoleculeCSVDataset(object):
    def __init__(self, df, smiles_2_graph, atom_featurizer, bond_featurizer, mol_featurizer, cache_file_path,
                 load=False, fragmentation=None):
        self.df = df
        self.cache_file_path = cache_file_path
        self._prepare(smiles_2_graph, atom_featurizer, bond_featurizer, mol_featurizer, load)
        self.whe_frag = False
        if fragmentation is not None:
            self.whe_frag = True
            self._prepare_frag(fragmentation, load)

    def _prepare(self, smiles_2_graph, atom_featurizer, bond_featurizer, mol_featurizer, load):
        '''
        :param
        '''
        if os.path.exists(self.cache_file_path) and load:
            print('Loading saved dgl graphs ...')
            self.origin_graphs, label_dict = load_graphs(self.cache_file_path)
            valid_idx = label_dict['valid_idx']
            self.valid_idx = valid_idx.detach().numpy().tolist()
        else:
            print('Preparing dgl by featurizers ...')
            self.origin_graphs = []
            for i, s in enumerate(self.df['smile']):
                self.origin_graphs.append(smiles_2_graph(s, atom_featurizer, bond_featurizer, mol_featurizer))

            # Check failed featurization
            # Keep successful featurization
            self.valid_idx = []
            origin_graphs, failed_smiles = [], []
            for i, g in enumerate(self.origin_graphs):
                if g is not None:
                    self.valid_idx.append(i)
                    origin_graphs.append(g)
                else:
                    failed_smiles.append((i, self.df['smile'][i]))

            self.origin_graphs = origin_graphs
            valid_idx = torch.tensor(self.valid_idx)
            save_graphs(self.cache_file_path, self.origin_graphs, labels={'valid_idx': valid_idx})

        self.smiles = [self.df['smile'][i] for i in self.valid_idx]

    def _prepare_frag(self, fragmentation, load):
        _frag_cache_file_path = self.cache_file_path + '_frag'
        _motif_cache_file_path = self.cache_file_path + '_motif'
        _atom_mask_cache_file_path = self.cache_file_path + '_atom_mask.npy'
        _frag_flag_file_path = self.cache_file_path + '_frag_flag.npy'
        if os.path.exists(_frag_cache_file_path) and os.path.exists(_motif_cache_file_path) and os.path.exists(
                _atom_mask_cache_file_path) and os.path.exists(_frag_flag_file_path) and load:
            print('Loading saved fragments and graphs ...')
            unbatched_frag_graphs, frag_label_dict = load_graphs(_frag_cache_file_path)
            self.motif_graphs, motif_label_dict = load_graphs(_motif_cache_file_path)
            self.atom_mask_list = np.load(_atom_mask_cache_file_path, allow_pickle=True)
            self.frag_flag_list = np.load(_frag_flag_file_path, allow_pickle=True)
            frag_graph_idx = frag_label_dict['frag_graph_idx'].detach().numpy().tolist()
            self.batched_frag_graphs = self.batch_frag_graph(unbatched_frag_graphs, frag_graph_idx)
        else:
            print('Preparing fragmentation ...')
            self.batched_frag_graphs = []
            unbatched_frag_graphs_list = []  # unbatched_* variables prepared for storage of graphs
            self.motif_graphs = []
            self.atom_mask_list = []
            self.frag_flag_list = []
            for i, s in enumerate(self.df['smile']):
                try:
                    frag_graph, motif_graph, atom_mask, frag_flag = graph_2_frag(s, self.origin_graphs[i],
                                                                                 fragmentation)
                except:
                    print('Failed to deal with  ', s)
                self.batched_frag_graphs.append(dgl.batch(frag_graph))
                unbatched_frag_graphs_list.append(frag_graph)
                self.motif_graphs.append(motif_graph)
                self.atom_mask_list.append(atom_mask)
                self.frag_flag_list.append(frag_flag)

            # Check failed fragmentation
            batched_frag_graphs = []
            unbatched_frag_graphs = []
            motif_graphs = []
            atom_masks = []
            frag_failed_smiles = []
            for i, g in enumerate(self.motif_graphs):
                if g is not None:
                    motif_graphs.append(g)
                    atom_masks.append(self.atom_mask_list[i])
                    batched_frag_graphs.append(self.batched_frag_graphs[i])
                    unbatched_frag_graphs.append(unbatched_frag_graphs_list[i])
                else:
                    frag_failed_smiles.append((i, self.df['smile'][i]))
                    self.valid_idx.remove(i)

            self.batched_frag_graphs = batched_frag_graphs
            self.motif_graphs = motif_graphs
            self.atom_mask_list = np.array(atom_masks, dtype=object)
            self.frag_flag_list = np.array(self.frag_flag_list, dtype=object)
            unbatched_frag_graphs, frag_graph_idx = self.merge_frag_list(unbatched_frag_graphs)
            valid_idx = torch.tensor(self.valid_idx)
            save_graphs(_frag_cache_file_path, unbatched_frag_graphs,
                        labels={'valid_idx': valid_idx, 'frag_graph_idx': frag_graph_idx})
            save_graphs(_motif_cache_file_path, self.motif_graphs,
                        labels={'valid_idx': valid_idx})
            np.save(_atom_mask_cache_file_path, self.atom_mask_list)
            np.save(_frag_flag_file_path, self.frag_flag_list)

    def __getitem__(self, index):
        # return self.df['SMILES'][item], self.graphs[item], self.values[item]
        if self.whe_frag:
            # self.frag_graphs_list = self.convert_frag_list()
            return self.origin_graphs[index], self.batched_frag_graphs[index], self.motif_graphs[index], self.smiles[
                index]
        else:
            return self.origin_graphs[index], self.smiles[index]

    def __len__(self):
        return len(self.df['smile'])

    def merge_frag_list(self, frag_graphs_list):
        # flatten all fragment lists in self.frag_graphs_lists for saving, [[...], [...], [...], ...] --> [..., ..., ...]
        frag_graphs = []
        idx = []
        for i, item in enumerate(frag_graphs_list):
            for _ in range(len(item)):
                idx.append(i)
            frag_graphs.extend(item)
        idx = torch.Tensor(idx)
        return frag_graphs, idx

    def convert_frag_list(self):
        # convert flattened list into 2-D list [[], [], []], inner list represents small subgraph of fragment while outer list denotes the index of molecule
        frag_graphs_list = [[] for _ in range(len(self))]
        for i, item in enumerate(self.frag_graph_idx):
            frag_graphs_list[int(item)].append(self.frag_graphs[i])
        return frag_graphs_list

    def batch_frag_graph(self, unbatched_graph, frag_graph_idx):
        batched_frag_graphs = []
        for i in range(len(self)):
            batched_frag_graph = dgl.batch(
                [unbatched_graph[idx] for idx, value in enumerate(frag_graph_idx) if int(value) == i])
            batched_frag_graphs.append(batched_frag_graph)
        return batched_frag_graphs


def load_AFP():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir,
                             "library/Ensembles/transmittance(400)_AFN/Ensemble_transmittance(400)_AFN_PI_descriptors_825_OP59_2023-11-24-09-13.csv")
    data = pd.read_csv(data_path).values
    smiles_list = data[:, 0].tolist()
    targets_list = data[:, 2].tolist()
    tag_list = data[:, 1].tolist()
    predictions_list = data[:, 74].tolist()
    attentions_list_array = data[:, 103:]
    time_step = attentions_list_array.shape[1] - 1
    return smiles_list, targets_list, tag_list, predictions_list, attentions_list_array, time_step


def load_FraGAT(dataset, OP):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir,
                             f"library/Ensembles/{dataset}_FraGAT/Ensemble_{dataset}_FraGAT_PI_descriptors_OP{OP}.csv")
    atom_mask_path = os.path.join(base_dir, f"cache/{dataset}_atom_mask.npy")
    data = pd.read_csv(data_path).values
    smiles_list = data[:, 0].tolist()
    targets_list = data[:, 1].tolist()
    predictions_list = data[:, OP + 2].tolist()
    attentions_list_array = data[:, 102:]
    atom_mask_list = np.load(atom_mask_path, allow_pickle=True)
    time_step = attentions_list_array.shape[1] - 1
    return dataset, smiles_list, targets_list, predictions_list, attentions_list_array, atom_mask_list, time_step


def load_Attention(dataset):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, f"raw_data/{dataset}.csv")
    attention_path = os.path.join(base_dir, f"QSPR/GNN/cache/{dataset}_attentions.npy")
    atom_mask_path = os.path.join(base_dir, f"QSPR/GNN/cache/{dataset}_atom_mask.npy")
    frag_flag_path = os.path.join(base_dir, f"QSPR/GNN/cache/{dataset}_frag_flag.npy")
    smiles_list = pd.read_csv(data_path).values[:, 0].tolist()
    attentions_list_array = np.load(attention_path, allow_pickle=True)
    atom_mask_list = np.load(atom_mask_path, allow_pickle=True)
    frag_flag_list = np.load(frag_flag_path, allow_pickle=True)
    time_step = attentions_list_array[0].shape[1] - 1
    return dataset, smiles_list, attentions_list_array, atom_mask_list, frag_flag_list, time_step


def load_Pred(dataset, OP):
    data_path = f"result/{dataset}_descriptors_OP{OP}.csv"
    attention_path = f"cache/predict/{dataset}_attentions.npy"
    atom_mask_path = f"cache/predict/{dataset}_atom_mask.npy"
    data = pd.read_csv(data_path).values
    smiles_list = data[:, 0].tolist()
    predictions_list = data[:, OP + 1].tolist()
    attentions_list_array = np.load(attention_path, allow_pickle=True)
    atom_mask_list = np.load(atom_mask_path, allow_pickle=True)
    time_step = attentions_list_array[0].shape[1] - 1
    return dataset, smiles_list, predictions_list, attentions_list_array, atom_mask_list, time_step
