import os
import re
import pickle
import gzip
import math
import torch
import pandas as pd
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors
from model.utils.mol2graph import smiles_2_bigraph
from model.src.feature.atom_featurizer import classic_atom_featurizer
from model.src.feature.bond_featurizer import classic_bond_featurizer
from model.src.feature.mol_featurizer import classic_mol_featurizer
from model.networks.AttentiveFP import AttentiveFPNet as AFP


class MoleculeCSVDataset(object):
    def __init__(self, smiles, smiles_2_graph, atom_featurizer, bond_featurizer, mol_featurizer):
        self.smiles = smiles
        self._prepare(smiles_2_graph, atom_featurizer, bond_featurizer, mol_featurizer)
        self.whe_frag = False

    def _prepare(self, smiles_2_graph, atom_featurizer, bond_featurizer, mol_featurizer):
        '''
        :param
        '''
        print('Preparing dgl by featurizers ...')
        self.origin_graphs = []
        for i, s in enumerate(self.smiles):
            self.origin_graphs.append(smiles_2_graph(s, atom_featurizer, bond_featurizer, mol_featurizer))

        # Check failed featurization; Keep successful featurization
        self.valid_idx = []
        origin_graphs, failed_smiles = [], []
        for i, g in enumerate(self.origin_graphs):
            if g is not None:
                self.valid_idx.append(i)
                origin_graphs.append(g)
            else:
                failed_smiles.append((i, self.smiles[i]))

        self.origin_graphs = origin_graphs

        self.smiles = [self.smiles[i] for i in self.valid_idx]

    def __getitem__(self, index):
        if self.whe_frag:
            # self.frag_graphs_list = self.convert_frag_list()
            return self.origin_graphs[index], self.batched_frag_graphs[index], self.motif_graphs[index], \
                   self.channel_graphs[index], self.values[index], self.smiles[index]
        else:
            return self.origin_graphs[index], self.smiles[index]

    def __len__(self):
        return len(self.smiles)


class Benchmark(object):
    def __init__(self, smiles: list):
        self.smiles = smiles
        self.dataset = MoleculeCSVDataset(self.smiles, smiles_2_bigraph, classic_atom_featurizer,
                                          classic_bond_featurizer, classic_mol_featurizer)
        self.params = dict()
        self.scaler_path = None
        self.property_name = None
        self.model_id = [43, 56, 84, 64]
        self.Score = 0.0

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.score()

    def score(self):
        self.transmittance = round(self.pred_transmittance()[0], 2)
        self.cte = round(self.pred_cte()[0], 2)
        self.strength = round(self.pred_strength()[0], 2)
        self.tg = round(self.pred_tg()[0], 2)
        self.ScoreSA = round(self.pred_SA()[0], 2)

        coef = self.transmittance / 100

        self.Score_cte = self.score_cte(self.cte)
        self.Score_strength = self.score_strength(self.strength)
        self.Score_tg = self.score_tg(self.tg)
        self.Score_SA = self.score_SA(self.ScoreSA)

        self.Score = round((coef * (self.Score_cte * self.Score_strength * self.Score_tg * self.Score_SA) ** 0.25), 4)

        # self.Score = [round((c + c * (s_cte + s_str + s_tg + s_sa)) / 5, 4)
        #               for c, s_cte, s_str, s_tg, s_sa in zip(
        #         coef, self.Score_cte, self.Score_strength, self.Score_tg, self.Score_SA
        #     )]

    def pred_transmittance(self):
        self.property_name = 'transmittance(400)'
        self.params.update({'sigmoid': True})
        self.params.update({'Dataset': self.property_name})
        self.scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        f'model/{self.property_name}_scaler.pkl')
        with open(self.scaler_path, 'rb') as fw:
            self.scaling = pickle.load(fw)
        self.model_name = 'Ensemble_{}_AFP_{}'.format(self.params['Dataset'], self.model_id[0])
        _, _, self.model = self.load_model(self.model_name)
        pred, smiles = self.evaluate(self.model, self.scaling, self.dataset.origin_graphs, self.dataset.smiles)

        return pred.numpy().flatten().tolist()

    def pred_cte(self):
        self.property_name = 'cte'
        self.params.update({'sigmoid': False})
        self.params.update({'Dataset': self.property_name})
        self.scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        f'model/{self.property_name}_scaler.pkl')
        with open(self.scaler_path, 'rb') as fw:
            self.scaling = pickle.load(fw)
        self.model_name = 'Ensemble_{}_AFP_{}'.format(self.params['Dataset'], self.model_id[1])
        _, _, self.model = self.load_model(self.model_name)
        pred, smiles = self.evaluate(self.model, self.scaling, self.dataset.origin_graphs, self.dataset.smiles)

        return pred.numpy().flatten().tolist()

    def pred_strength(self):
        self.property_name = 'strength'
        self.params.update({'sigmoid': False})
        self.params.update({'Dataset': self.property_name})
        self.scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        f'model/{self.property_name}_scaler.pkl')
        with open(self.scaler_path, 'rb') as fw:
            self.scaling = pickle.load(fw)
        self.model_name = 'Ensemble_{}_AFP_{}'.format(self.params['Dataset'], self.model_id[2])
        _, _, self.model = self.load_model(self.model_name)
        pred, smiles = self.evaluate(self.model, self.scaling, self.dataset.origin_graphs, self.dataset.smiles)

        return pred.numpy().flatten().tolist()

    def pred_tg(self):
        self.property_name = 'tg'
        self.params.update({'sigmoid': False})
        self.params.update({'Dataset': self.property_name})
        self.scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        f'model/{self.property_name}_scaler.pkl')
        with open(self.scaler_path, 'rb') as fw:
            self.scaling = pickle.load(fw)
        self.model_name = 'Ensemble_{}_AFP_{}'.format(self.params['Dataset'], self.model_id[3])
        _, _, self.model = self.load_model(self.model_name)
        pred, smiles = self.evaluate(self.model, self.scaling, self.dataset.origin_graphs, self.dataset.smiles)

        return pred.numpy().flatten().tolist()

    def pred_SA(self):
        FpScoresPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/fpscores.pkl.gz')
        FpScoresData = pickle.load(gzip.open(FpScoresPath))
        FpScores = {}
        for i in FpScoresData:
            for j in range(1, len(i)):
                FpScores[i[j]] = float(i[0])

        mols = [Chem.MolFromSmiles(smi) for smi in self.smiles]
        scoreSA = [self.calculateSAScore(mol, FpScores) for mol in mols]

        return scoreSA

    def load_model(self, name):
        model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        model_file_path = os.path.join(model_dir_path, f'{name}.pt')

        directory, filename = os.path.split(model_file_path)
        idx = int(re.findall('_(\d+).pt', filename)[0])
        name = filename.split(f"_{idx}")[0]
        setting_file_path = os.path.join(model_dir_path, name + '_settings' + '.csv')
        df = pd.read_csv(setting_file_path, sep=',', encoding='windows-1250', index_col=-1)
        params = {}
        net_params = {}
        for _, item in enumerate(df.columns):
            if ':' in item:
                if item.split(':')[0] == 'param':
                    params[item.split(':')[1]] = df[item][idx]
                if item.split(':')[0] == 'net_param':
                    net_params[item.split(':')[1]] = df[item][idx]

        if "sigmoid" not in net_params.keys():
            net_params["sigmoid"] = False

        network = re.findall('\w*_(\w*)_\d*.pt', filename)[0]
        model = eval(network)(net_params).to(device=self.device)

        model.load_state_dict(torch.load(model_file_path), strict=False)
        return params, net_params, model

    def evaluate(self, model, scaling, origin_graph, smiles):
        model.eval()
        score_list = []
        for i in range(len(smiles)):
            batch_origin_node = origin_graph[i].ndata['feat'].to(device=self.device)
            batch_origin_edge = origin_graph[i].edata['feat'].to(device=self.device)
            batch_origin_global = origin_graph[i].ndata['global_feat'].to(device=self.device)
            batch_origin_graph = origin_graph[i].to(device=self.device)

            torch.autograd.set_detect_anomaly(False)
            score = model.forward(batch_origin_graph, batch_origin_node, batch_origin_edge)
            score_list.append(score)
        score_list = torch.cat(score_list, dim=0)

        predict = scaling.ReScaler(score_list.detach().to(device='cpu'))

        return predict, smiles

    @staticmethod
    def calculateSAScore(_mol, FpScores):
        # fragment score
        fp = rdMolDescriptors.GetMorganFingerprint(_mol, 2)  # 2 is the *radius* of the circular fingerprint
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += FpScores.get(sfp, -4) * v
        score1 /= nf

        # features score
        def numBridgeheadsAndSpiro(mol, ri=None):
            nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
            nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            return nBridgehead, nSpiro

        nAtoms = _mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(_mol, includeUnassigned=True))
        ri = _mol.GetRingInfo()
        nBridgeheads, nSpiro = numBridgeheadsAndSpiro(_mol, ri)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore

    @staticmethod
    def score_cte(cte):
        cte = math.fabs(cte)
        if cte == 0:
            score = 1.0
        elif cte >= 80:
            score = 0.0
        else:
            score = 1 - cte / 80
        return score

    @staticmethod
    def score_strength(strength):
        if strength > 500:
            score = 1.0
        elif strength <= 30:
            score = 0.0
        else:
            score = (strength - 30) / 440
        return score

    @staticmethod
    def score_tg(tg):
        if tg > 600:
            score = 1.0
        elif tg <= 100:
            score = 0.0
        else:
            score = (tg - 100) / 500
        return score

    @staticmethod
    def score_SA(scoreSA):
        if scoreSA < 2:
            score = 1.0
        elif scoreSA >= 5:
            score = 0.0
        else:
            score = 1 - (scoreSA - 2) / 3
        return score


if __name__ == "__main__":
    res = Benchmark(["Cc1ccc(-c2c3ccccc3nc3cc(N4C(=O)CC(C)(c5ccc6c(c5)C(=O)N(C)C6=O)C4=O)ccc23)cc1"])
    print(res.Score)
    print(res.transmittance)
    print(res.cte)
    print(res.strength)
    print(res.tg)
