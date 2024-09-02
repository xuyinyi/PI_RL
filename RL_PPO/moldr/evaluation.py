import os
import rdkit
import scipy.sparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from functools import partial
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from scipy.spatial.distance import cosine as cos_distance
from RL_PPO.moldr.utils import mapper, get_mol, canonic_smiles


class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError


class FragMetric(Metric):
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        Frag = cos_similarity(pref['frag'], pgen['frag'])
        print("Frag:", Frag)
        return Frag


class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """
    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs, fp_type=self.fp_type)}

    def metric(self, pref, pgen):
        SNN = average_agg_tanimoto(pref['fps'], pgen['fps'], device=self.device)
        print("SNN:", SNN)
        return SNN


def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = AllChem.FragmentOnBRICSBonds(get_mol(mol))
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


def compute_fragments(mol_list, n_jobs=1):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments


def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:
     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)


def fingerprint(smiles_or_mol, fp_type='maccs', dtype=None, morgan__r=3, morgan__n=2048, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits
    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n), dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args, **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array, return_inverse=True)

    fps = mapper(n_jobs)(
        partial(fingerprint, *args, **kwargs), smiles_mols_array
    )

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps


def average_agg_tanimoto(stock_vecs, gen_vecs, batch_size=5000, agg='max', device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac ** p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)
    return np.mean(agg_tanimoto)


def cal_novelty(train_PI_fps, gen_PI_fps):
    fraction_similar = 0
    for i in range(len(gen_PI_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(gen_PI_fps[i], train_PI_fps)
        if max(sims) >= 0.4:
            fraction_similar += 1
    novelty = 1 - fraction_similar / len(gen_PI_fps)
    print('novelty:', novelty)

    return novelty


def cal_div(gen_PI_fps):
    similarity = 0
    for i in tqdm(range(len(gen_PI_fps))):
        sims = DataStructs.BulkTanimotoSimilarity(gen_PI_fps[i], gen_PI_fps[:i])
        similarity += sum(sims)

    n = len(gen_PI_fps)
    n_pairs = n * (n - 1) / 2
    diversity = 1 - similarity / n_pairs
    print('diversity:', diversity)

    return diversity


def cal_fraction_unique(gen, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique.")
    unique = len(canonic) / len(gen)
    print('unique:', unique)

    return unique


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    train_PI_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "raw_data\PI.csv")
    train_PI_smiles = pd.read_csv(train_PI_path)["smile"].values.tolist()
    train_PI_mols = [Chem.MolFromSmiles(s) for s in train_PI_smiles]
    train_PI_mols = [x for x in train_PI_mols if x is not None]
    train_PI_fps = [Morgan(x, 3, 2048) for x in train_PI_mols]

    gen_PI_path = r"xxx\xxx\xxx\xxx.csv"
    gen_PI_df = pd.read_csv(gen_PI_path)
    gen_PI_smiles = gen_PI_df[gen_PI_df["PI"] != "None"]["PI"].values.tolist()
    gen_PI_mols = [Chem.MolFromSmiles(s) for s in gen_PI_smiles]
    gen_PI_mols = [x for x in gen_PI_mols if x is not None]
    gen_PI_fps = [Morgan(x, 3, 2048) for x in gen_PI_mols]

    novelty = cal_novelty(train_PI_fps=train_PI_fps, gen_PI_fps=gen_PI_fps)
    diversity = cal_div(gen_PI_fps=gen_PI_fps)
    unique = cal_fraction_unique(gen=gen_PI_smiles)
    Frag = FragMetric()(ref=train_PI_smiles, gen=gen_PI_smiles)
    SNN = SNNMetric()(ref=train_PI_smiles, gen=gen_PI_smiles)
