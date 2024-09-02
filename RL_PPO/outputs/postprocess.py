import os
import math
import gzip
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from umap import UMAP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem as Chem
from RL_PPO.utils.polyBERT import Embedding_smiles
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FpScoresPath = r'xxx\xxx\xxx\model\fpscores.pkl.gz'
FpScoresData = pickle.load(gzip.open(FpScoresPath))
outDict = {}
for i in FpScoresData:
    for j in range(1, len(i)):
        outDict[i[j]] = float(i[0])
FpScores = outDict


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


def merge_property_epoch(basedir):
    property_list = ["reward", "transmittance", "cte", "strength", "tg", "SaScore"]
    for p in property_list:
        df = pd.DataFrame()
        if p == "reward":
            pass
        elif p == "transmittance":
            data = pd.read_csv(r'raw_data\transmittance(400).csv')
            property = data["value_mean"].values
            df = pd.concat([df, pd.DataFrame(data=property, columns=[f"dataset"])], axis=1)
        else:
            data = pd.read_csv(f'raw_data\{p}.csv')
            property = data["value_mean"].values
            df = pd.concat([df, pd.DataFrame(data=property, columns=[f"dataset"])], axis=1)
        data = pd.read_csv(os.path.join(basedir, f'random/generate_dark_random.csv'))
        property = data[data["PI"] != "None"][p].values
        df = pd.concat([df, pd.DataFrame(data=property, columns=[f"random"])], axis=1)

        for i in range(1, 8):
            epoch = i * 10
            data = pd.read_csv(os.path.join(basedir, f'xxx/xxx/generate_{epoch}.csv'))
            property = data[data["PI"] != "None"][p].values
            df = pd.concat([df, pd.DataFrame(data=property, columns=[f"Epoch-{epoch}"])], axis=1)

        df.to_csv(os.path.join(basedir, f'xxx/{p}.csv'), index=False)


def random_sample(size=50000):
    base_path = 'forward_generation'
    file_path = os.path.join(base_path, 'Summary.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    columns = pd.read_csv(file_path).columns.tolist()
    data = pd.read_csv(file_path).values
    total_samples = data.shape[0]
    if size > total_samples:
        raise ValueError(f"Requested sample size {size} exceeds total number of samples {total_samples}.")

    sampled_indices = np.random.choice(total_samples, size=size, replace=False)
    sampled_data = data[sampled_indices]

    save_path = os.path.join(base_path, f'Summary_{size}.csv')
    pd.DataFrame(data=sampled_data, columns=columns).to_csv(save_path, index=False)


def Umap(smile_list, reward_list, epoch=None, fit=False, batch_size=50):
    model_path = Path(__file__).resolve().parent.parent / "models"
    embedding_dim = 600

    # record_temp = dict()

    results = []
    for i in tqdm(range(0, len(smile_list), batch_size)):
        batch = smile_list[i:i + batch_size]
        batch_vecs = Embedding_smiles(model_path, batch)
        reshape_batch_vecs = batch_vecs.reshape(batch_size, embedding_dim)
        results.append(reshape_batch_vecs)
        # for smile, vec in zip(batch, batch_vecs):
        #     record_temp[smile] = vec

    matrix = np.vstack(results)
    print(matrix.shape)

    if fit:
        with open('PPO/models/random/transformer_umap_light_random.pkl', 'rb') as fw:
            umap_ = pickle.load(fw)
        embedding = umap_.transform(matrix)
        print(embedding.shape)
        result = np.concatenate((embedding, reward_list), axis=1)
        # pd.DataFrame(data=result, columns=["umap1", "umap2", "reward"]).to_csv(
        #     f'PPO/models/2024-01-08_00-49-04/epoch_{epoch}/umap_trained.csv', index=False)
        pd.DataFrame(data=result, columns=["umap1", "umap2"] + columns).to_csv(
            f'forward_generation/Summary_50000_umap.csv', index=False)

    else:
        # Create a UMAP converter and reduce the dimensions
        umap_ = UMAP(n_components=2)
        embedding = umap_.fit_transform(matrix)
        with open('PPO/models/random/transformer_umap_dark_random.pkl', 'wb') as fw:
            pickle.dump(umap_, fw)

        print(embedding.shape)
        result = np.concatenate((embedding, reward_list), axis=1)
        pd.DataFrame(data=result, columns=["umap1", "umap2", "reward"]).to_csv(
            'PPO/models/random/dark_random_umap.csv', index=False)


def Cluster(k=None):
    data = pd.read_csv(os.path.join(basedir, f'xxx/xxx/xxx.csv'))
    if not k:
        n_value = [_ for _ in range(2, 11)]
        wcss, silhouette_scores = [], []
        for n in n_value:
            kmeans = KMeans(n_clusters=n)
            kmeans.fit(data.values[:, :2])
            labels = kmeans.labels_
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, labels))

        pd.DataFrame.from_dict({"n": n_value, "silhouette_score": silhouette_scores, "wcss": wcss}).to_csv(
            os.path.join(basedir, f'xxx/xxx/xxx.csv'), index=False)
    else:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data.values[:, :2])
        labels = kmeans.labels_
        pd.concat((data, pd.DataFrame(labels.reshape(-1, 1), columns=["label"])), axis=1).to_csv(
            os.path.join(basedir, f'xxx/xxx/xxx.csv'), index=False)


if __name__ == "__main__":
    basedir = "PPO/"

    df = pd.read_csv('forward_generation/Summary_50000.csv')
    columns = df.columns.tolist()
    smile_list = df.values[:, 0].tolist()
    reward_list = df.values
    Umap(smile_list=smile_list, reward_list=reward_list, fit=True)
