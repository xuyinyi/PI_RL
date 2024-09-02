"""
This script aims to load data and generate map-style dataset for further implementation.
    -Import raw dataset
"""
import os
import pandas as pd
from rdkit import Chem
from sklearn import preprocessing
from scaler import Standardization, zero_oneNormalization

ATOM_REMOVE = ['Sn', 'As', 'Ti', 'Ca', 'Fe']


def get_canonical_smiles(smiles):
    smi_list = []
    for s in smiles:
        try:
            smi_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
        except:
            print('Failed to generate the canonical smiles from ', s, ' . Please check the inputs.')

    return smi_list


def import_dataset(params):
    CurrentPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_NAME = params['Dataset']
    path = os.path.join(CurrentPath, 'datasets/' + DATASET_NAME + '_descriptors_ridge_origin.csv')
    df = pd.read_csv(path)

    # remove heavy metal atoms:
    for i in ATOM_REMOVE:
        df.drop(df[df.smile.str.contains(i)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    x = df.iloc[:, 4:]
    num_des = x.shape[-1]
    Norm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x = Norm.fit_transform(x)
    df.iloc[:, 4:] = x

    if not params['sigmoid']:
        Scaling = Standardization(df['value_mean'])
        df['value_mean'] = Scaling.Scaler(df['value_mean'])
    else:
        Scaling = zero_oneNormalization()
        df['value_mean'] = Scaling.Scaler(df['value_mean'])
    df['smile'] = get_canonical_smiles(df['smile'])

    return df, Scaling, Norm, num_des
