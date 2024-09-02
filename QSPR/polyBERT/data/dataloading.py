import os
import pandas as pd
from rdkit import Chem
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
    CurrentPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATASET_NAME = params['Dataset']
    path = os.path.join(CurrentPath, 'raw_data/' + DATASET_NAME + '.csv')
    df = pd.read_csv(path)

    # remove heavy metal atoms:
    for i in ATOM_REMOVE:
        df.drop(df[df.smile.str.contains(i)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    if not params['sigmoid']:
        Scaling = Standardization(df['value_mean'])
        df['value_mean'] = Scaling.Scaler(df['value_mean'])
    else:
        Scaling = zero_oneNormalization()
        df['value_mean'] = Scaling.Scaler(df['value_mean'])
    df['smile'] = get_canonical_smiles(df['smile'])

    return df, Scaling
