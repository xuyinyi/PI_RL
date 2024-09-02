import os
import numpy as np
import pandas as pd


class MoleculeCSVDataset(object):
    def __init__(self, df, cache_file_path, load=True, error_log=None):
        self.df = df
        self.cache_file_path = cache_file_path
        self._prepare(load, error_log)

    def _prepare(self, load, error_log):
        if os.path.exists(self.cache_file_path) and load:
            print('Loading vectors ...')
            vectors = np.load(self.cache_file_path)
            self.vectors = vectors[:, :-1]
            self.values = vectors[:, -1]
        else:
            print('Prepare to convert smiles into vectors ...')
            self.vectors = self.df.iloc[:, 4:]

            if error_log is not None:
                failed_idx, failed_smis = [], []
                df = pd.DataFrame({'raw_id': failed_idx, 'smiles': failed_smis})
                df.to_csv(error_log, index=False)

            _label_values = self.df['value_mean']
            self.values = np.nan_to_num(_label_values).astype(np.float32)
            vectors = np.concatenate((self.vectors, (self.values).reshape(-1, 1)), axis=1)
            np.save(self.cache_file_path, vectors)

        self.smiles = self.df['smile']

    def __getitem__(self, index):
        return self.vectors[index], self.values[index], self.smiles[index]

    def __len__(self):
        return len(self.df['smile'])
