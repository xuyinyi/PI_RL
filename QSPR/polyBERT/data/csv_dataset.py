import os
import numpy as np
import pandas as pd
import torch


class MoleculeCSVDataset(object):
    def __init__(self, df, model_path, Embedding_smiles, cache_file_path, load=False, log_every=100, error_log=None):
        self.df = df
        self.model = model_path
        self.cache_file_path = cache_file_path
        self._prepare(Embedding_smiles, load, log_every, error_log)

    def _prepare(self, Embedding_smiles, load, log_every, error_log):
        if os.path.exists(self.cache_file_path) and load:
            print('Loading vectors ...')
            vectors = np.load(self.cache_file_path)
            self.vectors = vectors[:, :-2]
            self.values = vectors[:, -2]
            self.valid_idx = vectors[:, -1]
        else:
            print('Prepare to convert smiles into vectors ...')
            self.vectors = []
            for i, s in enumerate(self.df['smile']):
                if (i + 1) % log_every == 0:
                    print('Currently preparing molecule {:d}/{:d}'.format(i + 1, len(self)))
                self.vectors.append(Embedding_smiles(self.model, s))

            self.valid_idx = []
            vectors, failed_smiles = [], []
            for i, vector in enumerate(self.vectors):
                if len(vector) == 600:
                    self.valid_idx.append(i)
                    vectors.append(vector)
                else:
                    failed_smiles.append((i, self.df['smile'][i]))

            if error_log is not None:
                if len(failed_smiles) > 0:
                    failed_idx, failed_smis = map(list, zip(*failed_smiles))
                else:
                    failed_idx, failed_smis = [], []
                df = pd.DataFrame({'raw_id': failed_idx, 'smiles': failed_smis})
                df.to_csv(error_log, index=False)

            self.vectors = np.array(vectors)
            _label_values = self.df['value_mean']
            self.values = (np.nan_to_num(_label_values).astype(np.float32))[self.valid_idx]
            self.valid_idx = np.array(self.valid_idx).reshape(-1, 1)
            vectors = np.concatenate((self.vectors, (self.values).reshape(-1, 1), self.valid_idx), axis=1)
            np.save(self.cache_file_path, vectors)

        self.smiles = [self.df['smile'][i] for i in self.valid_idx]

    def __getitem__(self, index):
        return self.vectors[index], self.values[index], self.smiles[index]

    def __len__(self):
        return len(self.df['smile'])