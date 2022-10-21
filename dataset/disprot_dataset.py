import os
import warnings
from typing import List

import torch
from joblib import load
from torch.utils.data import Dataset

from dataset.encoding import Uniprot21
from dataset.utils import parse_target, read_pssm, read_spd3


class Sequence:
    def __init__(self, seq_id, sequence, target=None, load_pssm=False, load_spd3=False, features_path=None,
                 data_transform=None, target_transform=None, pssm_pca=None, spd3_pca=None, pca=False):
        self.seq_id = seq_id
        self.sequence = sequence
        self.encoded_sequence = torch.tensor(Uniprot21().encode(sequence), dtype=torch.uint8).reshape(1, -1)
        self._target = None
        if target is not None:
            self._target = parse_target(target)

        self.data_transform = data_transform
        self.target_transform = target_transform

        self.pssm, self.spd3 = None, None
        if load_pssm and features_path is not None:
            pssm = read_pssm(os.path.join(features_path, 'pssm/{}.pssm'.format(self.seq_id)))
            if pca:
                pssm = pssm_pca.transform(pssm)
            self.pssm = torch.tensor(pssm, dtype=torch.float32)

        if load_spd3:
            spd3 = read_spd3(os.path.join(features_path, 'spd3/{}.spd3'.format(self.seq_id)))
            if pca:
                spd3 = spd3_pca.transform(spd3)
            self.spd3 = torch.tensor(spd3, dtype=torch.float32)

    @property
    def data(self):
        data = self.encoded_sequence

        if self.pssm is not None:
            data = torch.cat((self.encoded_sequence, self.pssm.T), dim=0)

        if self.spd3 is not None:
            data = torch.cat((data, self.spd3.T), dim=0)

        if self.data_transform is not None:
            data = self.data_transform(data)
        return data.float()

    @property
    def target(self):
        target = self._target
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target.float()

    @property
    def clean_target(self):
        if self._target is None:
            return self._target
        return self._target.numpy()

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        return 'Sequence({}, {})'.format(self.seq_id, self.sequence)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, i):
        return self.data, self.target

    def as_dict(self):
        return {"seq_id": self.seq_id, "sequence": self.sequence, "target": self.target, "data": self.data}


# Base class for the two datasets, with common functionality
class DisprotDataset(Dataset):
    def __init__(self, data, feature_root='../data/features', transform=None, target_transform=None, pssm=False,
                 spd3=False, pca=False):
        # Define the encoder fot the sequence
        self.encoder = Uniprot21()

        self.transform = transform
        self.target_transform = target_transform

        self.raw_data = data

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            pssm_pca = load(os.path.join(feature_root, 'pssm_pca.joblib'))
            spd3_pca = load(os.path.join(feature_root, 'spd3_pca.joblib'))

        # Create sequences objects
        self.data = []
        for seq_id, sequence, target in self.raw_data.itertuples(index=False):
            # Split the sequence with a moving window of size 20, add padding at the end
            self.data.append(
                    Sequence(seq_id, sequence, target, load_pssm=pssm, load_spd3=spd3, features_path=feature_root,
                             data_transform=self.transform, target_transform=self.target_transform,
                             pssm_pca=pssm_pca, spd3_pca=spd3_pca, pca=pca))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: List[Sequence]):
    data = torch.stack([item.data for item in batch])
    target = torch.stack([item.target for item in batch])
    return batch, data, target
