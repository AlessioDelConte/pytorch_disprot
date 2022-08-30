import numpy as np
import pandas as pd
import torch


class PadRightTo(object):
    """Pad the tensor to a given size.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        padding = self.output_size - sample.size()[-1]
        return torch.nn.functional.pad(sample, (0, padding), 'constant', 0)


def read_pssm(pssm_path):
    pssm = pd.read_csv(pssm_path, sep='\s+', skiprows=3, header=None)
    pssm = pssm.dropna().values[:, 2:22].astype(np.int8)

    return pssm


def parse_target(x):
    return torch.tensor([int(y) for y in x], dtype=torch.uint8)


def spd3_feature_sincos(x, seq):
    ASA = x[:, 0]
    asa_rnm = {'A': 115, 'C': 135, 'D': 150, 'E': 190, 'F': 210, 'G': 75, 'H': 195, 'I': 175, 'K': 200, 'L': 170,
               'M': 185, 'N': 160, 'P': 145, 'Q': 180, 'R': 225, 'S': 115, 'T': 140, 'V': 155, 'W': 255, 'Y': 230}

    ASA_div = np.array([asa_rnm[aa] if aa in asa_rnm else 1 for aa in seq])
    ASA = (ASA / ASA_div)[:, None]
    angles = x[:, 1:5]
    HCEprob = x[:, -3:]
    angles = np.deg2rad(angles)
    angles = np.concatenate([np.sin(angles), np.cos(angles)], 1)
    return np.concatenate([ASA, angles, HCEprob], 1)


def read_spd3(spd3_path):
    spd3_features = pd.read_csv(spd3_path, sep='\s+')
    seq = spd3_features.AA.to_list()
    spd3_features = spd3_features.values[:, 3:].astype(float)
    spd3 = spd3_feature_sincos(spd3_features, seq)
    return spd3
