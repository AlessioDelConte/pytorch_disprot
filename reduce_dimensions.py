import os
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from tqdm import tqdm

from parse_features import read_pssm, read_spd3


def pca_features(features, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(features)
    return pca


def train_pca(pca, features):
    pca.fit(features)
    return pca


if __name__ == '__main__':
    feature = 'spd3'
    feature_parse_f = read_spd3

    pca = PCA(n_components=5)
    feature_files = Path(f'data/features/{feature}').glob(f'*.{feature}')
    disorder_train = set(pd.read_csv('data/splits/disorder_train.tsv', sep='\t', header=None).iloc[:, 0])
    feature_train_files = [p for p in feature_files if p.stem in disorder_train]
    for file in tqdm(feature_train_files):
        pca = train_pca(pca, feature_parse_f(file))

    print(pca.explained_variance_ratio_)

    dump(pca, f'data/features/{feature}_pca.joblib')

