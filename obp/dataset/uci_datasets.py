# Copyright (C) 2021 Alberto Maria Metelli, Alessio Russo, and Politecnico di Milano. All rights reserved.
# Licensed under the Apache 2.0 License.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle


DATASETS = {'ecoli': (['ecoli/ecoli.data'], [0], -1, '\s+'),
            'glass': (['glass/glass.data'], [0], -1, ','),
            'isolet': (['isolet/isolet1+2+3+4.data', 'isolet/isolet5.data'], [], -1, ','),
            'kropt': (['kropt/kropt.data'], [], -1, ' '),
            'letter': (['letter/letter-recognition.data'], [], 0, ','),
            'optdigits': (['optdigits/optdigits.tra', 'optdigits/optdigits.tes'], [], -1, ','),
            'page-blocks': (['page-blocks/page-blocks.data'], [], -1, '\s+'),
            'pendigits': (['pendigits/pendigits.tra', 'pendigits/pendigits.tes'], [], -1, ','),
            'satimage': (['satimage/sat.trn', 'satimage/sat.tst'], [], -1, '\s+'),
            'vehicle': (['vehicle/xaa.dat', 'vehicle/xab.dat', 'vehicle/xac.dat',
                         'vehicle/xad.dat', 'vehicle/xae.dat', 'vehicle/xaf.dat',
                         'vehicle/xag.dat', 'vehicle/xah.dat', 'vehicle/xai.dat'], [], -1, '\s+'),
            'yeast': (['yeast/yeast.data'], [0], -1, '\s+')}

DATASETS_REGRESSION = {'ailerons': (['ailerons/ailerons.data'], [], -1, ','),
            'boston': (['boston/Dataset.data'], [], -1, '\s+'),
            'delta-ailerons': (['delta_ailerons/delta_ailerons.data'], [], -1, '\s+'),
            'delta-elevators': (['delta_elevators/delta_elevators.data'], [], -1, '\s+'),
            'housing': (['housing/housing.csv'], [], -1, ','),
            'red-wine': (['red_wine/winequality-red.csv'], [], -1, ';'),
            'white-wine': (['white_wine/winequality-white.csv'], [], -1, ';'),
                       }


def get_uci_datasets_names():
    return DATASETS.keys()

def get_uci_datasets_regression_names():
    return DATASETS_REGRESSION.keys()

def load_uci_dataset_regression(name, base_path, num_samples):
    paths, exclude, target, delimiter = DATASETS_REGRESSION[name]

    dfs = []
    for path in paths:
        _df = pd.read_csv(base_path / path, sep=delimiter, header=None)
        dfs.append(_df)

    df = pd.concat(dfs)

    if target < 0:
        target = df.shape[1] + target

    X = np.delete(df.values, exclude + [target], axis=1).astype(float)
    y = df.iloc[:, target].astype(float)

    to_remove = np.where(np.isnan(X))[0]
    mask = np.ones_like(y, dtype=bool)
    mask[to_remove] = False
    X = X[mask]
    y = y[mask]

    dataset_size = X.shape[0]
    if num_samples < dataset_size:
        mask = np.full(dataset_size, False)
        mask[:num_samples] = True
        np.random.shuffle(mask)
        X = X[mask, :]
        y = y[mask]

    return X, y


def load_uci_dataset(name, base_path, num_samples):
    paths, exclude, target, delimiter = DATASETS[name]

    dfs = []
    for path in paths:
        _df = pd.read_csv(base_path / path, sep=delimiter, header=None)
        dfs.append(_df)

    df = pd.concat(dfs)

    if target < 0:
        target = df.shape[1] + target

    X = np.delete(df.values, exclude + [target], axis=1).astype(float)
    y = df.iloc[:, target].astype('category').cat.codes.values
    n_actions = np.unique(y).shape[0]
    print(f"n_actions in dataset {name} is {n_actions}")
    
    dataset_size = X.shape[0]
    #if num_samples < n_actions:
    #    raise ValueError
    if num_samples < dataset_size:

        mask = np.array([True] * num_samples + [False] * (dataset_size - num_samples))
        while True:
            np.random.shuffle(mask)
            X_ = X[mask, :]
            y_ = y[mask]

            if np.unique(y_).shape[0] >= 2:
                n_actions = np.unique(y_).shape[0]
                X = X_
                y = pd.Series(y_).astype('category').cat.codes.values
                break

    return X, y, n_actions

if __name__ == '__main__':
    ll = []
    for name in get_uci_datasets_regression_names():
        #print(name)
        X, y = load_uci_dataset_regression(name, Path('../../datasets/'))
        print(name, X.shape, len(set(y.tolist())))
        ll.append([X.shape[0], X.shape[1], len(set(y.tolist()))])

    ll = np.array(ll)
    for i in range(ll.shape[1]):
        ss = ""
        for j in range(ll.shape[0]):
           ss = ss + str(ll[j, i]) + ' & '
        print(ss + '\n')
