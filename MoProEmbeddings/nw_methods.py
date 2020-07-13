#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:     2020-02-17 16:45:59
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 16:50:20

import ipdb
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

cmap = sns.color_palette()
logging.basicConfig(level='INFO', format='.. %(message)s')


def propagate(X, A, n_steps, feature_names=None):
    """Propagates the features of matrix X, using the adjacency matrix. """

    n_features = X.shape[1]

    if feature_names is None:
        F = [[f'feature_{i}' for i in range(n_features)]]
    else:
        F = [feature_names]

    # get degree for normalization.
    D = 1/np.sum(A, axis=1)
    D_ = np.tile(D, n_features)
    W = [X]
    for x in range(n_steps):
        W_ = A * W[-1] 
        W_ = np.multiply(W_, D_)
        W.append(W_)
        F_ = [f'{n}_WLstep_{x+1}' for n in F[0]]
        F.append(F_)
    W_out = np.concatenate(W, axis=1)
    F_out = [x for y in F for x in y]
    return W_out, F_out


def main():
    pass

if __name__ == '__main__':
    main()
