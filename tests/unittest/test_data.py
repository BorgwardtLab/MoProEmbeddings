#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:   2020-02-17 16:59:38
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 17:34:17

import ipdb
import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from MoProEmbeddings import basegraph, features, utils, data

cmap = sns.color_palette()
logging.basicConfig(level='INFO', format='.. %(message)s')


class TestBasegraph(unittest.TestCase):
    """Tests the function data_splits.multisplits_fraction.py  """
    
    def setUp(self):

        self.edges = np.loadtxt('../data/network.txt', dtype=str)
        self.scores = pd.read_csv(
            '../data/features.txt', 
            delim_whitespace=True,
            index_col=0
        )
        self.basegraph_pkl = '../data/graph.pkl'

        if not os.path.isfile(self.basegraph_pkl):
            # create the data.
            bg = basegraph.create(self.edges, self.scores)
            # 1. log-transformation.
            bg = features.log_transform(bg, 'pvalue')
            # 2. Create all features. 
            k_lst = [1, 2]
            features_lst = ['pvalue', 'log_pvalue']
            moment_lst = ['mean', 'std', 'skew', 'kurt']
            for k in k_lst:
                for f in features_lst:
                    for m in moment_lst:
                        bg, _ = features.weighted_attr_kx(
                            bg, f, stat=m, 
                            k=k, weight='none', weight_dict=None
                        )
            # set random phenotype.
            bg.vs['class_label'] = [1, 0, 1, 0, 0, 0]
            # save the pickle.
            utils.write_pickle(bg, self.basegraph_pkl)


    def testAdjacencyUnweighted(self):

        adj = data.Adjacency(self.basegraph_pkl, False, False)
        self.assertTrue(adj.matrix.shape == (6, 6))
        self.assertTrue(np.sum(adj.matrix) == 10)
        self.assertTrue(np.sum(adj.matrix.trace()) == 0)
        pass


    def testAdjacencyWeighted(self):
        # TODO
        pass


    def testMoProDimensions(self):

        n_hops = 2
        n_steps = 2
        moments = ['mean', 'std', 'skew', 'kurt']

        mp_data = data.MoPro(
            self.basegraph_pkl, 'pvalue', n_steps, n_hops=n_hops,
            moments=moments, edge_weight='none', path_weight='max'
        )

        self.assertTrue(
            mp_data.X.shape == (6, (1 + len(moments)*n_hops) * (n_steps+1))
        )

        pass


def main():
    pass
if __name__ == '__main__':
    main()
