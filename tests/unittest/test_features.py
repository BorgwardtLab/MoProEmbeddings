#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:   2020-02-17 16:04:11
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 16:26:13

import ipdb
import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from MomPropEmbeddings import basegraph, features

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
        self.weight_dict = {
            'A': {'B': 0.5, 'C': 0.25}, 
            'B': {'A': 0.5, 'C': 0.5, 'D': 0.05},
            'C': {'A': 0.25, 'B': 0.5, 'D': 0.1, 'E': 0.1}, 
            'D': {'B': 0.05, 'C': 0.1, 'E': 1.0, 'F': 1.0},
            'E': {'C': 0.1, 'D': 1.0, 'F': 1.0},
            'F': {'D': 1.0, 'E': 1.0} 
        }


    def testFeaturesNoneWeight1Hop(self):
        """Test generation of features without using weights, and 
        without prior log-transformation. """

        bg = basegraph.create(self.edges, self.scores)

        # test mean.
        bg_mean, _ = features.weighted_attr_kx(
            bg, 'pvalue', stat='mean', 
            k=1, weight='none', weight_dict=None
        )

        # check that the values match.
        attribute = f'mean_k1_pvalue_noneWeight'
        true_mean = {
            'A': 0.2, 'B': 0.3, 'C': 0.5, 'D': 0.06, 'E':0.6, 'F': 0.02
        }
        for v in bg.vs:
            self.assertTrue(
                np.abs(v[attribute] - true_mean[v['name']]) < 1e-8
            )
        pass


    def testFeaturesNoneWeight2Hop(self):
        """Test generation of features without using weights, and 
        without prior log-transformation. """

        bg = basegraph.create(self.edges, self.scores)

        # test mean.
        bg_mean, _ = features.weighted_attr_kx(
            bg, 'pvalue', stat='mean', 
            k=2, weight='none', weight_dict=None
        )

        # check that the values match.
        attribute = f'mean_k2_pvalue_noneWeight'
        true_mean = {
            'A': 0.1, 'B': 0.8, 'C': 0.26, 'D': 0.3, 'E':0.1, 'F': 0.8
        }
        for v in bg.vs:
            self.assertTrue(
                np.abs(v[attribute] - true_mean[v['name']]) < 1e-8
            )
        pass


def main():

    pass

if __name__ == '__main__':
    main()