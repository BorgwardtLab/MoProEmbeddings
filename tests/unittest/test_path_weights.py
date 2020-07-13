#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:   2020-02-17 12:02:34
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 15:43:49

from collections import defaultdict
import ipdb
import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from MoProEmbeddings import basegraph, path_weights

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


    def testShortestPaths(self):
        """Compute shortest path weights. """

        true_weights = {
            'A': {'B': 0.5, 'C': 0.25}, 
            'B': {'A': 0.5, 'C': 0.5, 'D': 0.05},
            'C': {'A': 0.25, 'B': 0.5, 'D': 0.1, 'E': 0.1}, 
            'D': {'B': 0.05, 'C': 0.1, 'E': 1.0, 'F': 1.0},
            'E': {'C': 0.1, 'D': 1.0, 'F': 1.0},
            'F': {'D': 1.0, 'E': 1.0} 
        }

        bg = basegraph.create(self.edges, self.scores)

        weights = path_weights.ShortestPathWeights(
            bg, max_k=2, mode='max', weight='std')

        for root, d in weights.shortest_path_weights.items():
            root_node = bg.vs[root]['name']
            for neigh, weight in d.items():
                neigh_node = bg.vs[neigh]['name']
                self.assertTrue(
                    true_weights[root_node][neigh_node] == weight
                )
        pass


    def testShortestPathsMaxModified(self):
        """Add additional node and compute shortest path. """

        # add new node G, with connections to B and D.
        new_edges = np.asarray([['B', 'G', '0.5'], ['G', 'D', '0.3']])
        self.edges = np.concatenate((self.edges, new_edges), axis=0)
        self.scores.at['G'] = 0.3

        true_weights = {
            'A': {'B': 0.5, 'C': 0.25, 'G': 0.25}, 
            'B': {'A': 0.5, 'C': 0.5, 'G': 0.5, 'D': 0.15},
            'C': {'A': 0.25, 'B': 0.5, 'D': 0.1, 'E': 0.1, 'G': 0.25}, 
            'D': {'B': 0.15, 'C': 0.1, 'G': 0.3, 'E': 1.0, 'F': 1.0},
            'E': {'C': 0.1, 'D': 1.0, 'F': 1.0, 'G': 0.3},
            'F': {'D': 1.0, 'E': 1.0},
            'G': {'A': 0.25, 'B': 0.5, 'C': 0.25,  'D': 0.3, 'E': 0.3} 
        }

        bg = basegraph.create(self.edges, self.scores)

        weights = path_weights.ShortestPathWeights(
            bg, max_k=2, mode='max', weight='std')

        for root, d in weights.shortest_path_weights.items():
            root_node = bg.vs[root]['name']
            for neigh, weight in d.items():
                neigh_node = bg.vs[neigh]['name']
                self.assertTrue(
                    true_weights[root_node][neigh_node] == weight
                )
        pass


    def testShortestPathsMeanModified(self):
        """Add additional node and compute shortest path unding mean.
        This affects the paths between B - D and C - G."""

        # add new node G, with connections to B and D.
        new_edges = np.asarray([['B', 'G', '0.5'], ['G', 'D', '0.3']])
        self.edges = np.concatenate((self.edges, new_edges), axis=0)
        self.scores.at['G'] = 0.3

        true_weights = {
            'A': {'B': 0.5, 'C': 0.25, 'G': 0.25}, 
            'B': {'A': 0.5, 'C': 0.5, 'G': 0.5, 'D': 0.1},
            'C': {'A': 0.25, 'B': 0.5, 'D': 0.1, 'E': 0.1, 'G': 0.14}, 
            'D': {'B': 0.1, 'C': 0.1, 'G': 0.3, 'E': 1.0, 'F': 1.0},
            'E': {'C': 0.1, 'D': 1.0, 'F': 1.0, 'G': 0.3},
            'F': {'D': 1.0, 'E': 1.0},
            'G': {'A': 0.25, 'B': 0.5, 'C': 0.14,  'D': 0.3, 'E': 0.3} 
        }

        bg = basegraph.create(self.edges, self.scores)

        weights = path_weights.ShortestPathWeights(
            bg, max_k=2, mode='mean', weight='std')

        for root, d in weights.shortest_path_weights.items():
            root_node = bg.vs[root]['name']
            for neigh, weight in d.items():
                neigh_node = bg.vs[neigh]['name']
                self.assertTrue(
                    true_weights[root_node][neigh_node] == weight
                )
        pass





if __name__ == '__main__':
    main()
