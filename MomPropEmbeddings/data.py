#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:     2020-02-17 16:31:49
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 17:29:40

import ipdb
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
import seaborn as sns

from MomPropEmbeddings import nw_methods

cmap = sns.color_palette()
logging.basicConfig(level='INFO', format='.. %(message)s')


class CancerGenes():

    def __init__(self, cosmic_file):
        self.cosmic_genes = self._init_cosmic(cosmic_file)


    def _init_cosmic(self, cosmic_file):
        df = pd.read_csv(cosmic_file, sep='\t') 
        return list(df['Gene Symbol'].values)


class BaseData():

    def __init__(self, basegraph_pkl):
        self.basegraph_pkl = basegraph_pkl


    def extract_pickle(self):
        """Reads the data from the pickled igraph object into a dict. """
        with open(self.basegraph_pkl, 'rb') as fin:
            data = pickle.load(fin)
        data = {x: data.vs[x] for x in data.vs.attributes()}
        return data


class Adjacency():

    def __init__(self, basegraph_pkl, weighted, sparse):
        self.basegraph_pkl = basegraph_pkl
        self.weighted = weighted
        self.sparse = sparse

        # load the data, stored in igraph.
        self.data = self.load_data()
        self.n_nodes = len(self.data.vs)
        self.matrix = self.adjacency()


    def load_data(self):
        """Loads the data from pickle file into memory. """
        with open(self.basegraph_pkl, 'rb') as fin:
            data = pickle.load(fin)
        return data


    def adjacency(self):
        """Creates the adjacency matrix. """
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=float)

        # fill values in adjacency.
        for v_idx, vertex in enumerate(self.data.vs):
            for n_idx, weight in \
                    zip(vertex['k1_neighbors'], vertex['k1_stdWeights']):
                if self.weighted:
                    adj_mat[v_idx, n_idx] = weight
                else:
                    adj_mat[v_idx, n_idx] = 1

        # create sparse adjacency.
        if self.sparse:
            adj_mat = scipy.sparse.lil_matrix(adj_mat)

        return adj_mat


class MomProp(BaseData):
    """Class to create the WL feature set. 

    Args:
        basegraph_pkl: full-path filename to the pickle-file 
            containing the igraph data object.
        feature: feature that should be used to 
            represent each node in the network.
        n_steps: number of propagation steps.
        n_hops: number of neighborhood-hops that should be used.
        moments: list of moments to represent neighborhood with.
        edge_weight: which type of edge-weight to use.
        path_weight: which type of path-weight to use.
    """

    def __init__(self, basegraph_pkl, feature, n_steps, n_hops=2,
        moments=['mean'], edge_weight='none', path_weight='mean'):

        super().__init__(basegraph_pkl)

        self.feature = feature
        self.moments = moments

        # extract the igraph data.
        self.__data_dict = self.extract_pickle()

        # Init the phenotypes and sample names.
        self.y = np.asarray(self.__data_dict['class_label'])
        self.samples = np.asarray(self.__data_dict['name'])

        F = self.create_feature_lst(moments, edge_weight, n_hops)
        X = self.query_features_from_graph(F)

        # load the adjacency.
        A = Adjacency(
            basegraph_pkl=basegraph_pkl, 
            weighted=False, 
            sparse=True
        )
        # Propagation.
        W, F = nw_methods.propagate(X, A.matrix, n_steps, F)
        # store.
        self.X = W
        self.features = F
        

    def create_feature_lst(self, moments, edge_weight, n_hops):
        """Creates the names of features to propagate. """

        F = [self.feature]
        if len(moments) == 1 and moments[0] == 'anchor':
            return F

        for m in moments:
            for k in range(1, n_hops+1):
                feature = f'{m}_k{k}_{self.feature}_{edge_weight}Weight'
                F.append(feature)
        return F


    def query_features_from_graph(self, feature_names):
        """Creates a dataset with those features in self.features. """
        data_list = []
        for x in feature_names:
          data_list.append(self.__data_dict[x])
        data_array = np.asarray(data_list).T
        return data_array




def main():
        pass

if __name__ == '__main__':
        main()
