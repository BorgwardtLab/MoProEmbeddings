#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:   2020-02-17 12:02:34
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 14:09:35

from collections import defaultdict
import ipdb
import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from MomPropEmbeddings import basegraph, weights

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


    def testDimensions(self):
        """Tests that the dimensions of the output matrix are correct. """

        self.assertTrue(self.edges.shape==(5, 3))
        self.assertTrue(self.scores.shape==(6, 1))

        pass


    def testVertexMap(self):
        """Tests the generation of a vertex map, assigining an int index 
        to each unique vertex. 
        """
        v_map, vertices = basegraph.make_vertice_map_from_edge_arr(
            self.edges
        )
        true_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5}

        self.assertTrue(true_map == v_map)
        self.assertTrue(list(vertices) == ['A', 'B', 'C', 'D', 'E', 'F'])

        pass


    def testBuildGraph(self):
        """Tests that the dimensions of the output matrix are correct. """

        graph = basegraph.igraph_from_edge_arr(self.edges)

        # test that gene names are equal.
        self.assertTrue(
            graph.vs['name'].sort() == list(self.scores.index).sort()
        )

        # test that number of edges is identical.
        self.assertTrue(len(graph.es) == len(self.edges))

        # test that edges are correct (1. create true edges, 
        # check that all edges in graph are in data set). 
        edge_list = [(x[0], x[1]) for x in self.edges] + \
            [(x[1], x[0]) for x in self.edges]

        for e in graph.es:
            source_node = graph.vs[e.source]['name']
            target_node = graph.vs[e.target]['name']
            assert((source_node, target_node) in edge_list)
        
        pass


    def testRemoveNodes(self):
        """Test whether removing nodes from graph works. """
        remove_node = 'B'
        graph = basegraph.igraph_from_edge_arr(self.edges)
        orig_vs = graph.vs['name']
        orig_es = graph.es['name']

        # remove vertex.
        vert_to_keep = [x for x in graph.vs['name'] if x != remove_node]
        new_graph = basegraph.filter_igraph_by_vertex_names(graph, 
            vert_to_keep)

        # test that number of genes has been decreased by one.
        self.assertTrue(len(new_graph.vs) == len(orig_vs) - 1 )

        # test that number of edges has decreased by 2.
        self.assertTrue(len(new_graph.es) == len(orig_es) - 2 )

        # test that remove-node is not in gene names.
        self.assertTrue(remove_node not in new_graph.vs['name'])

        # assert that no edge to remove_node exists.
        for e in new_graph.es:
            source_node = new_graph.vs[e.source]['name']
            target_node = new_graph.vs[e.target]['name']
            self.assertTrue(source_node != remove_node)
            self.assertTrue(target_node != remove_node)

        pass


    def testCreateEdgeDict(self):
        """Test creation of edge-dictionary"""
        true_dict = defaultdict(dict)
        for x in self.edges:
            true_dict[x[0]][x[1]] = float(x[2])
            true_dict[x[1]][x[0]] = float(x[2])

        test_dict = basegraph.create_edge_dict_from_array(self.edges)
        self.assertTrue(true_dict == test_dict)


    def testBaseGraph(self):

        # test generated basegraph.
        bg = basegraph.create(self.edges, self.scores)

        # check the neighbors and weights.
        edge_dict = basegraph.create_edge_dict_from_array(self.edges)

        # check the weights.
        for v_idx, v in enumerate(bg.vs):
            graph_weights = \
                {x: y for x, y in 
                zip(v['k1_neighbors'], v['k1_stdWeights'])}
            for n, w in graph_weights.items():
                neigh_name = bg.vs[n]['name']
                self.assertTrue(edge_dict[v['name']][neigh_name] == w)
        pass




if __name__ == '__main__':
    main()