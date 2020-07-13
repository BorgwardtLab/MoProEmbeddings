#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:     2020-02-17 14:37:18
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 16:27:21

from collections import defaultdict
import ipdb
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

cmap = sns.color_palette()
logging.basicConfig(level='INFO', format='.. %(message)s')



class ShortestPathWeights():
    """Class to retrieve the shortest path weights of k-hop neighbors in 
    the network.

    The shortest path weight is not defined in the classical sense. It 
    corresponds to the maximal weight between two nodes that can be 
    connected by at least (k-1) edges.
    The computations are done iteratively, by increasing the depth of the 
    search.
    """

    def __init__(self, graph, max_k=2, mode='max', weight='std'):

        self.max_k = max_k
        self.mode = mode
        self.weight = weight
        self.graph = graph
        self.edge_dict = self.create_edge_dict()

        if self.max_k > 2 and self.mode == 'mean':
            raise NotImplementedError (
                'The generation of mean weights with the current ' \
                'iterative scheme is incorrect for values of k>2.'
            )

        # compute the shortest-path-weights.
        self.shortest_path_weights = defaultdict(dict)
        for vertex in graph.vs:
            print(f'@ vertex {vertex.index} ({max_k} hops)', end='\r')
            self.shortest_path_weights[vertex.index] = \
                self.process_vertex(vertex)
        pass


    def create_edge_dict(self):
        """Creates an edge-dict (constant time access).
        """
        edict = defaultdict(dict)
        for edge in self.graph.es:
            edict[edge.source][edge.target] = edge['weight']
            edict[edge.target][edge.source] = edge['weight'] 
        return edict


    def process_vertex(self, vert):
        # initialize the weight-dictionary with the 1-hop weights. This will 
        # be updated iteratively.
        if 'doremi' in self.weight:
            weight_dict = {n: w for n, w in 
                zip(vert['k1_neighbors'], vert['k1_doremiWeights'])}
        elif 'std' in self.weight:
            weight_dict = {n: w for n, w in 
                zip(vert['k1_neighbors'], vert['k1_stdWeights'])}
        else:
            raise NotImplementedError

        k = 2
        while k <= self.max_k:

            # get the k-hop neighbors for which weights should be computed.
            next_hop = self.graph.neighborhood(vert, order=k)

            # filter already-processed and anchor.
            next_hop = [x for x in next_hop 
                if not x in weight_dict and not x == vert.index]
            weight_dict = self.get_k_layer_weights(weight_dict, next_hop)
            k += 1

        return weight_dict


    def get_k_layer_weights(self, weight_dict, k_layer_vertices):
        """Finds the weights for vertices in the k-hop neighborhood, 
        given the vertices in the k-hop neighborhood, and the weights of 
        the nodes in the (k-1)-hop neighborhood. 
        """

        for tmp_vertex in k_layer_vertices:

            # get the parents in the (k-1)-th layer for the vertex in the 
            # k-th layer.
            targets = self.graph.vs[tmp_vertex]['k1_neighbors']
            parents = list(
                set.intersection(set(targets), set(weight_dict.keys()))
            )
            parents = [x for x in parents if not x in k_layer_vertices]

            if 'doremi' in self.weight:
                # compute penalty of the target node.
                target_correction = \
                    1 + sum(self.edge_dict[tmp_vertex].values())
                weights = [
                    self.edge_dict[tmp_vertex][x] / ((target_correction - \
                    self.edge_dict[tmp_vertex][x])**0.34) * weight_dict[x] 
                    for x in parents
                ]
            elif 'std' in self.weight:
                weights = [
                    self.edge_dict[tmp_vertex][x] * weight_dict[x] 
                    for x in parents
                ]
            else:
                raise NotImplementedError

            if self.mode == 'max':
                weight_dict[tmp_vertex] = max(weights)
            elif self.mode == 'mean':
                weight_dict[tmp_vertex] = np.mean(weights)

        return weight_dict


def main():
    pass

if __name__ == '__main__':
        main()
