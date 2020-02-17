#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:     2020-02-17 10:26:08
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 16:28:51

from collections import defaultdict
import ipdb
import logging
import os

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from MomPropEmbeddings import weights

cmap = sns.color_palette()
logging.basicConfig(level='INFO', format='.. %(message)s')


def create(edge_array, node_features):
    """Initializes the graph from an edge-array and a pd.DataFrame 
    containing the node-scores.

    Args:
        edge_arr: array-like, shape=(n_edges, 2), contains edges of 
            network.
        node_features: pd.DataFrame with indices corresponding to genes, 
            columns corresponding to node-features. 

    Returns:
        graph: igraph object.
    """

    tmp_graph = igraph_from_edge_arr(edge_array)

    # find the nodes with a score.
    vertices_to_keep = node_features.index.values

    # remove those nodes from the network that do not have features.
    graph = filter_igraph_by_vertex_names(tmp_graph, 
        vertices_to_keep)

    # find the nodes with degree > 0:
    vertices_to_keep = [v['name'] for v in graph.vs if v.degree() > 0 ]
    graph = filter_igraph_by_vertex_names(graph, vertices_to_keep)

    # add the scores to the vertices.
    vertices = graph.vs['name']

    for x in node_features.columns:
        feat = [float(node_features.loc[v][x]) for v in vertices]
        graph.vs[x] = feat

    # add the k1-neighbors.
    neigh = [graph.neighborhood(v.index, order=1) for v in graph.vs]
    neigh = [[x for x in lst if x != graph.vs[v].index] for v, lst in 
        enumerate(neigh)]
    graph.vs['k1_neighbors'] = neigh

    # add the weights of k1-neighbors.
    edge_dict = create_edge_dict_from_array(edge_array)
    k1_weights = [[edge_dict[v['name']][graph.vs[x]['name']] for
        x in v['k1_neighbors']] for v in graph.vs]
    graph.vs['k1_stdWeights'] = k1_weights

    # add the doremi-weights to the k1-neighbors.
    graph.vs['k1_doremiWeights'] = weights.doremi_weights_k1(graph)

    return graph


# This function can be simplified, there is no need to generate the edge-map,
# but the graph can be generated from edges directly.
def igraph_from_edge_arr(edge_arr):

    v_map, vertices = make_vertice_map_from_edge_arr(edge_arr)
    edges_int = [(v_map[x[0]], v_map[x[1]]) for x in edge_arr if x[1] != x[0]]
    edges_names = [f'{x[0]}_{x[1]}' for x in edge_arr if x[1] != x[0]]
    edges_weights = [float(x[2]) for x in edge_arr if x[1] != x[0]]

    graph = ig.Graph()
    graph.add_vertices(len(vertices))
    graph.vs['name'] = vertices
    graph.add_edges(edges_int)

    graph.es['weight'] = edges_weights 
    graph.es['name'] = edges_names

    return graph


def filter_igraph_by_vertex_names(graph_obj, vertices_to_keep):
    """Deletes vertices from a graph (by attribute 'name') if they are not 
    listed in vertices_to_keep.
    """
    del_vertices = []
    for vertex in graph_obj.vs:
        if vertex['name'] not in vertices_to_keep:
            del_vertices.append(vertex.index)
    graph_obj.delete_vertices(del_vertices)

    return graph_obj


def create_edge_dict_from_array(edge_array):
    """Edge dict is way faster for requesting single edge-weights.
    """
    edict = defaultdict(dict)
    for edge in edge_array:
        edict[edge[0]][edge[1]] = float(edge[2])
        edict[edge[1]][edge[0]] = float(edge[2])
    return edict


def make_vertice_map_from_edge_arr(edge_arr):
    # read the vertex-names.
    vertices = np.unique(edge_arr[:, 0:2])
    v_map = {vertex: idx for idx, vertex in enumerate(vertices)}
    return v_map, vertices


def main():
    pass

if __name__ == '__main__':
        main()