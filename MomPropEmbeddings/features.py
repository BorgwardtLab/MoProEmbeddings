#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:     2020-02-17 15:47:04
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 17:08:26

import ipdb
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import seaborn as sns

cmap = sns.color_palette()
logging.basicConfig(level='INFO', format='.. %(message)s')


def log_transform(graph, attr):
    """Creates a log-transformed version of the attribute.

    Args:
        graph: igraph object.
        attr: attribute that should be -log10 transformed.
        overwrite: bool, whether or not to overwrite the existing value.

    Returns:
        graph: updated graph.
        bool: indicator of whether the attribute did exist from the start.
    """
    new_attr = f'log_{attr}'
    graph.vs[new_attr] = [-np.log10(v[f'{attr}']) for v in graph.vs]
    return graph


def weight_transform(p, w):
    """Logarithmic weight transform between a p-value in p, and a weight 
    in w.
    """
    log_p = -np.log10(p)
    weight = p / (w ** log_p)
    return np.min((1, weight))


def get_k1_attribute(graph, vertex, attr):
    """Obtain the vertex-attribute 'attr' of the k1-neighbors of the 
    vertex. 
    """
    return [graph.vs[x][attr] for x in vertex['k1_neighbors']]


def get_weight_k1_attribute(graph, vertex, attr, weight='none'):
    """Obtain the vertex-attribute 'attr' of the k1-neighbors of the 
    vertex. The weight argument determines which type of weighting should 
    be done.

    Args:
        weight: which weighting technique to use, i.e. standard, doremi, 
            depth.
    """
    attr_lst = get_k1_attribute(graph, vertex, attr)

    if 'none' in weight:
        return attr_lst
    if 'std' in weight or 'depth' in weight:
        weights = vertex['k1_stdWeights']
    if 'doremi' in weight:
        weights = vertex['k1_doremiWeights']
    
    if 'log' in attr:
        return [w*s for w, s in zip(weights, attr_lst)]
    else:
        return [utils.weight_transform(p, w) for p, w in \
            zip(attr_lst, weights)]


def get_weight_kx_attribute(graph, vertex, attr, weight_dict=None, 
    weight='none', k=2):
    """Returns the weighted vertex attributes in the incremental k-hop 
    neighborhood of the given vertex. 

    Args:
        graph: igraph object
        vertex: vertex-ID (int)
        attr: attribute to obtain (has to be in vertex.attributes())
        weight_dict: weights to use.
        weight: name of the weight to use.
        k: k-value in k-hop neighborhood.
        stat: statistic to compute.

    Returns:
        list of weighted attributes in incremental k-hop neighborhood.
    """

    # Step 1: Get all attributes of the nodes in the k-hop neighborhood.
    if k == 1:
        attr_lst = get_k1_attribute(graph, vertex, attr)
    elif k > 1:
        tmp_kx_vert = incremental_khop_neighbors(graph, vertex, k=k)
        attr_lst = [graph.vs[x][attr] for x in tmp_kx_vert]
    else:
        raise ValueError('k must be in range [1, inf)')

    # If there are no k-hop neighbors, return the least significant value 
    # possible, i.e. 1.0 in the case of raw p-values, 0.0 in the case of 
    # log-transformed p-values.
    if len(attr_lst) == 0:
        if 'log' in attr:
            return [0.0]
        else:
            return [1.0]

    # Step 2. Weight the attributes in the k-hop neighborhood according to 
    # the chosen weights.
    if 'none' in weight:
        return [graph.vs[x][attr] for x in tmp_kx_vert]
    if 'std' in weight or 'doremi' in weight:
        weights = [weight_dict[vertex.index][x] for x in tmp_kx_vert]
    if 'depth' in weight:
        weights = [weight_dict[vertex.index][x]*(0.9**(k-1)) for x in 
                tmp_kx_vert]
    
    if 'log' in attr:
        return [w*s for w, s in zip(weights, attr_lst)]
    else:
        return [utils.weight_transform(p, w) for p, w in \
                zip(attr_lst, weights)]


def weighted_attr_kx(graph, attr, stat='max', k=2, weight='none', 
        weight_dict=None, overwrite=False):
    """Computes the minimum, maximum and median weighted attributes in a 
    vertex' incremental k-hop neighborhood.

    Args:
        graph: igraph object.
        attr: attributes to consider
        k: value of k-hop
        weight: type of weight to use (defaults to none)
        weight_dict: weight-dictionary (defaults to None)
        overwrite: bool, whether or not to overwrite an exisitng 
            attribute.

    Returns:
        graph: input graph with new attributes added.
    """
    stat_dict = {
        'max': np.max,
        'min': np.min,
        'mean': np.mean,
        'std': np.std,
        'skew': skew,
        'kurt': kurtosis,
        'median': np.median
    }

    assert(stat in stat_dict), 'Statistic not available.'

    # set the name of the new attribute to create.
    out_attr = f'{stat}_k{k}_{attr}_{weight}Weight'
    
    # If the attributes already exist and overwrite is not chosen, return.
    if out_attr in graph.vs.attributes() and not overwrite:
        logging.info(f'attributes {out_attr} already exists.')
        return graph, False

    if k == 1:
        if weight == 'depth':
            return graph, False
        neighbor_values = [get_weight_k1_attribute(graph, v, attr, weight) 
            for v in graph.vs
        ]     
    elif k > 1:
        neighbor_values = [
            get_weight_kx_attribute(graph, v, attr, weight_dict, weight, k)
            for v in graph.vs
        ]
    else:
        raise ValueError

    graph.vs[out_attr] = [stat_dict[stat](x) for x in neighbor_values]

    return graph, True


def incremental_khop_neighbors(graph, vertex, k=2):
    """Finds the vertices in a root vertex's k-hop neighborhood. Vertices 
    that are also part in the vertices (k-1) neighborhood are removed.

    Example:
        given the root vertex R, and two more nodes A and B, and assume 
        the following edges: (R,A), (R,B), (A,B). Then the 1-hop 
        neighborhood are vertices (A, B), but (A, B) are also in the 
        two-hop neighborhood. That means, the 2-hop neighborhood is empty.

    Args:
        graph: a igraph object.
        vertex: the root-vertex identifier in graph.
        k: the k-hop value.

    Returns:
        k_hop: list of all vertex-identifiers in the k-hop neighborhood.
    """
    if k==1:
        return graph.neighborhood(vertex, k=1)

    # find the lower k-hop neighbors:
    lower_k_hop = []
    for k_tmp in range(1, k):
        lower_k_hop.extend(graph.neighborhood(vertex, order=k_tmp))
    k_hop = graph.neighborhood(vertex, order=k)

    # find all nodes that are exclusively in the k-hop neighborhood. 
    k_hop = [x for x in k_hop if not x in lower_k_hop]

    return k_hop


def main():
    pass


if __name__ == '__main__':
        main()