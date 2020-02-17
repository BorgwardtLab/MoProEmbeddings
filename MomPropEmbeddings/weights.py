#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:     2020-02-17 10:35:06
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 10:35:47

import ipdb
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

cmap = sns.color_palette()
logging.basicConfig(level='INFO', format='.. %(message)s')


def doremi_weights_k1(graph):
    """Computes the 1-hop doremi-weights for each node (see DoReMi paper for
    details) and adds them as features to the network. 
    The doremi correction corrects for hubs in the network.
    """

    doremi_weights = []

    for vertex in graph.vs:
        print(f'vertex: {vertex.index}', end='\r')
        vertex_doremi_weights = []
        for x, neighbor in enumerate(vertex['k1_neighbors']):
            denom = 1 + sum(graph.vs[neighbor]['k1_stdWeights'])
            weight = vertex['k1_stdWeights'][x] / \
                ((denom-vertex['k1_stdWeights'][x])**0.34)                   
            vertex_doremi_weights.append(weight)
        assert(len(vertex_doremi_weights) == len(vertex['k1_stdWeights']))
        doremi_weights.append(vertex_doremi_weights)

    return doremi_weights


def main():

        pass

if __name__ == '__main__':
        main()