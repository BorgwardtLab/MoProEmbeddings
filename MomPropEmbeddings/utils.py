#!/usr/bin/env python3
# @Author: Anja Gumpinger
# @Date:   2020-02-17 17:05:53
# @Last Modified by:   Anja Gumpinger
# @Last Modified time: 2020-02-17 17:12:04

import ipdb
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

cmap = sns.color_palette()
logging.basicConfig(level='INFO', format='.. %(message)s')


def write_pickle(obj, filename):
    """Loads a pickled file. """
    with open(filename, 'wb') as fin:
        pickle.dump(obj, fin)
    pass


def main():

    pass

if __name__ == '__main__':
    main()