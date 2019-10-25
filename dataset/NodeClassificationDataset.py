#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import save_npz, load_npz
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from torch.utils.data import Dataset, DataLoader
from utils import *

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)).astype(float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

class NodeClassificationDataset(Dataset):
    """
    Extend the Dataset class for graph datasets
    """
    def __init__(self, args, logger):
        self.args = args
        self.load_data(self.args.dataset_str)

    def load_data(self, dataset_str):
        """
        Loads input data from data directory
        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
        All objects above must be saved using python pickle module.
        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/node/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/node/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset_str == 'citeseer':  # or dataset_str == 'nell.0.001'
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = [[i] for i in range(features.shape[0])]
        weight = [[1] for i in range(features.shape[0])]
        for node, neighbor in graph.items():
            for n in neighbor:
                adj[node].append(n)
                weight[node].append(1)

        max_len = max([len(i) for i in adj])
        normalize_weight(adj, weight)

        adj_label = []
        for i in range(len(adj)):
            for j in range(len(adj)):
                if j in adj[i]:
                    adj_label.append(1)
                else:
                    adj_label.append(0)
        adj = pad_sequence(adj, max_len)
        weight = pad_sequence(weight, max_len)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        self.adj = np.array(adj)
        self.weight = np.array(weight)
        features = np.array(features.todense().tolist())

        self.features = preprocess_features(features)
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_mask = train_mask.astype(int)
        self.val_mask = val_mask.astype(int)
        self.test_mask = test_mask.astype(int)
        # set up paramaters
        self.args.node_num = features.shape[0]
        self.args.input_dim = features.shape[1]
        self.args.num_class = y_train.shape[1]
        self.adj_label = np.array(adj_label)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return  {
                  'adj': self.adj,
                  'weight': self.weight,
                  'features': self.features,
                  'y_train' : self.y_train,
                  'y_val' : self.y_val,
                  'y_test' : self.y_test,
                  'train_mask' : self.train_mask,
                  'val_mask' : self.val_mask,
                  'test_mask' : self.test_mask,
                  'adj_label': self.adj_label
                }
