#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os
import math
import networkx as nx
from utils import *
from collections import defaultdict
import json

class GraphDataset(Dataset):

	def __init__(self, args, logger, split):
		self.args = args
		self.logger = logger

		if split == 'train':
			self.dataset = json.load(open(self.args.train_file))
		elif split == 'dev':
			self.dataset = json.load(open(self.args.dev_file))
		elif split == 'test':
			self.dataset = json.load(open(self.args.test_file))

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		graph = self.dataset[idx]
		node_num = len(graph['node_features'])
		# add self connection and a virtual node
		virtual_weight = self.args.edge_type - 1 if hasattr(self.args, 'edge_type') else 1
		adj_mat = [[i, node_num] for i in range(node_num)]
		weight = [[1, virtual_weight] for _ in range(node_num)]
		adj_mat.append([i for i in range(node_num + 1)])
		weight.append([virtual_weight for i in range(node_num + 1)])
		for src, w, dst in graph['graph']:
			adj_mat[src].append(dst)
			weight[src].append(w)
			adj_mat[dst].append(src)
			weight[dst].append(w)
		if self.args.normalization:
			normalize_weight(adj_mat, weight)
		node_feature = graph['node_features']
		if isinstance(node_feature[0], int):
			new_node_feature = np.zeros((len(node_feature), self.args.num_feature))
			for i in range(len(node_feature)):
				new_node_feature[i][node_feature[i]] = 1
			node_feature = new_node_feature.tolist()
		if len(node_feature[0]) < self.args.num_feature:
			zeros = np.zeros((len(node_feature), self.args.num_feature - len(node_feature[0])))
			node_feature = np.concatenate((node_feature, zeros), axis=1).tolist()
		node_feature.append(one_hot_vec(self.args.num_feature, -1)) # virtual node
		return  {
		          'node': node_feature,
		          'adj_mat': adj_mat,
		          'weight': weight,
		          'label': graph['targets']
		        }
