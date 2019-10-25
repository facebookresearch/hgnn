#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os
import math
import networkx as nx
from utils import *
from collections import defaultdict
import json

def one_hot_vec(length, pos):
	vec = [0] * length
	vec[pos] = 1
	return vec

class SyntheticDataset(Dataset):
	"""
	Extend the Dataset class for graph datasets
	"""
	def __init__(self, args, logger, split):
		self.args = args
		self.logger = logger

		if split == 'train':
			self.dataset = pickle_load(self.args.train_file)
		elif split == 'dev':
			self.dataset = pickle_load(self.args.dev_file)
		elif split == 'test':
			self.dataset = pickle_load(self.args.test_file)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		graph = self.dataset[idx]
		num_feature = self.args.num_feature
		# add self connection and an virtual node
		adj_mat = [[i] for i in range(num_feature)]
		weight = [[1] for _ in range(num_feature)]
		degrees = [0] * num_feature
		for src, w, dst in graph['graph']:
			adj_mat[src].append(dst)
			weight[src].append(w)
			adj_mat[dst].append(src)
			weight[dst].append(w)
			degrees[src] += 1
			degrees[dst] += 1
		degree_mat = [one_hot_vec(self.args.num_feature, d) for d in degrees]
		normalize_weight(adj_mat, weight)
		return  {
		          'node': degree_mat,
		          'adj_mat': adj_mat,
		          'weight': weight,
		          'label': graph['targets'],
		          'mask': graph['num_node']
		        }
