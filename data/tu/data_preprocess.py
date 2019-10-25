#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ssl
ssl.match_hostname = lambda cert, hostname: True
from torch_geometric.datasets import TUDataset
import json
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict
import sys

def one_hot_vec(length, pos):
	vec = [0] * length
	vec[pos] = 1
	return vec

if __name__ == "__main__":
	use_valid_set = False
	assert sys.argv[1] in ['REDDIT-MULTI-12K', 'PROTEINS_full', 'ENZYMES', 'DD', 'COLLAB']
	filename = sys.argv[1]
	random_seed = 0
	max_degree = 8000
	if filename == 'REDDIT-MULTI-12K':
		max_degree = 30

	dataset = TUDataset(root='/tmp/%s' % filename, name=filename)
	all_graph = []
	for i in range(len(dataset)):
		print(i)
		edges = dataset[i]['edge_index'].numpy()

		graph = []
		nodes = set()
		degrees = defaultdict(int)
		for j in range(edges.shape[1]):
			src, target = edges[:, j]
			nodes.add(src)
			nodes.add(target)
			if src <= target:
				graph.append((int(src), 1, int(target)))
				degrees[src] += 1
				degrees[target] += 1
		targets = int(dataset[i]['y'].item())
		try:
			node_features = dataset[i]['x'].numpy().tolist()
		except: # dataset without node tags. Use degree as the tag.
			node_features = []
			for n in range(len(nodes)):
				if degrees[n] < max_degree:
					node_features.append(degrees[n])
				else:
					node_features.append(max_degree)
		all_graph.append({
	    	'targets': targets,
	    	'graph': graph,
	    	'node_features': node_features,
		})

	kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
	kf.get_n_splits(all_graph)
	fold = 1
	for train_idx, test_idx in kf.split(all_graph):
		print(fold)
		if use_valid_set:
			valid_len = int(len(train_idx) / 9)
			valid_idx =  train_idx[-valid_len:]
			train_idx = train_idx[:-valid_len]
		else:
			valid_idx = test_idx
		with open('%s_%s_%d.json' % (filename, 'train', fold), 'w') as f:
			json.dump([all_graph[i] for i in train_idx], f)
		with open('%s_%s_%d.json' % (filename, 'test', fold), 'w') as f:
			json.dump([all_graph[i] for i in test_idx], f)
		with open('%s_%s_%d.json' % (filename, 'valid', fold), 'w') as f:
			json.dump([all_graph[i] for i in valid_idx], f)
		fold += 1
