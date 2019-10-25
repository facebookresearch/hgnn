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
from datetime import datetime

def read_ether_price(args, f_name):
	"""
	Read weighted average price of Ether
	"""
	with open(f_name) as f:
		json_data = json.load(f)
	date_prices = {}
	for chart_data in json_data:
		date = datetime.utcfromtimestamp(int(chart_data["date"]))
		date_prices[date.strftime("%Y.%m.%d.%H")] = float(chart_data["weightedAverage"])
	return date_prices

def read_truncated_ether_price(args, truncate):
	"""
	Read Ether price, sort it and truncate it
	"""
	price_file = sorted(read_ether_price(args, args.price_file).items())
	price_file = [(date, price) for date, price in price_file if date <= '2018.09.09.16']
	price_file = price_file[truncate:]
	return price_file

class EthereumDataset(Dataset):

	def __init__(self, args, logger, split):
		self.args = args
		# price data: format [(date, price) ...]
		self.price_file = read_truncated_ether_price(self.args, 0)
		self.date2idx = {date: idx for idx, (date, price) in enumerate(self.price_file)}

		dev_days, test_days = pickle_load("data/ethereum/dev_tst_split.pickle")

		if split == 'train':
			dataset_days = {date for date, price in self.price_file[args.prev_days:]}
			self.dataset_days = sorted(list(dataset_days - set(dev_days) - set(test_days)))
		elif split == 'dev':
			self.dataset_days = dev_days
		elif split == 'test':
			self.dataset_days = test_days

	def __len__(self):
		return len(self.dataset_days)

	def get_price_feature(self, initial_day, prediction_day):
		"""
		get price feature
		"""
		up_down_price_feature = [self.price_file[i][1] > self.price_file[i-1][1]
										for i in range(initial_day, prediction_day)]
		exact_price_feature = [self.price_file[i][1] / self.args.max_price
										for i in range(initial_day, prediction_day)]
		price_feature = exact_price_feature + up_down_price_feature
		return price_feature

	def defaultdict2list(self, adj_mat, weight, id2nodes,
						 trim=True, neighbor_trim=200, node_trim=60000):
		new_adj_mat = [[] for _ in range(len(id2nodes))]
		new_weight = [[] for _ in range(len(id2nodes))]

		for dst in range(len(id2nodes)):
			for src in adj_mat[dst]:
				new_adj_mat[dst].append(src)
				new_weight[dst].append(weight[dst][src])
		# trim the adjacency list for smaller graphs here
		if trim:
			# trim node
			new_weight, new_adj_mat, id2nodes = \
				new_weight[:node_trim], new_adj_mat[:node_trim], id2nodes[:node_trim]
			# trim neighbor
			for i in range(len(id2nodes)):
				# sort and remove invalid neighbors
				sorted_list = [(w, n) for w, n in sorted(zip(new_weight[i], new_adj_mat[i])) if n < len(id2nodes)]
				if self.args.neighbor_selection == 'max':
					trim_list = sorted_list[-neighbor_trim:]
				elif self.args.neighbor_selection == 'random':
					if len(sorted_list) <= neighbor_trim:
						trim_list = sorted_list
					else:
						idx_list = np.random.choice(range(len(sorted_list)), neighbor_trim)
						trim_list = [sorted_list[idx] for idx in idx_list]
				weight, adj = zip(*trim_list)
				new_weight[i], new_adj_mat[i] = list(weight), list(adj)

		normalize_weight(new_adj_mat, new_weight)
		assert len(new_weight) == len(new_adj_mat) == len(id2nodes)
		return new_adj_mat, new_weight, id2nodes

	def filter_graph(self, graph):
		degrees = defaultdict(int)
		new_graph = nx.MultiDiGraph()
		for (src, dst, data) in graph.edges(data=True):
			degrees[src] += 1
			degrees[dst] += 1
		for (src, dst, data) in graph.edges(data=True):
			if degrees[src] > 1 and degrees[dst] > 1:
				new_graph.add_edge(src, dst, weight=data['weight'])
		return new_graph

	def __getitem__(self, idx):
		prediction_day = self.date2idx[self.dataset_days[idx]]
		initial_day = prediction_day - self.args.prev_days

		# price feature
		price_feature = self.get_price_feature(initial_day, prediction_day)
		# label
		label = int(self.price_file[prediction_day][1] > self.price_file[prediction_day-1][1])

		# store the graph feature for previous days
		node_list = []
		adj_mat_list = []
		weight_list = []
		node_num_list = []
		# graph feature
		for date_idx in range(initial_day, prediction_day):
			# read graph
			graph_file = os.path.join(*["data", "ethereum", "graph", "%s.edgelist" % self.price_file[date_idx][0]])
			graph = nx.read_edgelist(graph_file, create_using=nx.MultiDiGraph(), data=True)
			graph = self.filter_graph(graph)
			# from graph node id to local node id
			nodes2id = {node: idx for idx, node in enumerate(graph.nodes())}
			# node mapping
			id2nodes = [int(n) for n in graph.nodes()]

			# graph adj and weight. using local ids
			adj_mat = defaultdict(set)
			weight = defaultdict(lambda: defaultdict(int))
			for (src, dst, data) in graph.edges(data=True):
				if data['weight'] < 0:
					continue
				# add one here as most smart contract invocation do not have Ether transferred
				w = math.log(max(float(data['weight']), 1 + 1e-10), 10)
				assert w > 0
				adj_mat[nodes2id[dst]].add(nodes2id[src])
				weight[nodes2id[dst]][nodes2id[src]] += w
				if self.args.add_neg_edge:
					adj_mat[nodes2id[src]].add(nodes2id[dst])
					weight[nodes2id[src]][nodes2id[dst]] += -w
			# add identity matrix I + A
			for i, _ in enumerate(graph.nodes()):
				adj_mat[i].add(i)
				weight[i][i] += 1

			# defaultdict to list
			adj_mat, weight, id2nodes = self.defaultdict2list(adj_mat, weight, id2nodes)

			adj_mat_list.append(adj_mat)
			node_list.append(id2nodes)
			weight_list.append(weight)
			node_num_list.append(len(id2nodes))

		return  {
		          'node_list': node_list,
		          'adj_mat_list': adj_mat_list,
		          'weight_list': weight_list,
		          'node_num_list': np.array(node_num_list, dtype=int),
		          'price_feature': np.array(price_feature, dtype=float),
		          'label': label,
		          'date': self.price_file[prediction_day][0]
		        }
