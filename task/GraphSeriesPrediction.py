#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils import * 
from hyperbolic_module.CentroidDistance import CentroidDistance

class GraphSeriesPrediction(nn.Module):

	def __init__(self, args, logger, rgnn, manifold):
		super(GraphSeriesPrediction, self).__init__()
		self.args = args
		self.logger = logger
		self.manifold = manifold

		self.embedding = nn.Embedding(
            args.total_addr, args.embed_size,
            sparse=False,
            scale_grad_by_freq=False
        )
		if self.args.embed_manifold == 'hyperbolic':
			self.manifold.init_embed(self.embedding)
			self.args.hyp_vars.append(self.embedding)
		elif self.args.embed_manifold == 'euclidean':
			nn_init(self.embedding, self.args.proj_init)
			self.args.eucl_vars.append(self.embedding)

		self.distance = CentroidDistance(args, logger, manifold)

		self.rgnn = rgnn

		self.output = nn.Linear(args.prev_days * self.args.num_centroid + 2 * args.prev_days,
								self.args.num_class
					  )
		nn_init(self.output, self.args.proj_init)
		self.args.eucl_vars.append(self.output)	

		self.log_softmax = nn.LogSoftmax(dim=1)
		
	def forward(self, node_list, adj_mat_list, weight_list, node_num_list, price_feature):
		"""
		Since the transaction graphs are large, the batch size is set to 1 on each GPU
		Args:
			node_list: the node ids [1, prev, node_num]
			adj_mat_list: the neighbor ids of each node [1, prev, node_num, max_neighbor]
			weight_list: the weight of each neighbor [1, prev, node_num, max_neighbor]
			node_num_list: [1, prev] node num before padding
			price_feature: [1, 2 * prev]
		"""
		assert node_list.size(0) == 1
		node_list, adj_mat_list, weight_list, node_num_list = \
			node_list.squeeze(0), adj_mat_list.squeeze(0), weight_list.squeeze(0), node_num_list.squeeze(0)
		
		tx_graph = []
		for idx in range(self.args.prev_days):
			node, adj_mat, weight, node_num = node_list[idx], adj_mat_list[idx], weight_list[idx], node_num_list[idx]
			max_neighbor = adj_mat.size(1)
			mask = (th.arange(1, node.size(0) + 1) <= node_num.item()).view(-1, 1).float().cuda() # [node_num, 1]
			
			if self.args.embed_manifold == 'hyperbolic':
				node_repr = self.manifold.log_map_zero(self.embedding(node)) * mask
			elif self.args.embed_manifold == 'euclidean':
				node_repr = self.embedding(node) * mask
			node_repr = self.rgnn(node_repr, adj_mat, weight, mask)	
			graph_repr, _ = self.distance(node_repr, mask)
			tx_graph.append(graph_repr.view(1, self.args.num_centroid))
		logit = self.output(th.cat(tx_graph + [price_feature], dim=1))
		return self.log_softmax(logit)
