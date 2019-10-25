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

class GraphPrediction(nn.Module):

	def __init__(self, args, logger, rgnn, manifold):
		super(GraphPrediction, self).__init__()
		self.args = args
		self.logger = logger
		self.manifold = manifold

		if not self.args.remove_embed:
			self.embedding = nn.Linear(
		            args.num_feature, args.embed_size,
		            bias=False
		    )
			if self.args.embed_manifold == 'hyperbolic':
				self.manifold.init_embed(self.embedding)
				self.args.hyp_vars.append(self.embedding)
			elif self.args.embed_manifold == 'euclidean':
				nn_init(self.embedding, self.args.proj_init)
				self.args.eucl_vars.append(self.embedding)

		self.distance = CentroidDistance(args, logger, manifold)

		self.rgnn = rgnn

		if self.args.is_regression:
			self.output_linear = nn.Linear(self.args.num_centroid, 1)
		else:
			self.output_linear = nn.Linear(self.args.num_centroid, self.args.num_class)
		nn_init(self.output_linear, self.args.proj_init)
		self.args.eucl_vars.append(self.output_linear)			

	def forward(self, node, adj, weight, mask):
		"""
		Args:
			adj: the neighbor ids of each node [1, node_num, max_neighbor]
			weight: the weight of each neighbor [1, node_num, max_neighbor]
			features: [1, node_num, input_dim]
		"""
		assert adj.size(0) == 1
		node, adj, weight = node.squeeze(0), adj.squeeze(0), weight.squeeze(0)
		node_num, max_neighbor = adj.size(0), adj.size(1)
		mask = (th.arange(1, node_num + 1) <= mask.item()).view(-1, 1).float().cuda() # [node_num, 1]

		if self.args.embed_manifold == 'hyperbolic':
			node_repr = self.manifold.log_map_zero(self.embedding(node)) * mask
		elif self.args.embed_manifold == 'euclidean':		
			node_repr = self.embedding(node) * mask if not self.args.remove_embed else (node * mask) 
		node_repr = self.rgnn(node_repr, adj, weight, mask) # [node_num, embed_size]
		graph_repr, _ = self.distance(node_repr, mask)
		return self.output_linear(graph_repr)
