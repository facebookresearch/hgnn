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

class NodeClassification(nn.Module):

	def __init__(self, args, logger, rgnn, manifold):
		super(NodeClassification, self).__init__()
		self.args = args
		self.logger = logger
		self.manifold = manifold

		self.feature_linear = nn.Linear(self.args.input_dim,
										self.args.embed_size
							  )
		nn_init(self.feature_linear, self.args.proj_init)
		self.args.eucl_vars.append(self.feature_linear)			

		self.distance = CentroidDistance(args, logger, manifold)

		self.rgnn = rgnn
		self.output_linear = nn.Linear(self.args.num_centroid,
										self.args.num_class
							  )
		nn_init(self.output_linear, self.args.proj_init)
		self.args.eucl_vars.append(self.output_linear)			

		self.log_softmax = nn.LogSoftmax(dim=1)
		self.activation = get_activation(self.args)

	def forward(self, adj, weight, features):
		"""
		Args:
			adj: the neighbor ids of each node [1, node_num, max_neighbor]
			weight: the weight of each neighbor [1, node_num, max_neighbor]
			features: [1, node_num, input_dim]
		"""
		assert adj.size(0) == 1
		adj, weight, features = adj.squeeze(0), weight.squeeze(0), features.squeeze(0)
		
		node_repr = self.activation(self.feature_linear(features))
		mask = th.ones((self.args.node_num, 1)).cuda() # [node_num, 1]
		node_repr = self.rgnn(node_repr, adj, weight, mask) # [node_num, embed_size]
		_, node_centroid_sim = self.distance(node_repr, mask) # [1, node_num, num_centroid]
		class_logit = self.output_linear(node_centroid_sim.squeeze())
		return self.log_softmax(class_logit)
