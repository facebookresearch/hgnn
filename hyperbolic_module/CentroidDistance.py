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

class CentroidDistance(nn.Module):
	"""
	Implement a model that calculates the pairwise distances between node representations
	and centroids
	"""
	def __init__(self, args, logger, manifold):
		super(CentroidDistance, self).__init__()
		self.args = args
		self.logger = logger
		self.manifold = manifold

		# centroid embedding
		self.centroid_embedding = nn.Embedding(
			args.num_centroid, args.embed_size,
			sparse=False,
			scale_grad_by_freq=False,
		)
		if args.embed_manifold == 'hyperbolic':
			args.manifold.init_embed(self.centroid_embedding)
			args.hyp_vars.append(self.centroid_embedding)
		elif args.embed_manifold == 'euclidean':
			nn_init(self.centroid_embedding, self.args.proj_init)
			if hasattr(args, 'eucl_vars'):
				args.eucl_vars.append(self.centroid_embedding)

	def forward(self, node_repr, mask):
		"""
		Args:
			node_repr: [node_num, embed_size]
			mask: [node_num, 1] 1 denote real node, 0 padded node
		return:
			graph_centroid_dist: [1, num_centroid]
			node_centroid_dist: [1, node_num, num_centroid]
		"""
		node_num = node_repr.size(0)

		# broadcast and reshape node_repr to [node_num * num_centroid, embed_size]
		node_repr =  node_repr.unsqueeze(1).expand(
												-1,
												self.args.num_centroid,
												-1).contiguous().view(-1, self.args.embed_size)

		# broadcast and reshape centroid embeddings to [node_num * num_centroid, embed_size]
		if self.args.embed_manifold == 'hyperbolic':
			centroid_repr = self.centroid_embedding(th.arange(self.args.num_centroid).cuda())
		else:
			centroid_repr = self.manifold.exp_map_zero(self.centroid_embedding(th.arange(self.args.num_centroid).cuda()))
		centroid_repr = centroid_repr.unsqueeze(0).expand(
												node_num,
												-1,
												-1).contiguous().view(-1, self.args.embed_size)
		# get distance
		node_centroid_dist = self.manifold.distance(node_repr, centroid_repr)
		node_centroid_dist = node_centroid_dist.view(1, node_num, self.args.num_centroid) * mask
		# average pooling over nodes
		graph_centroid_dist = th.sum(node_centroid_dist, dim=1) / th.sum(mask)
		return graph_centroid_dist, node_centroid_dist
