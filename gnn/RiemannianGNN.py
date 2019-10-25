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

class RiemannianGNN(nn.Module):

	def __init__(self, args, logger, manifold):
		super(RiemannianGNN, self).__init__()
		self.args = args
		self.logger = logger
		self.manifold = manifold
		self.set_up_params()
		self.activation = get_activation(self.args)
		self.dropout = nn.Dropout(self.args.dropout)

	def create_params(self):
		"""
		create the GNN params for a specific msg type
		"""
		msg_weight = []
		layer = self.args.gnn_layer if not self.args.tie_weight else 1
		for _ in range(layer):
			# weight in euclidean space
			if self.args.select_manifold in {'poincare', 'euclidean'}:
				M = th.zeros([self.args.embed_size, self.args.embed_size], requires_grad=True)
			elif self.args.select_manifold == 'lorentz': # one degree of freedom less
				M = th.zeros([self.args.embed_size, self.args.embed_size - 1], requires_grad=True)
			init_weight(M, self.args.proj_init)
			M = nn.Parameter(M)
			self.args.eucl_vars.append(M)
			msg_weight.append(M)
		return nn.ParameterList(msg_weight)

	def set_up_params(self):
		"""
		set up the params for all message types
		"""
		if not self.args.add_neg_edge and not self.args.apply_edge_type:
			self.type_of_msg = 1
		elif self.args.apply_edge_type and self.args.add_neg_edge:
			self.type_of_msg = self.args.edge_type
		elif self.args.add_neg_edge and not self.args.apply_edge_type:
			self.type_of_msg = 2
		else:
			raise Exception('Not implemented')

		for i in range(0, self.type_of_msg):
			setattr(self, "msg_%d_weight" % i, self.create_params())


	def retrieve_params(self, weight, step):
		"""
		Args:
			weight: a list of weights
			step: a certain layer
		"""
		if self.args.select_manifold in {'poincare', 'euclidean'}:
			layer_weight = weight[step]
		elif self.args.select_manifold == 'lorentz': # Ensure valid tangent vectors for (1, 0, ...)
			layer_weight = th.cat((th.zeros((self.args.embed_size, 1)).cuda(), weight[step]), dim=1)
		return layer_weight

	def apply_activation(self, node_repr):
		"""
		apply non-linearity for different manifolds
		"""
		if self.args.select_manifold in {"poincare", "euclidean"}:
			return self.activation(node_repr)
		elif self.args.select_manifold == "lorentz":
			return self.manifold.from_poincare_to_lorentz(
				self.activation(self.manifold.from_lorentz_to_poincare(node_repr))
			)

	def split_graph_by_negative_edge(self, adj_mat, weight):
		"""
		Split the graph according to positive and negative edges.
		"""
		mask = weight > 0
		neg_mask = weight < 0

		pos_adj_mat = adj_mat * mask.long()
		neg_adj_mat = adj_mat * neg_mask.long()
		pos_weight = weight * mask.float()
		neg_weight = -weight * neg_mask.float()
		return pos_adj_mat, pos_weight, neg_adj_mat, neg_weight

	def split_graph_by_type(self, adj_mat, weight):
		"""
		split the graph according to edge type for multi-relational datasets
		"""
		multi_relation_adj_mat = []
		multi_relation_weight = []
		for relation in range(1, self.args.edge_type):
			mask = (weight.int() == relation)
			multi_relation_adj_mat.append(adj_mat * mask.long())
			multi_relation_weight.append(mask.float())
		return multi_relation_adj_mat, multi_relation_weight

	def split_input(self, adj_mat, weight):
		"""
		Split the adjacency matrix and weight matrix for multi-relational datasets
		and datasets with enhanced inverse edges, e.g. Ethereum.
		"""
		if not self.args.add_neg_edge and not self.args.apply_edge_type:
			return [adj_mat], [weight]
		elif self.args.apply_edge_type and self.args.add_neg_edge:
			adj_mat, weight, neg_adj_mat, neg_weight = self.split_graph_by_negative_edge(adj_mat, weight)
			adj_mat, weight = self.split_graph_by_type(adj_mat, weight)
			adj_mat.append(neg_adj_mat)
			weight.append(neg_weight)
			return adj_mat, weight
		elif self.args.add_neg_edge and not self.args.apply_edge_type:
			pos_adj_mat, pos_weight, neg_adj_mat, neg_weight = self.split_graph_by_negative_edge(adj_mat, weight)
			return [pos_adj_mat, neg_adj_mat], [pos_weight, neg_weight]
		else:
			raise Exception('Not implemented')

	def aggregate_msg(self, node_repr, adj_mat, weight, layer_weight, mask):
		"""
		message passing for a specific message type.
		"""
		node_num, max_neighbor = adj_mat.size(0), adj_mat.size(1)
		msg = th.mm(node_repr, layer_weight) * mask
		# select out the neighbors of each node
		neighbors = th.index_select(msg, 0, adj_mat.view(-1)) # [node_num * max_neighbor, embed_size]
		neighbors = neighbors.view(node_num, max_neighbor, -1)
		# weighted sum of the neighbors' representations
		neighbors = weight.unsqueeze(2) * neighbors # [node_num, max_neighbor, embed_size]
		combined_msg = th.sum(neighbors, dim=1)  # [node_num, embed_size]
		return combined_msg

	def get_combined_msg(self, step, node_repr, adj_mat, weight, mask):
		"""
		perform message passing in the tangent space of x'
		"""
		# use the first layer only if tying weights
		gnn_layer = 0 if self.args.tie_weight else step
		combined_msg = None
		for relation in range(0, self.type_of_msg):
			layer_weight = self.retrieve_params(getattr(self, "msg_%d_weight" % relation), gnn_layer)
			aggregated_msg = self.aggregate_msg(node_repr,
												adj_mat[relation],
												weight[relation],
												layer_weight, mask)
			combined_msg = aggregated_msg if combined_msg is None else (combined_msg + aggregated_msg)
		return combined_msg

	def forward(self, node_repr, adj_list, weight, mask):
		"""
		Args:
			node_repr: [node_num, embed_size]
					   node_repr is in Euclidean space.
					   If node_repr is in hyperbolic space, invoke log_map_zero first.
			adj_list: [node_num, max_neighbor] adjacency list
			weight:  [node_num, max_neighbor]  weights of the adjacency list
			mask:    [node_num, 1] 1 denote real node, 0 padded node
		Return:
			[node_num, embed_size] in hyperbolic space
		"""
		# split the adjacency list and weights based on edge types
		adj_list, weight = self.split_input(adj_list, weight)
		# gnn layers
		for step in range(self.args.gnn_layer):
			node_repr = self.manifold.log_map_zero(node_repr) * mask if step > 0 else node_repr * mask
			combined_msg = self.get_combined_msg(step, node_repr, adj_list, weight, mask)
			combined_msg = self.dropout(combined_msg) * mask
			node_repr = self.manifold.exp_map_zero(combined_msg) * mask
			node_repr = self.apply_activation(node_repr) * mask
		return node_repr
