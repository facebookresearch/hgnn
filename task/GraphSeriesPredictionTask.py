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
from task.BaseTask import BaseTask
from dataset.EthereumDataset import EthereumDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import default_collate
from task.GraphSeriesPrediction import GraphSeriesPrediction

def collate_fn(batch):
	max_node_num = -1
	max_neighbor_num = -1
	for data in batch:
		for nodes in data['node_list']:
			max_node_num = max(max_node_num, len(nodes))
		for adj_mat in data['adj_mat_list']:
			for row in adj_mat:
				max_neighbor_num = max(max_neighbor_num, len(row))

	for data in batch:
		# pad node
		data['node_list'] = pad_sequence(data['node_list'], maxlen=max_node_num)

		# pad adj matrix and weight matrix
		def pad_mat(data, mat_name, max_node_num, max_neighbor_num):
			for idx, _ in enumerate(data[mat_name]):
				mat = data[mat_name][idx]
				for _ in range(max_node_num - len(mat)):
					mat.append([])
				data[mat_name][idx] = pad_sequence(mat, maxlen=max_neighbor_num)
		pad_mat(data, 'adj_mat_list', max_node_num, max_neighbor_num)
		pad_mat(data, 'weight_list', max_node_num, max_neighbor_num)

		data['node_list'] = np.array(data['node_list']).astype(np.int32)
		data['adj_mat_list'] = np.array(data['adj_mat_list']).astype(np.int32)
		data['weight_list'] = np.array(data['weight_list']).astype(np.float32)
	return default_collate(batch)

class GraphSeriesPredictionTask(BaseTask):

	def __init__(self, args, logger, rgnn, manifold):
		super(GraphSeriesPredictionTask, self).__init__(args , logger)
		self.hyperbolic = False if args.select_manifold == "euclidean" else True
		self.rgnn = rgnn
		self.manifold = manifold

	def forward(self, model, sample, loss_function):
		scores = model(
					   sample['node_list'].cuda().long(),
			           sample['adj_mat_list'].cuda().long(),
			           sample['weight_list'].cuda().float(),
			           sample['node_num_list'].cuda().long(),
			           sample['price_feature'].cuda().float())		
		loss = loss_function(scores, sample['label'].cuda())
		return scores, loss

	def run_gnn(self):
		train_loader, dev_loader, test_loader= self.load_data()

		task_model = GraphSeriesPrediction(self.args, self.logger, self.rgnn, self.manifold).cuda()
		model = nn.parallel.DistributedDataParallel(task_model,
                                                  device_ids=[self.args.device_id],
                                                  output_device=self.args.device_id)
		loss_function = nn.NLLLoss()

		optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler = \
								set_up_optimizer_scheduler(self.hyperbolic, self.args, model)
		
		for epoch in range(self.args.max_epochs):
			self.reset_epoch_stats(epoch, 'train')
			model.train()
			for i, sample in enumerate(train_loader):
				model.zero_grad()
				scores, loss = self.forward(model, sample, loss_function)
				loss.backward(retain_graph=True)
				if self.args.grad_clip > 0.0:
					th.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)

				optimizer.step()
				if self.hyperbolic and len(self.args.hyp_vars) != 0:
					hyperbolic_optimizer.step()
				self.update_epoch_stats(loss, scores, sample['label'].cuda())			
				if i % 20 ==0:
					self.report_epoch_stats()

			dev_acc, _ = self.evaluate(epoch, dev_loader, 'dev', model, loss_function)
			test_acc, _ = self.evaluate(epoch, test_loader, 'test', model, loss_function)
			
			if not self.early_stop.step(dev_acc, test_acc, epoch):
				break
		self.report_best()

	def evaluate(self, epoch, data_loader, prefix, model, loss_function):
		model.eval()
		with th.no_grad():
			self.reset_epoch_stats(epoch, prefix)
			for i, sample in enumerate(data_loader):
				scores, loss = self.forward(model, sample, loss_function)
				self.update_epoch_stats(loss, scores, sample['label'].cuda())
			return self.report_epoch_stats()

	def load_data(self):
		return self.load_dataset(EthereumDataset, collate_fn)
