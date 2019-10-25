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
from dataset.GraphDataset import GraphDataset
from dataset.SyntheticDataset import SyntheticDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import default_collate
from task.GraphPrediction import GraphPrediction
import torch.distributed as dist

def collate_fn(batch):
	max_neighbor_num = -1
	for data in batch:
		for row in data['adj_mat']:
			max_neighbor_num = max(max_neighbor_num, len(row))

	for data in batch:
		# pad the adjacency list
		data['adj_mat'] = pad_sequence(data['adj_mat'], maxlen=max_neighbor_num)
		data['weight'] = pad_sequence(data['weight'], maxlen=max_neighbor_num)

		data['node'] = np.array(data['node']).astype(np.float32)
		data['adj_mat'] = np.array(data['adj_mat']).astype(np.int32)
		data['weight'] = np.array(data['weight']).astype(np.float32)
		data['label'] = np.array(data['label'])
	return default_collate(batch)

class GraphPredictionTask(BaseTask):

	def __init__(self, args, logger, rgnn, manifold):
		if args.is_regression:
			super(GraphPredictionTask, self).__init__(args, logger, criterion='min')
		else:
			super(GraphPredictionTask, self).__init__(args, logger, criterion='max')
		self.hyperbolic = False if args.select_manifold == "euclidean" else True
		self.rgnn = rgnn
		self.manifold = manifold

	def forward(self, model, sample, loss_function):
		mask = sample['mask'].int() if 'mask' in sample else th.Tensor([sample['adj_mat'].size(1)]).cuda()
		scores = model(
					   sample['node'].cuda().float(),
			           sample['adj_mat'].cuda().long(),
			           sample['weight'].cuda().float(),
			           mask)
		if self.args.is_regression:
			loss = loss_function(scores.view(-1) * self.args.std[self.args.prop_idx] + self.args.mean[self.args.prop_idx], 
						     th.Tensor([sample['label'].view(-1)[self.args.prop_idx]]).float().cuda())
		else:
			loss = loss_function(scores, th.Tensor([sample['label'].view(-1)[self.args.prop_idx]]).long().cuda())
		return scores, loss

	def run_gnn(self):
		train_loader, dev_loader, test_loader = self.load_data()

		task_model = GraphPrediction(self.args, self.logger, self.rgnn, self.manifold).cuda()
		model = nn.parallel.DistributedDataParallel(task_model,
                                                  device_ids=[self.args.device_id],
                                                  output_device=self.args.device_id)
		if self.args.is_regression:
			loss_function = nn.MSELoss(reduction='sum')
		else:
			loss_function = nn.CrossEntropyLoss(reduction='sum')

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
				if self.args.is_regression and self.args.metric == "mae":
					loss = th.sqrt(loss)
				self.update_epoch_stats(loss, scores, sample['label'].cuda(), is_regression=self.args.is_regression)			
				if i % 400 ==0:
					self.report_epoch_stats()
			
			dev_acc, dev_loss = self.evaluate(epoch, dev_loader, 'dev', model, loss_function)
			test_acc, test_loss = self.evaluate(epoch, test_loader, 'test', model, loss_function)
			
			if self.args.is_regression and not self.early_stop.step(dev_loss, test_loss, epoch):		
				break
			elif not self.args.is_regression and not self.early_stop.step(dev_acc, test_acc, epoch):
				break

			lr_scheduler.step()
			if self.hyperbolic and len(self.args.hyp_vars) != 0:
				hyperbolic_lr_scheduler.step()
			th.cuda.empty_cache()
		self.report_best()

	def evaluate(self, epoch, data_loader, prefix, model, loss_function):
		model.eval()
		with th.no_grad():
			self.reset_epoch_stats(epoch, prefix)
			for i, sample in enumerate(data_loader):
				scores, loss = self.forward(model, sample, loss_function)
				if self.args.is_regression and self.args.metric == "mae":
					loss = th.sqrt(loss)
				self.update_epoch_stats(loss, scores, sample['label'].cuda(), is_regression=self.args.is_regression)
			accuracy, loss = self.report_epoch_stats()
		if self.args.is_regression and self.args.metric == "rmse":
			loss = np.sqrt(loss)
		return accuracy, loss

	def load_data(self):
		if self.args.task == 'synthetic':
			return self.load_dataset(SyntheticDataset, collate_fn)
		else:
			return self.load_dataset(GraphDataset, collate_fn)	
