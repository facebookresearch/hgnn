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
from torch.utils.data import DataLoader
import torch.optim as optim
from task.BaseTask import BaseTask
import numpy as np
from dataset.NodeClassificationDataset import NodeClassificationDataset
from task.NodeClassification import NodeClassification
import time

def cross_entropy(log_prob, label, mask):
	label, mask = label.squeeze(), mask.squeeze()
	negative_log_prob = -th.sum(label * log_prob, dim=1)
	return th.sum(mask * negative_log_prob, dim=0) / th.sum(mask)

def get_accuracy(label, log_prob, mask):
	label = label.squeeze()
	pred_class = th.argmax(log_prob, dim=1)
	real_class = th.argmax(label, dim=1)
	acc = th.eq(pred_class, real_class).float() * mask
	return (th.sum(acc) / th.sum(mask)).cpu().detach().numpy()

class NodeClassificationTask(BaseTask):

	def __init__(self, args, logger, rgnn, manifold):
		super(NodeClassificationTask, self).__init__(args, logger, criterion='max')
		self.args = args
		self.logger = logger
		self.manifold = manifold
		self.hyperbolic = False if args.select_manifold == "euclidean" else True
		self.rgnn = rgnn

	def forward(self, model, sample, loss_function):
		scores = model(
					sample['adj'].cuda().long(),
			        sample['weight'].cuda().float(),
			        sample['features'].cuda().float(),
					)
		loss = loss_function(scores,
						 sample['y_train'].cuda().float(), 
						 sample['train_mask'].cuda().float()) 
		return scores, loss

	def run_gnn(self):
		loader = self.load_data()

		model = NodeClassification(self.args, self.logger, self.rgnn, self.manifold).cuda()

		loss_function = cross_entropy

		optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler = \
								set_up_optimizer_scheduler(self.hyperbolic, self.args, model)
		

		for epoch in range(self.args.max_epochs):
			model.train()
			for i, sample in enumerate(loader):
				model.zero_grad()
				scores, loss = self.forward(model, sample, loss_function)
				loss.backward()
				if self.args.grad_clip > 0.0:
					th.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
				optimizer.step()
				if self.hyperbolic and len(self.args.hyp_vars) != 0:
					hyperbolic_optimizer.step()
				accuracy = get_accuracy(
									sample['y_train'].cuda().float(), 
									scores, 
									sample['train_mask'].cuda().float())
				self.logger.info("%s epoch %d: accuracy %.4f \n" % (
					'train', 
					epoch, 
					accuracy))
			dev_acc = self.evaluate(epoch, loader, 'dev', model, loss_function)
			test_acc = self.evaluate(epoch, loader, 'test', model, loss_function)
			lr_scheduler.step()
			if self.hyperbolic and len(self.args.hyp_vars) != 0:
				hyperbolic_lr_scheduler.step()
			if not self.early_stop.step(dev_acc, test_acc, epoch):		
				break
		self.report_best()
			
	def evaluate(self, epoch, data_loader, prefix, model, loss_function):
		model.eval()
		with th.no_grad():
			for i, sample in enumerate(data_loader):
				scores, _ = self.forward(model, sample, loss_function)
				if prefix == 'dev':
					accuracy = get_accuracy(
									sample['y_val'].cuda().float(), 
									scores, 
									sample['val_mask'].cuda().float())
				elif prefix == 'test':
					accuracy = get_accuracy(
									sample['y_test'].cuda().float(), 
									scores, 
									sample['test_mask'].cuda().float())
				if prefix == 'dev':
					self.logger.info("%s epoch %d: accuracy %.4f \n" % (
						prefix, 
						epoch, 
						accuracy))
		return accuracy

	def load_data(self):
		dataset = NodeClassificationDataset(self.args, self.logger)
		return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
