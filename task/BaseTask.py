#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from utils import *
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class BaseTask(object):
	"""
	A base class that supports loading datasets, early stop and reporting statistics
	"""
	def __init__(self, args, logger, criterion='max'):
		"""
		criterion: min/max
		"""
		self.args = args
		self.logger = logger
		self.early_stop = EarlyStoppingCriterion(self.args.patience, criterion)

	def reset_epoch_stats(self, epoch, prefix):
		"""
		prefix: train/dev/test
		"""
		self.epoch_stats = {
			'prefix': prefix,
			'epoch': epoch,
			'loss': 0,
			'num_correct': 0,
			'num_total': 0,
		}

	def update_epoch_stats(self, loss, score, label, is_regression=False):
		with th.no_grad():
			self.epoch_stats['loss'] += loss.item()
			self.epoch_stats['num_total'] += label.size(0)
			if not is_regression:
				self.epoch_stats['num_correct'] += th.sum(th.eq(th.argmax(score, dim=1), label)).item()
	
	def report_epoch_stats(self):
		if self.epoch_stats['prefix'] == 'train':
			statistics = [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']]
		else:
			# aggregate the results from all nodes
			group = dist.new_group(range(self.args.world_size))
			statistics = th.tensor(
				[self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']],
				dtype=th.float32
			).cuda()
			
			if self.args.dist_method == 'reduce':
				dist.reduce(tensor=statistics, dst=0, op=dist.ReduceOp.SUM, group=group)
			elif self.args.dist_method == 'all_gather':
				all_statistics = [th.zeros((1, 3)).cuda() for _ in range(self.args.world_size)]	
				dist.all_gather(tensor=statistics, tensor_list=all_statistics, group=group)
				statistics = th.sum(th.cat(all_statistics, dim=0), dim=0).cpu().numpy()
		
		accuracy = float(statistics[0])/statistics[1]
		loss = statistics[2] / statistics[1]
		if self.epoch_stats['prefix'] != 'test':
			self.logger.info(
				"rank %d, %s phase of epoch %d: accuracy %.6f, loss %.6f, num_correct %d, total %d" % (
				self.args.distributed_rank,
				self.epoch_stats['prefix'],
				self.epoch_stats['epoch'],
				accuracy, 
				loss,
				statistics[0], 
				statistics[1]))
		return accuracy, loss

	def report_best(self):
		self.logger.info("best dev %.6f, best test %.6f" 
			% (self.early_stop.best_dev_score, self.early_stop.best_test_score))

	def load_dataset(self, dataset_class, collate_fn, distributed=True):
		train_dataset = dataset_class(self.args, self.logger, split='train')
		dev_dataset = dataset_class(self.args, self.logger, split='dev')
		test_dataset = dataset_class(self.args, self.logger, split='test')
		if distributed:
			train_sampler = DistributedSampler(train_dataset, num_replicas=self.args.world_size, rank=self.args.distributed_rank)
			dev_sampler = DistributedSampler(dev_dataset, num_replicas=self.args.world_size, rank=self.args.distributed_rank)
			test_sampler = DistributedSampler(test_dataset, num_replicas=self.args.world_size, rank=self.args.distributed_rank)
		else:
			train_sampler, dev_sampler, test_sampler = None, None, None
		train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn,
                                         num_workers=0, sampler=train_sampler)
		dev_loader = DataLoader(dev_dataset, batch_size=1, collate_fn=collate_fn,
                                         num_workers=0, sampler=dev_sampler)
		test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn,
                                         num_workers=0, sampler=test_sampler)
		self.logger.info("train data size: %d" % len(train_dataset))
		self.logger.info("dev data size: %d" % len(dev_dataset))
		self.logger.info("test data size: %d" % len(test_dataset))
		return train_loader, dev_loader, test_loader
