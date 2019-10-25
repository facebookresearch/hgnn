#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import os
import pickle
import json
import torch.nn as nn
import torch as th
import torch.optim as optim
import numpy as np
import random
from optimizer.ramsgrad import RiemannianAMSGrad
from optimizer.rsgd import RiemannianSGD
import math
import subprocess

def str2bool(v):
    return v.lower() == "true"

def make_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            pass

def pickle_dump(file_name, content):
    with open(file_name, 'wb') as out_file:        
        pickle.dump(content, out_file, pickle.HIGHEST_PROTOCOL)
        
def pickle_load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def init_weight(weight, method):
    """
    Initialize parameters
    Args:
        weight: a Parameter object
        method: initialization method 
    """
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')


def nn_init(nn_module, method='orthogonal'):
    """
    Initialize a Sequential or Module object
    Args:
        nn_module: Sequential or Module
        method: initialization method
    """
    if method == 'none':
        return
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            # for a Sequential object, the param_name contains both id and param name
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)

class NoneScheduler:
	def step(self):
		pass

def get_lr_scheduler(args, optimizer):
	if args.lr_scheduler == 'exponential':
		return optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
	elif args.lr_scheduler == 'cosine':
		return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
	elif args.lr_scheduler == 'cycle':
		return optim.lr_scheduler.CyclicLR(optimizer, 0, max_lr=args.lr, step_size_up=20, cycle_momentum=False)
	elif args.lr_scheduler == 'none':
		return NoneScheduler()

def get_optimizer(args, params):
	if args.optimizer == 'sgd':
		optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
	elif args.optimizer == 'adam':
		optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
	elif args.optimizer == 'amsgrad':
		optimizer = optim.Adam(params, lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
	return optimizer

def get_hyperbolic_optimizer(args, params):
    if args.hyper_optimizer == 'rsgd':
        optimizer = RiemannianSGD(
            args,
            params,
            lr=args.lr_hyperbolic,
        )
    elif args.hyper_optimizer == 'ramsgrad':
        optimizer = RiemannianAMSGrad(
            args,
            params,
            lr=args.lr_hyperbolic,
        )
    else:
        print("unsupported hyper optimizer")
        exit(1)        
    return optimizer

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def pad_sequence(data_list, maxlen, value=0):
	return [row + [value] * (maxlen - len(row)) for row in data_list]

def normalize_weight(adj_mat, weight):
	degree = [1 / math.sqrt(sum(np.abs(w))) for w in weight]
	for dst in range(len(adj_mat)):
		for src_idx in range(len(adj_mat[dst])):
			src = adj_mat[dst][src_idx]
			weight[dst][src_idx] = degree[dst] * weight[dst][src_idx] * degree[src]

def set_up_distributed_training_slurm(args):
    # get node and host
    node_list = os.environ.get('SLURM_JOB_NODELIST')
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
    # master id
    args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hostnames.split()[0].decode('utf-8'),
                    port=13349)
   	# global id 
    args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
    # id on the local machine
    args.device_id = int(os.environ.get('SLURM_LOCALID'))
    th.cuda.set_device(args.device_id)
    th.distributed.init_process_group(backend='nccl',
                                     init_method=args.distributed_init_method,
                                     world_size=args.world_size,
                                     rank=args.distributed_rank) 

def set_up_distributed_training_multi_gpu(args): 
    args.device_id = args.local_rank
    th.cuda.set_device(args.device_id)
    args.distributed_rank = args.device_id
    th.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

def save_model_weights(args, model, path):
	"""
	save model weights out to file
	"""
	if args.distributed_rank == 0:
		make_dir(path)
		th.save(model.state_dict(), os.path.join(path, args.name))

def load_model_weights(model, path):
	"""
	load saved weights
	"""
	model.load_state_dict(th.load(path))

def th_atanh(x, EPS):
	values = th.min(x, th.Tensor([1.0 - EPS]).cuda())
	return 0.5 * (th.log(1 + values + EPS) - th.log(1 - values + EPS))
	
def th_norm(x, dim=1):
	"""
	Args
		x: [batch size, dim]
	Output:	
		[batch size, 1]
	"""
	return th.norm(x, 2, dim, keepdim=True)

def th_dot(x, y, keepdim=True):
	return th.sum(x * y, dim=1, keepdim=keepdim)

def clip_by_norm(x, clip_norm):
	return th.renorm(x, 2, 0, clip_norm)

def get_params(params_list, vars_list):
	"""
	Add parameters in vars_list to param_list
	"""
	for i in vars_list:
		if issubclass(i.__class__, nn.Module):
			params_list.extend(list(i.parameters()))
		elif issubclass(i.__class__, nn.Parameter):
			params_list.append(i)
		else:
			print("Encounter unknown objects")
			exit(1)

def categorize_params(args):
	"""
	Categorize parameters into hyperbolic ones and euclidean ones
	"""
	hyperbolic_params, euclidean_params = [], []
	get_params(euclidean_params, args.eucl_vars)
	get_params(hyperbolic_params, args.hyp_vars)
	return hyperbolic_params, euclidean_params

def get_activation(args):
	if args.activation == 'leaky_relu':
		return nn.LeakyReLU(args.leaky_relu)
	elif args.activation == 'rrelu':
		return nn.RReLU()
	elif args.activation == 'relu':
		return nn.ReLU()
	elif args.activation == 'elu':
		return nn.ELU()
	elif args.activation == 'prelu':
		return nn.PReLU()
	elif args.activation == 'selu':
		return nn.SELU()

def set_up_optimizer_scheduler(hyperbolic, args, model):
	if hyperbolic:
		hyperbolic_params, euclidean_params = categorize_params(args)
		assert(len(list(model.parameters())) == len(hyperbolic_params) + len(euclidean_params))
		optimizer = get_optimizer(args, euclidean_params)
		lr_scheduler = get_lr_scheduler(args, optimizer)
		if len(hyperbolic_params) > 0:
			hyperbolic_optimizer = get_hyperbolic_optimizer(args, hyperbolic_params)
			hyperbolic_lr_scheduler = get_lr_scheduler(args, hyperbolic_optimizer)
		else:
			hyperbolic_optimizer, hyperbolic_lr_scheduler = None, None
		return optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler
	else:
		optimizer = get_optimizer(args, model.parameters())
		lr_scheduler = get_lr_scheduler(args, optimizer)
		return optimizer, lr_scheduler, None, None

# reimplement clamp functions to avoid killing gradient during backpropagation
def clamp_max(x, max_value):
	t = th.clamp(max_value - x.detach(), max=0)
	return x + t

def clamp_min(x, min_value):
	t = th.clamp(min_value - x.detach(), min=0)
	return x + t

def one_hot_vec(length, pos):
	vec = [0] * length
	vec[pos] = 1
	return vec
