#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from datetime import datetime
import random
import numpy as np
from task import *
import os
import time
from utils import *
from params import *
import sys
from manifold import *
from gnn import RiemannianGNN

def set_up_fold(args):
    if args.task in ['dd', 'enzymes', 'proteins', 'reddit', 'collab']:
        args.train_file = (args.train_file % args.fold)
        args.dev_file = (args.dev_file % args.fold)
        args.test_file = (args.test_file % args.fold)

def add_embed_size(args):
    # add 1 for Lorentz as the degree of freedom is d - 1 with d dimensions
    if args.select_manifold == 'lorentz':
        args.embed_size += 1

def parse_default_args():
    parser = argparse.ArgumentParser(description='RiemannianGNN')
    parser.add_argument('--name', type=str, default='{:%Y_%m_%d_%H_%M_%S_%f}'.format(datetime.now()))
    parser.add_argument('--task', type=str, choices=['qm8',
                                                     'qm9',
                                                     'zinc',
                                                     'ethereum',
                                                     'node_classification',
                                                     'synthetic',
                                                     'dd',
                                                     'enzymes',
                                                     'proteins',
                                                     'reddit',
                                                     'collab',
                                                     ])
    parser.add_argument('--select_manifold', type=str, default='lorentz', choices=['poincare', 'lorentz', 'euclidean'])
    parser.add_argument('--seed', type=int, default=int(time.time()))
    # for distributed training
    parser.add_argument('--world_size', type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--distributed_method", default='multi_gpu', choices=['multi_gpu', 'slurm'])
    args, _ = parser.parse_known_args()
    # model-specific params
    if args.task == 'ethereum' and args.select_manifold == 'euclidean':
        EthereumEuclideanParams.add_params(parser)
    elif args.task == 'ethereum' and args.select_manifold != 'euclidean':
        EthereumHyperbolicParams.add_params(parser)
    elif args.task == 'node_classification':
        NodeClassificationHyperbolicParams.add_params(parser)
    elif args.task == 'qm9' and args.select_manifold == 'euclidean':
        QM9EuclideanParams.add_params(parser)
    elif args.task == 'qm9' and args.select_manifold != 'euclidean':
        QM9HyperbolicParams.add_params(parser)
    elif args.task == 'qm8' and args.select_manifold == 'euclidean':
        QM8EuclideanParams.add_params(parser)
    elif args.task == 'qm8' and args.select_manifold != 'euclidean':
        QM8HyperbolicParams.add_params(parser)
    elif args.task == 'zinc' and args.select_manifold == 'euclidean':
        ZINCEuclideanParams.add_params(parser)
    elif args.task == 'zinc' and args.select_manifold != 'euclidean':
        ZINCHyperbolicParams.add_params(parser)
    elif args.task == 'synthetic' and args.select_manifold == 'euclidean':
        SyntheticEuclideanParams.add_params(parser)
    elif args.task == 'synthetic' and args.select_manifold != 'euclidean':
        SyntheticHyperbolicParams.add_params(parser)
    elif args.task == 'dd' and args.select_manifold == 'euclidean':
        DDEuclideanParams.add_params(parser)
    elif args.task == 'dd' and args.select_manifold != 'euclidean':
        DDHyperbolicParams.add_params(parser)
    elif args.task == 'enzymes' and args.select_manifold == 'euclidean':
        EnzymesEuclideanParams.add_params(parser)
    elif args.task == 'enzymes' and args.select_manifold != 'euclidean':
        EnzymesHyperbolicParams.add_params(parser)
    elif args.task == 'proteins' and args.select_manifold == 'euclidean':
        ProteinsEuclideanParams.add_params(parser)
    elif args.task == 'proteins' and args.select_manifold != 'euclidean':
        ProteinsHyperbolicParams.add_params(parser)
    elif args.task == 'reddit' and args.select_manifold == 'euclidean':
        RedditEuclideanParams.add_params(parser)
    elif args.task == 'reddit' and args.select_manifold != 'euclidean':
        RedditHyperbolicParams.add_params(parser)
    elif args.task == 'collab' and args.select_manifold == 'euclidean':
        CollabEuclideanParams.add_params(parser)
    elif args.task == 'collab' and args.select_manifold != 'euclidean':
        CollabHyperbolicParams.add_params(parser)
    args = parser.parse_args()
    set_up_fold(args)
    add_embed_size(args)
    if args.task != 'node_classification':
        if args.distributed_method == 'slurm':
            set_up_distributed_training_slurm(args)
        elif args.distributed_method == 'multi_gpu':
            set_up_distributed_training_multi_gpu(args)
    return args

def create_manifold(args, logger):
    if args.select_manifold == 'poincare':
        return PoincareManifold(args, logger)
    elif args.select_manifold == 'lorentz':
        return LorentzManifold(args, logger)
    elif args.select_manifold == 'euclidean':
        return EuclideanManifold(args, logger)

if __name__ == '__main__':
    args = parse_default_args()
    set_seed(args.seed)

    logger = create_logger('log/%s.log' % args.name)
    logger.info("save debug info to log/%s.log" % args.name)
    logger.info(args)

    manifold = create_manifold(args, logger)
    rgnn = RiemannianGNN(args, logger, manifold)

    if args.task == 'ethereum':
        gnn_task = GraphSeriesPredictionTask(args, logger, rgnn, manifold)
    elif args.task == 'node_classification':
        gnn_task = NodeClassificationTask(args, logger, rgnn, manifold)
    elif args.task in {'qm8', 'qm9', 'zinc', 'dd', 'enzymes', 'proteins', 'reddit', 'collab', 'synthetic'}:
        gnn_task = GraphPredictionTask(args, logger, rgnn, manifold)
    else:
        raise Exception("Unknown task")
    gnn_task.run_gnn()
