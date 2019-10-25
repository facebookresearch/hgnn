#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from utils import str2bool

def add_params(parser):
    parser.add_argument('--lr', type=float, default=1e-2) 
    parser.add_argument('--lr_hyperbolic', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max_epochs', type=int, default=6)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--num_centroid", type=int, default=2) 
    parser.add_argument('--optimizer', type=str, 
                        default='amsgrad', choices=['sgd', 'adam', 'amsgrad'])  
    parser.add_argument('--hyper_optimizer', type=str,
                        default='ramsgrad', 
                        choices=['rsgd', 'ramsgrad']) 
    parser.add_argument('--lr_scheduler', type=str, 
                        default='none', choices=['exponential', 'cosine', 'cycle', 'none'])                   
    parser.add_argument('--neighbor_selection', type=str, default='random', choices=['random', 'max'])
    parser.add_argument('--gnn_layer', type=int, default=2) 
    parser.add_argument('--embed_size', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--activation', type=str, default='leaky_relu', choices=['leaky_relu', 'rrelu', 'elu', 'selu'])
    parser.add_argument('--leaky_relu', type=float, default=0.5)
    parser.add_argument('--apply_edge_type', type=str2bool, default="False") 
    parser.add_argument('--embed_manifold', type=str, default='euclidean', choices=['euclidean', 'hyperbolic']) 
    parser.add_argument('--proj_init', type=str, 
                        default='xavier', 
                        choices=['xavier', 'orthogonal', 'kaiming', 'none'])
    parser.add_argument('--eucl_vars', type=list, default=[])    
    parser.add_argument('--hyp_vars', type=list, default=[])
    parser.add_argument('--add_neg_edge', type=str2bool, default='True')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--tie_weight', type=str2bool, default="False") 
    parser.add_argument('--price_file', type=str, default='data/ethereum/POLO_USDT_ETH_FOURHOURLY.json')
    parser.add_argument('--total_addr', type=int, default=38920591)
    parser.add_argument('--max_price', type=float, default=1400)
    parser.add_argument('--prev_days', type=int, default=2)
    parser.add_argument('--dev_days', type=int, default=700)
    parser.add_argument('--test_days', type=int, default=1400) 
    parser.add_argument('--dist_method', type=str, default='all_gather', choices=['all_gather', 'reduce'])    
