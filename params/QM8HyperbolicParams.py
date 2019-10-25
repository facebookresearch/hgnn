#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from utils import str2bool
import sys

def add_params(parser):
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_hyperbolic', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--optimizer', type=str, 
                        default='amsgrad', choices=['sgd', 'adam', 'amsgrad', 'noam']) 
    parser.add_argument('--hyper_optimizer', type=str,
                        default='ramsgrad', 
                        choices=['rsgd', 'ramsgrad'])  
    parser.add_argument('--lr_scheduler', type=str, 
                        default='none', choices=['exponential', 'cosine', 'cycle', 'none'])    
    parser.add_argument("--num_centroid", type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--gnn_layer', type=int, default=4) 
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='leaky_relu', choices=['leaky_relu', 'rrelu', 'selu'])
    parser.add_argument('--leaky_relu', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--add_neg_edge', type=str2bool, default='True')
    parser.add_argument('--proj_init', type=str, 
                        default='xavier', 
                        choices=['xavier', 'orthogonal', 'kaiming', 'none'])
    parser.add_argument('--embed_size', type=int, default=256)    
    parser.add_argument('--eucl_vars', type=list, default=[])    
    parser.add_argument('--hyp_vars', type=list, default=[])
    parser.add_argument('--tie_weight', type=str2bool, default="False") 
    parser.add_argument('--apply_edge_type', type=str2bool, default="True") 
    parser.add_argument('--edge_type', type=int, default=6) 
    parser.add_argument('--embed_manifold', type=str, default='euclidean', choices=['euclidean', 'hyperbolic']) 
    parser.add_argument('--mean', type=list, default=[0.04316444, 0.21669717, 0.24225989, 0.02028711, 0.0331261, 0.21057869, 0.23447066, 0.01869214, 0.03175573, 0.21681479, 0.24463635, 0.02345177])
    parser.add_argument('--std', type=list, default=[0.07305397, 0.04787911, 0.04030875, 0.05117046, 0.06029926, 0.04306428, 0.03571646, 0.04692344, 0.05757811, 0.04458193, 0.03597614, 0.05786732])        
    parser.add_argument('--metric', type=str, default="mae", choices=['rmse', 'mae']) 
    parser.add_argument('--train_file', type=str, default='data/qm8/molecules_train_qm8.json')
    parser.add_argument('--dev_file', type=str, default='data/qm8/molecules_valid_qm8.json')
    parser.add_argument('--test_file', type=str, default='data/qm8/molecules_test_qm8.json')
    parser.add_argument('--total_atom', type=int, default=6)
    parser.add_argument('--num_feature', type=int, default=6)
    parser.add_argument('--num_property', type=int, default=12)
    parser.add_argument('--prop_idx', type=int, default=0)
    parser.add_argument('--is_regression', type=str2bool, default=True)
    parser.add_argument('--normalization', type=str2bool, default=False) 
    parser.add_argument('--remove_embed', type=str2bool, default=False) 
    parser.add_argument('--dist_method', type=str, default='all_gather', choices=['all_gather', 'reduce'])   
