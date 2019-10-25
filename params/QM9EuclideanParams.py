#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from utils import str2bool

def add_params(parser):
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, 
                        default='amsgrad', choices=['sgd', 'adam', 'amsgrad']) 
    parser.add_argument('--lr_scheduler', type=str, 
                        default='none', choices=['exponential', 'cosine', 'cycle', 'none'])             
    parser.add_argument('--patience', type=int, default=25)   
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--gnn_layer', type=int, default=4)
    parser.add_argument('--embed_size', type=int, default=256)  
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='leaky_relu', choices=['leaky_relu', 'rrelu'])
    parser.add_argument('--leaky_relu', type=float, default=0.5)                 
    parser.add_argument('--mean', type=list, default=[2.70603747e+00, 7.51912962e+01, -2.39976699e-01, 1.11237674e-02, 2.51100371e-01, 1.18952745e+03, 1.48524389e-01, -4.11543985e+02, -4.11535513e+02, -4.11534569e+02, -4.11577397e+02, 3.16006759e+01])
    parser.add_argument('--std', type=list, default=[1.53038828e+00, 8.18776222e+00, 2.21313514e-02, 4.69358946e-02, 4.75187104e-02, 2.79756127e+02, 3.32737721e-02, 4.00600808e+01, 4.00598619e+01, 4.00598619e+01, 4.00605912e+01, 4.06245625e+00])    
    parser.add_argument('--proj_init', type=str, 
                        default='xavier', 
                        choices=['xavier', 'orthogonal', 'kaiming', 'none'])         
    parser.add_argument('--tie_weight', type=str2bool, default="False")  
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--apply_edge_type', type=str2bool, default="True") 
    parser.add_argument('--edge_type', type=int, default=6) 
    parser.add_argument('--add_neg_edge', type=str2bool, default='True') 
    parser.add_argument('--embed_manifold', type=str, default='euclidean', choices=['euclidean']) 
    parser.add_argument('--metric', type=str, default="mae", choices=['rmse', 'mae']) 
    parser.add_argument('--train_file', type=str, default='data/qm9/molecules_train_qm9.json')
    parser.add_argument('--dev_file', type=str, default='data/qm9/molecules_valid_qm9.json')
    parser.add_argument('--test_file', type=str, default='data/qm9/molecules_test_qm9.json')
    parser.add_argument('--total_atom', type=int, default=6)
    parser.add_argument('--num_feature', type=int, default=6)
    parser.add_argument('--num_property', type=int, default=12)
    parser.add_argument('--prop_idx', type=int, default=0)
    parser.add_argument("--num_centroid", type=int, default=30)
    parser.add_argument('--eucl_vars', type=list, default=[]) 
    parser.add_argument('--is_regression', type=str2bool, default=True) 
    parser.add_argument('--normalization', type=str2bool, default=False) 
    parser.add_argument('--remove_embed', type=str2bool, default=False) 
    parser.add_argument('--dist_method', type=str, default='all_gather', choices=['all_gather', 'reduce'])  
