#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch.optim.optimizer import Optimizer, required
from utils import *
import os
import math

class RiemannianSGD(Optimizer):
    """Riemannian stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rgrad (Function): Function to compute the Riemannian gradient from
            an Euclidean gradient
        retraction (Function): Function to update the parameters via a
            retraction of the Riemannian gradient
        lr (float): learning rate
    """

    def __init__(self, args, params, lr):
        defaults = dict(lr=lr)
        self.args = args
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """
        Performs a single optimization step.
        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p = self.args.manifold.rgrad(p, d_p)
                if lr is None:
                    lr = group['lr']
                p.data = self.args.manifold.normalize(
                            self.args.manifold.exp_map_x(p, -lr * d_p)
                         )
        return loss
