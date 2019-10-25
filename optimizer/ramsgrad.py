#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Implement a AMSGrad: https://openreview.net/pdf?id=r1eiqi09K7
"""
import torch as th
from torch.optim.optimizer import Optimizer, required
import os
import math
import numpy as np

class RiemannianAMSGrad(Optimizer):
    """
    Riemannian AMS gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """

    def __init__(self, args, params, lr, betas=(0.9, 0.99), eps=1e-8):
        self.args = args
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(RiemannianAMSGrad, self).__init__(params, defaults)

    def step(self, lr=None):
        """Performs a single optimization step.
        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None
        with th.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    grad = self.args.manifold.rgrad(p, grad)
                    if lr is None:
                        lr = group['lr']

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['tau'] = th.zeros_like(p.data)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = th.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = th.zeros_like(p.data)
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = th.zeros_like(p.data)

                    exp_avg, exp_avg_sq, tau, max_exp_avg_sq = \
                    			state['exp_avg'], state['exp_avg_sq'], state['tau'], state['max_exp_avg_sq']

                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    # Decay the first and second moment running average coefficient
                    exp_avg.data = beta1 * tau + (1 - beta1) * grad
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, self.args.manifold.metric_tensor(p, grad, grad))
                    th.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().clamp_(min=group['eps'])

                    step_size = group['lr']

                    p_original = p.clone()
                    before_proj = self.args.manifold.exp_map_x(p, (-step_size * exp_avg).div_(denom))
                    p.data = self.args.manifold.normalize(before_proj)
                    tau.data = self.args.manifold.parallel_transport(p_original, p, exp_avg)
            return loss
