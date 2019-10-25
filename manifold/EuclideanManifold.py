#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Function, Variable
from utils import *

class EuclideanManifold:

    def __init__(self, args, logger, max_norm=1, EPS=1e-8):
        self.args = args
        self.logger = logger
        self.max_norm = max_norm
        self.EPS = EPS

    def init_embed(self, embed, irange=1e-3):
        embed.weight.data.uniform_(-irange, irange)
        embed.weight.data.copy_(self.normalize(embed.weight.data))

    def distance(self, u, v):
        return th.sqrt(clamp_min(th.sum((u - v).pow(2), dim=1), self.EPS))

    def log_map_zero(self, y):
        return y

    def log_map_x(self, x, y):
        return y - x

    def metric_tensor(self, x, u, v):
        return th_dot(u, v)

    def exp_map_zero(self, v):
        return self.normalize(v)

    def exp_map_x(self, x, v, approximate=False):
        return self.normalize(x + v)

    def parallel_transport(self, src, dst, v):
        return v

    def rgrad(self, p, d_p):
        return d_p

    def normalize(self, w):
        if self.max_norm:
            return clip_by_norm(w, self.max_norm)
        else:
            return w
