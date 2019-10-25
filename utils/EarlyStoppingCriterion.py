#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class EarlyStoppingCriterion(object):
    """
    Arguments:
        patience (int): The maximum number of epochs with no improvement before early stopping should take place
        mode (str, can only be 'max' or 'min'): To take the maximum or minimum of the score for optimization
        min_delta (float, optional): Minimum change in the score to qualify as an improvement (default: 0.0)
    """

    def __init__(self, patience, mode, min_delta=0.0):
        assert patience >= 0
        assert mode in {'min', 'max'}
        assert min_delta >= 0.0
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self._count = 0
        self.best_dev_score = None
        self.best_test_score = None
        self.best_epoch = None
        self.is_improved = None

    def step(self, cur_dev_score, cur_test_score, epoch):
        """
        Checks if training should be continued given the current score.

        Arguments:
            cur_dev_score (float): the current development score
            cur_test_score (float): the current test score
        Output:
            bool: if training should be continued
        """
        if self.best_dev_score is None:
            self.best_dev_score = cur_dev_score
            self.best_test_score = cur_test_score
            self.best_epoch = epoch
            return True
        else:
            if self.mode == 'max':
                self.is_improved = (cur_dev_score > self.best_dev_score + self.min_delta)
            else:
                self.is_improved = (cur_dev_score < self.best_dev_score - self.min_delta)

            if self.is_improved:
                self._count = 0
                self.best_dev_score = cur_dev_score
                self.best_test_score = cur_test_score
                self.best_epoch = epoch
            else:
                self._count += 1
            return self._count <= self.patience
