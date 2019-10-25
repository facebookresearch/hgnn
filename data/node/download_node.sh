#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

git clone https://github.com/tkipf/gcn.git
mv gcn/gcn/data/* .
rm -rf gcn
