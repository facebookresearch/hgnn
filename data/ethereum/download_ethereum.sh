#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

wget http://dl.fbaipublicfiles.com/ethereum_data.zip
curl "https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=0&end=1546646400&period=14400" -o POLO_USDT_ETH_FOURHOURLY.json
unzip ethereum_data.zip
mv ethereum_data/graph .
mv ethereum_data/dev_tst_split.pickle .
rm -rf ethereum_data
rm ethereum_data.zip
