#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# download and install miniconda
# please update the link below according to the platform you are using (https://conda.io/miniconda.html)
# e.g. for Mac, change to https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# create a new environment named hgnn
conda create --name hgnn python=3.7 pip
source activate hgnn

# install rdkit
conda install -c rdkit rdkit 

# install requirements
pip install -r requirements.txt

# remove conda bash
rm ./Miniconda3-latest-Linux-x86_64.sh
