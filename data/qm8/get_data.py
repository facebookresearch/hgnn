#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import QED
import glob
import json
import numpy as np
from utils import bond_dict, dataset_info, need_kekulize, to_graph
import utils
import pickle
import random

dataset = 'qm8'
download_path = 'gdb8'

def get_validation_file_names():
    print('loading valid split')
    with open('valid_idx_qm8.json', 'r') as f:
        valid_idx = json.load(f)['idxs']
    valid_files = [int(i) for i in valid_idx]
    return valid_files

def get_test_file_names():
    print('loading test split')
    with open('test_idx_qm8.json', 'r') as f:
        test_idx = json.load(f)['idxs']
    test_files = [int(i) for i in test_idx]
    return test_files


all_values = []

def read_xyz(file_path, mol, properties):
    smiles = mol
    prop = properties[file_path]
    all_values.append(prop)
    return {'smiles': smiles, 'prop': prop}

def read_csv():
    properties = {}
    with open("qm8.sdf.csv", 'r') as f:
        for idx, line in enumerate(f.readlines()[1:]):
            prop = line.split(",")[1:13]
            properties[idx] = [float(i) for i in prop]
    return properties

def train_valid_split():
    print('reading data...')
    raw_data = {'train': [], 'valid': [], 'test': []}

    all_files = Chem.SDMolSupplier(str("qm8.sdf"), False, False, False)
    properties = read_csv()
    valid_files = get_validation_file_names()
    test_files = get_test_file_names()

    file_count = 0
    for file_idx, mol in enumerate(all_files):
        if file_idx in valid_files:
            raw_data['valid'].append(read_xyz(file_idx, mol, properties))
        elif file_idx in test_files:
            raw_data['test'].append(read_xyz(file_idx, mol, properties))
        else:
            raw_data['train'].append(read_xyz(file_idx, mol, properties))
        file_count += 1
        if file_count % 2000 == 0:
            print('finished reading: %d' % file_count, end='\r')
    return raw_data

def preprocess(raw_data, dataset):
    global all_values
    mean = np.mean(all_values, axis=0).tolist()
    std = np.std(all_values, axis=0).tolist()
    mean = np.array(mean)
    std = np.array(std)

    print('parsing smiles as graphs...')
    processed_data = {'train': [], 'valid': [], 'test': []}

    file_count = 0
    for section in ['train', 'valid', 'test']:
        for i, (smiles, prop) in enumerate([(mol['smiles'], mol['prop'])                                        for mol in raw_data[section]]):
            nodes, edges = to_graph(smiles, dataset)
            if len(edges) <= 0:
                continue
            prop = np.array(prop)
            processed_data[section].append({
                'targets': prop.tolist(),
                'graph': edges,
                'node_features': nodes,
            })
            if file_count % 2000 == 0:
                print('finished processing: %d' % file_count, end='\r')
            file_count += 1
        print('%s: 100 %%      ' % (section))
        with open('molecules_%s_%s.json' % (section, dataset), 'w') as f:
            json.dump(processed_data[section], f)


def download_file():
    print('downloading data to %s ...' % download_path)
    source = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb8.tar.gz'
    os.system('wget %s' % source)
    os.system('tar xvzf %s.tar.gz' % download_path)
    print('finished downloading')

if __name__ == "__main__":
    download_file()
    raw_data = train_valid_split()
    preprocess(raw_data, dataset)
