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
import csv, json
import numpy as np
from utils import bond_dict, dataset_info, need_kekulize, to_graph
import utils
import pickle
import random

dataset = "zinc"
all_values = []

def preprocess(raw_data, dataset, all_values):
    # mean and std
    mean = np.mean(all_values, axis=0).tolist()
    std = np.std(all_values, axis=0).tolist()
    mean = np.array(mean)
    std = np.array(std)

    print('parsing smiles as graphs...')
    processed_data = {'train': [], 'valid': [], 'test': []}

    file_count = 0
    for section in ['train', 'valid', 'test']:
        for i, (smiles, prop) in enumerate([(mol['smiles'], mol['prop'])
                                          for mol in raw_data[section]]):
            nodes, edges = to_graph(smiles, dataset)
            if len(edges) <= 0:
                continue
            prop = np.array(prop)
            processed_data[section].append({
                'targets': prop.tolist(),
                'graph': edges,
                'node_features': nodes,
                'smiles': smiles
            })
            if file_count % 2000 == 0:
                print('finished processing: %d' % file_count, end='\r')
            file_count += 1
        print('%s: 100 %%      ' % (section))
        # save the dataset
        with open('molecules_%s_%s.json' % (section, dataset), 'w') as f:
            json.dump(processed_data[section], f)

def dataset_split(download_path):
    # load validation dataset
    with open("valid_idx_zinc.json", 'r') as f:
        valid_idx = json.load(f)
    with open("test_idx_zinc.json", 'r') as f:
        test_idx = json.load(f)

    print('reading data...')
    with open(download_path, 'r') as f:
        all_data = list(csv.DictReader(f))

    file_count=0
    raw_data = {'train': [], 'valid': [], 'test': []} # save the train, valid dataset.
    for i, data_item in enumerate(all_data):
        smiles = data_item['smiles'].strip()
        prop = [float(data_item['qed']), float(data_item['logP']), float(data_item['SAS'])]
        all_values.append(prop)
        if i in valid_idx:
            raw_data['valid'].append({'smiles': smiles, 'prop': prop})
        elif i in test_idx:
            raw_data['test'].append({'smiles': smiles, 'prop': prop})
        else:
            raw_data['train'].append({'smiles': smiles, 'prop': prop})

        file_count += 1
        if file_count % 2000 ==0:
            print('finished reading: %d' % file_count, end='\r')
    return raw_data

def download_file(download_path):

    if not os.path.exists(download_path):
        print('downloading data to %s ...' % download_path)
        source = 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'
        os.system('wget -O %s %s' % (download_path, source))
        print('finished downloading')

if __name__ == "__main__":
    download_path = '250k_rndm_zinc_drugs_clean_3.csv'
    download_file(download_path)
    raw_data = dataset_split(download_path)
    preprocess(raw_data, dataset, all_values)
