#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import queue
import threading
import pickle
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import rdmolops
from collections import defaultdict, deque
import os
import heapq
from rdkit.Chem import Crippen
from rdkit.Chem import QED

# bond mapping
bond_dict = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, "AROMATIC": 4}
number_to_bond= {1: Chem.rdchem.BondType.SINGLE, 2:Chem.rdchem.BondType.DOUBLE,
                 3: Chem.rdchem.BondType.TRIPLE, 4:Chem.rdchem.BondType.AROMATIC}

def dataset_info(dataset):

    return { 'atom_types': ["H", "C", "N", "O", "F"],
             'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
             'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
           }


def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z

def need_kekulize(mol):
    for bond in mol.GetBonds():
        if bond_dict[str(bond.GetBondType())] >= 3:
            return True
    return False

def to_graph(mol, dataset):
    if mol is None:
        return [], []
    edges = []
    nodes = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
    n_atoms = mol.GetNumAtoms()
    dist_matrix = Chem.rdmolops.Get3DDistanceMatrix(mol)
    for i in range(n_atoms):
      for j in range(n_atoms):
        if i < j:
            dist = dist_matrix[i][j]
            gaussian_dist = np.exp(-dist)
            edges.append((i, -gaussian_dist, j))
        else:
          continue
    for atom in mol.GetAtoms():
        nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom.GetSymbol()), len(dataset_info(dataset)['atom_types'])))
    return nodes, edges
