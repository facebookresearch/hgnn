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
    return { 'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
            'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)','S4(0)', 'S6(0)'],
            'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5:1, 6:2, 7:3, 8:4, 9:1, 10:2, 11:2, 12:4, 13:6, 14:3},
            'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5:'I', 6:'N', 7:'N', 8:'N', 9:'O', 10:'O', 11:'S', 12:'S', 13:'S'},
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

def to_graph(smiles, dataset):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol is None:
        return [], []
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return [], []
    Chem.RemoveStereochemistry(mol)

    edges = []
    nodes = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        valence = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        atom_str = "%s%i(%i)" % (symbol, valence, charge)

        if atom_str not in dataset_info(dataset)['atom_types']:
            return [], []
        nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom_str), len(dataset_info(dataset)['atom_types'])))
    return nodes, edges
