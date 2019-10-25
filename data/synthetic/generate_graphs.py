#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import networkx as nx
import json
import numpy as np
from collections import deque
import pickle

def pickle_dump(file_name, content):
    with open(file_name, 'wb') as out_file:
        pickle.dump(content, out_file, pickle.HIGHEST_PROTOCOL)

def graph_to_edges(graph):
    return [(src, 1, dst) for src, dst in graph.edges()]

if __name__ == "__main__":
    max_node_num = 500
    min_node_num = 200
    graph_num = 6000
    dev_num = 2000
    test_num = 2000

    print("erdos_renyi")
    erdos_renyi = list()
    for i in range(graph_num):
        print(i)
        num_node = np.random.randint(min_node_num, max_node_num)
        graph = nx.erdos_renyi_graph(num_node, np.random.uniform(0.01, 1))
        erdos_renyi.append({
                    'targets': 0,
                    'graph': graph_to_edges(graph),
                    'num_node': num_node
                })

    print("small_world")
    small_world = list()
    for i in range(graph_num):
        print(i)
        num_node = np.random.randint(min_node_num, max_node_num)
        graph = nx.watts_strogatz_graph(num_node, np.random.randint(low=1, high=200), np.random.uniform(0.01, 1))
        small_world.append({
                    'targets': 1,
                    'graph': graph_to_edges(graph),
                    'num_node': num_node
                })

    print("barabasi_albert")
    barabasi_albert = list()
    for i in range(graph_num):
        print(i)
        num_node = np.random.randint(min_node_num, max_node_num)
        graph = nx.barabasi_albert_graph(num_node, np.random.randint(low=1, high=200))
        barabasi_albert.append({
                    'targets': 2,
                    'graph': graph_to_edges(graph),
                    'num_node': num_node
                })

    dev_section = erdos_renyi[:dev_num] + small_world[:dev_num] + barabasi_albert[:dev_num]
    test_section = erdos_renyi[dev_num:(dev_num + test_num)] + small_world[dev_num:(dev_num + test_num)] + barabasi_albert[dev_num:(dev_num + test_num)]
    train_section = erdos_renyi[(dev_num + test_num):] + small_world[(dev_num + test_num):] + barabasi_albert[(dev_num + test_num):]

    pickle_dump('synthetic_train.pkl', train_section)
    pickle_dump('synthetic_dev.pkl', dev_section)
    pickle_dump('synthetic_test.pkl', test_section)

