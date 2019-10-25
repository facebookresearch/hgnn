# Hyperbolic Graph Neural Networks

## Requirements

* Python 3.7
* PyTorch >= 1.1
* RDKit
* numpy
* networkx
* scikit-learn

A recipe about installing the requirements is provided in `install.sh`.

## Data Preprocess

For the Ethereum dataset, go to `data/ethereum` and run
`download_ethereum.sh`

For the node classification dataset, go to `data/node` and run
`download_node.sh`

For QM8, QM9 and ZINC, go to `data/qm8`, `data/qm9` and `data/zinc`, respectively and run 
```python get_data.py```

For the synthetic dataset, go to `data/synthetic` and run
```python generate_graphs.py```

For TU Dortmund datasets, go to `data/tu` and run
```python data_preprocess.py {REDDIT-MULTI-12K, PROTEINS_full, ENZYMES, DD, COLLAB}```

## Run Experiments
The code can be run on SLURM and on multiple GPUs. To run on multi GPUs, use 

```python -m torch.distributed.launch --nproc_per_node=NUM_GPU main.py --task {qm8, qm9, zinc, ethereum, node_classification, synthetic, dd, enzymes, proteins, reddit, collab}```

## Inputs of Riemannian GNN

Here we introduce the inputs of Riemannian GNN:

* `node_repr`: representations of each node. 
* `adj_list`: an adjacency list, of which each row `i` consists of the neighbor IDs of node `i`. `adj_list` is padded using 0 to make each row of the same size.
* `weight`: a weight list for weighted graphs, of which each row `i` contains the weights of neighbors. `weight` is padded using 0 to make each row of the same size.
* `mask`: the `i`-th row of `mask` is 0 if the node `i` is padded. Otherwise, the `i`-th row is 1.


## Directory

* `dataset`: dataset files.
* `gnn`: Riemannian graph neural network implementation.
* `hyperbolic_module`: centroid-based classification and Poincaré distance.
* `manifold`: Poincaré, Lorentz and Euclidean manifolds.
* `optimizer`: Riemannian SGD and Riemannian AMSGrad.
* `params`: parameters for each task.
* `task`: task code.
* `utils`: utility modules and functions.

## Hyperparameters

Some notable hyperparameters are listed here.

* `lr`: learning rate for Euclidean variables.
* `lr_hyperbolic`: learning rate for hyperbolic variables.
* `optimizer`: optimizer for Euclidean variables.
* `hyper_optimizer`: optimizer rate for hyperbolic variables.
* `num_centroid`: the number of centroids for centroid-based prediction.
* `gnn_layer`: the number of GNN layers.
* `embed_size`: the embedding size.
* `apply_edge_type`: a boolean value denotes multi-relational or single-relational.
* `edge_type`: the number of relations for multi-relational datasets.
* `select_manifold`: use the Euclidean, Poincaré or Lorentz manifold.
* `activation`: the activation function.

## License
HGNN is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.
