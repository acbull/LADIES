## Overview

LADIES (Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks) is a novel sampling algorithm for training GNN. It considers the previously sampled nodes for calculating layer-dependent sampling probability.

Based on the sampled nodes in the upper layer, LADIES selects their neighborhood nodes, compute the importance probability accordingly and samples a fixed number of nodes within them. We prove theoretically and experimentally, that our proposed sampling algorithm outperforms the previous sampling methods regarding time, memory and accuracy.

You can see our NeurIPS 2019 paper [“**Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks**”](https://arxiv.org/abs/1911.07323) for more details.

## Setup
This implementation is based on Pytorch We assume that you're using Python 3 with pip installed. To run the code, you need the following dependencies:

- [Pytorch 1.0](https://pytorch.org/)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [networkx](https://networkx.github.io/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://numpy.org/)


We upload the three small benchmark node classification datasets, cora, pubmed and citeseer in /data for usage. You can upload any graph datasets as you want, and change the data loader function in /utils.py

## Usage
Execute the following scripts to train and evaluate the model:

```bash
python3 pytorch_ladies.py --cuda 0 --dataset cora  # Train GCN with LADIES on cora.
```
There's also other hyperparameters to be tuned, which can be found in 'pytorch_ladies.py' for details.

The main function is ``ladies_sampler()``, which sample a fixed number of nodes per layer. The sampling probability (importance) is computed adaptively according to the nodes sampled in the upper layer. We currently implement it using numpy sparse matrix multiplication. One can also implement it via pytorch multiplication or dictionary operation later. 

### Citation

Please consider citing the following paper when using our code for your application.

```
@inproceedings{ladies2019,
  title={Few-Shot Representation Learning for Out-Of-Vocabulary Words},
  author={Difan Zou and Ziniu Hu and Yewen Wang and Song Jiang and Yizhou Sun and Quanquan Gu},
  booktitle={Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems, NeurIPS},
  year={2019}
}
```
