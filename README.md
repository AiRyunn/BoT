# Bag of Tricks for Node Classification with Graph Neural Networks

The official implementation for [Bag of Tricks for Node Classification with Graph Neural Networks](https://arxiv.org/abs/2103.13355) (Best Paper Award at DLG-KDD'21 workshop) based on [Deep Graph Library](https://www.dgl.ai/).

## Dependencies

* dgl 0.5.*
* torch 1.6.0
* ogb 1.3.1

## How to run

### Cora, Citeseer, Pubmed, Reddit, ogbn-arxiv

Run

```bash
cd src/no-sampling/
python3 src/no-sampling/run.py [args]
```

For example,

```bash
python3 run.py --optimizer=rmsprop --lr=0.002 --loss=loge --labels --mask-rate=0.5 --model=gat --linear --n-heads=3 --n-hidden=250 --dropout=0.75 --input-drop=0.25 --attn-drop=0.1 --norm-adj=symm
```

More details of the hyperparameters and experimental results can be found at the end of `run.py`.

### ogbn-proteins

Run

```bash
cd src/ogbn-proteins/
python3 gat.py [args]
```

For the results in the paper, run

```bash
python3 gat.py
```

or

```bash
python3 gat.py --use-labels
```

### ogbn-products

First change the directory

```bash
cd src/ogbn-products/
```

For GAT, run

```bash
python3 gat.py [args]
```

For MLP, run

```bash
python3 mlp.py [args]
```

## Citing our work

If you find this work helpful in your research, please consider citing our work.

```tex
@article{wang2021bag,
  title={Bag of Tricks for Node Classification with Graph Neural Networks},
  author={Wang, Yangkun and Jin, Jiarui and Zhang, Weinan and Yu, Yong and Zhang, Zheng and Wipf, David},
  journal={arXiv preprint arXiv:2103.13355},
  year={2021}
}
```
