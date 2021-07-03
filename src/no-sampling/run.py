#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import sys
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl import function as fn
from dgl.data import (
    AmazonCoBuyComputerDataset,
    AmazonCoBuyPhotoDataset,
    CiteseerGraphDataset,
    CoauthorCSDataset,
    CoraFullDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    RedditDataset,
)
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from tqdm import tqdm

from models import GAT, GCN, MLP

epsilon = 1 - math.log(2)

device = None

n_node_feats, n_edge_feats, n_classes = 0, 0, 0


def sum_w2(model):
    w2 = 0
    for param in model.parameters():
        w2 += param.pow(2).sum().item()
    return w2


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def compute_acc(pred, labels):
    return ((torch.argmax(pred, dim=1) == labels[:, 0]).float().sum() / len(pred)).item()


def load_data(dataset, split):
    global n_node_feats, n_classes

    if dataset in ["ogbn-arxiv"]:
        data = DglNodePropPredDataset(name=dataset)
    elif dataset == "cora":
        data = CoraGraphDataset()
    elif dataset == "citeseer":
        data = CiteseerGraphDataset()
    elif dataset == "pubmed":
        data = PubmedGraphDataset()
    elif dataset == "cora-full":
        data = CoraFullDataset()
    elif dataset == "reddit":
        data = RedditDataset()
    elif dataset == "amazon-co-computer":
        data = AmazonCoBuyComputerDataset()
    elif dataset == "amazon-co-photo":
        data = AmazonCoBuyPhotoDataset()
    elif dataset == "coauthor-cs":
        data = CoauthorCSDataset()
    else:
        assert False

    if dataset in ["ogbn-arxiv"]:
        graph, labels = data[0]
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]

        evaluator_ = Evaluator(name=dataset)
        evaluator = lambda pred, labels: evaluator_.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
        )["acc"]

    elif dataset in ["cora", "citeseer", "pubmed", "reddit"]:
        graph = data[0]
        labels = graph.ndata["label"].reshape(-1, 1)
        train_mask, val_mask, test_mask = graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"]
        train_idx, val_idx, test_idx = map(
            lambda mask: torch.nonzero(mask, as_tuple=False).squeeze_(), [train_mask, val_mask, test_mask]
        )

        evaluator = compute_acc
    elif dataset == "cora-full":
        graph = data[0]
        labels = graph.ndata["label"].reshape(-1, 1)
    elif dataset in ["amazon-co-computer", "amazon-co-photo", "coauthor-cs"]:
        graph = data[0]
        labels = graph.ndata["label"].reshape(-1, 1)
        train_idx, val_idx, test_idx = None, None, None
        assert split == "random"
        # train_mask, val_mask, test_mask = graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"]
        # train_idx, val_idx, test_idx = map(
        #     lambda mask: torch.nonzero(mask, as_tuple=False).squeeze_(), [train_mask, val_mask, test_mask]
        # )

        evaluator = compute_acc
    else:
        assert False

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    print(f"#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}, #Classes: {n_classes}")
    if split != "random":
        print(f"#Train/Val/Test nodes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


def random_split(graph):
    """
    6:2:2 for traing/val/test
    """
    n = graph.number_of_nodes()
    perm = torch.randperm(n, device=device)
    val_offset, test_offset = int(n * 0.6), int(n * 0.8)
    train_idx, val_idx, test_idx = perm[:val_offset], perm[val_offset:test_offset], perm[test_offset:]

    print(f"#Train/Val/Test nodes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    return train_idx, val_idx, test_idx


def build_model(args):
    if args.labels:
        n_input_feats = n_node_feats + n_classes
    else:
        n_input_feats = n_node_feats
    if args.activation == "relu":
        activation = F.relu
    elif args.activation == "elu":
        activation = F.elu
    else:
        assert False

    if args.model == "mlp":
        model = MLP(
            in_feats=n_input_feats,
            n_hidden=args.n_hidden,
            n_classes=n_classes,
            n_layers=args.n_layers,
            activation=activation,
            norm=args.norm,
            input_drop=args.input_drop,
            dropout=args.dropout,
            residual=args.residual,
        )
    elif args.model == "gcn":
        model = GCN(
            in_feats=n_input_feats,
            n_classes=n_classes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            activation=activation,
            norm=args.norm,
            norm_adj=args.norm_adj,
            input_drop=args.input_drop,
            dropout=args.dropout,
            use_linear=args.linear,
            residual=args.residual,
        )
    elif args.model == "gat":
        model = GAT(
            dim_node=n_input_feats,
            dim_edge=n_edge_feats,
            dim_output=n_classes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=activation,
            norm=args.norm,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            edge_drop=args.edge_drop,
            non_interactive_attn=args.non_interactive_attn,
            # negative_slope=args.negative_slope,
            use_symmetric_norm=args.norm_adj == "symm",
            linear=args.linear,
            residual=args.residual,
        )
    else:
        assert False

    return model


def compute_loss(args, x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    if args.loss == "loge":
        y = torch.log(epsilon + y) - math.log(epsilon)
    elif args.loss == "savage":
        y = (1 - torch.exp(-y)) ** 2
    else:
        assert args.loss == "logit"
    return torch.mean(y)


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, evaluator):
    model.train()

    feat = graph.ndata["feat"]

    if args.labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_pred_idx = train_idx[mask]

    if args.model == "mlp":
        pred = model(feat)
    else:
        pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)

    loss = compute_loss(args, pred[train_pred_idx], labels[train_pred_idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return evaluator(pred[train_idx], labels[train_idx]), loss.item()


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, evaluator, epoch):
    model.eval()

    feat = graph.ndata["feat"]

    if args.labels:
        feat = add_labels(feat, labels, train_idx)

    if args.model == "mlp":
        pred = model(feat)
    else:
        pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)

    train_loss = compute_loss(args, pred[train_idx], labels[train_idx])
    val_loss = compute_loss(args, pred[val_idx], labels[val_idx])
    test_loss = compute_loss(args, pred[test_idx], labels[test_idx])

    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        pred,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    if args.split == "random":
        train_idx, val_idx, test_idx = random_split(graph)

    # define model and optimizer
    model = build_model(args).to(device)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.8)
    else:
        assert False

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
    final_pred = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        tic = time.time()

        if args.optimizer == "rmsprop":
            adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss = train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, labels, train_idx, val_idx, test_idx, evaluator, epoch
        )

        toc = time.time()
        total_time += toc - tic

        if (
            args.dataset != "ogbn-arxiv"
            and val_acc > best_val_acc
            or args.dataset == "ogbn-arxiv"
            and val_loss < best_val_loss
        ):
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_pred = pred

        if epoch == args.epochs or epoch % args.log_every == 0:
            print(
                f"Run: {n_running}/{args.runs}, Epoch: {epoch}/{args.epochs}, Average epoch time: {total_time / epoch:.4f}s\n"
                f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

    print("*" * 50)
    print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    print("*" * 50)

    # plot learning curves
    if args.plot:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([accs, train_accs, val_accs, test_accs], ["acc", "train acc", "val acc", "test acc"]):
            plt.plot(range(args.epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.model}_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
        ):
            plt.plot(range(args.epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.model}_loss_{n_running}.png")

    if args.save_pred:
        os.makedirs("./output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{n_running}.pt")

    return best_val_acc, final_test_acc


def count_parameters(args):
    model = build_model(args)
    return sum([p.numel() for p in model.parameters()])


def main():
    global device

    argparser = argparse.ArgumentParser(
        "Implementation of MLP, GCN and GAT with Bag of Tricks (arXiv:2103.13355)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # basic settings
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument("--runs", type=int, default=10, help="running times")
    argparser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "ogbn-arxiv",
            "cora",
            "citeseer",
            "pubmed",
            "cora-full",
            "reddit",
            "amazon-co-computer",
            "amazon-co-photo",
            "coauthor-cs",
        ],
        default="ogbn-arxiv",
        help="dataset",
    )
    argparser.add_argument("--split", type=str, choices=["std", "random"], default="std", help="split")
    # training
    argparser.add_argument("--epochs", type=int, default=2000, help="number of epochs")
    argparser.add_argument(
        "--loss", type=str, choices=["logit", "loge", "savage"], default="logit", help="loss function"
    )
    argparser.add_argument(
        "--optimizer", type=str, choices=["adam", "rmsprop", "sgd"], default="adam", help="optimizer"
    )
    argparser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    # model
    argparser.add_argument("--labels", action="store_true", help="use labels in the training set as input features")
    argparser.add_argument("--n-label-iters", type=int, default=0, help="number of label iterations")
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument("--model", type=str, choices=["mlp", "gcn", "gat"], default="gat", help="model")
    argparser.add_argument("--residual", action="store_true", help="residual")
    argparser.add_argument("--linear", action="store_true", help="use linear layer")
    argparser.add_argument(
        "--norm-adj",
        type=str,
        choices=["symm", "rw", "default"],
        default="default",
        help="symmetric normalized (symm) or randon walk normalized (rw) adjacency matrix; default for GCN: symm, default for GAT: rw",
    )
    argparser.add_argument("--non-interactive-attn", action="store_true", help="non-interactive attention")
    argparser.add_argument("--norm", type=str, choices=["none", "batch"], default="batch", help="norm")
    argparser.add_argument("--activation", type=str, choices=["relu", "elu"], default="relu", help="activation")
    argparser.add_argument("--n-prop", type=int, default=7, help="number of props")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--n-hidden", type=int, default=256, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.0, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention drop rate")
    argparser.add_argument("--edge-drop", type=float, default=0.0, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    # output
    argparser.add_argument("--log-every", type=int, default=20, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--tune", type=str, default="", help="tune")
    args = argparser.parse_args()

    if not args.labels and args.n_label_iters > 0:
        raise ValueError("'--labels' must be enabled when n_label_iters > 0")

    if args.model == "gcn":
        if args.non_interactive_attn > 0:
            raise ValueError("'no_attn_dst' is not supported for GCN")
        if args.attn_drop > 0:
            raise ValueError("'attn_drop' is not supported for GCN")
        if args.edge_drop > 0:
            raise ValueError("'edge_drop' is not supported for GCN")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    if args.norm_adj == "default":
        if args.model == "gcn":
            args.norm_adj = "symm"
        elif args.model == "gat":
            args.norm_adj = "rw"

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(args.dataset, args.split)
    graph = preprocess(graph)

    graph, labels = map(lambda x: x.to(device), (graph, labels))
    if args.split != "random":
        train_idx, val_idx, test_idx = map(lambda x: x.to(device), (train_idx, val_idx, test_idx))

    # run
    val_accs, test_accs = [], []

    for i in range(args.runs):
        seed(args.seed + i)
        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    # print results
    print(" ".join(sys.argv))
    print(args)
    if args.runs > 0:
        print(f"Runned {args.runs} times")
        print("Val Accs:", val_accs)
        print("Test Accs:", test_accs)
        print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
        print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"#Params: {count_parameters(args)}")


if __name__ == "__main__":
    main()


### MLP

## Cora

# Logistic loss

# run.py --dataset=cora --runs=100 --optimizer=adam --lr=2e-3 --wd=5e-4 --epochs=500 --loss=logit --model=mlp --n-layers=2 --n-hidden=256 --dropout=0.8 --input-drop=0.8 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='cora', dropout=0.8, edge_drop=0.0, epochs=500, gpu=0, input_drop=0.8, labels=False, linear=False, log_every=20, loss='logit', lr=0.002, mask_rate=1.0, model='mlp', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.6320000290870667, 0.6200000047683716, 0.6200000047683716, 0.6200000047683716, 0.6200000047683716, 0.6300000548362732, 0.6260000467300415, 0.6200000047683716, 0.6100000143051147, 0.628000020980835, 0.6260000467300415, 0.6260000467300415, 0.6300000548362732, 0.6200000047683716, 0.6100000143051147, 0.6260000467300415, 0.6260000467300415, 0.6160000562667847, 0.6100000143051147, 0.628000020980835, 0.64000004529953, 0.628000020980835, 0.6160000562667847, 0.6160000562667847, 0.6180000305175781, 0.6060000061988831, 0.6200000047683716, 0.6140000224113464, 0.6200000047683716, 0.6320000290870667, 0.628000020980835, 0.6200000047683716, 0.6180000305175781, 0.6220000386238098, 0.628000020980835, 0.6300000548362732, 0.6180000305175781, 0.6100000143051147, 0.6260000467300415, 0.628000020980835, 0.6180000305175781, 0.6260000467300415, 0.6260000467300415, 0.6200000047683716, 0.6260000467300415, 0.6200000047683716, 0.6180000305175781, 0.6220000386238098, 0.6240000128746033, 0.6300000548362732, 0.6100000143051147, 0.6260000467300415, 0.6260000467300415, 0.6100000143051147, 0.6220000386238098, 0.6320000290870667, 0.6340000033378601, 0.6160000562667847, 0.6240000128746033, 0.6220000386238098, 0.6140000224113464, 0.6260000467300415, 0.612000048160553, 0.6260000467300415, 0.6020000576972961, 0.6200000047683716, 0.6320000290870667, 0.6160000562667847, 0.6240000128746033, 0.612000048160553, 0.6100000143051147, 0.6240000128746033, 0.6220000386238098, 0.6180000305175781, 0.6240000128746033, 0.6240000128746033, 0.6220000386238098, 0.6180000305175781, 0.6080000400543213, 0.612000048160553, 0.6220000386238098, 0.6260000467300415, 0.6220000386238098, 0.6080000400543213, 0.6140000224113464, 0.6160000562667847, 0.6180000305175781, 0.628000020980835, 0.6320000290870667, 0.6260000467300415, 0.6160000562667847, 0.6200000047683716, 0.6240000128746033, 0.6220000386238098, 0.6340000033378601, 0.6200000047683716, 0.6160000562667847, 0.628000020980835, 0.6300000548362732, 0.6220000386238098]
# Test Accs: [0.6110000014305115, 0.593000054359436, 0.6070000529289246, 0.6020000576972961, 0.6020000576972961, 0.5960000157356262, 0.5900000333786011, 0.6070000529289246, 0.5730000138282776, 0.597000002861023, 0.609000027179718, 0.6070000529289246, 0.597000002861023, 0.5879999995231628, 0.593000054359436, 0.5980000495910645, 0.5950000286102295, 0.5800000429153442, 0.5910000205039978, 0.5950000286102295, 0.612000048160553, 0.6140000224113464, 0.6070000529289246, 0.5890000462532043, 0.5850000381469727, 0.5750000476837158, 0.609000027179718, 0.5870000123977661, 0.6140000224113464, 0.5820000171661377, 0.5940000414848328, 0.6270000338554382, 0.5950000286102295, 0.5990000367164612, 0.597000002861023, 0.5900000333786011, 0.6060000061988831, 0.5860000252723694, 0.6080000400543213, 0.6080000400543213, 0.5950000286102295, 0.593000054359436, 0.6020000576972961, 0.5980000495910645, 0.6060000061988831, 0.593000054359436, 0.6020000576972961, 0.5860000252723694, 0.5879999995231628, 0.6000000238418579, 0.5850000381469727, 0.5990000367164612, 0.6020000576972961, 0.5860000252723694, 0.612000048160553, 0.6210000514984131, 0.6070000529289246, 0.5879999995231628, 0.6030000448226929, 0.581000030040741, 0.597000002861023, 0.5800000429153442, 0.6070000529289246, 0.5870000123977661, 0.5840000510215759, 0.6080000400543213, 0.6030000448226929, 0.6140000224113464, 0.5940000414848328, 0.5910000205039978, 0.5890000462532043, 0.5980000495910645, 0.5980000495910645, 0.5790000557899475, 0.5960000157356262, 0.5990000367164612, 0.6030000448226929, 0.5890000462532043, 0.5960000157356262, 0.6000000238418579, 0.6050000190734863, 0.6010000109672546, 0.5940000414848328, 0.5830000042915344, 0.5940000414848328, 0.5980000495910645, 0.5990000367164612, 0.5920000076293945, 0.6040000319480896, 0.5920000076293945, 0.5940000414848328, 0.6040000319480896, 0.6060000061988831, 0.5840000510215759, 0.6080000400543213, 0.5950000286102295, 0.5800000429153442, 0.6100000143051147, 0.6100000143051147, 0.5960000157356262]
# Average val accuracy: 0.62144003033638 ± 0.007054531683158277
# Average test accuracy: 0.5972300326824188 ± 0.010144807924476095
# #Params: 368903

# Savage loss

# run.py --dataset=cora --runs=100 --optimizer=adam --lr=2e-3 --wd=5e-4 --epochs=500 --loss=loge --model=mlp --n-layers=2 --n-hidden=256 --dropout=0.8 --input-drop=0.8 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='cora', dropout=0.8, edge_drop=0.0, epochs=500, gpu=0, input_drop=0.8, labels=False, linear=False, log_every=20, loss='loge', lr=0.002, mask_rate=1.0, model='mlp', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.6180000305175781, 0.6320000290870667, 0.628000020980835, 0.6160000562667847, 0.6340000033378601, 0.6320000290870667, 0.64000004529953, 0.6240000128746033, 0.6380000114440918, 0.628000020980835, 0.628000020980835, 0.6240000128746033, 0.628000020980835, 0.6320000290870667, 0.6140000224113464, 0.6260000467300415, 0.6220000386238098, 0.6420000195503235, 0.6220000386238098, 0.6260000467300415, 0.6320000290870667, 0.6260000467300415, 0.6300000548362732, 0.6240000128746033, 0.6240000128746033, 0.6140000224113464, 0.6320000290870667, 0.6380000114440918, 0.6180000305175781, 0.6320000290870667, 0.6380000114440918, 0.6260000467300415, 0.628000020980835, 0.6220000386238098, 0.6380000114440918, 0.6300000548362732, 0.6300000548362732, 0.6180000305175781, 0.6320000290870667, 0.6340000033378601, 0.6380000114440918, 0.6340000033378601, 0.6340000033378601, 0.6320000290870667, 0.6320000290870667, 0.6260000467300415, 0.6240000128746033, 0.6360000371932983, 0.6260000467300415, 0.628000020980835, 0.6220000386238098, 0.628000020980835, 0.628000020980835, 0.6180000305175781, 0.6320000290870667, 0.6380000114440918, 0.6320000290870667, 0.6140000224113464, 0.6340000033378601, 0.6260000467300415, 0.6160000562667847, 0.628000020980835, 0.6160000562667847, 0.6320000290870667, 0.6260000467300415, 0.6240000128746033, 0.6420000195503235, 0.6240000128746033, 0.6260000467300415, 0.628000020980835, 0.6200000047683716, 0.628000020980835, 0.6260000467300415, 0.6360000371932983, 0.628000020980835, 0.6300000548362732, 0.628000020980835, 0.6200000047683716, 0.6200000047683716, 0.6260000467300415, 0.628000020980835, 0.6240000128746033, 0.6300000548362732, 0.6240000128746033, 0.6240000128746033, 0.6320000290870667, 0.6340000033378601, 0.6340000033378601, 0.6240000128746033, 0.6340000033378601, 0.6420000195503235, 0.6200000047683716, 0.6360000371932983, 0.6240000128746033, 0.628000020980835, 0.6240000128746033, 0.6260000467300415, 0.64000004529953, 0.6320000290870667, 0.6180000305175781]
# Test Accs: [0.6070000529289246, 0.597000002861023, 0.609000027179718, 0.5910000205039978, 0.6070000529289246, 0.6020000576972961, 0.6070000529289246, 0.593000054359436, 0.6000000238418579, 0.6080000400543213, 0.6060000061988831, 0.6020000576972961, 0.6020000576972961, 0.609000027179718, 0.6020000576972961, 0.5940000414848328, 0.6070000529289246, 0.5910000205039978, 0.6160000562667847, 0.6020000576972961, 0.5990000367164612, 0.609000027179718, 0.6070000529289246, 0.6050000190734863, 0.5990000367164612, 0.593000054359436, 0.6200000047683716, 0.6050000190734863, 0.6140000224113464, 0.6030000448226929, 0.609000027179718, 0.6050000190734863, 0.6160000562667847, 0.5990000367164612, 0.6110000014305115, 0.6020000576972961, 0.609000027179718, 0.6020000576972961, 0.5990000367164612, 0.6020000576972961, 0.6010000109672546, 0.6080000400543213, 0.6060000061988831, 0.6170000433921814, 0.6070000529289246, 0.6080000400543213, 0.6010000109672546, 0.6210000514984131, 0.5940000414848328, 0.612000048160553, 0.6030000448226929, 0.5920000076293945, 0.597000002861023, 0.5840000510215759, 0.6170000433921814, 0.6060000061988831, 0.6100000143051147, 0.612000048160553, 0.6070000529289246, 0.6040000319480896, 0.6030000448226929, 0.5850000381469727, 0.593000054359436, 0.5910000205039978, 0.6080000400543213, 0.5950000286102295, 0.6050000190734863, 0.6040000319480896, 0.6020000576972961, 0.6100000143051147, 0.593000054359436, 0.6000000238418579, 0.6060000061988831, 0.5920000076293945, 0.6150000095367432, 0.6160000562667847, 0.6160000562667847, 0.5990000367164612, 0.609000027179718, 0.5990000367164612, 0.609000027179718, 0.6110000014305115, 0.6010000109672546, 0.5950000286102295, 0.6050000190734863, 0.6040000319480896, 0.6060000061988831, 0.5990000367164612, 0.6050000190734863, 0.6130000352859497, 0.6070000529289246, 0.6050000190734863, 0.6000000238418579, 0.6070000529289246, 0.5860000252723694, 0.6050000190734863, 0.6040000319480896, 0.6070000529289246, 0.6060000061988831, 0.6040000319480896]
# Average val accuracy: 0.6280400264263153 ± 0.006615010742289189
# Average test accuracy: 0.603870033621788 ± 0.00743727830779088
# #Params: 368903

# Loge loss

# run.py --dataset=cora --runs=100 --optimizer=adam --lr=2e-3 --wd=5e-4 --epochs=500 --loss=savage --model=mlp --n-layers=2 --n-hidden=256 --dropout=0.8 --input-drop=0.8 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='cora', dropout=0.8, edge_drop=0.0, epochs=500, gpu=0, input_drop=0.8, labels=False, linear=False, log_every=20, loss='savage', lr=0.002, mask_rate=1.0, model='mlp', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.6340000033378601, 0.6340000033378601, 0.6260000467300415, 0.628000020980835, 0.6380000114440918, 0.6260000467300415, 0.6460000276565552, 0.6380000114440918, 0.6320000290870667, 0.6360000371932983, 0.6420000195503235, 0.628000020980835, 0.6260000467300415, 0.6380000114440918, 0.628000020980835, 0.6260000467300415, 0.6300000548362732, 0.6340000033378601, 0.6220000386238098, 0.6220000386238098, 0.6300000548362732, 0.6260000467300415, 0.628000020980835, 0.6420000195503235, 0.6240000128746033, 0.6180000305175781, 0.6340000033378601, 0.6420000195503235, 0.628000020980835, 0.6240000128746033, 0.6380000114440918, 0.6300000548362732, 0.628000020980835, 0.6220000386238098, 0.6320000290870667, 0.6380000114440918, 0.628000020980835, 0.6220000386238098, 0.6300000548362732, 0.6340000033378601, 0.6380000114440918, 0.6340000033378601, 0.6340000033378601, 0.64000004529953, 0.6420000195503235, 0.6300000548362732, 0.6320000290870667, 0.6420000195503235, 0.6300000548362732, 0.6380000114440918, 0.6260000467300415, 0.6300000548362732, 0.6320000290870667, 0.6240000128746033, 0.628000020980835, 0.628000020980835, 0.6320000290870667, 0.628000020980835, 0.6380000114440918, 0.6320000290870667, 0.6180000305175781, 0.6320000290870667, 0.6220000386238098, 0.628000020980835, 0.6380000114440918, 0.6220000386238098, 0.6460000276565552, 0.6300000548362732, 0.6360000371932983, 0.6340000033378601, 0.628000020980835, 0.6320000290870667, 0.6260000467300415, 0.6380000114440918, 0.628000020980835, 0.6320000290870667, 0.6420000195503235, 0.6260000467300415, 0.6240000128746033, 0.6320000290870667, 0.628000020980835, 0.6200000047683716, 0.6320000290870667, 0.6220000386238098, 0.6320000290870667, 0.6300000548362732, 0.6360000371932983, 0.628000020980835, 0.6320000290870667, 0.6360000371932983, 0.6380000114440918, 0.628000020980835, 0.64000004529953, 0.6360000371932983, 0.6380000114440918, 0.6260000467300415, 0.6320000290870667, 0.6300000548362732, 0.6340000033378601, 0.6240000128746033]
# Test Accs: [0.6080000400543213, 0.6070000529289246, 0.6000000238418579, 0.6020000576972961, 0.6180000305175781, 0.5990000367164612, 0.6110000014305115, 0.6150000095367432, 0.6050000190734863, 0.6150000095367432, 0.6100000143051147, 0.5870000123977661, 0.6030000448226929, 0.6180000305175781, 0.6220000386238098, 0.5990000367164612, 0.6210000514984131, 0.6030000448226929, 0.6150000095367432, 0.6080000400543213, 0.6080000400543213, 0.6060000061988831, 0.6200000047683716, 0.6000000238418579, 0.6100000143051147, 0.6060000061988831, 0.6210000514984131, 0.6100000143051147, 0.6260000467300415, 0.6100000143051147, 0.625, 0.6170000433921814, 0.6080000400543213, 0.6210000514984131, 0.6030000448226929, 0.612000048160553, 0.6240000128746033, 0.5920000076293945, 0.6060000061988831, 0.6130000352859497, 0.6100000143051147, 0.6080000400543213, 0.6050000190734863, 0.6170000433921814, 0.6210000514984131, 0.6020000576972961, 0.5950000286102295, 0.6220000386238098, 0.6070000529289246, 0.6150000095367432, 0.6030000448226929, 0.6190000176429749, 0.6130000352859497, 0.6040000319480896, 0.6060000061988831, 0.6000000238418579, 0.6190000176429749, 0.609000027179718, 0.6240000128746033, 0.6210000514984131, 0.6210000514984131, 0.6010000109672546, 0.6230000257492065, 0.6030000448226929, 0.609000027179718, 0.612000048160553, 0.6200000047683716, 0.6020000576972961, 0.6210000514984131, 0.6380000114440918, 0.578000009059906, 0.6050000190734863, 0.6140000224113464, 0.6200000047683716, 0.6150000095367432, 0.6070000529289246, 0.6180000305175781, 0.612000048160553, 0.6110000014305115, 0.6220000386238098, 0.6000000238418579, 0.6080000400543213, 0.6180000305175781, 0.609000027179718, 0.6150000095367432, 0.6040000319480896, 0.6070000529289246, 0.6140000224113464, 0.6140000224113464, 0.6130000352859497, 0.6170000433921814, 0.6330000162124634, 0.6040000319480896, 0.6060000061988831, 0.6050000190734863, 0.6100000143051147, 0.6080000400543213, 0.6190000176429749, 0.6130000352859497, 0.6100000143051147]
# Average val accuracy: 0.6312600272893906 ± 0.006155675797602882
# Average test accuracy: 0.61103002846241 ± 0.009131763274171811
# #Params: 368903

## Citeseer

# Logistic loss

# run.py --dataset=citeseer --runs=100 --optimizer=adam --lr=2e-3 --wd=5e-4 --epochs=500 --loss=logit --model=mlp --n-layers=2 --n-hidden=256 --dropout=0.8 --input-drop=0.8 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='citeseer', dropout=0.8, edge_drop=0.0, epochs=500, gpu=0, input_drop=0.8, labels=False, linear=False, log_every=20, loss='logit', lr=0.002, mask_rate=1.0, model='mlp', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.5700000524520874, 0.578000009059906, 0.5800000429153442, 0.5920000076293945, 0.5940000414848328, 0.5840000510215759, 0.5940000414848328, 0.578000009059906, 0.5960000157356262, 0.5800000429153442, 0.5820000171661377, 0.5960000157356262, 0.5900000333786011, 0.5940000414848328, 0.5980000495910645, 0.6000000238418579, 0.5860000252723694, 0.5860000252723694, 0.5760000348091125, 0.5840000510215759, 0.5860000252723694, 0.5720000267028809, 0.5860000252723694, 0.6040000319480896, 0.5900000333786011, 0.5800000429153442, 0.5740000009536743, 0.5900000333786011, 0.5900000333786011, 0.578000009059906, 0.5760000348091125, 0.5900000333786011, 0.5879999995231628, 0.5879999995231628, 0.5860000252723694, 0.6020000576972961, 0.578000009059906, 0.5900000333786011, 0.5840000510215759, 0.5840000510215759, 0.5800000429153442, 0.5920000076293945, 0.5900000333786011, 0.5900000333786011, 0.5820000171661377, 0.5860000252723694, 0.5680000185966492, 0.5700000524520874, 0.5800000429153442, 0.5840000510215759, 0.5940000414848328, 0.5900000333786011, 0.5860000252723694, 0.5920000076293945, 0.578000009059906, 0.5840000510215759, 0.578000009059906, 0.5720000267028809, 0.5940000414848328, 0.5800000429153442, 0.5920000076293945, 0.5840000510215759, 0.5800000429153442, 0.5900000333786011, 0.578000009059906, 0.5920000076293945, 0.5920000076293945, 0.5920000076293945, 0.5900000333786011, 0.5800000429153442, 0.5640000104904175, 0.6020000576972961, 0.578000009059906, 0.5740000009536743, 0.6020000576972961, 0.5820000171661377, 0.5840000510215759, 0.6020000576972961, 0.6000000238418579, 0.6000000238418579, 0.5879999995231628, 0.5760000348091125, 0.5860000252723694, 0.5960000157356262, 0.5900000333786011, 0.5680000185966492, 0.5879999995231628, 0.5900000333786011, 0.5980000495910645, 0.6000000238418579, 0.5820000171661377, 0.5860000252723694, 0.5920000076293945, 0.5800000429153442, 0.5900000333786011, 0.578000009059906, 0.5840000510215759, 0.5879999995231628, 0.5879999995231628, 0.578000009059906]
# Test Accs: [0.5850000381469727, 0.5870000123977661, 0.593000054359436, 0.5680000185966492, 0.5640000104904175, 0.581000030040741, 0.5750000476837158, 0.5670000314712524, 0.5820000171661377, 0.5800000429153442, 0.562000036239624, 0.5890000462532043, 0.581000030040741, 0.5820000171661377, 0.5820000171661377, 0.5750000476837158, 0.5860000252723694, 0.5990000367164612, 0.562000036239624, 0.5890000462532043, 0.5660000443458557, 0.5680000185966492, 0.5770000219345093, 0.5900000333786011, 0.581000030040741, 0.5790000557899475, 0.5680000185966492, 0.5690000057220459, 0.5640000104904175, 0.5879999995231628, 0.5900000333786011, 0.5750000476837158, 0.5730000138282776, 0.5690000057220459, 0.578000009059906, 0.5830000042915344, 0.5730000138282776, 0.5800000429153442, 0.5890000462532043, 0.5790000557899475, 0.5700000524520874, 0.5730000138282776, 0.5730000138282776, 0.5879999995231628, 0.5600000023841858, 0.5640000104904175, 0.5830000042915344, 0.5580000281333923, 0.5950000286102295, 0.5660000443458557, 0.5730000138282776, 0.5910000205039978, 0.581000030040741, 0.5870000123977661, 0.5610000491142273, 0.5670000314712524, 0.5630000233650208, 0.5840000510215759, 0.581000030040741, 0.5850000381469727, 0.5920000076293945, 0.5950000286102295, 0.5770000219345093, 0.5740000009536743, 0.5730000138282776, 0.578000009059906, 0.5770000219345093, 0.5750000476837158, 0.5740000009536743, 0.5649999976158142, 0.5610000491142273, 0.5920000076293945, 0.5680000185966492, 0.5640000104904175, 0.5790000557899475, 0.5580000281333923, 0.5920000076293945, 0.5730000138282776, 0.6010000109672546, 0.5870000123977661, 0.5840000510215759, 0.5690000057220459, 0.5600000023841858, 0.5720000267028809, 0.597000002861023, 0.5660000443458557, 0.5830000042915344, 0.5740000009536743, 0.5800000429153442, 0.6030000448226929, 0.5710000395774841, 0.5770000219345093, 0.5850000381469727, 0.5820000171661377, 0.5740000009536743, 0.5730000138282776, 0.5850000381469727, 0.5770000219345093, 0.5890000462532043, 0.5640000104904175]
# Average val accuracy: 0.5860800278186798 ± 0.008522537536197172
# Average test accuracy: 0.5775100249052048 ± 0.010450355652656713
# #Params: 949766

# Savage loss

# run.py --dataset=citeseer --runs=100 --optimizer=adam --lr=2e-3 --wd=5e-4 --epochs=500 --loss=loge --model=mlp --n-layers=2 --n-hidden=256 --dropout=0.8 --input-drop=0.8 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='citeseer', dropout=0.8, edge_drop=0.0, epochs=500, gpu=0, input_drop=0.8, labels=False, linear=False, log_every=20, loss='loge', lr=0.002, mask_rate=1.0, model='mlp', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.6020000576972961, 0.6020000576972961, 0.6020000576972961, 0.6200000047683716, 0.5980000495910645, 0.6000000238418579, 0.5980000495910645, 0.6000000238418579, 0.6020000576972961, 0.5960000157356262, 0.5960000157356262, 0.5980000495910645, 0.5980000495910645, 0.6040000319480896, 0.612000048160553, 0.6060000061988831, 0.5960000157356262, 0.6060000061988831, 0.5860000252723694, 0.6000000238418579, 0.5980000495910645, 0.5960000157356262, 0.5940000414848328, 0.6040000319480896, 0.5940000414848328, 0.6000000238418579, 0.5940000414848328, 0.5980000495910645, 0.5940000414848328, 0.5980000495910645, 0.6000000238418579, 0.5980000495910645, 0.6060000061988831, 0.6000000238418579, 0.5940000414848328, 0.6020000576972961, 0.5960000157356262, 0.6100000143051147, 0.6080000400543213, 0.6040000319480896, 0.5960000157356262, 0.6080000400543213, 0.5960000157356262, 0.6000000238418579, 0.5800000429153442, 0.5980000495910645, 0.6040000319480896, 0.6000000238418579, 0.6060000061988831, 0.6040000319480896, 0.6140000224113464, 0.6100000143051147, 0.6040000319480896, 0.6080000400543213, 0.6100000143051147, 0.6020000576972961, 0.6000000238418579, 0.6020000576972961, 0.6000000238418579, 0.5940000414848328, 0.612000048160553, 0.6080000400543213, 0.5960000157356262, 0.6040000319480896, 0.6040000319480896, 0.6100000143051147, 0.6060000061988831, 0.5980000495910645, 0.6060000061988831, 0.5940000414848328, 0.6040000319480896, 0.6060000061988831, 0.6060000061988831, 0.5940000414848328, 0.6020000576972961, 0.6040000319480896, 0.5960000157356262, 0.6080000400543213, 0.6100000143051147, 0.6020000576972961, 0.6020000576972961, 0.6020000576972961, 0.6080000400543213, 0.6020000576972961, 0.5980000495910645, 0.5940000414848328, 0.6020000576972961, 0.6020000576972961, 0.5960000157356262, 0.6080000400543213, 0.5879999995231628, 0.6040000319480896, 0.6080000400543213, 0.6040000319480896, 0.612000048160553, 0.5840000510215759, 0.6020000576972961, 0.5920000076293945, 0.6020000576972961, 0.5980000495910645]
# Test Accs: [0.5920000076293945, 0.597000002861023, 0.5960000157356262, 0.5730000138282776, 0.5530000329017639, 0.5870000123977661, 0.5770000219345093, 0.581000030040741, 0.5960000157356262, 0.5879999995231628, 0.5770000219345093, 0.5950000286102295, 0.5950000286102295, 0.5920000076293945, 0.5980000495910645, 0.6100000143051147, 0.5940000414848328, 0.5860000252723694, 0.597000002861023, 0.5920000076293945, 0.5960000157356262, 0.5980000495910645, 0.5910000205039978, 0.5990000367164612, 0.5850000381469727, 0.5730000138282776, 0.5820000171661377, 0.5770000219345093, 0.5840000510215759, 0.6010000109672546, 0.5890000462532043, 0.597000002861023, 0.5900000333786011, 0.5900000333786011, 0.5860000252723694, 0.5870000123977661, 0.5760000348091125, 0.5960000157356262, 0.5950000286102295, 0.593000054359436, 0.5879999995231628, 0.5950000286102295, 0.5890000462532043, 0.5760000348091125, 0.593000054359436, 0.5770000219345093, 0.581000030040741, 0.5840000510215759, 0.6070000529289246, 0.5950000286102295, 0.6070000529289246, 0.5910000205039978, 0.5990000367164612, 0.6080000400543213, 0.5860000252723694, 0.5890000462532043, 0.5990000367164612, 0.5820000171661377, 0.5910000205039978, 0.5760000348091125, 0.609000027179718, 0.5940000414848328, 0.593000054359436, 0.5980000495910645, 0.581000030040741, 0.578000009059906, 0.5980000495910645, 0.5879999995231628, 0.5980000495910645, 0.597000002861023, 0.593000054359436, 0.5850000381469727, 0.5860000252723694, 0.5980000495910645, 0.6050000190734863, 0.6030000448226929, 0.593000054359436, 0.5940000414848328, 0.5980000495910645, 0.5960000157356262, 0.5890000462532043, 0.5860000252723694, 0.5980000495910645, 0.6100000143051147, 0.5770000219345093, 0.5860000252723694, 0.6040000319480896, 0.6020000576972961, 0.5820000171661377, 0.6030000448226929, 0.5870000123977661, 0.6070000529289246, 0.5820000171661377, 0.5760000348091125, 0.6000000238418579, 0.5860000252723694, 0.5840000510215759, 0.5879999995231628, 0.5730000138282776, 0.593000054359436]
# Average val accuracy: 0.6012400341033936 ± 0.006342111204686473
# Average test accuracy: 0.5907200294733047 ± 0.009809264280487425
# #Params: 949766

# Loge loss

# run.py --dataset=citeseer --runs=100 --optimizer=adam --lr=2e-3 --wd=5e-4 --epochs=500 --loss=savage --model=mlp --n-layers=2 --n-hidden=256 --dropout=0.8 --input-drop=0.8 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='citeseer', dropout=0.8, edge_drop=0.0, epochs=500, gpu=0, input_drop=0.8, labels=False, linear=False, log_every=20, loss='savage', lr=0.002, mask_rate=1.0, model='mlp', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.6020000576972961, 0.6020000576972961, 0.6060000061988831, 0.6100000143051147, 0.5940000414848328, 0.5980000495910645, 0.6020000576972961, 0.6040000319480896, 0.6040000319480896, 0.5960000157356262, 0.5900000333786011, 0.6060000061988831, 0.6060000061988831, 0.6000000238418579, 0.6180000305175781, 0.6060000061988831, 0.6040000319480896, 0.6000000238418579, 0.5900000333786011, 0.5980000495910645, 0.5960000157356262, 0.5960000157356262, 0.5980000495910645, 0.6000000238418579, 0.5900000333786011, 0.5980000495910645, 0.5940000414848328, 0.5900000333786011, 0.6040000319480896, 0.5960000157356262, 0.5960000157356262, 0.5900000333786011, 0.6060000061988831, 0.6060000061988831, 0.6000000238418579, 0.6100000143051147, 0.6060000061988831, 0.6020000576972961, 0.6080000400543213, 0.5960000157356262, 0.6000000238418579, 0.6100000143051147, 0.5940000414848328, 0.5980000495910645, 0.5920000076293945, 0.6020000576972961, 0.6040000319480896, 0.6040000319480896, 0.5960000157356262, 0.612000048160553, 0.6080000400543213, 0.6080000400543213, 0.6040000319480896, 0.6100000143051147, 0.5879999995231628, 0.5879999995231628, 0.5940000414848328, 0.5920000076293945, 0.5940000414848328, 0.5960000157356262, 0.6020000576972961, 0.6080000400543213, 0.5879999995231628, 0.6040000319480896, 0.6020000576972961, 0.6140000224113464, 0.5960000157356262, 0.5980000495910645, 0.6020000576972961, 0.6020000576972961, 0.6060000061988831, 0.5960000157356262, 0.6020000576972961, 0.5879999995231628, 0.6020000576972961, 0.5960000157356262, 0.6040000319480896, 0.6000000238418579, 0.6000000238418579, 0.5980000495910645, 0.5980000495910645, 0.6040000319480896, 0.6040000319480896, 0.6040000319480896, 0.5920000076293945, 0.5900000333786011, 0.6000000238418579, 0.6060000061988831, 0.6020000576972961, 0.5980000495910645, 0.5900000333786011, 0.612000048160553, 0.6020000576972961, 0.5940000414848328, 0.6020000576972961, 0.5879999995231628, 0.6000000238418579, 0.5940000414848328, 0.6040000319480896, 0.6000000238418579]
# Test Accs: [0.5910000205039978, 0.6050000190734863, 0.593000054359436, 0.5960000157356262, 0.6030000448226929, 0.5920000076293945, 0.597000002861023, 0.6010000109672546, 0.5920000076293945, 0.5890000462532043, 0.5860000252723694, 0.6040000319480896, 0.6150000095367432, 0.5900000333786011, 0.6050000190734863, 0.6060000061988831, 0.6000000238418579, 0.593000054359436, 0.6000000238418579, 0.597000002861023, 0.5870000123977661, 0.5850000381469727, 0.609000027179718, 0.5980000495910645, 0.593000054359436, 0.5890000462532043, 0.5870000123977661, 0.5870000123977661, 0.5980000495910645, 0.593000054359436, 0.5990000367164612, 0.5910000205039978, 0.5950000286102295, 0.5840000510215759, 0.6060000061988831, 0.5960000157356262, 0.5879999995231628, 0.5950000286102295, 0.612000048160553, 0.5910000205039978, 0.5870000123977661, 0.6000000238418579, 0.597000002861023, 0.5860000252723694, 0.5820000171661377, 0.5890000462532043, 0.6080000400543213, 0.5890000462532043, 0.6020000576972961, 0.5870000123977661, 0.6030000448226929, 0.6020000576972961, 0.6040000319480896, 0.6140000224113464, 0.5900000333786011, 0.6100000143051147, 0.5960000157356262, 0.5960000157356262, 0.597000002861023, 0.593000054359436, 0.6050000190734863, 0.5990000367164612, 0.6000000238418579, 0.5910000205039978, 0.5740000009536743, 0.6040000319480896, 0.6020000576972961, 0.5910000205039978, 0.6070000529289246, 0.5920000076293945, 0.5950000286102295, 0.5900000333786011, 0.5790000557899475, 0.5940000414848328, 0.6050000190734863, 0.5980000495910645, 0.6010000109672546, 0.5800000429153442, 0.5760000348091125, 0.6080000400543213, 0.5990000367164612, 0.6000000238418579, 0.5980000495910645, 0.6130000352859497, 0.6070000529289246, 0.5890000462532043, 0.5900000333786011, 0.6040000319480896, 0.5850000381469727, 0.6150000095367432, 0.581000030040741, 0.597000002861023, 0.6110000014305115, 0.5790000557899475, 0.6040000319480896, 0.5770000219345093, 0.6010000109672546, 0.6100000143051147, 0.5890000462532043, 0.5890000462532043]
# Average val accuracy: 0.6000400304794311 ± 0.006424829249352705
# Average test accuracy: 0.5959900289773941 ± 0.009179862339404387
# #Params: 949766

## Pubmed

# Logistic loss

# run.py --dataset=pubmed --runs=100 --optimizer=adam --lr=1e-2 --wd=5e-4 --epochs=500 --loss=logit --model=mlp --n-layers=2 --n-hidden=256 --dropout=0.5 --input-drop=0.1 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='pubmed', dropout=0.5, edge_drop=0.0, epochs=500, gpu=0, input_drop=0.1, labels=False, linear=False, log_every=20, loss='logit', lr=0.01, mask_rate=1.0, model='mlp', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.7400000095367432, 0.7400000095367432, 0.7360000610351562, 0.7420000433921814, 0.734000027179718, 0.7420000433921814, 0.7360000610351562, 0.7380000352859497, 0.7320000529289246, 0.7320000529289246, 0.7360000610351562, 0.7400000095367432, 0.7360000610351562, 0.734000027179718, 0.7360000610351562, 0.7360000610351562, 0.7320000529289246, 0.734000027179718, 0.7400000095367432, 0.7400000095367432, 0.7360000610351562, 0.7400000095367432, 0.734000027179718, 0.7360000610351562, 0.734000027179718, 0.7400000095367432, 0.7380000352859497, 0.7360000610351562, 0.7360000610351562, 0.7360000610351562, 0.7360000610351562, 0.734000027179718, 0.7380000352859497, 0.7380000352859497, 0.7420000433921814, 0.7400000095367432, 0.7360000610351562, 0.7360000610351562, 0.7360000610351562, 0.7420000433921814, 0.7420000433921814, 0.7380000352859497, 0.7380000352859497, 0.7400000095367432, 0.734000027179718, 0.7380000352859497, 0.7380000352859497, 0.7360000610351562, 0.7380000352859497, 0.7400000095367432, 0.7380000352859497, 0.734000027179718, 0.7420000433921814, 0.7380000352859497, 0.7380000352859497, 0.7360000610351562, 0.7360000610351562, 0.7320000529289246, 0.7380000352859497, 0.7360000610351562, 0.734000027179718, 0.7360000610351562, 0.7380000352859497, 0.7320000529289246, 0.7380000352859497, 0.734000027179718, 0.7400000095367432, 0.734000027179718, 0.7420000433921814, 0.7400000095367432, 0.7360000610351562, 0.7380000352859497, 0.7380000352859497, 0.7380000352859497, 0.7400000095367432, 0.734000027179718, 0.7360000610351562, 0.7320000529289246, 0.734000027179718, 0.7360000610351562, 0.7460000514984131, 0.7300000190734863, 0.7380000352859497, 0.734000027179718, 0.7360000610351562, 0.734000027179718, 0.7360000610351562, 0.7360000610351562, 0.7400000095367432, 0.7380000352859497, 0.7380000352859497, 0.7380000352859497, 0.734000027179718, 0.7360000610351562, 0.7320000529289246, 0.7380000352859497, 0.734000027179718, 0.734000027179718, 0.7380000352859497, 0.734000027179718]
# Test Accs: [0.7240000367164612, 0.734000027179718, 0.7300000190734863, 0.7320000529289246, 0.7350000143051147, 0.737000048160553, 0.7350000143051147, 0.7230000495910645, 0.7360000610351562, 0.7190000414848328, 0.7240000367164612, 0.7260000109672546, 0.7390000224113464, 0.7410000562667847, 0.7280000448226929, 0.7290000319480896, 0.7210000157356262, 0.7350000143051147, 0.7450000643730164, 0.7430000305175781, 0.7380000352859497, 0.7360000610351562, 0.7320000529289246, 0.7200000286102295, 0.7430000305175781, 0.7360000610351562, 0.7300000190734863, 0.7310000061988831, 0.7350000143051147, 0.7300000190734863, 0.7380000352859497, 0.7330000400543213, 0.7350000143051147, 0.7350000143051147, 0.737000048160553, 0.7300000190734863, 0.7390000224113464, 0.7410000562667847, 0.7270000576972961, 0.7330000400543213, 0.7300000190734863, 0.7270000576972961, 0.7330000400543213, 0.7460000514984131, 0.7240000367164612, 0.7200000286102295, 0.7290000319480896, 0.7330000400543213, 0.7350000143051147, 0.7150000333786011, 0.7360000610351562, 0.7280000448226929, 0.7420000433921814, 0.7300000190734863, 0.7300000190734863, 0.7300000190734863, 0.7220000624656677, 0.7260000109672546, 0.7320000529289246, 0.7220000624656677, 0.7330000400543213, 0.7220000624656677, 0.7270000576972961, 0.7440000176429749, 0.7330000400543213, 0.7310000061988831, 0.737000048160553, 0.7230000495910645, 0.7300000190734863, 0.7320000529289246, 0.7400000095367432, 0.7210000157356262, 0.7330000400543213, 0.7410000562667847, 0.737000048160553, 0.7350000143051147, 0.7350000143051147, 0.7260000109672546, 0.7320000529289246, 0.7240000367164612, 0.7310000061988831, 0.7300000190734863, 0.7230000495910645, 0.7350000143051147, 0.7320000529289246, 0.7290000319480896, 0.7450000643730164, 0.7260000109672546, 0.7170000076293945, 0.7270000576972961, 0.7350000143051147, 0.7390000224113464, 0.7360000610351562, 0.7170000076293945, 0.7250000238418579, 0.7330000400543213, 0.7270000576972961, 0.7280000448226929, 0.7400000095367432, 0.7380000352859497]
# Average val accuracy: 0.736840038895607 ± 0.0028730425292467277
# Average test accuracy: 0.7315400344133377 ± 0.0067740995693401935
# #Params: 129027

# Savage loss

# run.py --dataset=pubmed --runs=100 --optimizer=adam --lr=5e-3 --wd=5e-4 --epochs=500 --loss=loge --model=mlp --n-layers=2 --n-hidden=256 --dropout=0.5 --input-drop=0.1 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='pubmed', dropout=0.5, edge_drop=0.0, epochs=500, gpu=0, input_drop=0.1, labels=False, linear=False, log_every=20, loss='loge', lr=0.005, mask_rate=1.0, model='mlp', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.7320000529289246, 0.7300000190734863, 0.7360000610351562, 0.7300000190734863, 0.7360000610351562, 0.734000027179718, 0.734000027179718, 0.734000027179718, 0.734000027179718, 0.7300000190734863, 0.7360000610351562, 0.7300000190734863, 0.7360000610351562, 0.7380000352859497, 0.734000027179718, 0.7360000610351562, 0.734000027179718, 0.734000027179718, 0.7380000352859497, 0.7360000610351562, 0.7360000610351562, 0.734000027179718, 0.734000027179718, 0.7300000190734863, 0.7300000190734863, 0.7320000529289246, 0.7320000529289246, 0.7360000610351562, 0.734000027179718, 0.7300000190734863, 0.7320000529289246, 0.7380000352859497, 0.7320000529289246, 0.7420000433921814, 0.734000027179718, 0.734000027179718, 0.7320000529289246, 0.7360000610351562, 0.734000027179718, 0.7320000529289246, 0.7320000529289246, 0.7380000352859497, 0.734000027179718, 0.734000027179718, 0.7360000610351562, 0.7360000610351562, 0.7380000352859497, 0.7320000529289246, 0.734000027179718, 0.7380000352859497, 0.7320000529289246, 0.7320000529289246, 0.734000027179718, 0.7300000190734863, 0.7360000610351562, 0.734000027179718, 0.7400000095367432, 0.7360000610351562, 0.734000027179718, 0.7360000610351562, 0.7360000610351562, 0.734000027179718, 0.734000027179718, 0.7280000448226929, 0.7320000529289246, 0.734000027179718, 0.7300000190734863, 0.734000027179718, 0.734000027179718, 0.7400000095367432, 0.7320000529289246, 0.7400000095367432, 0.7360000610351562, 0.7300000190734863, 0.734000027179718, 0.734000027179718, 0.7360000610351562, 0.734000027179718, 0.7360000610351562, 0.7360000610351562, 0.7320000529289246, 0.7300000190734863, 0.7300000190734863, 0.7360000610351562, 0.7300000190734863, 0.7320000529289246, 0.7360000610351562, 0.7320000529289246, 0.7320000529289246, 0.734000027179718, 0.7380000352859497, 0.7380000352859497, 0.7320000529289246, 0.7320000529289246, 0.734000027179718, 0.7300000190734863, 0.7360000610351562, 0.7320000529289246, 0.7320000529289246, 0.7320000529289246]
# Test Accs: [0.7220000624656677, 0.7260000109672546, 0.7330000400543213, 0.7220000624656677, 0.7270000576972961, 0.7220000624656677, 0.7310000061988831, 0.7360000610351562, 0.7310000061988831, 0.7260000109672546, 0.7260000109672546, 0.7330000400543213, 0.7230000495910645, 0.737000048160553, 0.7330000400543213, 0.734000027179718, 0.7280000448226929, 0.7170000076293945, 0.7300000190734863, 0.7350000143051147, 0.7310000061988831, 0.7300000190734863, 0.7320000529289246, 0.7310000061988831, 0.7360000610351562, 0.734000027179718, 0.7300000190734863, 0.7320000529289246, 0.7290000319480896, 0.7300000190734863, 0.718000054359436, 0.7250000238418579, 0.7250000238418579, 0.7270000576972961, 0.7390000224113464, 0.7320000529289246, 0.7360000610351562, 0.7330000400543213, 0.7290000319480896, 0.7290000319480896, 0.7330000400543213, 0.734000027179718, 0.7240000367164612, 0.7230000495910645, 0.7360000610351562, 0.7240000367164612, 0.7320000529289246, 0.7230000495910645, 0.734000027179718, 0.7220000624656677, 0.7310000061988831, 0.7250000238418579, 0.7300000190734863, 0.7160000205039978, 0.734000027179718, 0.7270000576972961, 0.7240000367164612, 0.7320000529289246, 0.7310000061988831, 0.7260000109672546, 0.734000027179718, 0.7380000352859497, 0.7170000076293945, 0.7320000529289246, 0.7130000591278076, 0.7220000624656677, 0.7310000061988831, 0.7260000109672546, 0.734000027179718, 0.7330000400543213, 0.7260000109672546, 0.7280000448226929, 0.7230000495910645, 0.7310000061988831, 0.7260000109672546, 0.7230000495910645, 0.7350000143051147, 0.7240000367164612, 0.7280000448226929, 0.7350000143051147, 0.7330000400543213, 0.7200000286102295, 0.7190000414848328, 0.7350000143051147, 0.7310000061988831, 0.7300000190734863, 0.737000048160553, 0.734000027179718, 0.7280000448226929, 0.7300000190734863, 0.7280000448226929, 0.7320000529289246, 0.7210000157356262, 0.7320000529289246, 0.7350000143051147, 0.7250000238418579, 0.7230000495910645, 0.734000027179718, 0.7270000576972961, 0.7220000624656677]
# Average val accuracy: 0.733960039615631 ± 0.0027126401384930934
# Average test accuracy: 0.7287600338459015 ± 0.00544815361418519
# #Params: 129027

# Loge loss

# run.py --dataset=pubmed --runs=100 --optimizer=adam --lr=1e-2 --wd=5e-4 --epochs=500 --loss=savage --model=mlp --n-layers=2 --n-hidden=256 --dropout=0.5 --input-drop=0.1 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='pubmed', dropout=0.5, edge_drop=0.0, epochs=500, gpu=0, input_drop=0.1, labels=False, linear=False, log_every=20, loss='savage', lr=0.01, mask_rate=1.0, model='mlp', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.7360000610351562, 0.7460000514984131, 0.7360000610351562, 0.7360000610351562, 0.7380000352859497, 0.7380000352859497, 0.7400000095367432, 0.7380000352859497, 0.7380000352859497, 0.7400000095367432, 0.7400000095367432, 0.7400000095367432, 0.7380000352859497, 0.7400000095367432, 0.7380000352859497, 0.7420000433921814, 0.7380000352859497, 0.7380000352859497, 0.7420000433921814, 0.7380000352859497, 0.7400000095367432, 0.7460000514984131, 0.7360000610351562, 0.7380000352859497, 0.7400000095367432, 0.7400000095367432, 0.7400000095367432, 0.734000027179718, 0.7380000352859497, 0.7380000352859497, 0.7400000095367432, 0.7440000176429749, 0.7420000433921814, 0.7380000352859497, 0.7400000095367432, 0.7400000095367432, 0.7360000610351562, 0.7380000352859497, 0.7360000610351562, 0.7360000610351562, 0.7440000176429749, 0.7400000095367432, 0.7400000095367432, 0.7420000433921814, 0.7420000433921814, 0.7400000095367432, 0.7380000352859497, 0.7440000176429749, 0.7380000352859497, 0.7380000352859497, 0.7440000176429749, 0.7320000529289246, 0.7420000433921814, 0.7460000514984131, 0.7380000352859497, 0.7420000433921814, 0.7400000095367432, 0.734000027179718, 0.7380000352859497, 0.7380000352859497, 0.7400000095367432, 0.7400000095367432, 0.7360000610351562, 0.7380000352859497, 0.7380000352859497, 0.7360000610351562, 0.7380000352859497, 0.7400000095367432, 0.7360000610351562, 0.7420000433921814, 0.734000027179718, 0.7400000095367432, 0.7360000610351562, 0.7400000095367432, 0.7400000095367432, 0.7400000095367432, 0.7360000610351562, 0.734000027179718, 0.7360000610351562, 0.734000027179718, 0.7400000095367432, 0.7360000610351562, 0.7360000610351562, 0.7360000610351562, 0.7400000095367432, 0.7380000352859497, 0.7400000095367432, 0.7360000610351562, 0.7420000433921814, 0.7360000610351562, 0.7360000610351562, 0.7380000352859497, 0.7380000352859497, 0.7380000352859497, 0.734000027179718, 0.7420000433921814, 0.7380000352859497, 0.7360000610351562, 0.734000027179718, 0.734000027179718]
# Test Accs: [0.7230000495910645, 0.7450000643730164, 0.7310000061988831, 0.7290000319480896, 0.7380000352859497, 0.737000048160553, 0.7280000448226929, 0.7260000109672546, 0.7290000319480896, 0.7430000305175781, 0.7230000495910645, 0.7320000529289246, 0.7390000224113464, 0.7260000109672546, 0.7290000319480896, 0.7380000352859497, 0.7300000190734863, 0.737000048160553, 0.7330000400543213, 0.7410000562667847, 0.7300000190734863, 0.7330000400543213, 0.7310000061988831, 0.7480000257492065, 0.7300000190734863, 0.737000048160553, 0.7470000386238098, 0.7250000238418579, 0.7330000400543213, 0.7270000576972961, 0.7380000352859497, 0.7270000576972961, 0.7290000319480896, 0.7460000514984131, 0.7410000562667847, 0.7360000610351562, 0.7360000610351562, 0.7380000352859497, 0.737000048160553, 0.7320000529289246, 0.7240000367164612, 0.7350000143051147, 0.7350000143051147, 0.7310000061988831, 0.7310000061988831, 0.7380000352859497, 0.7240000367164612, 0.7350000143051147, 0.7350000143051147, 0.7390000224113464, 0.7420000433921814, 0.7300000190734863, 0.7390000224113464, 0.737000048160553, 0.737000048160553, 0.7270000576972961, 0.7330000400543213, 0.7350000143051147, 0.7350000143051147, 0.7310000061988831, 0.7300000190734863, 0.7380000352859497, 0.7330000400543213, 0.7300000190734863, 0.7250000238418579, 0.734000027179718, 0.734000027179718, 0.737000048160553, 0.7310000061988831, 0.7350000143051147, 0.7460000514984131, 0.7350000143051147, 0.7430000305175781, 0.7410000562667847, 0.734000027179718, 0.7430000305175781, 0.7290000319480896, 0.7450000643730164, 0.7260000109672546, 0.7250000238418579, 0.7240000367164612, 0.7400000095367432, 0.7320000529289246, 0.7300000190734863, 0.7420000433921814, 0.7360000610351562, 0.7460000514984131, 0.7250000238418579, 0.7350000143051147, 0.7320000529289246, 0.7300000190734863, 0.7300000190734863, 0.7270000576972961, 0.7360000610351562, 0.7260000109672546, 0.7280000448226929, 0.734000027179718, 0.7490000128746033, 0.7290000319480896, 0.737000048160553]
# Average val accuracy: 0.7386200338602066 ± 0.0028382324180536707
# Average test accuracy: 0.7339300334453582 ± 0.006200415046527465
# #Params: 129027

## ogbn-arxiv

# Logistic loss

# run.py --gpu=2 --dataset=ogbn-arxiv --epochs=1000 --loss=logit --model=mlp --n-hidden=512 --input-drop=0.1 --dropout=0.5 --lr=0.005
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='ogbn-arxiv', dropout=0.5, edge_drop=0.0, epochs=1000, gpu=2, input_drop=0.1, labels=False, linear=False, log_every=20, loss='logit', lr=0.005, mask_rate=0.5, model='mlp', n_heads=3, n_hidden=512, n_label_iters=0, n_layers=3, n_prop=7, non_interactive_attn=False, norm='batch', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=10, save_pred=False, seed=0, split='std', tune='', wd=0)
# Runned 10 times
# Val Accs: [0.5842477935501191, 0.5817644887412329, 0.5832410483573274, 0.5830396993187691, 0.5837444209537233, 0.583543071915165, 0.5832074901842343, 0.582368535856908, 0.5849525151850733, 0.5833417228766066]
# Test Accs: [0.5604592309116722, 0.5608501532827191, 0.563010513754295, 0.5613233751003025, 0.562640166244882, 0.5632162623706356, 0.560047733678991, 0.559697961031212, 0.5637717836347551, 0.5628870645844907]
# Average val accuracy: 0.5833450786939159 ± 0.0008468968641504751
# Average test accuracy: 0.5617904244593956 ± 0.0014022051237183562
# #Params: 351272

# Savage loss

# run.py --gpu=4 --dataset=ogbn-arxiv --epochs=1000 --loss=loge --model=mlp --n-hidden=512 --input-drop=0.1 --dropout=0.5 --lr=0.005
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='ogbn-arxiv', dropout=0.5, edge_drop=0.0, epochs=1000, gpu=4, input_drop=0.1, labels=False, linear=False, log_every=20, loss='loge', lr=0.005, mask_rate=0.5, model='mlp', n_heads=3, n_hidden=512, n_label_iters=0, n_layers=3, n_prop=7, non_interactive_attn=False, norm='batch', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=10, save_pred=False, seed=0, split='std', tune='', wd=0)
# Runned 10 times
# Val Accs: [0.5857243531662136, 0.5874693781670526, 0.5856236786469344, 0.5848853988388872, 0.584717607973422, 0.5887110305714957, 0.5880398671096345, 0.5879056344172623, 0.5852545387429109, 0.5827041175878385]
# Test Accs: [0.5661996173075736, 0.5676604324835915, 0.5696561940620949, 0.5650679999177005, 0.5660350184145012, 0.5690183733514392, 0.5670020369113018, 0.5687097504269284, 0.5655617965969179, 0.5666111145402547]
# Average val accuracy: 0.5861035605221653 ± 0.001780927535124825
# Average test accuracy: 0.5671522334012303 ± 0.0014758372774703156
# #Params: 351272

# Loge loss

# run.py --gpu=3 --dataset=ogbn-arxiv --epochs=1000 --loss=savage --model=mlp --n-hidden=512 --input-drop=0.1 --dropout=0.5 --lr=0.005
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='ogbn-arxiv', dropout=0.5, edge_drop=0.0, epochs=1000, gpu=3, input_drop=0.1, labels=False, linear=False, log_every=20, loss='savage', lr=0.005, mask_rate=0.5, model='mlp', n_heads=3, n_hidden=512, n_label_iters=0, n_layers=3, n_prop=7, non_interactive_attn=False, norm='batch', norm_adj='default', optimizer='adam', plot=False, residual=False, runs=10, save_pred=False, seed=0, split='std', tune='', wd=0)
# Runned 10 times
# Val Accs: [0.5343803483338367, 0.5340112084298131, 0.534615255545488, 0.5319977180442297, 0.5335078358334172, 0.5320312762173227, 0.5329037887177422, 0.5329373468908353, 0.5374341420853048, 0.5346823718916742]
# Test Accs: [0.5180544410838838, 0.5236713783099809, 0.5217579161780137, 0.5220048145176224, 0.5190008847190503, 0.5184659383165648, 0.5193095076435611, 0.5189803098574162, 0.5165524761845977, 0.5190831841655865]
# Average val accuracy: 0.5338501291989665 ± 0.001517096677746897
# Average test accuracy: 0.5196880850976278 ± 0.0020247211085293263
# #Params: 351272

### GCN

## Cora

# Logistic loss

# run.py --gpu=1 --dataset=cora --runs=100 --optimizer=adam --lr=1e-1 --wd=5e-4 --epochs=500 --loss=logit --model=gcn --n-layers=2 --n-hidden=32 --dropout=0.8 --input-drop=0.8 --mask-rate=1.0 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='cora', dropout=0.8, edge_drop=0.0, epochs=500, gpu=1, input_drop=0.8, labels=False, linear=False, log_every=20, loss='logit', lr=0.1, mask_rate=1.0, model='gcn', n_heads=3, n_hidden=32, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.8280000686645508, 0.8240000605583191, 0.8320000171661377, 0.8320000171661377, 0.8260000348091125, 0.8200000524520874, 0.8240000605583191, 0.8260000348091125, 0.8360000252723694, 0.8220000267028809, 0.8220000267028809, 0.8200000524520874, 0.8280000686645508, 0.8220000267028809, 0.8220000267028809, 0.8300000429153442, 0.8200000524520874, 0.8280000686645508, 0.8240000605583191, 0.8180000185966492, 0.8280000686645508, 0.8300000429153442, 0.8260000348091125, 0.8220000267028809, 0.8280000686645508, 0.8180000185966492, 0.8160000443458557, 0.8280000686645508, 0.8260000348091125, 0.8240000605583191, 0.8280000686645508, 0.8220000267028809, 0.8220000267028809, 0.8220000267028809, 0.8300000429153442, 0.8280000686645508, 0.8280000686645508, 0.8220000267028809, 0.8300000429153442, 0.8200000524520874, 0.8300000429153442, 0.8300000429153442, 0.8280000686645508, 0.8360000252723694, 0.8280000686645508, 0.8280000686645508, 0.8240000605583191, 0.8260000348091125, 0.8220000267028809, 0.8240000605583191, 0.8380000591278076, 0.8200000524520874, 0.8240000605583191, 0.8220000267028809, 0.8200000524520874, 0.8240000605583191, 0.8240000605583191, 0.8200000524520874, 0.8280000686645508, 0.8220000267028809, 0.8280000686645508, 0.8200000524520874, 0.8180000185966492, 0.8160000443458557, 0.8260000348091125, 0.8240000605583191, 0.8300000429153442, 0.8260000348091125, 0.8280000686645508, 0.8220000267028809, 0.8280000686645508, 0.8200000524520874, 0.8220000267028809, 0.8240000605583191, 0.8200000524520874, 0.8320000171661377, 0.8220000267028809, 0.8200000524520874, 0.8260000348091125, 0.8260000348091125, 0.8280000686645508, 0.8300000429153442, 0.8240000605583191, 0.8300000429153442, 0.8220000267028809, 0.8220000267028809, 0.8140000104904175, 0.8280000686645508, 0.8300000429153442, 0.8260000348091125, 0.8240000605583191, 0.8240000605583191, 0.8320000171661377, 0.8220000267028809, 0.8200000524520874, 0.8240000605583191, 0.8240000605583191, 0.8220000267028809, 0.8320000171661377, 0.8200000524520874]
# Test Accs: [0.8270000219345093, 0.8360000252723694, 0.8250000476837158, 0.8140000104904175, 0.8250000476837158, 0.8300000429153442, 0.8230000138282776, 0.815000057220459, 0.8320000171661377, 0.8320000171661377, 0.8160000443458557, 0.8040000200271606, 0.812000036239624, 0.8370000123977661, 0.8370000123977661, 0.8100000619888306, 0.8260000348091125, 0.8220000267028809, 0.8270000219345093, 0.8260000348091125, 0.8190000653266907, 0.8210000395774841, 0.8340000510215759, 0.8300000429153442, 0.8200000524520874, 0.7940000295639038, 0.8060000538825989, 0.8280000686645508, 0.8390000462532043, 0.8230000138282776, 0.8170000314712524, 0.8210000395774841, 0.8180000185966492, 0.8210000395774841, 0.812000036239624, 0.8170000314712524, 0.8240000605583191, 0.8270000219345093, 0.8180000185966492, 0.8130000233650208, 0.8270000219345093, 0.8210000395774841, 0.8180000185966492, 0.8080000281333923, 0.8230000138282776, 0.8320000171661377, 0.8360000252723694, 0.8110000491142273, 0.8090000152587891, 0.8300000429153442, 0.8330000638961792, 0.8210000395774841, 0.8250000476837158, 0.8250000476837158, 0.8210000395774841, 0.8280000686645508, 0.8410000205039978, 0.8300000429153442, 0.8320000171661377, 0.8180000185966492, 0.8260000348091125, 0.8160000443458557, 0.8230000138282776, 0.8140000104904175, 0.8290000557899475, 0.8170000314712524, 0.8330000638961792, 0.8200000524520874, 0.8250000476837158, 0.8170000314712524, 0.8130000233650208, 0.8230000138282776, 0.8140000104904175, 0.8220000267028809, 0.8160000443458557, 0.8220000267028809, 0.8300000429153442, 0.8130000233650208, 0.8280000686645508, 0.8160000443458557, 0.8090000152587891, 0.8290000557899475, 0.8200000524520874, 0.8200000524520874, 0.8220000267028809, 0.831000030040741, 0.815000057220459, 0.8350000381469727, 0.8240000605583191, 0.8250000476837158, 0.8170000314712524, 0.8260000348091125, 0.8210000395774841, 0.8410000205039978, 0.8290000557899475, 0.8140000104904175, 0.8270000219345093, 0.8260000348091125, 0.8220000267028809, 0.8230000138282776]
# Average val accuracy: 0.8249600452184677 ± 0.00449426546516535
# Average test accuracy: 0.822600035071373 ± 0.008382124915616535
# #Params: 46119

# Savage loss

# run.py --gpu=2 --dataset=cora --runs=100 --optimizer=adam --lr=1e-1 --wd=5e-4 --epochs=500 --loss=loge --model=gcn --n-layers=2 --n-hidden=32 --dropout=0.8 --input-drop=0.8 --mask-rate=1.0 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='cora', dropout=0.8, edge_drop=0.0, epochs=500, gpu=2, input_drop=0.8, labels=False, linear=False, log_every=20, loss='loge', lr=0.1, mask_rate=1.0, model='gcn', n_heads=3, n_hidden=32, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.8280000686645508, 0.8240000605583191, 0.8240000605583191, 0.8240000605583191, 0.8280000686645508, 0.8240000605583191, 0.8320000171661377, 0.8240000605583191, 0.8260000348091125, 0.8300000429153442, 0.8220000267028809, 0.8200000524520874, 0.8260000348091125, 0.8300000429153442, 0.8260000348091125, 0.8220000267028809, 0.8240000605583191, 0.8320000171661377, 0.8280000686645508, 0.8240000605583191, 0.8200000524520874, 0.8280000686645508, 0.8280000686645508, 0.8200000524520874, 0.8280000686645508, 0.8300000429153442, 0.8240000605583191, 0.8280000686645508, 0.8340000510215759, 0.8240000605583191, 0.8240000605583191, 0.8280000686645508, 0.8240000605583191, 0.8200000524520874, 0.8280000686645508, 0.8320000171661377, 0.8300000429153442, 0.8300000429153442, 0.8280000686645508, 0.8240000605583191, 0.8260000348091125, 0.8300000429153442, 0.8280000686645508, 0.8300000429153442, 0.8240000605583191, 0.8280000686645508, 0.8300000429153442, 0.8260000348091125, 0.8220000267028809, 0.8200000524520874, 0.8300000429153442, 0.8220000267028809, 0.8380000591278076, 0.8260000348091125, 0.8260000348091125, 0.8260000348091125, 0.8180000185966492, 0.8280000686645508, 0.8220000267028809, 0.8300000429153442, 0.8220000267028809, 0.8260000348091125, 0.8220000267028809, 0.8280000686645508, 0.8280000686645508, 0.8240000605583191, 0.8200000524520874, 0.8220000267028809, 0.8300000429153442, 0.8240000605583191, 0.8240000605583191, 0.8260000348091125, 0.8300000429153442, 0.8220000267028809, 0.8200000524520874, 0.8280000686645508, 0.8280000686645508, 0.8340000510215759, 0.8200000524520874, 0.8220000267028809, 0.8200000524520874, 0.8240000605583191, 0.8220000267028809, 0.8300000429153442, 0.8240000605583191, 0.8280000686645508, 0.8240000605583191, 0.8240000605583191, 0.8380000591278076, 0.8220000267028809, 0.8280000686645508, 0.8220000267028809, 0.8180000185966492, 0.8260000348091125, 0.8240000605583191, 0.8240000605583191, 0.8280000686645508, 0.8240000605583191, 0.8180000185966492, 0.8280000686645508]
# Test Accs: [0.8220000267028809, 0.8260000348091125, 0.8330000638961792, 0.8320000171661377, 0.8210000395774841, 0.8290000557899475, 0.8210000395774841, 0.8140000104904175, 0.8190000653266907, 0.8220000267028809, 0.8210000395774841, 0.8250000476837158, 0.8250000476837158, 0.8170000314712524, 0.831000030040741, 0.8400000333786011, 0.8240000605583191, 0.8340000510215759, 0.8320000171661377, 0.8250000476837158, 0.8210000395774841, 0.8350000381469727, 0.831000030040741, 0.8160000443458557, 0.8300000429153442, 0.8350000381469727, 0.8080000281333923, 0.8190000653266907, 0.8440000414848328, 0.8270000219345093, 0.8340000510215759, 0.8290000557899475, 0.8420000672340393, 0.8180000185966492, 0.8230000138282776, 0.8380000591278076, 0.8270000219345093, 0.8260000348091125, 0.8330000638961792, 0.8280000686645508, 0.8160000443458557, 0.8330000638961792, 0.8050000667572021, 0.8100000619888306, 0.8240000605583191, 0.8200000524520874, 0.8420000672340393, 0.831000030040741, 0.8190000653266907, 0.8220000267028809, 0.8330000638961792, 0.8260000348091125, 0.8290000557899475, 0.8270000219345093, 0.8130000233650208, 0.8180000185966492, 0.812000036239624, 0.8330000638961792, 0.8220000267028809, 0.8350000381469727, 0.8080000281333923, 0.8210000395774841, 0.812000036239624, 0.8240000605583191, 0.8250000476837158, 0.831000030040741, 0.8340000510215759, 0.8340000510215759, 0.8240000605583191, 0.8200000524520874, 0.8360000252723694, 0.8230000138282776, 0.8300000429153442, 0.8180000185966492, 0.8390000462532043, 0.8250000476837158, 0.8370000123977661, 0.8240000605583191, 0.8160000443458557, 0.8340000510215759, 0.8140000104904175, 0.8240000605583191, 0.815000057220459, 0.8220000267028809, 0.8240000605583191, 0.8300000429153442, 0.831000030040741, 0.8250000476837158, 0.8290000557899475, 0.8370000123977661, 0.8290000557899475, 0.8270000219345093, 0.8390000462532043, 0.8210000395774841, 0.8380000591278076, 0.8180000185966492, 0.8270000219345093, 0.8290000557899475, 0.8410000205039978, 0.8140000104904175]
# Average val accuracy: 0.8257600492238999 ± 0.00403267005427958
# Average test accuracy: 0.8259600412845611 ± 0.008329372287805718
# #Params: 46119

# Loge loss

# run.py --gpu=2 --dataset=cora --runs=100 --optimizer=adam --lr=1e-1 --wd=5e-4 --epochs=500 --loss=savage --model=gcn --n-layers=2 --n-hidden=32 --dropout=0.8 --input-drop=0.8 --mask-rate=1.0 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='cora', dropout=0.8, edge_drop=0.0, epochs=500, gpu=2, input_drop=0.8, labels=False, linear=False, log_every=20, loss='savage', lr=0.1, mask_rate=1.0, model='gcn', n_heads=3, n_hidden=32, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.8240000605583191, 0.8160000443458557, 0.8160000443458557, 0.8140000104904175, 0.812000036239624, 0.8160000443458557, 0.8160000443458557, 0.8200000524520874, 0.812000036239624, 0.8220000267028809, 0.8260000348091125, 0.8140000104904175, 0.8160000443458557, 0.8140000104904175, 0.8160000443458557, 0.812000036239624, 0.812000036239624, 0.8200000524520874, 0.8100000619888306, 0.8140000104904175, 0.8140000104904175, 0.8100000619888306, 0.8140000104904175, 0.8160000443458557, 0.8060000538825989, 0.8140000104904175, 0.8160000443458557, 0.8220000267028809, 0.8160000443458557, 0.8160000443458557, 0.8140000104904175, 0.8140000104904175, 0.8160000443458557, 0.8060000538825989, 0.8260000348091125, 0.8240000605583191, 0.8160000443458557, 0.8220000267028809, 0.8180000185966492, 0.8160000443458557, 0.8180000185966492, 0.8180000185966492, 0.8180000185966492, 0.8140000104904175, 0.8200000524520874, 0.8160000443458557, 0.8140000104904175, 0.8140000104904175, 0.8140000104904175, 0.8200000524520874, 0.8220000267028809, 0.8200000524520874, 0.8200000524520874, 0.812000036239624, 0.8140000104904175, 0.8140000104904175, 0.8180000185966492, 0.8180000185966492, 0.8140000104904175, 0.8100000619888306, 0.8180000185966492, 0.8200000524520874, 0.8200000524520874, 0.8220000267028809, 0.8200000524520874, 0.8160000443458557, 0.8140000104904175, 0.8180000185966492, 0.812000036239624, 0.8180000185966492, 0.8080000281333923, 0.8080000281333923, 0.8160000443458557, 0.8280000686645508, 0.8200000524520874, 0.812000036239624, 0.8140000104904175, 0.8160000443458557, 0.8200000524520874, 0.8200000524520874, 0.8080000281333923, 0.8140000104904175, 0.8160000443458557, 0.8140000104904175, 0.8160000443458557, 0.8260000348091125, 0.8140000104904175, 0.8260000348091125, 0.8200000524520874, 0.8240000605583191, 0.812000036239624, 0.8320000171661377, 0.812000036239624, 0.812000036239624, 0.8180000185966492, 0.8100000619888306, 0.8200000524520874, 0.8200000524520874, 0.8180000185966492, 0.8140000104904175]
# Test Accs: [0.8060000538825989, 0.8230000138282776, 0.7920000553131104, 0.8170000314712524, 0.8220000267028809, 0.8050000667572021, 0.8070000410079956, 0.8220000267028809, 0.8130000233650208, 0.8190000653266907, 0.8100000619888306, 0.8290000557899475, 0.8190000653266907, 0.8170000314712524, 0.8230000138282776, 0.8200000524520874, 0.8140000104904175, 0.8230000138282776, 0.8170000314712524, 0.8180000185966492, 0.8130000233650208, 0.8210000395774841, 0.8070000410079956, 0.8110000491142273, 0.8100000619888306, 0.8100000619888306, 0.8030000329017639, 0.8130000233650208, 0.8250000476837158, 0.8170000314712524, 0.812000036239624, 0.8300000429153442, 0.8280000686645508, 0.8110000491142273, 0.8140000104904175, 0.7990000247955322, 0.8250000476837158, 0.8240000605583191, 0.8040000200271606, 0.8180000185966492, 0.8110000491142273, 0.8050000667572021, 0.8180000185966492, 0.8210000395774841, 0.8210000395774841, 0.8180000185966492, 0.8180000185966492, 0.8070000410079956, 0.8250000476837158, 0.8220000267028809, 0.815000057220459, 0.8250000476837158, 0.8170000314712524, 0.8200000524520874, 0.8230000138282776, 0.8160000443458557, 0.812000036239624, 0.8080000281333923, 0.8200000524520874, 0.8160000443458557, 0.8210000395774841, 0.8200000524520874, 0.8160000443458557, 0.812000036239624, 0.8050000667572021, 0.8250000476837158, 0.8180000185966492, 0.815000057220459, 0.815000057220459, 0.8240000605583191, 0.8060000538825989, 0.8210000395774841, 0.8090000152587891, 0.831000030040741, 0.8130000233650208, 0.815000057220459, 0.8190000653266907, 0.8210000395774841, 0.8220000267028809, 0.8160000443458557, 0.8170000314712524, 0.8200000524520874, 0.8170000314712524, 0.8210000395774841, 0.8290000557899475, 0.812000036239624, 0.8010000586509705, 0.815000057220459, 0.8160000443458557, 0.8240000605583191, 0.831000030040741, 0.8300000429153442, 0.8160000443458557, 0.8050000667572021, 0.815000057220459, 0.812000036239624, 0.8220000267028809, 0.8230000138282776, 0.8100000619888306, 0.8210000395774841]
# Average val accuracy: 0.8165200340747834 ± 0.004704213809155496
# Average test accuracy: 0.8165000408887864 ± 0.007383086773028536
# #Params: 46119

## Citeseer

# Logistic loss

# run.py --gpu=2 --dataset=citeseer --runs=100 --optimizer=adam --lr=1e-1 --wd=5e-4 --epochs=500 --loss=logit --model=gcn --n-layers=2 --n-hidden=32 --dropout=0.8 --input-drop=0.8 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='citeseer', dropout=0.8, edge_drop=0.0, epochs=500, gpu=2, input_drop=0.8, labels=False, linear=False, log_every=20, loss='logit', lr=0.1, mask_rate=1.0, model='gcn', n_heads=3, n_hidden=32, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.7260000109672546, 0.7320000529289246, 0.7360000610351562, 0.734000027179718, 0.7440000176429749, 0.7380000352859497, 0.7280000448226929, 0.7380000352859497, 0.7280000448226929, 0.7360000610351562, 0.7320000529289246, 0.7260000109672546, 0.7360000610351562, 0.7260000109672546, 0.7400000095367432, 0.7260000109672546, 0.7300000190734863, 0.7360000610351562, 0.7300000190734863, 0.734000027179718, 0.7300000190734863, 0.7320000529289246, 0.734000027179718, 0.7220000624656677, 0.7280000448226929, 0.7240000367164612, 0.7360000610351562, 0.7360000610351562, 0.7280000448226929, 0.734000027179718, 0.7360000610351562, 0.7260000109672546, 0.7320000529289246, 0.7380000352859497, 0.734000027179718, 0.7300000190734863, 0.7280000448226929, 0.734000027179718, 0.7320000529289246, 0.7300000190734863, 0.734000027179718, 0.734000027179718, 0.7380000352859497, 0.7220000624656677, 0.7380000352859497, 0.734000027179718, 0.7280000448226929, 0.7360000610351562, 0.7220000624656677, 0.7280000448226929, 0.7360000610351562, 0.7620000243186951, 0.7320000529289246, 0.7320000529289246, 0.7420000433921814, 0.7260000109672546, 0.7300000190734863, 0.7360000610351562, 0.7540000081062317, 0.7360000610351562, 0.734000027179718, 0.7300000190734863, 0.7280000448226929, 0.7240000367164612, 0.7360000610351562, 0.7280000448226929, 0.7320000529289246, 0.7220000624656677, 0.7280000448226929, 0.734000027179718, 0.7280000448226929, 0.7320000529289246, 0.7320000529289246, 0.7280000448226929, 0.7360000610351562, 0.7440000176429749, 0.7260000109672546, 0.7260000109672546, 0.7240000367164612, 0.7320000529289246, 0.7360000610351562, 0.7320000529289246, 0.7320000529289246, 0.7320000529289246, 0.734000027179718, 0.7300000190734863, 0.7280000448226929, 0.7300000190734863, 0.7240000367164612, 0.7300000190734863, 0.7460000514984131, 0.7300000190734863, 0.7300000190734863, 0.7320000529289246, 0.7280000448226929, 0.7360000610351562, 0.734000027179718, 0.7400000095367432, 0.7240000367164612, 0.7260000109672546]
# Test Accs: [0.7240000367164612, 0.7070000171661377, 0.7250000238418579, 0.7090000510215759, 0.7210000157356262, 0.718000054359436, 0.7140000462532043, 0.7220000624656677, 0.7170000076293945, 0.7290000319480896, 0.6960000395774841, 0.7200000286102295, 0.7110000252723694, 0.7040000557899475, 0.7160000205039978, 0.7160000205039978, 0.6940000057220459, 0.7020000219345093, 0.7190000414848328, 0.7230000495910645, 0.7050000429153442, 0.7050000429153442, 0.7000000476837158, 0.7130000591278076, 0.7170000076293945, 0.7130000591278076, 0.7210000157356262, 0.6970000267028809, 0.7140000462532043, 0.6850000619888306, 0.6950000524520874, 0.7150000333786011, 0.7250000238418579, 0.7120000123977661, 0.7100000381469727, 0.6980000138282776, 0.6910000443458557, 0.7050000429153442, 0.7160000205039978, 0.7020000219345093, 0.7130000591278076, 0.7280000448226929, 0.734000027179718, 0.687000036239624, 0.7190000414848328, 0.7090000510215759, 0.6960000395774841, 0.7210000157356262, 0.687000036239624, 0.7050000429153442, 0.7040000557899475, 0.7280000448226929, 0.7110000252723694, 0.7130000591278076, 0.7100000381469727, 0.7050000429153442, 0.7130000591278076, 0.7040000557899475, 0.7280000448226929, 0.7100000381469727, 0.7050000429153442, 0.7080000042915344, 0.7280000448226929, 0.7100000381469727, 0.7100000381469727, 0.7010000348091125, 0.6980000138282776, 0.7200000286102295, 0.7050000429153442, 0.7130000591278076, 0.7140000462532043, 0.7000000476837158, 0.7250000238418579, 0.7360000610351562, 0.706000030040741, 0.7150000333786011, 0.718000054359436, 0.7080000042915344, 0.7120000123977661, 0.7240000367164612, 0.7000000476837158, 0.7280000448226929, 0.7210000157356262, 0.7080000042915344, 0.7320000529289246, 0.7120000123977661, 0.7140000462532043, 0.7050000429153442, 0.6860000491142273, 0.6950000524520874, 0.7040000557899475, 0.7330000400543213, 0.718000054359436, 0.7120000123977661, 0.7150000333786011, 0.706000030040741, 0.7260000109672546, 0.7010000348091125, 0.7100000381469727, 0.6960000395774841]
# Average val accuracy: 0.7321800380945206 ± 0.00622957438857565
# Average test accuracy: 0.7112900364398956 ± 0.011194010947676717
# #Params: 118726

# Savage loss

# run.py --gpu=2 --dataset=citeseer --runs=100 --optimizer=adam --lr=1e-1 --wd=5e-4 --epochs=500 --loss=loge --model=gcn --n-layers=2 --n-hidden=32 --dropout=0.8 --input-drop=0.8 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='citeseer', dropout=0.8, edge_drop=0.0, epochs=500, gpu=2, input_drop=0.8, labels=False, linear=False, log_every=20, loss='loge', lr=0.1, mask_rate=1.0, model='gcn', n_heads=3, n_hidden=32, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.7440000176429749, 0.7480000257492065, 0.7400000095367432, 0.7480000257492065, 0.7420000433921814, 0.7400000095367432, 0.7460000514984131, 0.7480000257492065, 0.7600000500679016, 0.7460000514984131, 0.734000027179718, 0.7460000514984131, 0.7420000433921814, 0.7440000176429749, 0.7500000596046448, 0.7380000352859497, 0.7420000433921814, 0.7380000352859497, 0.7440000176429749, 0.7440000176429749, 0.7540000081062317, 0.7360000610351562, 0.7460000514984131, 0.7400000095367432, 0.7440000176429749, 0.734000027179718, 0.7400000095367432, 0.7460000514984131, 0.7400000095367432, 0.7500000596046448, 0.7560000419616699, 0.7520000338554382, 0.7460000514984131, 0.7420000433921814, 0.7380000352859497, 0.7360000610351562, 0.7540000081062317, 0.7460000514984131, 0.7400000095367432, 0.7460000514984131, 0.7360000610351562, 0.7400000095367432, 0.7520000338554382, 0.7500000596046448, 0.7380000352859497, 0.7480000257492065, 0.7400000095367432, 0.7440000176429749, 0.7460000514984131, 0.7380000352859497, 0.7460000514984131, 0.7380000352859497, 0.7460000514984131, 0.7440000176429749, 0.7500000596046448, 0.7440000176429749, 0.7380000352859497, 0.7480000257492065, 0.7440000176429749, 0.7460000514984131, 0.7380000352859497, 0.7440000176429749, 0.7460000514984131, 0.7440000176429749, 0.7420000433921814, 0.7480000257492065, 0.7400000095367432, 0.7460000514984131, 0.734000027179718, 0.7400000095367432, 0.7460000514984131, 0.7420000433921814, 0.7480000257492065, 0.7460000514984131, 0.7440000176429749, 0.7440000176429749, 0.7440000176429749, 0.7400000095367432, 0.734000027179718, 0.7420000433921814, 0.7440000176429749, 0.7400000095367432, 0.7480000257492065, 0.7420000433921814, 0.7560000419616699, 0.7360000610351562, 0.7440000176429749, 0.7500000596046448, 0.7420000433921814, 0.7360000610351562, 0.7420000433921814, 0.7380000352859497, 0.7400000095367432, 0.7520000338554382, 0.7460000514984131, 0.7480000257492065, 0.7460000514984131, 0.7420000433921814, 0.7420000433921814, 0.7360000610351562]
# Test Accs: [0.7140000462532043, 0.7300000190734863, 0.7100000381469727, 0.7250000238418579, 0.7190000414848328, 0.7170000076293945, 0.7380000352859497, 0.7110000252723694, 0.7210000157356262, 0.737000048160553, 0.7250000238418579, 0.7170000076293945, 0.7220000624656677, 0.7250000238418579, 0.7310000061988831, 0.7230000495910645, 0.7280000448226929, 0.7000000476837158, 0.737000048160553, 0.7230000495910645, 0.7300000190734863, 0.718000054359436, 0.7300000190734863, 0.7280000448226929, 0.7270000576972961, 0.7310000061988831, 0.7150000333786011, 0.737000048160553, 0.7070000171661377, 0.7260000109672546, 0.7270000576972961, 0.718000054359436, 0.7260000109672546, 0.7140000462532043, 0.7210000157356262, 0.7190000414848328, 0.706000030040741, 0.7130000591278076, 0.7360000610351562, 0.7320000529289246, 0.7150000333786011, 0.7290000319480896, 0.7250000238418579, 0.7380000352859497, 0.7270000576972961, 0.7210000157356262, 0.7210000157356262, 0.7240000367164612, 0.734000027179718, 0.7270000576972961, 0.7390000224113464, 0.7420000433921814, 0.7400000095367432, 0.7570000290870667, 0.7330000400543213, 0.7100000381469727, 0.7240000367164612, 0.7240000367164612, 0.7350000143051147, 0.7290000319480896, 0.718000054359436, 0.7090000510215759, 0.7250000238418579, 0.734000027179718, 0.7140000462532043, 0.7430000305175781, 0.7330000400543213, 0.7280000448226929, 0.7230000495910645, 0.7300000190734863, 0.7290000319480896, 0.7140000462532043, 0.7290000319480896, 0.7320000529289246, 0.7110000252723694, 0.7150000333786011, 0.7390000224113464, 0.7360000610351562, 0.7300000190734863, 0.7280000448226929, 0.7150000333786011, 0.7260000109672546, 0.6910000443458557, 0.7160000205039978, 0.7420000433921814, 0.7150000333786011, 0.7220000624656677, 0.7480000257492065, 0.7090000510215759, 0.703000009059906, 0.7500000596046448, 0.7040000557899475, 0.7120000123977661, 0.7400000095367432, 0.7320000529289246, 0.718000054359436, 0.7290000319480896, 0.7210000157356262, 0.7350000143051147, 0.7330000400543213]
# Average val accuracy: 0.7436800342798233 ± 0.0051824337703506845
# Average test accuracy: 0.7248900347948074 ± 0.011223986782884913
# #Params: 118726

# Loge loss

# run.py --gpu=2 --dataset=citeseer --runs=100 --optimizer=adam --lr=1e-1 --wd=5e-4 --epochs=500 --loss=savage --model=gcn --n-layers=2 --n-hidden=32 --dropout=0.8 --input-drop=0.8 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='citeseer', dropout=0.8, edge_drop=0.0, epochs=500, gpu=2, input_drop=0.8, labels=False, linear=False, log_every=20, loss='savage', lr=0.1, mask_rate=1.0, model='gcn', n_heads=3, n_hidden=32, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.7360000610351562, 0.7360000610351562, 0.7480000257492065, 0.7420000433921814, 0.7440000176429749, 0.734000027179718, 0.734000027179718, 0.7380000352859497, 0.7420000433921814, 0.7420000433921814, 0.7260000109672546, 0.7420000433921814, 0.7500000596046448, 0.7500000596046448, 0.7560000419616699, 0.7380000352859497, 0.734000027179718, 0.734000027179718, 0.7400000095367432, 0.7400000095367432, 0.7480000257492065, 0.7460000514984131, 0.7400000095367432, 0.7320000529289246, 0.7380000352859497, 0.7380000352859497, 0.7440000176429749, 0.7380000352859497, 0.734000027179718, 0.7400000095367432, 0.7520000338554382, 0.7300000190734863, 0.7460000514984131, 0.7320000529289246, 0.7440000176429749, 0.7460000514984131, 0.7300000190734863, 0.7280000448226929, 0.7360000610351562, 0.7360000610351562, 0.7440000176429749, 0.734000027179718, 0.7280000448226929, 0.7320000529289246, 0.7420000433921814, 0.7460000514984131, 0.7440000176429749, 0.7380000352859497, 0.7380000352859497, 0.7420000433921814, 0.734000027179718, 0.7360000610351562, 0.7320000529289246, 0.7360000610351562, 0.7360000610351562, 0.7420000433921814, 0.7260000109672546, 0.7320000529289246, 0.734000027179718, 0.734000027179718, 0.7300000190734863, 0.7500000596046448, 0.7380000352859497, 0.7320000529289246, 0.7480000257492065, 0.7540000081062317, 0.7460000514984131, 0.7420000433921814, 0.7460000514984131, 0.7360000610351562, 0.734000027179718, 0.7280000448226929, 0.7360000610351562, 0.7400000095367432, 0.7400000095367432, 0.7560000419616699, 0.7440000176429749, 0.7380000352859497, 0.7440000176429749, 0.7420000433921814, 0.7520000338554382, 0.7460000514984131, 0.7480000257492065, 0.7420000433921814, 0.7400000095367432, 0.7500000596046448, 0.7540000081062317, 0.7400000095367432, 0.7380000352859497, 0.7460000514984131, 0.7400000095367432, 0.7460000514984131, 0.7320000529289246, 0.7400000095367432, 0.7460000514984131, 0.7420000433921814, 0.7360000610351562, 0.7400000095367432, 0.7260000109672546, 0.7240000367164612]
# Test Accs: [0.7050000429153442, 0.718000054359436, 0.7160000205039978, 0.7290000319480896, 0.7050000429153442, 0.7190000414848328, 0.7110000252723694, 0.7100000381469727, 0.706000030040741, 0.7040000557899475, 0.7150000333786011, 0.7000000476837158, 0.7250000238418579, 0.7380000352859497, 0.7110000252723694, 0.6960000395774841, 0.7020000219345093, 0.7110000252723694, 0.6960000395774841, 0.7000000476837158, 0.7240000367164612, 0.7200000286102295, 0.7110000252723694, 0.7070000171661377, 0.6970000267028809, 0.7100000381469727, 0.7070000171661377, 0.7050000429153442, 0.7240000367164612, 0.7080000042915344, 0.7290000319480896, 0.6970000267028809, 0.6970000267028809, 0.6990000605583191, 0.7250000238418579, 0.7050000429153442, 0.6970000267028809, 0.7270000576972961, 0.7140000462532043, 0.718000054359436, 0.718000054359436, 0.7140000462532043, 0.7110000252723694, 0.7100000381469727, 0.7100000381469727, 0.6980000138282776, 0.7020000219345093, 0.7070000171661377, 0.706000030040741, 0.7100000381469727, 0.7070000171661377, 0.7250000238418579, 0.706000030040741, 0.7290000319480896, 0.6800000071525574, 0.6850000619888306, 0.6670000553131104, 0.7270000576972961, 0.7210000157356262, 0.7090000510215759, 0.7190000414848328, 0.706000030040741, 0.7170000076293945, 0.7070000171661377, 0.7230000495910645, 0.6980000138282776, 0.7210000157356262, 0.734000027179718, 0.7040000557899475, 0.7120000123977661, 0.7230000495910645, 0.7210000157356262, 0.6960000395774841, 0.7160000205039978, 0.7120000123977661, 0.7010000348091125, 0.7050000429153442, 0.7420000433921814, 0.7170000076293945, 0.7200000286102295, 0.7110000252723694, 0.7010000348091125, 0.7250000238418579, 0.7010000348091125, 0.7110000252723694, 0.7190000414848328, 0.7220000624656677, 0.7190000414848328, 0.7140000462532043, 0.7160000205039978, 0.7150000333786011, 0.7310000061988831, 0.718000054359436, 0.7000000476837158, 0.7220000624656677, 0.6850000619888306, 0.7150000333786011, 0.7100000381469727, 0.690000057220459, 0.703000009059906]
# Average val accuracy: 0.7396600359678268 ± 0.0070259808938795305
# Average test accuracy: 0.711020033955574 ± 0.012202441229059599
# #Params: 118726

## Pubmed

# Logistic loss

# run.py --dataset=pubmed --runs=100 --lr=1e-1 --wd=5e-4 --epochs=200 --loss=logit --model=gcn --n-layers=2 --n-hidden=32 --dropout=0.5 --input-drop=0.5 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='pubmed', dropout=0.5, edge_drop=0.0, epochs=200, gpu=0, input_drop=0.5, labels=False, linear=False, log_every=20, loss='logit', lr=0.1, mask_rate=1.0, model='gcn', n_heads=3, n_hidden=32, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.8240000605583191, 0.8140000104904175, 0.812000036239624, 0.8100000619888306, 0.8060000538825989, 0.8200000524520874, 0.812000036239624, 0.8100000619888306, 0.8160000443458557, 0.812000036239624, 0.8140000104904175, 0.8140000104904175, 0.8140000104904175, 0.8080000281333923, 0.8100000619888306, 0.8100000619888306, 0.8060000538825989, 0.8080000281333923, 0.812000036239624, 0.8140000104904175, 0.8060000538825989, 0.8200000524520874, 0.812000036239624, 0.8160000443458557, 0.8100000619888306, 0.8180000185966492, 0.8200000524520874, 0.8080000281333923, 0.8100000619888306, 0.8180000185966492, 0.8180000185966492, 0.812000036239624, 0.8140000104904175, 0.8220000267028809, 0.8160000443458557, 0.8040000200271606, 0.8100000619888306, 0.8180000185966492, 0.8080000281333923, 0.812000036239624, 0.8140000104904175, 0.8100000619888306, 0.8200000524520874, 0.812000036239624, 0.812000036239624, 0.8160000443458557, 0.8100000619888306, 0.812000036239624, 0.8060000538825989, 0.8080000281333923, 0.8180000185966492, 0.8140000104904175, 0.812000036239624, 0.8140000104904175, 0.8160000443458557, 0.8080000281333923, 0.8080000281333923, 0.8100000619888306, 0.8100000619888306, 0.8100000619888306, 0.8100000619888306, 0.8040000200271606, 0.812000036239624, 0.8100000619888306, 0.8240000605583191, 0.8080000281333923, 0.8240000605583191, 0.8080000281333923, 0.8140000104904175, 0.8160000443458557, 0.8160000443458557, 0.8020000457763672, 0.812000036239624, 0.8140000104904175, 0.8140000104904175, 0.8100000619888306, 0.8140000104904175, 0.8160000443458557, 0.8080000281333923, 0.8100000619888306, 0.8140000104904175, 0.8100000619888306, 0.8140000104904175, 0.8080000281333923, 0.8140000104904175, 0.8080000281333923, 0.8140000104904175, 0.8220000267028809, 0.8200000524520874, 0.8080000281333923, 0.8060000538825989, 0.812000036239624, 0.812000036239624, 0.812000036239624, 0.8100000619888306, 0.8100000619888306, 0.8100000619888306, 0.8080000281333923, 0.8080000281333923, 0.8180000185966492]
# Test Accs: [0.7880000472068787, 0.7950000166893005, 0.7830000519752502, 0.796000063419342, 0.7820000648498535, 0.796000063419342, 0.7780000567436218, 0.7870000600814819, 0.8050000667572021, 0.7980000376701355, 0.7930000424385071, 0.7730000615119934, 0.7890000343322754, 0.784000039100647, 0.7780000567436218, 0.781000018119812, 0.784000039100647, 0.7830000519752502, 0.800000011920929, 0.7910000085830688, 0.7860000133514404, 0.784000039100647, 0.7820000648498535, 0.7780000567436218, 0.7920000553131104, 0.7790000438690186, 0.7780000567436218, 0.7920000553131104, 0.7850000262260437, 0.7990000247955322, 0.7900000214576721, 0.7890000343322754, 0.7890000343322754, 0.7980000376701355, 0.7920000553131104, 0.7800000309944153, 0.7860000133514404, 0.7860000133514404, 0.7790000438690186, 0.7800000309944153, 0.7940000295639038, 0.800000011920929, 0.7910000085830688, 0.7830000519752502, 0.7850000262260437, 0.7770000100135803, 0.7890000343322754, 0.7950000166893005, 0.7900000214576721, 0.8060000538825989, 0.7940000295639038, 0.7920000553131104, 0.7910000085830688, 0.7780000567436218, 0.800000011920929, 0.7760000228881836, 0.7930000424385071, 0.7940000295639038, 0.7880000472068787, 0.7940000295639038, 0.7870000600814819, 0.7830000519752502, 0.7850000262260437, 0.7770000100135803, 0.7920000553131104, 0.7990000247955322, 0.781000018119812, 0.7930000424385071, 0.7910000085830688, 0.7930000424385071, 0.7950000166893005, 0.7890000343322754, 0.7970000505447388, 0.7850000262260437, 0.8010000586509705, 0.7870000600814819, 0.7910000085830688, 0.7940000295639038, 0.7930000424385071, 0.7910000085830688, 0.7930000424385071, 0.7880000472068787, 0.784000039100647, 0.7920000553131104, 0.7970000505447388, 0.7910000085830688, 0.7970000505447388, 0.7780000567436218, 0.7920000553131104, 0.8030000329017639, 0.7890000343322754, 0.7820000648498535, 0.7900000214576721, 0.7850000262260437, 0.7880000472068787, 0.800000011920929, 0.7900000214576721, 0.7790000438690186, 0.7850000262260437, 0.7890000343322754]
# Average val accuracy: 0.8124200373888015 ± 0.004554512322775121
# Average test accuracy: 0.7889400368928909 ± 0.007073639054637011
# #Params: 16131

# Savage loss

# run.py --dataset=pubmed --runs=100 --lr=1e-1 --wd=5e-4 --epochs=200 --loss=loge --model=gcn --n-layers=2 --n-hidden=32 --dropout=0.5 --input-drop=0.5 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='pubmed', dropout=0.5, edge_drop=0.0, epochs=200, gpu=0, input_drop=0.5, labels=False, linear=False, log_every=20, loss='loge', lr=0.1, mask_rate=1.0, model='gcn', n_heads=3, n_hidden=32, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.812000036239624, 0.812000036239624, 0.812000036239624, 0.8200000524520874, 0.8140000104904175, 0.8200000524520874, 0.8080000281333923, 0.8100000619888306, 0.8180000185966492, 0.8180000185966492, 0.8160000443458557, 0.812000036239624, 0.8140000104904175, 0.812000036239624, 0.812000036239624, 0.8160000443458557, 0.8180000185966492, 0.8140000104904175, 0.8180000185966492, 0.8140000104904175, 0.8140000104904175, 0.812000036239624, 0.8140000104904175, 0.812000036239624, 0.8140000104904175, 0.8140000104904175, 0.8140000104904175, 0.8100000619888306, 0.8160000443458557, 0.8140000104904175, 0.8080000281333923, 0.8180000185966492, 0.8200000524520874, 0.812000036239624, 0.812000036239624, 0.812000036239624, 0.8140000104904175, 0.8160000443458557, 0.8180000185966492, 0.8060000538825989, 0.8160000443458557, 0.8180000185966492, 0.8180000185966492, 0.812000036239624, 0.8060000538825989, 0.8220000267028809, 0.8160000443458557, 0.8080000281333923, 0.8080000281333923, 0.8140000104904175, 0.8140000104904175, 0.8140000104904175, 0.8200000524520874, 0.8140000104904175, 0.8180000185966492, 0.8100000619888306, 0.8080000281333923, 0.8160000443458557, 0.812000036239624, 0.812000036239624, 0.8260000348091125, 0.8080000281333923, 0.8060000538825989, 0.8100000619888306, 0.8220000267028809, 0.8040000200271606, 0.812000036239624, 0.8140000104904175, 0.8100000619888306, 0.8100000619888306, 0.8160000443458557, 0.7980000376701355, 0.8080000281333923, 0.8040000200271606, 0.8140000104904175, 0.812000036239624, 0.812000036239624, 0.8200000524520874, 0.812000036239624, 0.8140000104904175, 0.8100000619888306, 0.8140000104904175, 0.8200000524520874, 0.812000036239624, 0.8100000619888306, 0.8160000443458557, 0.8140000104904175, 0.8060000538825989, 0.8180000185966492, 0.8140000104904175, 0.8140000104904175, 0.8100000619888306, 0.8160000443458557, 0.8140000104904175, 0.8080000281333923, 0.8080000281333923, 0.8040000200271606, 0.800000011920929, 0.8160000443458557, 0.8160000443458557]
# Test Accs: [0.784000039100647, 0.7850000262260437, 0.796000063419342, 0.7850000262260437, 0.7860000133514404, 0.800000011920929, 0.781000018119812, 0.7830000519752502, 0.7940000295639038, 0.7890000343322754, 0.7790000438690186, 0.7950000166893005, 0.7890000343322754, 0.7900000214576721, 0.7940000295639038, 0.7930000424385071, 0.8020000457763672, 0.7900000214576721, 0.7940000295639038, 0.8010000586509705, 0.796000063419342, 0.7900000214576721, 0.7870000600814819, 0.7870000600814819, 0.7930000424385071, 0.7990000247955322, 0.796000063419342, 0.7980000376701355, 0.796000063419342, 0.796000063419342, 0.7870000600814819, 0.7880000472068787, 0.7920000553131104, 0.7900000214576721, 0.7910000085830688, 0.7800000309944153, 0.7900000214576721, 0.7950000166893005, 0.7970000505447388, 0.7830000519752502, 0.7820000648498535, 0.796000063419342, 0.7900000214576721, 0.7920000553131104, 0.7880000472068787, 0.7940000295639038, 0.7890000343322754, 0.7880000472068787, 0.7830000519752502, 0.7930000424385071, 0.7950000166893005, 0.7970000505447388, 0.7970000505447388, 0.7760000228881836, 0.7830000519752502, 0.7800000309944153, 0.7920000553131104, 0.800000011920929, 0.7820000648498535, 0.7930000424385071, 0.8020000457763672, 0.7920000553131104, 0.7890000343322754, 0.784000039100647, 0.7940000295639038, 0.7780000567436218, 0.781000018119812, 0.7820000648498535, 0.781000018119812, 0.7900000214576721, 0.7890000343322754, 0.7600000500679016, 0.7940000295639038, 0.7990000247955322, 0.7950000166893005, 0.7790000438690186, 0.7930000424385071, 0.7920000553131104, 0.7830000519752502, 0.7940000295639038, 0.7820000648498535, 0.7880000472068787, 0.7850000262260437, 0.7920000553131104, 0.7910000085830688, 0.7920000553131104, 0.7950000166893005, 0.7870000600814819, 0.7970000505447388, 0.7930000424385071, 0.781000018119812, 0.784000039100647, 0.7820000648498535, 0.784000039100647, 0.7870000600814819, 0.796000063419342, 0.781000018119812, 0.7770000100135803, 0.7800000309944153, 0.7900000214576721]
# Average val accuracy: 0.8130800318717957 ± 0.004651191784986798
# Average test accuracy: 0.7892600393295288 ± 0.006907416198303355
# #Params: 16131

# Loge loss

# run.py --dataset=pubmed --runs=100 --lr=1e-1 --wd=5e-4 --epochs=200 --loss=savage --model=gcn --n-layers=2 --n-hidden=32 --dropout=0.5 --input-drop=0.5 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='pubmed', dropout=0.5, edge_drop=0.0, epochs=200, gpu=0, input_drop=0.5, labels=False, linear=False, log_every=20, loss='savage', lr=0.1, mask_rate=1.0, model='gcn', n_heads=3, n_hidden=32, n_label_iters=0, n_layers=2, n_prop=7, non_interactive_attn=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, split='std', tune='', wd=0.0005)
# Runned 100 times
# Val Accs: [0.812000036239624, 0.8140000104904175, 0.8160000443458557, 0.8080000281333923, 0.8140000104904175, 0.8160000443458557, 0.812000036239624, 0.8160000443458557, 0.8180000185966492, 0.8140000104904175, 0.8140000104904175, 0.8140000104904175, 0.8140000104904175, 0.8080000281333923, 0.8100000619888306, 0.8180000185966492, 0.8200000524520874, 0.8200000524520874, 0.8160000443458557, 0.8080000281333923, 0.8160000443458557, 0.8180000185966492, 0.8220000267028809, 0.8100000619888306, 0.8240000605583191, 0.8100000619888306, 0.8180000185966492, 0.8100000619888306, 0.8220000267028809, 0.8100000619888306, 0.8160000443458557, 0.8140000104904175, 0.8160000443458557, 0.8220000267028809, 0.8180000185966492, 0.8140000104904175, 0.8140000104904175, 0.8160000443458557, 0.8140000104904175, 0.8100000619888306, 0.8040000200271606, 0.8080000281333923, 0.8160000443458557, 0.8140000104904175, 0.8180000185966492, 0.8160000443458557, 0.8160000443458557, 0.8140000104904175, 0.8140000104904175, 0.8240000605583191, 0.8220000267028809, 0.8180000185966492, 0.8180000185966492, 0.8160000443458557, 0.8140000104904175, 0.8100000619888306, 0.8100000619888306, 0.8200000524520874, 0.8220000267028809, 0.812000036239624, 0.8160000443458557, 0.8140000104904175, 0.8200000524520874, 0.8180000185966492, 0.8140000104904175, 0.8100000619888306, 0.8140000104904175, 0.8060000538825989, 0.8140000104904175, 0.8140000104904175, 0.8240000605583191, 0.8100000619888306, 0.812000036239624, 0.8220000267028809, 0.8180000185966492, 0.8140000104904175, 0.8180000185966492, 0.8160000443458557, 0.8160000443458557, 0.8180000185966492, 0.8160000443458557, 0.8160000443458557, 0.812000036239624, 0.8180000185966492, 0.8240000605583191, 0.812000036239624, 0.8100000619888306, 0.812000036239624, 0.8160000443458557, 0.8140000104904175, 0.8140000104904175, 0.8140000104904175, 0.8200000524520874, 0.8140000104904175, 0.812000036239624, 0.8160000443458557, 0.8140000104904175, 0.8080000281333923, 0.8100000619888306, 0.8100000619888306]
# Test Accs: [0.7910000085830688, 0.7900000214576721, 0.7850000262260437, 0.7920000553131104, 0.7850000262260437, 0.7950000166893005, 0.7830000519752502, 0.7900000214576721, 0.800000011920929, 0.7930000424385071, 0.7940000295639038, 0.7820000648498535, 0.7870000600814819, 0.7940000295639038, 0.7790000438690186, 0.781000018119812, 0.7970000505447388, 0.7880000472068787, 0.7860000133514404, 0.7920000553131104, 0.7870000600814819, 0.7980000376701355, 0.7930000424385071, 0.7870000600814819, 0.7970000505447388, 0.7870000600814819, 0.7950000166893005, 0.7950000166893005, 0.7920000553131104, 0.7920000553131104, 0.7950000166893005, 0.784000039100647, 0.7930000424385071, 0.7850000262260437, 0.7820000648498535, 0.7870000600814819, 0.7850000262260437, 0.7930000424385071, 0.7800000309944153, 0.7850000262260437, 0.7820000648498535, 0.7890000343322754, 0.781000018119812, 0.7910000085830688, 0.800000011920929, 0.7800000309944153, 0.7940000295639038, 0.7720000147819519, 0.7940000295639038, 0.7940000295639038, 0.784000039100647, 0.7980000376701355, 0.7920000553131104, 0.7880000472068787, 0.7890000343322754, 0.796000063419342, 0.7750000357627869, 0.7890000343322754, 0.7930000424385071, 0.7860000133514404, 0.7930000424385071, 0.7830000519752502, 0.7870000600814819, 0.800000011920929, 0.7790000438690186, 0.7970000505447388, 0.800000011920929, 0.781000018119812, 0.7790000438690186, 0.7850000262260437, 0.7870000600814819, 0.796000063419342, 0.7910000085830688, 0.7930000424385071, 0.7930000424385071, 0.7930000424385071, 0.7830000519752502, 0.7890000343322754, 0.7880000472068787, 0.7830000519752502, 0.7740000486373901, 0.7900000214576721, 0.7900000214576721, 0.7990000247955322, 0.7970000505447388, 0.7850000262260437, 0.781000018119812, 0.7830000519752502, 0.7890000343322754, 0.7860000133514404, 0.7790000438690186, 0.7860000133514404, 0.7920000553131104, 0.7920000553131104, 0.7900000214576721, 0.800000011920929, 0.8030000329017639, 0.7930000424385071, 0.7880000472068787, 0.7870000600814819]
# Average val accuracy: 0.8149200332164764 ± 0.004180142570095234
# Average test accuracy: 0.7890700370073318 ± 0.006336014070826452
# #Params: 16131

## ogbn-arxiv

# Logistic loss

# run.py --gpu=0 --dataset=ogbn-arxiv --epochs=1000 --model=gcn --n-hidden=256 --input-drop=0.1 --dropout=0.5 --lr=0.005
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='ogbn-arxiv', dropout=0.5, edge_drop=0.0, epochs=1000, gpu=0, input_drop=0.1, labels=False, linear=False, log_every=20, loss='ce', lr=0.005, mask_rate=0.5, model='gcn', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=3, non_interactive_attn=False, norm='batch', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=10, save_pred=False, seed=0, tune='', wd=0)
# Runned 10 times
# Val Accs: [0.7299573811201718, 0.7317359642941038, 0.7314674989093594, 0.7294204503506829, 0.7307627772744052, 0.7293197758314037, 0.7320379878519413, 0.7304271955434746, 0.7316352897748246, 0.7285814960233565]
# Test Accs: [0.719070839248606, 0.7243585786885584, 0.7201201571919429, 0.7157994362487913, 0.7200378577454066, 0.7148529926136247, 0.7160257597267659, 0.7166430055757875, 0.7186387671542909, 0.7112318169660309]
# Average val accuracy: 0.7305345816973724 ± 0.0011275546159266877
# Average test accuracy: 0.7176779211159804 ± 0.0034042667663424734
# #Params: 109608

# Savage loss

# run.py --gpu=1 --dataset=ogbn-arxiv --epochs=1000 --loss=savage --model=gcn --n-hidden=256 --input-drop=0.1 --dropout=0.5 --lr=0.005
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='ogbn-arxiv', dropout=0.5, edge_drop=0.0, epochs=1000, gpu=1, input_drop=0.1, labels=False, linear=False, log_every=20, loss='savage', lr=0.005, mask_rate=0.5, model='gcn', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=3, n_prop=7, non_interactive_attn=False, norm='batch', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=10, save_pred=False, seed=0, split='std', tune='', wd=0)
# Runned 10 times
# Val Accs: [0.6905936440820162, 0.6900567133125273, 0.6933454142756469, 0.6938823450451358, 0.6894191080237592, 0.6962314171616497, 0.6905600859089231, 0.6830094969629853, 0.6924393436021343, 0.6876405248498272]
# Test Accs: [0.6798757278357304, 0.6852869164454869, 0.684937143797708, 0.6865419830051643, 0.682365286093451, 0.6927967409419171, 0.6853897907536571, 0.68347632862169, 0.6835586280682262, 0.6828385079110343]
# Average val accuracy: 0.6907178093224605 ± 0.003483301187767395
# Average test accuracy: 0.6847067053474065 ± 0.003237528846203473
# #Params: 109608

# Loge loss

# run.py --gpu=1 --dataset=ogbn-arxiv --epochs=1000 --loss=loge --model=gcn --n-hidden=256 --input-drop=0.1 --dropout=0.5 --lr=0.005
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='ogbn-arxiv', dropout=0.5, edge_drop=0.0, epochs=1000, gpu=1, input_drop=0.1, labels=False, linear=False, log_every=20, loss='loge', lr=0.005, mask_rate=0.5, model='gcn', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=3, non_interactive_attn=False, norm='batch', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=10, save_pred=False, seed=0, tune='', wd=0)
# Runned 10 times
# Val Accs: [0.7348904325648512, 0.7354609215074331, 0.7372059465082721, 0.7366690157387832, 0.7370045974697137, 0.7370045974697137, 0.7349911070841303, 0.7360985267962012, 0.7356622705459914, 0.7388502969898318]
# Test Accs: [0.7240499557640475, 0.7249346748143118, 0.7232269612986852, 0.7249140999526779, 0.7244820278583627, 0.724605477028167, 0.7215398226446927, 0.7250375491224822, 0.7227125897578339, 0.7277534308581775]
# Average val accuracy: 0.7363837712674922 ± 0.0011504385862895954
# Average test accuracy: 0.7243256589099438 ± 0.001572334103032402
# #Params: 109608

# run.py --gpu=2 --dataset=ogbn-arxiv --epochs=1000 --loss=lce --model=gcn --labels --n-hidden=256 --input-drop=0.1 --dropout=0.5 --lr=0.005
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='ogbn-arxiv', dropout=0.5, edge_drop=0.0, epochs=1000, gpu=2, input_drop=0.1, labels=True, linear=False, log_every=20, loss='lce', lr=0.005, mask_rate=0.5, model='gcn', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=3, non_interactive_attn=False, norm='batch', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=10, save_pred=False, seed=0, tune='', wd=0)
# Runned 10 times
# Val Accs: [0.7383133662203429, 0.7405282056444847, 0.7393872277593208, 0.7402261820866473, 0.7401255075673681, 0.7398570421826236, 0.7397228094902514, 0.7393872277593208, 0.740729554683043, 0.7406288801637639]
# Test Accs: [0.7254284714935292, 0.7278357303047137, 0.7257782441413082, 0.7273830833487644, 0.7252021480155546, 0.7247289261979714, 0.7269510112544493, 0.7270744604242536, 0.7263543402670617, 0.7272596341789601]
# Average val accuracy: 0.7398906003557166 ± 0.0006962006767953424
# Average test accuracy: 0.7263996049626565 ± 0.0010042471723822777
# #Params: 119848

# run.py --gpu=1 --lr=0.005 --loss=lce --labels --n-label-iter=1 --model=gcn --n-hidden=256 --dropout=0.5 --input-drop=0.1 --plot
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='ogbn-arxiv', dropout=0.5, edge_drop=0.0, epochs=2000, gpu=1, input_drop=0.1, labels=True, linear=False, log_every=20, loss='lce', lr=0.005, mask_rate=0.5, model='gcn', n_heads=3, n_hidden=256, n_label_iters=1, n_layers=3, non_interactive_attn=False, norm='batch', norm_adj='symm', optimizer='adam', plot=True, residual=False, runs=10, save_pred=False, seed=0, split='std', tune='', wd=0)
# Runned 10 times
# Val Accs: [0.7413336017987181, 0.7409980200677875, 0.7401255075673681, 0.7404275311252055, 0.7414678344910903, 0.7404946474713917, 0.7407966710292292, 0.7409309037216014, 0.7385482734319944, 0.7403268566059263]
# Test Accs: [0.7278768800279818, 0.727938604612884, 0.7277328559965435, 0.7278974548896159, 0.7293788449272679, 0.7228977635125404, 0.728329526983931, 0.7296257432668766, 0.7282883772606629, 0.7276711314116413]
# Average val accuracy: 0.7405449847310313 ± 0.0007821646377936661
# Average test accuracy: 0.7277637182889946 ± 0.00174411853304139
# #Params: 119848

# run.py --lr=0.005 --loss=lce --model=gcn --linear --labels --plot --input-drop=0.1
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='ogbn-arxiv', dropout=0.5, edge_drop=0.0, epochs=2000, gpu=0, input_drop=0.1, labels=True, linear=True, log_every=20, loss='lce', lr=0.005, mask_rate=0.5, model='gcn', n_heads=3, n_hidden=256, n_label_iters=0, n_layers=3, non_interactive_attn=False, norm='batch', norm_adj='symm', optimizer='adam', plot=True, residual=False, runs=10, save_pred=False, seed=0, split='std', tune='', wd=0)
# Runned 10 times
# Val Accs: [0.746031746031746, 0.744823651800396, 0.7450250008389543, 0.7441524883385349, 0.7460653042048391, 0.7441189301654418, 0.7451256753582335, 0.7450921171851405, 0.7461995368972113, 0.7469378167052586]
# Test Accs: [0.7320947266629632, 0.7321153015245973, 0.7318272534617205, 0.7296051684052425, 0.7321976009711335, 0.7299960907762896, 0.7317655288768183, 0.7308808098265539, 0.7319301277698907, 0.7294817192354381]
# Average val accuracy: 0.7453572267525757 ± 0.0008756625729623333
# Average test accuracy: 0.7311894327510647 ± 0.0010452758715207328
# #Params: 238632

# run.py --gpu=1 --lr=0.005 --loss=lce --model=gcn --linear --labels --plot --input-drop=0.25 --n-label-iters=1
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='ogbn-arxiv', dropout=0.5, edge_drop=0.0, epochs=2000, gpu=1, input_drop=0.25, labels=True, linear=True, log_every=20, loss='lce', lr=0.005, mask_rate=0.5, model='gcn', n_heads=3, n_hidden=256, n_label_iters=1, n_layers=3, non_interactive_attn=False, norm='batch', norm_adj='symm', optimizer='adam', plot=True, residual=False, runs=10, save_pred=False, seed=0, split='std', tune='', wd=0)
# Runned 10 times
# Val Accs: [0.7475083056478405, 0.7467029094936072, 0.7458639551662808, 0.7457968388200946, 0.7464344441088627, 0.7438840229537904, 0.7468371421859794, 0.7471727239169099, 0.7468035840128864, 0.7468707003590724]
# Test Accs: [0.7341727876880028, 0.7327325473736189, 0.7328559965434233, 0.7339876139332963, 0.731662654568648, 0.7307573606567496, 0.7299343661913874, 0.7321564512478653, 0.7328765714050574, 0.7313128819208691]
# Average val accuracy: 0.7463874626665323 ± 0.0009733258711990791
# Average test accuracy: 0.7322449231528918 ± 0.0012857247333008925

# Params: 238632

### GAT

## Cora

# Vanilla GAT

# run.py --dataset=cora --runs=100 --lr=2e-2 --wd=5e-4 --epochs=500 --model=gat --n-layers=2 --n-heads=8 --n-hidden=8 --dropout=0.8 --input-drop=0.8 --edge-drop=0.5 --loss=ce --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='cora', dropout=0.8, edge_drop=0.5, epochs=500, gpu=0, input_drop=0.8, labels=False, linear=False, log_every=20, loss='ce', lr=0.02, mask_rate=1.0, model='gat', n_heads=8, n_hidden=8, n_label_iters=0, n_layers=2, no_attn_dst=False, norm='none', norm_adj='rw', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, wd=0.0005)
# Runned 100 times
# Val Accs: [0.8340000510215759, 0.8340000510215759, 0.8320000171661377, 0.8260000348091125, 0.8320000171661377, 0.8400000333786011, 0.8340000510215759, 0.8360000252723694, 0.8300000429153442, 0.8260000348091125, 0.8340000510215759, 0.8360000252723694, 0.8300000429153442, 0.8280000686645508, 0.8320000171661377, 0.8380000591278076, 0.8320000171661377, 0.8380000591278076, 0.8340000510215759, 0.8240000605583191, 0.8380000591278076, 0.8360000252723694, 0.8240000605583191, 0.8300000429153442, 0.8340000510215759, 0.8320000171661377, 0.8320000171661377, 0.8300000429153442, 0.8220000267028809, 0.8320000171661377, 0.8240000605583191, 0.8240000605583191, 0.8340000510215759, 0.8300000429153442, 0.8280000686645508, 0.8360000252723694, 0.8300000429153442, 0.8380000591278076, 0.8340000510215759, 0.8340000510215759, 0.8280000686645508, 0.8260000348091125, 0.8400000333786011, 0.8300000429153442, 0.8300000429153442, 0.8340000510215759, 0.8340000510215759, 0.8320000171661377, 0.8300000429153442, 0.8300000429153442, 0.8320000171661377, 0.8320000171661377, 0.8320000171661377, 0.8320000171661377, 0.8260000348091125, 0.8340000510215759, 0.8380000591278076, 0.8280000686645508, 0.8260000348091125, 0.8300000429153442, 0.8280000686645508, 0.8300000429153442, 0.8260000348091125, 0.8300000429153442, 0.8340000510215759, 0.8340000510215759, 0.8360000252723694, 0.8360000252723694, 0.8240000605583191, 0.8320000171661377, 0.8320000171661377, 0.8360000252723694, 0.8260000348091125, 0.8320000171661377, 0.8280000686645508, 0.8240000605583191, 0.8380000591278076, 0.8380000591278076, 0.8300000429153442, 0.8300000429153442, 0.8300000429153442, 0.8280000686645508, 0.8340000510215759, 0.8280000686645508, 0.8260000348091125, 0.8320000171661377, 0.8360000252723694, 0.8240000605583191, 0.8300000429153442, 0.8320000171661377, 0.8320000171661377, 0.8400000333786011, 0.8320000171661377, 0.8220000267028809, 0.8320000171661377, 0.8320000171661377, 0.8360000252723694, 0.8320000171661377, 0.8200000524520874, 0.8240000605583191]
# Test Accs: [0.8360000252723694, 0.8470000624656677, 0.8400000333786011, 0.8230000138282776, 0.8350000381469727, 0.8390000462532043, 0.8400000333786011, 0.8360000252723694, 0.8340000510215759, 0.8360000252723694, 0.831000030040741, 0.8410000205039978, 0.8380000591278076, 0.8250000476837158, 0.8400000333786011, 0.8330000638961792, 0.8250000476837158, 0.8280000686645508, 0.8440000414848328, 0.8440000414848328, 0.8400000333786011, 0.8390000462532043, 0.8270000219345093, 0.8470000624656677, 0.8290000557899475, 0.8350000381469727, 0.8290000557899475, 0.8440000414848328, 0.8110000491142273, 0.8350000381469727, 0.831000030040741, 0.8300000429153442, 0.8280000686645508, 0.8350000381469727, 0.8350000381469727, 0.8350000381469727, 0.8290000557899475, 0.8290000557899475, 0.8320000171661377, 0.8370000123977661, 0.8250000476837158, 0.8270000219345093, 0.8390000462532043, 0.8190000653266907, 0.8370000123977661, 0.8510000109672546, 0.8300000429153442, 0.8370000123977661, 0.8390000462532043, 0.8390000462532043, 0.8230000138282776, 0.8380000591278076, 0.8320000171661377, 0.8360000252723694, 0.8290000557899475, 0.8320000171661377, 0.8450000286102295, 0.8290000557899475, 0.8330000638961792, 0.8460000157356262, 0.8340000510215759, 0.8500000238418579, 0.8270000219345093, 0.8380000591278076, 0.8250000476837158, 0.8450000286102295, 0.8410000205039978, 0.8260000348091125, 0.8260000348091125, 0.8370000123977661, 0.8370000123977661, 0.831000030040741, 0.8340000510215759, 0.8230000138282776, 0.8300000429153442, 0.8220000267028809, 0.8450000286102295, 0.8480000495910645, 0.8290000557899475, 0.8340000510215759, 0.8280000686645508, 0.8320000171661377, 0.8350000381469727, 0.8380000591278076, 0.8420000672340393, 0.8350000381469727, 0.8380000591278076, 0.8320000171661377, 0.8280000686645508, 0.8290000557899475, 0.8360000252723694, 0.8270000219345093, 0.8410000205039978, 0.8260000348091125, 0.8540000319480896, 0.8250000476837158, 0.8270000219345093, 0.8330000638961792, 0.8270000219345093, 0.8350000381469727]
# Average val accuracy: 0.8312200403213501 ± 0.00438993964319963
# Average test accuracy: 0.8340800386667252 ± 0.007425198538639505
# #Params: 92373

# GAT+symm.norm.adj.

# run.py --dataset=cora --runs=100 --lr=2e-2 --wd=5e-4 --epochs=500 --model=gat --norm-adj=symm --n-layers=2 --n-heads=8 --n-hidden=8 --dropout=0.8 --input-drop=0.8 --edge-drop=0.5 --loss=ce --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='cora', dropout=0.8, edge_drop=0.5, epochs=500, gpu=0, input_drop=0.8, labels=False, linear=False, log_every=20, loss='ce', lr=0.02, mask_rate=1.0, model='gat', n_heads=8, n_hidden=8, n_label_iters=0, n_layers=2, no_attn_dst=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, wd=0.0005)
# Runned 100 times
# Val Accs: [0.8300000429153442, 0.8280000686645508, 0.8320000171661377, 0.8200000524520874, 0.8220000267028809, 0.8360000252723694, 0.8180000185966492, 0.8220000267028809, 0.8260000348091125, 0.8320000171661377, 0.8340000510215759, 0.8300000429153442, 0.8240000605583191, 0.8240000605583191, 0.8220000267028809, 0.8300000429153442, 0.8260000348091125, 0.8260000348091125, 0.8260000348091125, 0.8280000686645508, 0.8260000348091125, 0.8300000429153442, 0.8220000267028809, 0.8240000605583191, 0.8260000348091125, 0.8260000348091125, 0.8220000267028809, 0.8240000605583191, 0.8260000348091125, 0.8280000686645508, 0.8240000605583191, 0.8260000348091125, 0.8260000348091125, 0.8280000686645508, 0.8340000510215759, 0.8320000171661377, 0.8300000429153442, 0.8260000348091125, 0.8260000348091125, 0.8340000510215759, 0.8280000686645508, 0.8280000686645508, 0.8220000267028809, 0.8180000185966492, 0.8300000429153442, 0.8220000267028809, 0.8260000348091125, 0.8240000605583191, 0.8300000429153442, 0.8260000348091125, 0.8320000171661377, 0.8280000686645508, 0.8340000510215759, 0.8320000171661377, 0.8240000605583191, 0.8380000591278076, 0.8280000686645508, 0.8220000267028809, 0.8260000348091125, 0.8260000348091125, 0.8280000686645508, 0.8300000429153442, 0.8220000267028809, 0.8260000348091125, 0.8240000605583191, 0.8240000605583191, 0.8240000605583191, 0.8260000348091125, 0.8220000267028809, 0.8240000605583191, 0.8220000267028809, 0.8260000348091125, 0.8240000605583191, 0.8240000605583191, 0.8240000605583191, 0.8260000348091125, 0.8240000605583191, 0.8260000348091125, 0.8320000171661377, 0.8260000348091125, 0.8220000267028809, 0.8180000185966492, 0.8280000686645508, 0.8240000605583191, 0.8260000348091125, 0.8220000267028809, 0.8260000348091125, 0.8300000429153442, 0.8280000686645508, 0.8340000510215759, 0.8260000348091125, 0.8260000348091125, 0.8300000429153442, 0.8260000348091125, 0.8200000524520874, 0.8240000605583191, 0.8220000267028809, 0.8360000252723694, 0.8240000605583191, 0.8300000429153442]
# Test Accs: [0.8410000205039978, 0.8420000672340393, 0.8320000171661377, 0.8370000123977661, 0.8250000476837158, 0.8480000495910645, 0.8290000557899475, 0.8390000462532043, 0.8320000171661377, 0.859000027179718, 0.8290000557899475, 0.8420000672340393, 0.8360000252723694, 0.8390000462532043, 0.8380000591278076, 0.8380000591278076, 0.8460000157356262, 0.8260000348091125, 0.8500000238418579, 0.8400000333786011, 0.8300000429153442, 0.8480000495910645, 0.8420000672340393, 0.8450000286102295, 0.8390000462532043, 0.8410000205039978, 0.8300000429153442, 0.8340000510215759, 0.8300000429153442, 0.8360000252723694, 0.843000054359436, 0.8320000171661377, 0.8390000462532043, 0.8250000476837158, 0.8390000462532043, 0.8360000252723694, 0.8360000252723694, 0.8370000123977661, 0.8200000524520874, 0.8200000524520874, 0.8400000333786011, 0.8470000624656677, 0.843000054359436, 0.8280000686645508, 0.8330000638961792, 0.8410000205039978, 0.8320000171661377, 0.831000030040741, 0.8330000638961792, 0.8390000462532043, 0.8290000557899475, 0.8320000171661377, 0.8420000672340393, 0.8390000462532043, 0.831000030040741, 0.8440000414848328, 0.8530000448226929, 0.8380000591278076, 0.8290000557899475, 0.8350000381469727, 0.8410000205039978, 0.8410000205039978, 0.8410000205039978, 0.8340000510215759, 0.8380000591278076, 0.8320000171661377, 0.8440000414848328, 0.8480000495910645, 0.8390000462532043, 0.8460000157356262, 0.843000054359436, 0.8170000314712524, 0.8380000591278076, 0.8330000638961792, 0.843000054359436, 0.8450000286102295, 0.8450000286102295, 0.8410000205039978, 0.8450000286102295, 0.8370000123977661, 0.8220000267028809, 0.8390000462532043, 0.8460000157356262, 0.8380000591278076, 0.8460000157356262, 0.831000030040741, 0.8300000429153442, 0.8490000367164612, 0.8380000591278076, 0.8370000123977661, 0.8370000123977661, 0.831000030040741, 0.8370000123977661, 0.8240000605583191, 0.8380000591278076, 0.8230000138282776, 0.8370000123977661, 0.8390000462532043, 0.8380000591278076, 0.843000054359436]
# Average val accuracy: 0.8264600425958634 ± 0.004048260273544484
# Average test accuracy: 0.8372300392389298 ± 0.00743351098953742
# #Params: 92373

## Citeseer

# Vanilla GAT

# run.py --gpu=1 --dataset=citeseer --runs=100 --lr=2e-2 --wd=2e-3 --epochs=500 --model=gat --n-layers=2 --n-heads=8 --n-hidden=8 --dropout=0.6 --input-drop=0.8 --edge-drop=0.5 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='citeseer', dropout=0.6, edge_drop=0.5, epochs=500, gpu=1, input_drop=0.8, labels=False, linear=False, log_every=20, loss='ce', lr=0.02, mask_rate=1.0, model='gat', n_heads=8, n_hidden=8, n_label_iters=0, n_layers=2, no_attn_dst=False, norm='none', norm_adj='rw', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, tune='', wd=0.002)
# Runned 100 times
# Val Accs: [0.7300000190734863, 0.7200000286102295, 0.7320000529289246, 0.7200000286102295, 0.7240000367164612, 0.7280000448226929, 0.718000054359436, 0.7360000610351562, 0.7240000367164612, 0.7220000624656677, 0.7200000286102295, 0.7320000529289246, 0.7220000624656677, 0.7260000109672546, 0.7320000529289246, 0.7420000433921814, 0.7280000448226929, 0.7240000367164612, 0.7460000514984131, 0.718000054359436, 0.7260000109672546, 0.7300000190734863, 0.7320000529289246, 0.7240000367164612, 0.7320000529289246, 0.7160000205039978, 0.7380000352859497, 0.7280000448226929, 0.7360000610351562, 0.7200000286102295, 0.7160000205039978, 0.7240000367164612, 0.7320000529289246, 0.7200000286102295, 0.718000054359436, 0.7280000448226929, 0.7280000448226929, 0.7260000109672546, 0.7360000610351562, 0.7220000624656677, 0.7280000448226929, 0.7240000367164612, 0.7300000190734863, 0.7200000286102295, 0.734000027179718, 0.7200000286102295, 0.7360000610351562, 0.7300000190734863, 0.7240000367164612, 0.7380000352859497, 0.734000027179718, 0.7300000190734863, 0.7260000109672546, 0.7460000514984131, 0.7380000352859497, 0.7220000624656677, 0.7140000462532043, 0.7300000190734863, 0.7300000190734863, 0.7200000286102295, 0.7220000624656677, 0.7320000529289246, 0.7400000095367432, 0.7200000286102295, 0.734000027179718, 0.7440000176429749, 0.718000054359436, 0.7300000190734863, 0.734000027179718, 0.7360000610351562, 0.7480000257492065, 0.7440000176429749, 0.7320000529289246, 0.7240000367164612, 0.7500000596046448, 0.706000030040741, 0.7260000109672546, 0.7240000367164612, 0.7240000367164612, 0.7280000448226929, 0.7200000286102295, 0.734000027179718, 0.7260000109672546, 0.734000027179718, 0.7260000109672546, 0.734000027179718, 0.7300000190734863, 0.7300000190734863, 0.7260000109672546, 0.7260000109672546, 0.7320000529289246, 0.7200000286102295, 0.7240000367164612, 0.7300000190734863, 0.7300000190734863, 0.7420000433921814, 0.7280000448226929, 0.7380000352859497, 0.7140000462532043, 0.7200000286102295]
# Test Accs: [0.7220000624656677, 0.7280000448226929, 0.7220000624656677, 0.7170000076293945, 0.7200000286102295, 0.7280000448226929, 0.718000054359436, 0.7350000143051147, 0.7190000414848328, 0.7020000219345093, 0.7190000414848328, 0.7240000367164612, 0.7260000109672546, 0.718000054359436, 0.7170000076293945, 0.7240000367164612, 0.7190000414848328, 0.706000030040741, 0.7100000381469727, 0.7190000414848328, 0.6990000605583191, 0.7150000333786011, 0.7070000171661377, 0.7100000381469727, 0.7350000143051147, 0.7270000576972961, 0.7400000095367432, 0.7240000367164612, 0.7120000123977661, 0.7120000123977661, 0.7240000367164612, 0.7280000448226929, 0.7290000319480896, 0.7100000381469727, 0.7080000042915344, 0.7270000576972961, 0.7130000591278076, 0.7250000238418579, 0.7150000333786011, 0.7170000076293945, 0.7230000495910645, 0.7000000476837158, 0.7000000476837158, 0.7080000042915344, 0.734000027179718, 0.7110000252723694, 0.7300000190734863, 0.7280000448226929, 0.7310000061988831, 0.7380000352859497, 0.7260000109672546, 0.7270000576972961, 0.7160000205039978, 0.7210000157356262, 0.7290000319480896, 0.7270000576972961, 0.7240000367164612, 0.7140000462532043, 0.7120000123977661, 0.734000027179718, 0.7010000348091125, 0.7120000123977661, 0.7130000591278076, 0.690000057220459, 0.7250000238418579, 0.734000027179718, 0.7220000624656677, 0.7190000414848328, 0.7300000190734863, 0.7240000367164612, 0.7100000381469727, 0.7070000171661377, 0.7070000171661377, 0.7310000061988831, 0.7230000495910645, 0.6960000395774841, 0.7140000462532043, 0.7200000286102295, 0.706000030040741, 0.706000030040741, 0.7160000205039978, 0.7120000123977661, 0.7090000510215759, 0.7320000529289246, 0.7210000157356262, 0.7270000576972961, 0.7290000319480896, 0.7190000414848328, 0.7230000495910645, 0.718000054359436, 0.7300000190734863, 0.7160000205039978, 0.7140000462532043, 0.7150000333786011, 0.7250000238418579, 0.7230000495910645, 0.7300000190734863, 0.7320000529289246, 0.7150000333786011, 0.7170000076293945]
# Average val accuracy: 0.7283000355958938 ± 0.00803679140952777
# Average test accuracy: 0.7191600334644318 ± 0.00988607007697665
# #Params: 237586

# GAT+symm.norm.adj.

# run.py --gpu=2 --dataset=citeseer --runs=100 --lr=2e-2 --wd=2e-3 --epochs=500 --model=gat --norm-adj=symm --n-layers=2 --n-heads=8 --n-hidden=8 --dropout=0.6 --input-drop=0.8 --edge-drop=0.5 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.0, cpu=False, dataset='citeseer', dropout=0.6, edge_drop=0.5, epochs=500, gpu=2, input_drop=0.8, labels=False, linear=False, log_every=20, loss='ce', lr=0.02, mask_rate=1.0, model='gat', n_heads=8, n_hidden=8, n_label_iters=0, n_layers=2, no_attn_dst=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, tune='', wd=0.002)
# Runned 100 times
# Val Accs: [0.734000027179718, 0.7300000190734863, 0.7300000190734863, 0.7320000529289246, 0.7380000352859497, 0.7240000367164612, 0.7320000529289246, 0.7500000596046448, 0.7380000352859497, 0.7280000448226929, 0.7220000624656677, 0.7280000448226929, 0.7240000367164612, 0.7220000624656677, 0.7280000448226929, 0.7360000610351562, 0.7300000190734863, 0.7360000610351562, 0.7360000610351562, 0.7320000529289246, 0.7360000610351562, 0.734000027179718, 0.7240000367164612, 0.734000027179718, 0.7260000109672546, 0.7280000448226929, 0.734000027179718, 0.7320000529289246, 0.7260000109672546, 0.734000027179718, 0.7320000529289246, 0.7280000448226929, 0.734000027179718, 0.7400000095367432, 0.7280000448226929, 0.7280000448226929, 0.7420000433921814, 0.7300000190734863, 0.7280000448226929, 0.7260000109672546, 0.7300000190734863, 0.718000054359436, 0.7380000352859497, 0.7300000190734863, 0.7380000352859497, 0.7360000610351562, 0.734000027179718, 0.7320000529289246, 0.7220000624656677, 0.7320000529289246, 0.7400000095367432, 0.7280000448226929, 0.7220000624656677, 0.7320000529289246, 0.7300000190734863, 0.7320000529289246, 0.7300000190734863, 0.7360000610351562, 0.7300000190734863, 0.734000027179718, 0.7260000109672546, 0.734000027179718, 0.7320000529289246, 0.7300000190734863, 0.7400000095367432, 0.7320000529289246, 0.7280000448226929, 0.734000027179718, 0.7380000352859497, 0.7280000448226929, 0.734000027179718, 0.7360000610351562, 0.7320000529289246, 0.7320000529289246, 0.734000027179718, 0.718000054359436, 0.7280000448226929, 0.7280000448226929, 0.7280000448226929, 0.7300000190734863, 0.7260000109672546, 0.734000027179718, 0.7260000109672546, 0.7320000529289246, 0.718000054359436, 0.7260000109672546, 0.7440000176429749, 0.734000027179718, 0.7320000529289246, 0.7400000095367432, 0.7300000190734863, 0.7320000529289246, 0.7280000448226929, 0.7260000109672546, 0.7240000367164612, 0.7200000286102295, 0.7300000190734863, 0.7400000095367432, 0.718000054359436, 0.7320000529289246]
# Test Accs: [0.7200000286102295, 0.7260000109672546, 0.7100000381469727, 0.718000054359436, 0.7210000157356262, 0.7270000576972961, 0.7330000400543213, 0.7420000433921814, 0.7350000143051147, 0.7290000319480896, 0.7290000319480896, 0.7070000171661377, 0.7280000448226929, 0.7150000333786011, 0.7290000319480896, 0.7300000190734863, 0.7210000157356262, 0.7160000205039978, 0.706000030040741, 0.7330000400543213, 0.7280000448226929, 0.7360000610351562, 0.7240000367164612, 0.7320000529289246, 0.7200000286102295, 0.7100000381469727, 0.7310000061988831, 0.7290000319480896, 0.718000054359436, 0.7220000624656677, 0.7200000286102295, 0.6980000138282776, 0.7220000624656677, 0.7120000123977661, 0.7360000610351562, 0.7320000529289246, 0.7310000061988831, 0.7190000414848328, 0.7170000076293945, 0.706000030040741, 0.7010000348091125, 0.7130000591278076, 0.7080000042915344, 0.7260000109672546, 0.7210000157356262, 0.734000027179718, 0.7230000495910645, 0.7320000529289246, 0.6990000605583191, 0.7390000224113464, 0.7190000414848328, 0.718000054359436, 0.7220000624656677, 0.7330000400543213, 0.7210000157356262, 0.7170000076293945, 0.7240000367164612, 0.7240000367164612, 0.7390000224113464, 0.734000027179718, 0.690000057220459, 0.706000030040741, 0.7270000576972961, 0.7300000190734863, 0.7140000462532043, 0.7270000576972961, 0.7320000529289246, 0.7320000529289246, 0.7250000238418579, 0.718000054359436, 0.7140000462532043, 0.7090000510215759, 0.6980000138282776, 0.734000027179718, 0.7170000076293945, 0.7200000286102295, 0.7230000495910645, 0.718000054359436, 0.7360000610351562, 0.7140000462532043, 0.7360000610351562, 0.7310000061988831, 0.7160000205039978, 0.7230000495910645, 0.7130000591278076, 0.7280000448226929, 0.7100000381469727, 0.7400000095367432, 0.7200000286102295, 0.7140000462532043, 0.7260000109672546, 0.7320000529289246, 0.7310000061988831, 0.7270000576972961, 0.7320000529289246, 0.7110000252723694, 0.7300000190734863, 0.7330000400543213, 0.7260000109672546, 0.7220000624656677]
# Average val accuracy: 0.7309200370311737 ± 0.005730059083922087
# Average test accuracy: 0.7225000357627869 ± 0.01036677440198382
# #Params: 237586

## Pubmed

# Vanilla GAT

# run.py --gpu=3 --dataset=pubmed --runs=100 --lr=2e-2 --wd=2e-3 --epochs=500 --model=gat --n-layers=2 --n-heads=8 --n-hidden=8 --dropout=0.5 --input-drop=0.8 --attn-drop=0.1 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.1, cpu=False, dataset='pubmed', dropout=0.5, edge_drop=0.0, epochs=500, gpu=3, input_drop=0.8, labels=False, linear=False, log_every=20, loss='ce', lr=0.02, mask_rate=1.0, model='gat', n_heads=8, n_hidden=8, n_label_iters=0, n_layers=2, no_attn_dst=False, norm='none', norm_adj='rw', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, tune='', wd=0.002)
# Runned 100 times
# Val Accs: [0.8180000185966492, 0.8180000185966492, 0.8160000443458557, 0.8180000185966492, 0.8180000185966492, 0.8180000185966492, 0.8160000443458557, 0.8140000104904175, 0.8140000104904175, 0.8180000185966492, 0.8220000267028809, 0.8200000524520874, 0.8160000443458557, 0.8200000524520874, 0.8240000605583191, 0.8140000104904175, 0.8160000443458557, 0.8240000605583191, 0.8180000185966492, 0.8180000185966492, 0.8160000443458557, 0.8100000619888306, 0.8180000185966492, 0.8140000104904175, 0.8140000104904175, 0.812000036239624, 0.8160000443458557, 0.8140000104904175, 0.812000036239624, 0.8160000443458557, 0.812000036239624, 0.8140000104904175, 0.8240000605583191, 0.8200000524520874, 0.8200000524520874, 0.8180000185966492, 0.8180000185966492, 0.8140000104904175, 0.8060000538825989, 0.8220000267028809, 0.8160000443458557, 0.8240000605583191, 0.8180000185966492, 0.8180000185966492, 0.8220000267028809, 0.8160000443458557, 0.8180000185966492, 0.8160000443458557, 0.8160000443458557, 0.8140000104904175, 0.8140000104904175, 0.8200000524520874, 0.8180000185966492, 0.8180000185966492, 0.8200000524520874, 0.8140000104904175, 0.8140000104904175, 0.8160000443458557, 0.8160000443458557, 0.8160000443458557, 0.8240000605583191, 0.8220000267028809, 0.8200000524520874, 0.8200000524520874, 0.8160000443458557, 0.8180000185966492, 0.8240000605583191, 0.8180000185966492, 0.8200000524520874, 0.8200000524520874, 0.8140000104904175, 0.8200000524520874, 0.8200000524520874, 0.8160000443458557, 0.8200000524520874, 0.8160000443458557, 0.8280000686645508, 0.8200000524520874, 0.8160000443458557, 0.812000036239624, 0.8160000443458557, 0.8180000185966492, 0.8200000524520874, 0.8140000104904175, 0.8200000524520874, 0.8220000267028809, 0.8160000443458557, 0.8140000104904175, 0.812000036239624, 0.8220000267028809, 0.8200000524520874, 0.8160000443458557, 0.8200000524520874, 0.8280000686645508, 0.8200000524520874, 0.8140000104904175, 0.8140000104904175, 0.8240000605583191, 0.8200000524520874, 0.8160000443458557]
# Test Accs: [0.7760000228881836, 0.7910000085830688, 0.7670000195503235, 0.7790000438690186, 0.7850000262260437, 0.781000018119812, 0.7860000133514404, 0.7900000214576721, 0.8030000329017639, 0.7870000600814819, 0.7830000519752502, 0.7850000262260437, 0.784000039100647, 0.7780000567436218, 0.7820000648498535, 0.7760000228881836, 0.7800000309944153, 0.7900000214576721, 0.7800000309944153, 0.7860000133514404, 0.7800000309944153, 0.7800000309944153, 0.7870000600814819, 0.7880000472068787, 0.784000039100647, 0.7830000519752502, 0.7830000519752502, 0.7890000343322754, 0.7860000133514404, 0.7940000295639038, 0.7820000648498535, 0.7870000600814819, 0.7870000600814819, 0.7870000600814819, 0.7790000438690186, 0.7870000600814819, 0.7760000228881836, 0.7760000228881836, 0.7880000472068787, 0.7880000472068787, 0.7770000100135803, 0.7930000424385071, 0.784000039100647, 0.7750000357627869, 0.7850000262260437, 0.7790000438690186, 0.7820000648498535, 0.7850000262260437, 0.7830000519752502, 0.7750000357627869, 0.7880000472068787, 0.7770000100135803, 0.7790000438690186, 0.7860000133514404, 0.7880000472068787, 0.7780000567436218, 0.781000018119812, 0.7940000295639038, 0.7890000343322754, 0.7800000309944153, 0.7900000214576721, 0.7940000295639038, 0.7820000648498535, 0.7900000214576721, 0.7880000472068787, 0.800000011920929, 0.7890000343322754, 0.784000039100647, 0.7850000262260437, 0.784000039100647, 0.781000018119812, 0.7760000228881836, 0.7920000553131104, 0.784000039100647, 0.7780000567436218, 0.7850000262260437, 0.796000063419342, 0.7750000357627869, 0.7700000405311584, 0.7910000085830688, 0.8010000586509705, 0.7740000486373901, 0.7850000262260437, 0.7860000133514404, 0.7930000424385071, 0.7850000262260437, 0.7780000567436218, 0.7710000276565552, 0.7850000262260437, 0.7830000519752502, 0.7860000133514404, 0.781000018119812, 0.7900000214576721, 0.7900000214576721, 0.7880000472068787, 0.7860000133514404, 0.7920000553131104, 0.7780000567436218, 0.7820000648498535, 0.7850000262260437]
# Average val accuracy: 0.8176400357484818 ± 0.0036919454177915976
# Average test accuracy: 0.7843100363016129 ± 0.006371334478914139
# #Params: 32393

# GAT+symm.norm.adj.

# run.py --gpu=4 --dataset=pubmed --runs=100 --lr=2e-2 --wd=2e-3 --epochs=500 --model=gat --norm-adj=symm --n-layers=2 --n-heads=8 --n-hidden=8 --dropout=0.5 --input-drop=0.8 --attn-drop=0.1 --mask-rate=1 --norm=none
# Namespace(activation='relu', attn_drop=0.1, cpu=False, dataset='pubmed', dropout=0.5, edge_drop=0.0, epochs=500, gpu=4, input_drop=0.8, labels=False, linear=False, log_every=20, loss='ce', lr=0.02, mask_rate=1.0, model='gat', n_heads=8, n_hidden=8, n_label_iters=0, n_layers=2, no_attn_dst=False, norm='none', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=100, save_pred=False, seed=0, tune='', wd=0.002)
# Runned 100 times
# Val Accs: [0.8240000605583191, 0.8180000185966492, 0.8280000686645508, 0.8200000524520874, 0.8280000686645508, 0.8260000348091125, 0.8240000605583191, 0.8220000267028809, 0.8180000185966492, 0.8180000185966492, 0.8180000185966492, 0.8240000605583191, 0.8260000348091125, 0.8200000524520874, 0.8340000510215759, 0.8200000524520874, 0.8240000605583191, 0.8260000348091125, 0.8240000605583191, 0.8220000267028809, 0.8240000605583191, 0.8220000267028809, 0.8220000267028809, 0.8220000267028809, 0.8180000185966492, 0.8220000267028809, 0.8240000605583191, 0.8180000185966492, 0.8260000348091125, 0.8280000686645508, 0.8220000267028809, 0.8200000524520874, 0.8140000104904175, 0.8180000185966492, 0.8240000605583191, 0.8160000443458557, 0.8220000267028809, 0.8140000104904175, 0.8180000185966492, 0.8260000348091125, 0.8240000605583191, 0.8180000185966492, 0.8220000267028809, 0.8260000348091125, 0.8280000686645508, 0.8240000605583191, 0.8200000524520874, 0.8260000348091125, 0.8300000429153442, 0.8260000348091125, 0.8280000686645508, 0.8240000605583191, 0.8140000104904175, 0.8180000185966492, 0.8260000348091125, 0.8300000429153442, 0.8320000171661377, 0.8220000267028809, 0.8200000524520874, 0.8200000524520874, 0.8240000605583191, 0.8240000605583191, 0.8220000267028809, 0.8260000348091125, 0.8240000605583191, 0.8200000524520874, 0.8180000185966492, 0.8280000686645508, 0.8300000429153442, 0.8280000686645508, 0.8220000267028809, 0.8300000429153442, 0.8180000185966492, 0.8200000524520874, 0.8240000605583191, 0.8240000605583191, 0.8300000429153442, 0.8240000605583191, 0.8200000524520874, 0.8180000185966492, 0.8260000348091125, 0.8320000171661377, 0.8220000267028809, 0.8260000348091125, 0.8260000348091125, 0.8240000605583191, 0.8180000185966492, 0.8220000267028809, 0.8180000185966492, 0.8260000348091125, 0.8260000348091125, 0.8320000171661377, 0.8280000686645508, 0.8220000267028809, 0.8180000185966492, 0.8220000267028809, 0.8220000267028809, 0.8180000185966492, 0.8240000605583191, 0.8220000267028809]
# Test Accs: [0.7800000309944153, 0.7900000214576721, 0.7940000295639038, 0.781000018119812, 0.7930000424385071, 0.7790000438690186, 0.7930000424385071, 0.796000063419342, 0.7820000648498535, 0.7970000505447388, 0.7910000085830688, 0.7910000085830688, 0.7870000600814819, 0.796000063419342, 0.7910000085830688, 0.8030000329017639, 0.7830000519752502, 0.7900000214576721, 0.7940000295639038, 0.7890000343322754, 0.7980000376701355, 0.784000039100647, 0.8040000200271606, 0.7880000472068787, 0.781000018119812, 0.784000039100647, 0.7880000472068787, 0.7870000600814819, 0.7830000519752502, 0.784000039100647, 0.7900000214576721, 0.781000018119812, 0.7870000600814819, 0.7940000295639038, 0.7870000600814819, 0.7930000424385071, 0.7890000343322754, 0.7890000343322754, 0.7870000600814819, 0.7790000438690186, 0.7770000100135803, 0.7870000600814819, 0.7860000133514404, 0.7920000553131104, 0.7850000262260437, 0.8010000586509705, 0.7880000472068787, 0.7880000472068787, 0.7820000648498535, 0.7880000472068787, 0.7910000085830688, 0.781000018119812, 0.7850000262260437, 0.7930000424385071, 0.7880000472068787, 0.7890000343322754, 0.7850000262260437, 0.7870000600814819, 0.7870000600814819, 0.7870000600814819, 0.7770000100135803, 0.7830000519752502, 0.784000039100647, 0.7870000600814819, 0.7900000214576721, 0.7800000309944153, 0.781000018119812, 0.7920000553131104, 0.7860000133514404, 0.7900000214576721, 0.7890000343322754, 0.7820000648498535, 0.781000018119812, 0.7910000085830688, 0.784000039100647, 0.7890000343322754, 0.784000039100647, 0.7770000100135803, 0.7900000214576721, 0.7930000424385071, 0.7880000472068787, 0.7890000343322754, 0.7930000424385071, 0.7940000295639038, 0.7890000343322754, 0.7850000262260437, 0.784000039100647, 0.784000039100647, 0.781000018119812, 0.7890000343322754, 0.7860000133514404, 0.7940000295639038, 0.7900000214576721, 0.7820000648498535, 0.7790000438690186, 0.7930000424385071, 0.7860000133514404, 0.7900000214576721, 0.7890000343322754, 0.7890000343322754]
# Average val accuracy: 0.8231000393629074 ± 0.004227300245204239
# Average test accuracy: 0.7876800364255905 ± 0.005394220508257293
# #Params: 32393

## Reddit

# Vanilla GAT

# run.py --gpu=7 --dataset=reddit --model=gat --linear --n-layers=3 --n-heads=4 --n-hidden=64 --attn-drop=0.1 --dropout=0.6 --input-drop=0.1
# Namespace(activation='relu', attn_drop=0.1, cpu=False, dataset='reddit', dropout=0.6, edge_drop=0.0, epochs=2000, gpu=7, input_drop=0.1, labels=False, linear=True, log_every=20, loss='CE', lr=0.002, mask_rate=0.5, model='gat', n_heads=4, n_hidden=64, n_label_iters=0, n_layers=3, no_attn_dst=False, norm='batch', norm_adj='rw', optimizer='adam', plot=False, residual=False, runs=10, save_pred=False, seed=0, wd=0)
# Runned 10 times
# Val Accs: [0.9706684350967407, 0.970920205116272, 0.9707943201065063, 0.9708782434463501, 0.9707523584365845, 0.9707943201065063, 0.9703747034072876, 0.9704166650772095, 0.9700809717178345, 0.9702068567276001]
# Test Accs: [0.970127284526825, 0.9696605205535889, 0.9700375199317932, 0.9691937565803528, 0.9692296385765076, 0.970378577709198, 0.9702170491218567, 0.9696425199508667, 0.9694989323616028, 0.9692296385765076]
# Average val accuracy: 0.9705887079238892 ± 0.0002811431884765625
# Average test accuracy: 0.96972154378891 ± 0.00041996557477400894
# #Params: 462459

# GAT+symm.norm.adj.

# run.py --gpu=2 --dataset=reddit --model=gat --linear --norm-adj=symm --n-layers=3 --n-heads=4 --n-hidden=64 --attn-drop=0.1 --dropout=0.6 --input-drop=0.1
# Namespace(activation='relu', attn_drop=0.1, cpu=False, dataset='reddit', dropout=0.6, edge_drop=0.0, epochs=2000, gpu=2, input_drop=0.1, labels=False, linear=True, log_every=20, loss='CE', lr=0.002, mask_rate=0.5, model='gat', n_heads=4, n_hidden=64, n_label_iters=0, n_layers=3, no_attn_dst=False, norm='batch', norm_adj='symm', optimizer='adam', plot=False, residual=False, runs=10, save_pred=False, seed=0, wd=0)
# Runned 10 times
# Val Accs: [0.9725987315177917, 0.9715916514396667, 0.9729763865470886, 0.9725567698478699, 0.9720532298088074, 0.9720532298088074, 0.9716755747795105, 0.9724728465080261, 0.971255898475647, 0.9728924632072449]
# Test Accs: [0.9711325764656067, 0.9696784615516663, 0.9711146354675293, 0.9706299304962158, 0.9707555770874023, 0.9706299304962158, 0.9705042839050293, 0.9711325764656067, 0.9698400497436523, 0.9710248708724976]
# Average val accuracy: 0.972212678194046 ± 0.0005496281662627172
# Average test accuracy: 0.9706442892551422 ± 0.0004954735007051695
# #Params: 462459

## ogbn-arxiv

# Vanilla GAT+Logistic loss

# run.py --gpu=1 --optimizer=rmsprop --lr=0.002 --loss=logit --labels --mask-rate=0.5 --model=gat --linear --n-heads=3 --n-hidden=250 --dropout=0.75 --input-drop=0.25 --attn-drop=0.1
# Runned 10 times
# Val Accs: [0.7419376489143931, 0.7401255075673681, 0.7416020671834626, 0.739521460451693, 0.7392529950669485, 0.7429443941071848, 0.7417027417027418, 0.7381455753548777, 0.740729554683043, 0.738917413336018]
# Test Accs: [0.7337201407320536, 0.7318066786000864, 0.7319918523547929, 0.7303870131473366, 0.7307367857951155, 0.7338641647634919, 0.7303664382857026, 0.729461144373804, 0.7317861037384523, 0.724255704380388]
# Average val accuracy: 0.7404879358367731 ± 0.0014647517890811804
# Average test accuracy: 0.7308376026171224 ± 0.002574711089192581
# Number of params: 1383120
# #Params: 1383120

# Vanilla GAT+Savage loss

# run.py --gpu=1 --optimizer=rmsprop --lr=0.002 --loss=savage --labels --mask-rate=0.5 --model=gat --linear --n-heads=3 --n-hidden=250 --dropout=0.75 --input-drop=0.25 --attn-drop=0.1
# Namespace(activation='relu', attn_drop=0.1, cpu=False, dataset='ogbn-arxiv', dropout=0.75, edge_drop=0.0, epochs=2000, gpu=1, input_drop=0.25, labels=True, linear=True, log_every=20, loss='savage', lr=0.002, mask_rate=0.5, model='gat', n_heads=3, n_hidden=250, n_label_iters=0, n_layers=3, n_prop=7, non_interactive_attn=False, norm='batch', norm_adj='rw', optimizer='rmsprop', plot=False, residual=False, runs=10, save_pred=False, seed=0, split='std', tune='', wd=0)
# Runned 10 times
# Val Accs: [0.6981777912010471, 0.7137152253431324, 0.7135809926507601, 0.6944192758146247, 0.6850565455216618, 0.7089499647639183, 0.7002248397597235, 0.7123057820732239, 0.6991174200476526, 0.707607637840196]
# Test Accs: [0.6951834248914676, 0.7039894656708434, 0.7044215377651585, 0.6811719441186758, 0.6737238442071477, 0.7041334897022817, 0.6945661790424459, 0.7036396930230644, 0.6969528629919963, 0.7006974878093944]
# Average val accuracy: 0.703315547501594 ± 0.009003520255058533
# Average test accuracy: 0.6958479929222476 ± 0.010004182614870205
# #Params: 1441580

# Vanilla GAT+Loge loss

# run.py --gpu=1 --optimizer=rmsprop --lr=0.002 --loss=loge --labels --mask-rate=0.5 --model=gat --linear --n-heads=3 --n-hidden=250 --dropout=0.75 --input-drop=0.25 --attn-drop=0.1

# GAT+symm.norm.adj.+Logistic loss

# run.py --gpu=1 --optimizer=rmsprop --lr=0.002 --loss=logit --labels --mask-rate=0.5 --model=gat --linear --n-heads=3 --n-hidden=250 --dropout=0.75 --input-drop=0.25 --attn-drop=0.1 --norm-adj=symm

# GAT+symm.norm.adj.+Savage loss

# run.py --gpu=1 --optimizer=rmsprop --lr=0.002 --loss=savage --labels --mask-rate=0.5 --model=gat --linear --n-heads=3 --n-hidden=250 --dropout=0.75 --input-drop=0.25 --attn-drop=0.1 --norm-adj=symm

# GAT+symm.norm.adj.+Loge loss

# run.py --gpu=1 --optimizer=rmsprop --lr=0.002 --loss=loge --labels --mask-rate=0.5 --model=gat --linear --n-heads=3 --n-hidden=250 --dropout=0.75 --input-drop=0.25 --attn-drop=0.1 --norm-adj=symm
