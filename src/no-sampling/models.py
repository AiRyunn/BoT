import math

import dgl.nn.pytorch as dglnn
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.nn import init

torch.set_printoptions(precision=20)


class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_layers,
        n_hidden,
        activation,
        norm="none",
        dropout=0.0,
        input_drop=0.0,
        residual=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.linears = nn.ModuleList()
        if norm != "none:":
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes

            self.linears.append(nn.Linear(in_hidden, out_hidden))

            if i < n_layers - 1:
                if norm == "batch":
                    self.norms.append(nn.BatchNorm1d(out_hidden))

        self.activation = activation
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, h):
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):
            h = self.linears[i](h)

            if i < self.n_layers - 1:
                if self.residual and h_last is not None:
                    h += h_last
                h_last = h

                if self.norms:
                    h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        return h


# pylint: disable=W0235
class GraphConv(nn.Module):
    r"""

    Description
    -----------
    Graph convolution was introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and mathematically is defined as follows:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`c_{ij}` is the product of the square root of node degrees
    (i.e.,  :math:`c_{ij} = \sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`),
    and :math:`\sigma` is an activation function.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zere-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zere-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GraphConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> conv = GraphConv(10, 2, norm='both', weight=True, bias=True)
    >>> res = conv(g, feat)
    >>> print(res)
    tensor([[ 1.3326, -0.2797],
            [ 1.4673, -0.3080],
            [ 1.3326, -0.2797],
            [ 1.6871, -0.3541],
            [ 1.7711, -0.3717],
            [ 1.0375, -0.2178]], grad_fn=<AddBackward0>)
    >>> # allow_zero_in_degree example
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> conv = GraphConv(10, 2, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
    >>> res = conv(g, feat)
    >>> print(res)
    tensor([[-0.2473, -0.4631],
            [-0.3497, -0.6549],
            [-0.3497, -0.6549],
            [-0.4221, -0.7905],
            [-0.3497, -0.6549],
            [ 0.0000,  0.0000]], grad_fn=<AddBackward0>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_fea = th.rand(2, 5)
    >>> v_fea = th.rand(4, 5)
    >>> conv = GraphConv(5, 2, norm='both', weight=True, bias=True)
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    tensor([[-0.2994,  0.6106],
            [-0.4482,  0.5540],
            [-0.5287,  0.8235],
            [-0.2994,  0.6106]], grad_fn=<AddBackward0>)
    """

    def __init__(
        self, in_feats, out_feats, norm="both", weight=True, bias=True, activation=None, allow_zero_in_degree=False
    ):
        super(GraphConv, self).__init__()
        if norm not in ("none", "both", "right"):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both" or "right".' ' But got "{}".'.format(norm)
            )
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

            Case 2:
            External weight is provided while at the same time the module
            has defined its own weight parameter.

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            # print("111:", feat.sum())
            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == "both":
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata["h"] = feat_src
                # print("11:", graph.srcdata["h"].sum())
                graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
                # print("22:", graph.srcdata["h"].sum())
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                # print("1:", graph.srcdata["h"].sum())
                graph.update_all(fn.copy_src(src="h", out="m"), fn.sum(msg="m", out="h"))
                # print("2:", graph.srcdata["h"].sum())
                rst = graph.dstdata["h"]
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm != "none":
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        linear=True,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
        non_interactive_attn=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if non_interactive_attn:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if linear:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=graph.device)
                # print(graph.number_of_edges(), perm)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                # print(e.shape, eids.shape)
                a = torch.zeros_like(e)
                # print(e[eids])
                # print("============", a)
                a[eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
                # print(a)
                graph.edata.update({"a": a})
                # graph.edata["a"] = torch.zeros_like(e)
                # graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
                # print(graph.edata["a"])
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GCN(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        activation,
        norm="none",
        norm_adj="symm",
        dropout=0.0,
        input_drop=0,
        residual=False,
        use_linear=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear
        self.residual = residual

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        if norm != "none:":
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = norm == "none" or i == n_layers - 1
            norm_adj_ = "both" if norm_adj == "symm" else "right"

            self.convs.append(GraphConv(in_hidden, out_hidden, norm_adj_, bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                if norm == "batch":
                    self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                if self.residual and h_last is not None:
                    h += h_last
                h_last = h

                if self.norms:
                    h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h


class GAT(nn.Module):
    def __init__(
        self,
        dim_node,
        dim_edge,
        dim_output,
        n_hidden,
        n_layers,
        n_heads,
        activation,
        norm="none",
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        non_interactive_attn=False,
        use_symmetric_norm=False,
        linear=False,
        residual=False,
    ):
        super().__init__()
        self.n_node_feats = dim_node
        self.n_hidden = n_hidden
        self.n_classes = dim_output
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        if norm != "none:":
            self.norms = nn.ModuleList()
        else:
            self.norms = None
        self.biases = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else dim_node
            out_hidden = n_hidden if i < n_layers - 1 else dim_output
            num_heads = n_heads if i < n_layers - 1 else 1

            self.convs.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    non_interactive_attn=non_interactive_attn,
                    use_symmetric_norm=use_symmetric_norm,
                    linear=linear,
                )
            )

            if i < n_layers - 1:
                if norm == "batch":
                    self.norms.append(nn.BatchNorm1d(num_heads * out_hidden))
                elif norm == "none":
                    self.biases.append(ElementWiseLinear(num_heads * out_hidden, weight=False, bias=True))
            else:
                self.biases.append(ElementWiseLinear(out_hidden, weight=False, bias=True))

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.residual = residual

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            h = conv

            if i < self.n_layers - 1:
                if self.residual and h_last is not None:
                    h += h_last
                h_last = h

                h = h.flatten(1)
                if self.norms:
                    h = self.norms[i](h)
                else:
                    h = self.biases[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = h.mean(1)
        h = self.biases[-1](h)

        return h
