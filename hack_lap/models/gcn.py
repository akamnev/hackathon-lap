import torch
import torch.nn as nn
import torch_geometric.nn as gnn
# from torch_geometric.nn import GCNConv, global_mean_pool
from torch_sparse import SparseTensor
from ..utils import BondType, ATOM2IDX
from .dropout import VariationalNormalEpanechnikovDropout


class GCLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            normalize=False,
            bias=False
    ):
        super().__init__()
        self.dropout = VariationalNormalEpanechnikovDropout(
            input_size=hidden_size
        )
        self.conv_i = gnn.GCNConv(
            hidden_size,
            hidden_size,
            add_self_loops=False,
            normalize=normalize,
            bias=bias
        )
        self.conv_ii = gnn.GCNConv(
            hidden_size,
            hidden_size,
            add_self_loops=False,
            normalize=normalize,
            bias=bias
        )
        self.conv_iii = gnn.GCNConv(
            hidden_size,
            hidden_size,
            add_self_loops=False,
            normalize=normalize,
            bias=bias
        )
        self.conv_a = gnn.GCNConv(
            hidden_size,
            hidden_size,
            add_self_loops=False,
            normalize=normalize,
            bias=bias
        )

    def forward(
            self,
            hidden_states,
            adj_i,
            adj_ii,
            adj_iii,
            adj_a
    ):
        hidden_states = self.dropout(hidden_states)
        output = 0.0
        if adj_i.nnz() > 0:
            output = output + self.conv_i(hidden_states, adj_i)
        if adj_ii.nnz() > 0:
            output = output + self.conv_ii(hidden_states, adj_ii)
        if adj_iii.nnz() > 0:
            output = output + self.conv_iii(hidden_states, adj_iii)
        if adj_a.nnz() > 0:
            output = output + self.conv_a(hidden_states, adj_a)
        return output


class FeedForward(nn.Module):
    def __init__(
            self,
            hidden_size,
            intermediate_size
    ):
        super().__init__()
        self.dense_input = nn.Linear(hidden_size, intermediate_size)
        self.dropout = VariationalNormalEpanechnikovDropout(
            input_size=intermediate_size
        )
        self.act_fn = nn.LeakyReLU(negative_slope=0.01)
        self.dense_output = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        intermediate_states = self.dense_input(hidden_states)
        intermediate_states = self.dropout(intermediate_states)
        intermediate_states = self.act_fn(intermediate_states)
        output_states = self.dense_output(intermediate_states)
        return output_states


class EncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            intermediate_size,
            normalize=False,
            bias=False,
            layer_norm_eps=1e-6
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            hidden_size,
            eps=layer_norm_eps
        )
        self.gcn = GCLayer(
            hidden_size=hidden_size,
            normalize=normalize,
            bias=bias
        )
        self.feedforward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )

    def forward(
            self,
            hidden_states,
            adj_i,
            adj_ii,
            adj_iii,
            adj_a
    ):
        hidden_states = self.layer_norm(hidden_states)
        out_gcn = self.gcn(
            hidden_states,
            adj_i,
            adj_ii,
            adj_iii,
            adj_a
        )
        hidden_states = hidden_states + out_gcn
        out_ff = self.feedforward(hidden_states)
        hidden_states = hidden_states + out_ff
        return hidden_states


class Encoder(nn.Module):
    def __init__(
            self,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            normalize=False,
            bias=False,
            layer_norm_eps=1e-6
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            EncoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                normalize=normalize,
                bias=bias,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_hidden_layers)
        ])

    def forward(
            self,
            hidden_states,
            adj_i,
            adj_ii,
            adj_iii,
            adj_a,
            output_hidden_states=False
    ):
        all_hidden_states = []

        for layer_module in self.layer:

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states = layer_module(
                hidden_states,
                adj_i,
                adj_ii,
                adj_iii,
                adj_a
            )
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states


class HeadLogReg(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, 1)
        # self.act = nn.Sigmoid()

    def forward(self, hidden_states):
        output = self.lin(hidden_states)
        # output = self.act(output)
        return output


class GCNModel(nn.Module):
    def __init__(
            self,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            pooling='mean',
            normalize=False,
            bias=False,
            layer_norm_eps=1e-6
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=max(ATOM2IDX.values()) + 1,
            embedding_dim=hidden_size,
            padding_idx=0
        )
        self.encoder = Encoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            normalize=normalize,
            bias=bias,
            layer_norm_eps=layer_norm_eps
        )
        if pooling == 'mean':
            self.pooling = gnn.global_mean_pool
        elif pooling == 'max':
            self.pooling = gnn.global_max_pool
        elif pooling == 'add':
            self.pooling = gnn.global_add_pool
        elif pooling == 'sort':
            self.pooling = gnn.global_sort_pool
            raise NotImplementedError
        else:
            raise ValueError(pooling)
        self.head = HeadLogReg(hidden_size=hidden_size)

    def forward(
            self,
            data,
            output_hidden_states=False
    ):
        edge_index = data.edge_index
        device = edge_index.device
        m_i = (data.edge_attr == BondType.SINGLE).view(-1)
        adj_i = SparseTensor(
            row=edge_index[0, m_i],
            col=edge_index[1, m_i],
            value=torch.ones((torch.sum(m_i), ), device=device),
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )
        m_ii = (data.edge_attr == BondType.DOUBLE).view(-1)
        adj_ii = SparseTensor(
            row=edge_index[0, m_ii],
            col=edge_index[1, m_ii],
            value=torch.ones((torch.sum(m_ii), ), device=device),
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )
        m_iii = (data.edge_attr == BondType.TRIPLE).view(-1)
        adj_iii = SparseTensor(
            row=edge_index[0, m_iii],
            col=edge_index[1, m_iii],
            value=torch.ones((torch.sum(m_iii), ), device=device),
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )
        m_a = (data.edge_attr == BondType.AROMATIC).view(-1)
        adj_a = SparseTensor(
            row=edge_index[0, m_a],
            col=edge_index[1, m_a],
            value=torch.ones((torch.sum(m_a), ), device=device),
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )

        hidden_states = self.embedding(data.x.view(-1))
        output = self.encoder(
            hidden_states,
            adj_i,
            adj_ii,
            adj_iii,
            adj_a,
            output_hidden_states=output_hidden_states
        )
        hidden_states = output[0]
        vec = self.pooling(hidden_states, data.batch)
        prop = self.head(vec)
        output = (prop, vec, ) + output
        return output

