import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool

from .protein_cnn_simple import SimpleProteinCNN

class AttentiveFPConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=0, dropout=0.0):
        super(AttentiveFPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        self.W_node = nn.Linear(in_channels, out_channels)
        self.W_neigh = nn.Linear(in_channels, out_channels)
        att_input_dim = in_channels + edge_dim if edge_dim > 0 else in_channels
        self.W_att = nn.Linear(att_input_dim, out_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index

        if edge_attr is not None:
            att_input = torch.cat([x[col], edge_attr], dim=-1)
        else:
            att_input = x[col]

        alpha = torch.sigmoid(self.W_att(att_input))

        neigh_features = x[col] * alpha

        aggr = torch.zeros_like(x)
        aggr.index_add_(0, row, neigh_features)

        out = torch.tanh(self.W_node(x) + self.W_neigh(aggr))
        out = self.dropout(out)

        return out

class AttentiveFPNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_edge_features=6, output_dim=128, dropout=0.2, protein_encoder=None):
        super(AttentiveFPNet, self).__init__()

        dim = 64
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.node_init = nn.Linear(num_features_xd, dim)

        self.conv1 = AttentiveFPConv(dim, dim, edge_dim=num_edge_features, dropout=dropout)
        self.conv2 = AttentiveFPConv(dim, dim, edge_dim=num_edge_features, dropout=dropout)
        self.conv3 = AttentiveFPConv(dim, dim, edge_dim=num_edge_features, dropout=dropout)

        self.W_super = nn.Linear(dim, dim)
        self.W_output = nn.Linear(dim, dim)

        self.fc1_xd = nn.Linear(dim, output_dim)

        if protein_encoder is None:
            protein_encoder = SimpleProteinCNN(output_dim=output_dim, dropout=dropout)
        self.protein_encoder = protein_encoder

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if hasattr(data, 'raw_sequence') and isinstance(self.protein_encoder.__class__.__name__, str) and 'ESM2' in self.protein_encoder.__class__.__name__:
            target = data.raw_sequence
        else:
            target = data.target

        x = self.node_init(x)

        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)

        super_node = global_mean_pool(x, batch)

        att_weights = torch.sigmoid(self.W_super(super_node[batch]) * self.W_output(x))
        x_weighted = x * att_weights

        x = global_add_pool(x_weighted, batch)
        x = F.relu(self.fc1_xd(x))
        x = self.dropout(x)

        xt = self.protein_encoder(target)

        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
