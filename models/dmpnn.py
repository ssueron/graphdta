import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from .protein_cnn_simple import SimpleProteinCNN

class DMPNNConv(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(DMPNNConv, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        self.W_msg = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.W_node = nn.Linear(node_dim + hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_hidden):
        row, col = edge_index

        messages = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)

        for i in range(edge_index.size(1)):
            src, tgt = row[i], col[i]

            reverse_edges = (row == tgt) & (col == src)
            incoming_mask = (row == tgt) & ~reverse_edges

            incoming = edge_hidden[incoming_mask]
            msg_agg = incoming.sum(dim=0) if incoming.size(0) > 0 else torch.zeros(self.hidden_dim, device=x.device)

            msg_input = torch.cat([x[src], msg_agg], dim=-1)
            messages[i] = F.relu(self.W_msg(msg_input))

        node_messages = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        node_messages.index_add_(0, col, messages)

        node_features = torch.cat([x, node_messages], dim=-1)
        x_out = F.relu(self.W_node(node_features))

        return x_out, messages

class DMPNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_edge_features=6, output_dim=128, dropout=0.2, protein_encoder=None):
        super(DMPNNNet, self).__init__()

        hidden_dim = 64
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.edge_init = nn.Linear(num_edge_features, hidden_dim)

        self.conv1 = DMPNNConv(num_features_xd, hidden_dim, hidden_dim)
        self.conv2 = DMPNNConv(hidden_dim, hidden_dim, hidden_dim)
        self.conv3 = DMPNNConv(hidden_dim, hidden_dim, hidden_dim)

        self.fc1_xd = nn.Linear(hidden_dim, output_dim)

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

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_hidden = self.edge_init(data.edge_attr)
        else:
            edge_attr = torch.ones((edge_index.size(1), 6), dtype=torch.float, device=x.device)
            edge_hidden = self.edge_init(edge_attr)

        x, edge_hidden = self.conv1(x, edge_index, edge_hidden)
        x = self.dropout(x)
        x, edge_hidden = self.conv2(x, edge_index, edge_hidden)
        x = self.dropout(x)
        x, edge_hidden = self.conv3(x, edge_index, edge_hidden)
        x = self.dropout(x)

        x = global_add_pool(x, batch)
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
