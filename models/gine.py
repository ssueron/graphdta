import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINEConv, global_add_pool

from .protein_cnn_simple import SimpleProteinCNN

class GINENet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_edge_features=7, output_dim=128, dropout=0.2, protein_encoder=None):
        super(GINENet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        self.edge_encoder = Linear(num_edge_features, dim)

        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINEConv(nn1, edge_dim=dim)
        self.bn1 = BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINEConv(nn2, edge_dim=dim)
        self.bn2 = BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINEConv(nn3, edge_dim=dim)
        self.bn3 = BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINEConv(nn4, edge_dim=dim)
        self.bn4 = BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINEConv(nn5, edge_dim=dim)
        self.bn5 = BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

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
            edge_attr = self.edge_encoder(data.edge_attr)
        else:
            edge_attr = self.edge_encoder(torch.ones((edge_index.size(1), 7), dtype=torch.float, device=x.device))

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index, edge_attr))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

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
