import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from .protein_cnn import DeepProteinCNN

class GAT_GCN_Deep(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=27,
                 embed_dim=128, output_dim=128, dropout=0.1):
        super(GAT_GCN_Deep, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.protein_encoder = DeepProteinCNN(num_features_xt, embed_dim, output_dim, dropout)

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        xt = self.protein_encoder(target)

        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
