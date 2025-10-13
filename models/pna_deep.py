import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_mean_pool
from .protein_cnn import DeepProteinCNN

class PNANet_Deep(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=27,
                 embed_dim=128, output_dim=128, dropout=0.1, deg=None, protein_encoder=None):
        super(PNANet_Deep, self).__init__()

        if deg is None:
            raise ValueError("PNA requires degree histogram")

        aggregators = ['mean', 'max', 'min', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        pna_hidden = 64

        self.conv1 = PNAConv(num_features_xd, pna_hidden, aggregators=aggregators,
                            scalers=scalers, deg=deg, towers=1, pre_layers=1, post_layers=1)
        self.bn1 = nn.BatchNorm1d(pna_hidden)

        self.conv2 = PNAConv(pna_hidden, pna_hidden, aggregators=aggregators,
                            scalers=scalers, deg=deg, towers=1, pre_layers=1, post_layers=1)
        self.bn2 = nn.BatchNorm1d(pna_hidden)

        self.conv3 = PNAConv(pna_hidden, pna_hidden, aggregators=aggregators,
                            scalers=scalers, deg=deg, towers=1, pre_layers=1, post_layers=1)
        self.bn3 = nn.BatchNorm1d(pna_hidden)

        self.fc1_xd = nn.Linear(pna_hidden, output_dim)

        if protein_encoder is None:
            protein_encoder = DeepProteinCNN(num_features_xt, embed_dim, output_dim, dropout)
        self.protein_encoder = protein_encoder

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_output = n_output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.1, training=self.training)

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
