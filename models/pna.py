import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_mean_pool

class PNANet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=27,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, deg=None):
        super(PNANet, self).__init__()

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

        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*78, output_dim)

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

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
        x = F.dropout(x, p=0.2, training=self.training)

        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.permute(0, 2, 1)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 78)
        xt = self.fc1_xt(xt)

        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
