import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleProteinCNN(nn.Module):
    def __init__(self, num_features_xt=21, embed_dim=128, n_filters=32, kernel_size=8,
                 output_dim=128, dropout=0.2, seq_len=85):
        super(SimpleProteinCNN, self).__init__()

        self.embedding = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

        self._calc_fc_input_dim()
        self.fc = nn.Linear(self.fc_input_dim, output_dim)

    def _calc_fc_input_dim(self):
        dummy_input = torch.zeros(1, self.seq_len).long()
        x = self.embedding(dummy_input)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        self.fc_input_dim = x.size(1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
