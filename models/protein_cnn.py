import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepProteinCNN(nn.Module):
    def __init__(self, num_features_xt=21, embed_dim=128, output_dim=128, dropout=0.1):
        super(DeepProteinCNN, self).__init__()

        self.embedding = nn.Embedding(num_features_xt + 1, embed_dim)

        self.conv1_1 = nn.Conv1d(embed_dim, 32, kernel_size=4)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=4)
        self.conv1_3 = nn.Conv1d(32, 32, kernel_size=4)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=4)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=4)
        self.conv2_3 = nn.Conv1d(64, 64, kernel_size=4)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3_1 = nn.Conv1d(64, 96, kernel_size=4)
        self.conv3_2 = nn.Conv1d(96, 96, kernel_size=4)
        self.conv3_3 = nn.Conv1d(96, 96, kernel_size=4)
        self.pool3 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(dropout)

        self._calc_fc_input_dim()
        self.fc = nn.Linear(self.fc_input_dim, output_dim)

    def _calc_fc_input_dim(self):
        x = torch.zeros(1, 85)
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        self.fc_input_dim = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1_1(x))
        x = self.dropout(x)
        x = F.relu(self.conv1_2(x))
        x = self.dropout(x)
        x = F.relu(self.conv1_3(x))
        x = self.pool1(x)
        x = self.dropout(x)

        x = F.relu(self.conv2_1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2_2(x))
        x = self.dropout(x)
        x = F.relu(self.conv2_3(x))
        x = self.pool2(x)
        x = self.dropout(x)

        x = F.relu(self.conv3_1(x))
        x = self.dropout(x)
        x = F.relu(self.conv3_2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
