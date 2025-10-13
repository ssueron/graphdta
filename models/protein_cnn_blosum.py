import torch
import torch.nn as nn
import torch.nn.functional as F

BLOSUM62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
}

AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'
GAP_CHAR = '-'

def create_blosum62_embedding(num_tokens=22, embed_dim=128):
    embedding_matrix = torch.zeros(num_tokens, 20)

    for idx, aa in enumerate(AA_ORDER, start=1):
        if aa in BLOSUM62:
            embedding_matrix[idx] = torch.tensor(BLOSUM62[aa], dtype=torch.float32)

    embedding_matrix[21] = torch.zeros(20)

    embedding_matrix = embedding_matrix / 4.0

    embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
    projection = nn.Linear(20, embed_dim)

    return embedding, projection

class DeepProteinCNN_BLOSUM(nn.Module):
    def __init__(self, num_features_xt=21, embed_dim=128, n_filters=32, kernel_size=8,
                 output_dim=128, dropout=0.2, seq_len=85):
        super(DeepProteinCNN_BLOSUM, self).__init__()

        self.embedding, self.projection = create_blosum62_embedding(num_features_xt + 1, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

        self._calc_fc_input_dim()
        self.fc = nn.Linear(self.fc_input_dim, output_dim)

    def _calc_fc_input_dim(self):
        x = torch.zeros(1, self.seq_len).long()
        x = self.embedding(x)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        self.fc_input_dim = x.size(1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
