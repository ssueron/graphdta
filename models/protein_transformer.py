import torch
import torch.nn as nn
import math


class ProteinTransformer(nn.Module):
    def __init__(self, num_features_xt=21, embed_dim=384, n_layers=5, n_heads=8,
                 ffn_dim=1536, output_dim=128, dropout=0.15, seq_len=85, **kwargs):
        super(ProteinTransformer, self).__init__()

        self.embedding = nn.Embedding(num_features_xt + 1, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding

        x = self.transformer(x)

        attn_weights = self.attention_pool(x)
        x = (x * attn_weights).sum(dim=1)

        x = self.projection(x)
        return x
