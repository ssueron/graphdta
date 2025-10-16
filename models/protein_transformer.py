import torch
import torch.nn as nn
import math


class ProteinTransformer(nn.Module):
    def __init__(self, num_features_xt=21, embed_dim=384, n_layers=5, n_heads=8,
                 ffn_dim=1536, output_dim=128, dropout=0.15, attn_dropout=0.1,
                 seq_len=85, **kwargs):
        super(ProteinTransformer, self).__init__()

        self.embedding = nn.Embedding(num_features_xt + 1, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.max_seq_len = seq_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=attn_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.attention_score = nn.Linear(embed_dim, 1)

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds configured maximum {self.max_seq_len}")

        padding_mask = (x == 0)

        pos_indices = torch.arange(x.size(1), device=x.device)
        pos_embed = self.pos_embedding(pos_indices).unsqueeze(0)

        x = self.embedding(x) + pos_embed
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        attn_scores = self.attention_score(x).squeeze(-1)
        attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        x = (x * attn_weights).sum(dim=1)

        x = self.projection(x)
        return x
