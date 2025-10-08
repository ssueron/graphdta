import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, max_path_dist=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.spatial_encoder = nn.Embedding(max_path_dist, num_heads)
        self.edge_encoder = nn.Linear(1, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, distances, edge_features, attn_mask):
        batch_size, num_nodes, _ = x.shape

        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        spatial_bias = self.spatial_encoder(distances).permute(0, 3, 1, 2)
        attn_scores = attn_scores + spatial_bias

        edge_bias = self.edge_encoder(edge_features.unsqueeze(-1)).permute(0, 3, 1, 2)
        attn_scores = attn_scores + edge_bias

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        attn_output = self.o_proj(attn_output)

        x = self.layer_norm1(x + self.dropout(attn_output))
        x = self.layer_norm2(x + self.ffn(x))

        return x

class GraphormerNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=27,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2,
                 num_layers=3, num_heads=8, max_path_dist=10, max_degree=128):
        super().__init__()

        self.node_emb = nn.Linear(num_features_xd, output_dim)
        self.in_degree_emb = nn.Embedding(max_degree, output_dim)
        self.out_degree_emb = nn.Embedding(max_degree, output_dim)

        self.virtual_node = nn.Parameter(torch.randn(1, output_dim))

        self.layers = nn.ModuleList([
            GraphormerLayer(output_dim, num_heads, dropout, max_path_dist)
            for _ in range(num_layers)
        ])

        self.graph_proj = nn.Linear(output_dim, output_dim)

        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(embed_dim, n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*78, output_dim)

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_output = n_output
        self.max_degree = max_degree
        self.max_path_dist = max_path_dist

    def forward(self, data):
        x = data.x
        batch = data.batch
        distances = data.distances
        in_degree = data.in_degree
        out_degree = data.out_degree
        target = data.target

        batch_size = batch.max().item() + 1
        device = x.device

        node_features = self.node_emb(x)
        in_deg_clamp = torch.clamp(in_degree, max=self.max_degree - 1)
        out_deg_clamp = torch.clamp(out_degree, max=self.max_degree - 1)
        node_features = node_features + self.in_degree_emb(in_deg_clamp) + self.out_degree_emb(out_deg_clamp)

        graph_list = []
        dist_list = []
        edge_feat_list = []
        max_nodes = 0

        for i in range(batch_size):
            mask = batch == i
            num_nodes = mask.sum().item()
            max_nodes = max(max_nodes, num_nodes)

        for i in range(batch_size):
            mask = batch == i
            indices = mask.nonzero(as_tuple=True)[0]
            num_nodes = len(indices)

            graph_nodes = node_features[mask]
            vnode = self.virtual_node.expand(1, -1)
            graph_nodes = torch.cat([vnode, graph_nodes], dim=0)

            graph_dist = distances[i][:num_nodes, :num_nodes]
            vnode_dist = torch.zeros(num_nodes, dtype=torch.long, device=device)
            graph_dist = torch.cat([vnode_dist.unsqueeze(0), graph_dist], dim=0)
            graph_dist = torch.cat([torch.zeros(num_nodes + 1, 1, dtype=torch.long, device=device), graph_dist], dim=1)

            edge_feat = (graph_dist > 0).float()

            if num_nodes + 1 < max_nodes + 1:
                pad_size = max_nodes + 1 - (num_nodes + 1)
                graph_nodes = F.pad(graph_nodes, (0, 0, 0, pad_size))
                graph_dist = F.pad(graph_dist, (0, pad_size, 0, pad_size), value=self.max_path_dist - 1)
                edge_feat = F.pad(edge_feat, (0, pad_size, 0, pad_size))

            graph_list.append(graph_nodes)
            dist_list.append(graph_dist)
            edge_feat_list.append(edge_feat)

        h = torch.stack(graph_list, dim=0)
        distances_batch = torch.stack(dist_list, dim=0)
        edge_features_batch = torch.stack(edge_feat_list, dim=0)

        distances_batch = torch.clamp(distances_batch, max=self.max_path_dist - 1)

        attn_mask = (distances_batch == self.max_path_dist - 1) & (distances_batch > 0)

        for layer in self.layers:
            h = layer(h, distances_batch, edge_features_batch, attn_mask)

        graph_repr = h[:, 0, :]
        graph_repr = self.graph_proj(graph_repr)

        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.permute(0, 2, 1)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 78)
        xt = self.fc1_xt(xt)

        xc = torch.cat((graph_repr, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out
