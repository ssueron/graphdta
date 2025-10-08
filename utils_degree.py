import os
import torch
from torch_geometric.utils import degree
from utils import TestbedDataset

def compute_degree_histogram(dataset_name):
    train_data = TestbedDataset(root='data', dataset=f'{dataset_name}_train')

    max_degree = 0
    for data in train_data:
        d = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_data:
        d = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d.long(), minlength=deg.numel())

    return deg

def get_or_compute_degree(dataset_name):
    cache_path = f'data/degree_stats_{dataset_name}.pt'

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    print(f'Computing degree histogram for {dataset_name}...')
    deg = compute_degree_histogram(dataset_name)
    torch.save(deg, cache_path)
    print(f'Saved to {cache_path}')
    return deg
