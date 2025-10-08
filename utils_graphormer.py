import os
import torch
from collections import deque
from torch_geometric.utils import degree
from utils import TestbedDataset

def compute_shortest_paths(edge_index, num_nodes):
    distances = torch.full((num_nodes, num_nodes), 999, dtype=torch.long)
    distances.fill_diagonal_(0)

    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[u].append(v)

    for start in range(num_nodes):
        queue = deque([start])
        visited = {start}
        dist = 0

        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start, neighbor] = dist + 1
                        queue.append(neighbor)
            dist += 1

    return distances

def preprocess_graphormer_data(dataset_name):
    cache_path = f'data/graphormer_cache_{dataset_name}.pt'

    if os.path.exists(cache_path):
        print(f'Loading cached Graphormer data from {cache_path}')
        return torch.load(cache_path, weights_only=True)

    print(f'Preprocessing Graphormer data for {dataset_name}...')
    dataset = TestbedDataset(root='data', dataset=f'{dataset_name}_train')
    test_dataset = TestbedDataset(root='data', dataset=f'{dataset_name}_test')

    graph_data = {}

    for split_name, split_dataset in [('train', dataset), ('test', test_dataset)]:
        split_data = []
        for i, data in enumerate(split_dataset):
            if (i + 1) % 1000 == 0:
                print(f'  {split_name}: {i+1}/{len(split_dataset)}')

            distances = compute_shortest_paths(data.edge_index, data.num_nodes)
            in_degree = degree(data.edge_index[1], data.num_nodes, dtype=torch.long)
            out_degree = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)

            split_data.append({
                'distances': distances,
                'in_degree': in_degree,
                'out_degree': out_degree
            })

        graph_data[split_name] = split_data

    torch.save(graph_data, cache_path)
    print(f'Saved to {cache_path}')
    return graph_data

def get_or_preprocess_graphormer(dataset_name):
    cache_path = f'data/graphormer_cache_{dataset_name}.pt'

    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    return preprocess_graphormer_data(dataset_name)
