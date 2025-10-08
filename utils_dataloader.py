import torch
from torch_geometric.data import Data, Batch

class GraphormerDataLoader:
    def __init__(self, dataset, graphormer_cache, batch_size, shuffle=False):
        self.dataset = dataset
        self.graphormer_cache = graphormer_cache
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            torch.manual_seed(torch.initial_seed())
            indices = torch.randperm(len(self.dataset)).tolist()

        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]

            batch = Batch.from_data_list(batch_data)

            distances_list = []
            in_degree_list = []
            out_degree_list = []

            for idx in batch_indices:
                cache_item = self.graphormer_cache[idx]
                distances_list.append(cache_item['distances'])
                in_degree_list.append(cache_item['in_degree'])
                out_degree_list.append(cache_item['out_degree'])

            batch.distances = distances_list
            batch.in_degree = torch.cat(in_degree_list, dim=0)
            batch.out_degree = torch.cat(out_degree_list, dim=0)

            yield batch
