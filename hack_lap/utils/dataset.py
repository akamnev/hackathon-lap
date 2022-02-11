from math import ceil
import torch
import pickle
import random
from torch_geometric.data import Data


def _gcn_dataset(data, multiplier=1):
    dataset = []
    for d, y in data:
        x, edge_index, edge_attr = d
        x = torch.tensor(x, dtype=torch.long).view(-1, 1)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr, dtype=torch.long).view(-1, 1)
        y = torch.tensor(y, dtype=torch.float32)
        for _ in range(multiplier):
            dataset.append(
                Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            )

    return dataset


def get_gcn_dataset(filename, dev_frac=0.1, positive_multiplier=1, seed=None):
    random.seed(seed)

    with open(filename, 'rb') as fp:
        data = pickle.load(fp)

    random.shuffle(data)
    data_pos = [v for v in data if v[-1]]
    data_neg = [v for v in data if not v[-1]]
    dev_pos_num = ceil(dev_frac * len(data_pos))
    dev_neg_num = ceil(dev_frac * len(data_neg))

    dataset_train = _gcn_dataset(data_neg[dev_neg_num:]) + \
        _gcn_dataset(data_pos[dev_pos_num:], multiplier=positive_multiplier)

    dataset_dev = _gcn_dataset(data_pos[:dev_pos_num]) + \
        _gcn_dataset(data_neg[:dev_neg_num])

    return dataset_train, dataset_dev

