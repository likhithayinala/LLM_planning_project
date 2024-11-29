# create a dataloader that gives random data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def dataset(config):
    return RandomDataset()

class RandomDataset(Dataset):
    def __init__(self):
        self.len = 320
        self.data = torch.randn(320,35)

    def __getitem__(self, index):
        return self.data[index], torch.tensor(0)

    def __len__(self):
        return self.len
