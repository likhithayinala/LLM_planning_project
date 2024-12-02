# create a dataloader that gives random data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import gzip
import pickle 
import h5py
def dataset(data,config):
    return BTDataset(data,config)

class BTDataset(Dataset):
    
    def __init__(self,data,config):

        ls = os.listdir(config['data_path'])
        self.path = config['data_path']
        # Split the data into train and test
        self.data = data 
        self.len = len(data)
        self.hidden_states = config['hidden_states_path']
    def __getitem__(self, index):
        response = self.data[index]['prompt']
        safety_class = self.data[index]['logits']
        with h5py.File(self.hidden_states, 'r') as hdf:
            token_hidden_states = hdf['token_hidden_states'][index]
            original_size = hdf['original_size'][index]
            prompt_hidden_states = hdf['prompt_hidden_states'][index]
            # Cut the padded arrays to their original size
            prompt_hidden_states = prompt_hidden_states[:, :, :original_size, :]
        return (response, safety_class, token_hidden_states, prompt_hidden_states)
    
    def __len__(self):
        return self.len
