import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import gzip
import pickle
import h5py

def dataset(data, config):
    """
    Return the appropriate dataset based on the configuration.

    Args:
        data: The dataset.
        config: Configuration dictionary containing dataset information.

    Returns:
        An instance of RefusalDataset or SafetyDataset.
    """
    if 'refusal' in config['dataset']:
        return RefusalDataset(data, config)
    if 'safety' in config['dataset']:
        return SafetyDataset(data, config)
    raise ValueError('Dataset not recognized')

class RefusalDataset(Dataset):
    """
    Dataset class for refusal data.
    """

    def __init__(self, data, config):
        """
        Initialize the RefusalDataset.

        Args:
            data: The dataset.
            config: Configuration dictionary containing dataset paths.
        """
        self.path = config['dataset_path']
        self.data = data
        self.len = len(data)
        self.hidden_states = config['hidden_states_path']

    def __getitem__(self, index):
        """
        Retrieve a single data sample.

        Args:
            index: Index of the data sample.

        Returns:
            Tuple containing response, safety_class, token_hidden_states, and prompt_hidden_states.
        """
        response = self.data.loc[index]['output']
        og_index = self.data.loc[index]['og_index']
        safety_class = self.data.loc[index]['label']
        with h5py.File(self.hidden_states, 'r') as hdf:
            token_hidden_states = hdf['token_hidden_states'][og_index]
            prompt_hidden_states = 1  # Placeholder value
        return (response, safety_class, token_hidden_states, prompt_hidden_states)

    def __len__(self):
        """
        Return the total number of samples.
        """
        return self.len

class SafetyDataset(Dataset):
    """
    Dataset class for safety data.
    """

    def __init__(self, data, config):
        """
        Initialize the SafetyDataset.

        Args:
            data: The dataset.
            config: Configuration dictionary containing dataset paths.
        """
        self.path = config['dataset_path']
        self.data = data
        self.len = len(data)
        self.hidden_states = config['hidden_states_path']

    def __getitem__(self, index):
        """
        Retrieve a single data sample.

        Args:
            index: Index of the data sample.

        Returns:
            Tuple containing response, safety_class, token_hidden_states, and prompt_hidden_states.
        """
        response = self.data.loc[index]['prompt']
        safety_class = self.data.loc[index]['logits']
        with h5py.File(self.hidden_states, 'r') as hdf:
            token_hidden_states = hdf['token_hidden_states'][index]
            prompt_hidden_states = 1  # Placeholder value
        return (response, safety_class, token_hidden_states, prompt_hidden_states)

    def __len__(self):
        """
        Return the total number of samples.
        """
        return self.len
