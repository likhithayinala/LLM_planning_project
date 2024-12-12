import os
import pickle
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm 
import numpy as np
import h5py
def pad_to_max_size(arrays,max_size):
    original_sizes = []
    padded_arrays = []
    for arr in arrays:
        sizes = [arr.shape[2]]*arr.shape[1]
        original_sizes.append(sizes)  # Save original size
        # print(arr.shape)
        pad_width = ( (0, 0), (0, 0),(0, max_size - arr.shape[2]), (0, 0))  # Only pad 3rd dimension
        padded = np.pad(arr, pad_width, mode='constant', constant_values=0)
        padded_arrays.append(padded)
    return padded_arrays, original_sizes

folder_path = 'dataset/genai_project_total/safety-data/'
dataset = 'dataset.csv'
save_path = 'final_dataset'
hidden_states_path = 'final_dataset/hidden_states'
dirlist  = []
counter = 0
max_size = 38

with h5py.File('hidden_data.h5', 'a') as hdf:
    # Initialize datasets with an unknown final size using maxshape
    hdf.create_dataset('prompt_hidden_states', shape=(0, 33,max_size,4096), maxshape=(None, 33,max_size,4096), dtype='float32')
    hdf.create_dataset('original_sizes', shape=(0,1), maxshape=(None, 1), dtype='float32')

    
    #Append data from each file
    current_size = 0
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith('pkl.gz'):
            # print("Found ends with")
            with gzip.open(os.path.join(folder_path, file), 'rb') as f:
                data = pickle.load(f)
            arr1 = []

            data_slice = data
            for i in range(0, len(data_slice), 32):
                row1 = data_slice[i]['hidden_states'][0]
                arr2 = []
                for j in range(len(row1)):
                    row2 = data_slice[i]['hidden_states'][0][j].float().numpy()
                    arr2.append(row2)
                arr1.append(np.array(arr2))
            arr1,org = pad_to_max_size(arr1,max_size)
            arr1 = np.array(arr1)
            arr1 = np.swapaxes(arr1,1,2)
            arr1 = arr1.reshape(-1,33,max_size,4096)
            batch_size = arr1.shape[0]
            org = np.array(org).reshape(-1,1)
            print(org.shape)
            hdf['prompt_hidden_states'].resize(current_size + batch_size, axis=0)
            hdf['original_sizes'].resize(current_size + batch_size, axis=0)
            hdf['prompt_hidden_states'][current_size:current_size+batch_size] = arr1
            hdf['original_sizes'][current_size:current_size+batch_size] = org
            current_size += batch_size
