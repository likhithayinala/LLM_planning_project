def pad_to_max_size(arrays):
    original_sizes = []
    padded_arrays = []
    max_size = max(arr.shape[2] for arr in arrays)
    for arr in arrays:
        original_sizes.append(arr.shape[2])  # Save original size
        pad_width = ((0, 0), (0, 0), (0, max_size - arr.shape[2]), (0, 0))  # Only pad 3rd dimension
        padded = np.pad(arr, pad_width, mode='constant', constant_values=0)
        padded_arrays.append(padded)
    return padded_arrays, original_sizes

import os
import pickle
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm 
import numpy as np
import h5py
folder_path = '/home/la3073/data/safety-data'
dataset = 'dataset.csv'
save_path = 'final_dataset'
hidden_states_path = 'final_dataset/hidden_states'
dirlist  = []
counter = 0
#df = pd.read_csv(dataset)
with h5py.File('/home/la3073/data/hidden_data_token_new.h5', 'w') as hdf:
    # Initialize datasets with an unknown final size using maxshape
    hdf.create_dataset('token_hidden_states', shape=(0, 33,2,4096), maxshape=(None, 33,2,4096), dtype='float32')
    
    # Append data from each file
    current_size = 0
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith('pkl.gz'):
            # print("Found ends with")
            with gzip.open(os.path.join(folder_path, file), 'rb') as f:
                data = pickle.load(f)
            arr1 = []
            arr2 = []
            for i in range(0,len(data),32):
                l1 = []
                row1 = data[i]['hidden_states']
                # print(len(data[i]['output']))
                for j in range(len(row1)):
                    l2 = []
                    row2 = data[i]['hidden_states'][j]
                    for k in range(len(row2)): 
                        l2.append(data[i]['hidden_states'][j][k].float().numpy())
                    l1.append(np.array(l2))
                for m in range(len(l1)):
                    l1[m] = np.array(l1[m])
                new_arr = np.stack((l1[1],l1[2]),axis=2) 
                # squeeze the array
                new_arr = np.squeeze(new_arr,axis=3)
                arr1.append(l1[0])
                arr2.append(new_arr)
            arr2 = np.array(arr2)
            arr2 = np.array(arr2)
            arr2 = np.swapaxes(arr2,1,2)
            arr2 = arr2.reshape(-1,33,2,4096)
            batch_size = arr2.shape[0]
            hdf['token_hidden_states'].resize(current_size + batch_size, axis=0)
            hdf['token_hidden_states'][current_size:current_size+batch_size] = arr2
            current_size += batch_size
