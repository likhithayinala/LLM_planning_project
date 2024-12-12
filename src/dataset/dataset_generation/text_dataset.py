import numpy as np
import pandas as pd
from tqdm import tqdm 
import numpy as np
import h5py
dataset = 'safety-dataset.csv'
df = pd.read_csv(dataset)
with h5py.File('hidden_data.h5', 'a') as hdf:
    # Create 3 datasets to store strings of prompt, output and safety_class and add the corresponding data from df
    hdf.create_dataset('prompt', data=df['prompt'].values.astype('S'), dtype=h5py.string_dtype(encoding='utf-8'))
    hdf.create_dataset('output', data=df['output'].values.astype('S'), dtype=h5py.string_dtype(encoding='utf-8'))
    hdf.create_dataset('safety_class', data=df['safety_class'].values, dtype='S100')