import os
import gzip
import _pickle as cPickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    
    def __init__(self, file_dir, transform=None):
        self.file_dir = file_dir
        self.cwd = os.getcwd()
        self.fn = [os.path.join(file_dir, f) for f in os.listdir(self.cwd + file_dir)]
        self.transform = transform
    
    def __len__(self):
        return len(self.fn)
    
    def __getitem__(self, idx):

        with gzip.open(self.cwd + self.fn[idx], 'rb') as f:
            data = cPickle.load(f)
        
        return data

data_path = '/batches1/' #./safety-data/'
eval_ds = CustomDataset(data_path)
eval_loader = DataLoader(eval_ds, batch_size = 1)
    
model_name = "./checkpoint-37571" # "./qwen2.5_1.5B_judge"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", pad_token='<pad>', return_tensors='pt')
model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='cuda').to('cuda')  # Safe/Unsafe , num_labels=2

# # 4. Load and preprocess the dataset
# dataset = load_dataset("PKU-Alignment/BeaverTails")
# # Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=12)
# dataset = dataset.map(tokenize_function, batched=True)
# print('Entering loop...')

new_dataset = []
results = []
for batch in tqdm(eval_loader):
    
    res = []
    
    # batch_perpkl = 320
    for data in batch[0]:    
        
        print(data, batch)    
        prompt = data['output']
        # print(prompt)
        
        tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to('cuda')
        output = model(**tokens)
        # print('Model Loaded.')
        
        res.append(output.logits.cpu().detach().float().numpy())
        new_dataset.append((prompt, torch.argmax(output.logits.cpu().detach()).float().numpy()))
        # print()
    
    results.append(res)

results = np.array(results)
np.save('./safety-results.npy', results)

df = pd.DataFrame(new_dataset, columns=['prompt', 'logits'])
df.to_csv('./safety-dataset.csv', index=False)