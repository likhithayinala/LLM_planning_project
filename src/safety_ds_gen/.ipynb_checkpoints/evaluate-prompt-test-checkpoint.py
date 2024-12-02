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

model_name = "./checkpoint-37571" # "./qwen2.5_1.5B_judge"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", pad_token='<pad>', return_tensors='pt')
model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='cuda').to('cuda')  # Safe/Unsafe , num_labels=2

# # 4. Load and preprocess the dataset
# dataset = load_dataset("PKU-Alignment/BeaverTails")
# # Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=12)
# dataset = dataset.map(tokenize_function, batched=True)
prompt1 = "Hello, my name is Likhith."
print('Entering loop...')

tokens = tokenizer(prompt1, return_tensors="pt", padding=True, truncation=True).to('cuda')
output = model(**tokens)
print('Model Loaded.')
print(torch.argmax(output.logits.cpu().detach()).float().numpy())
