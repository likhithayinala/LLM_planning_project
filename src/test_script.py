# Imports
import torch 
import torch.nn as nn
import torch.nn.functional as F
from detection_model import select_det_model
from main_model import select_main_model
from torch.utils.data import Dataset, DataLoader
from dataloader import dataset
import logging
import wandb
import os
import json
from tqdm import tqdm
import argparse
import pandas as pd
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler


# +
# Util funcs
def create_dir(path):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(path):
        print("Creating path: ", path)
        os.makedirs(path)
        
def set_config(config_dict):
    with open(config_dict['config']) as f:
        config = json.load(f)
    config.update(config_dict)
    if config.get('debug', False):
        config.update({'wandb': False, 'log': False})
    if config['wandb']:
        wandb.init(project='bad_content_detection', config=config)
    config = results(config)
    return config

def results(config):
    return config


# -

# Inits
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

config = {
    'config': 'config.json',
    'detection_model': 'MLP',
    'main_model': 'Llama2',
    'dataset': 'CIFAR10',
    'debug': False,
    'wandb': True,
    'log': True,
    'tag': '',
    'layer': 0,
    'token': 0
}
config = set_config(config)
config

# Load model
detection_model = select_det_model('MLP', config).to(device)

# Load weights
detection_model.load_state_dict(torch.load('/home/sv2795_columbia_edu/GENAI_Project/src/results/MLP_Llama2_CIFAR10_MLP-last-layer-firsrt-token-refusal-big/best_model.pth'))

# Init test set
data = pd.read_csv(config['dataset_path'])
test_data = data[int(len(data)*0.9):].reset_index(drop=True)
test_dataloader = DataLoader(dataset(test_data,config), batch_size=config['batch_size'], shuffle=True)

# +
# Run test
total_samples = 0
total_correct = 0
misclassified = []
classified = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        response, safety_class, token_hidden_states, prompt_hidden_states = data
        state = token_hidden_states[:,config['layer'] - 1,config['token'] - 1,:]
        state, labels = state.to(device), safety_class.to(device)
        output = detection_model(state)
        labels = labels.to(torch.long)
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        #print(labels,predicted)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        for i in range(labels.size(0)):
            if predicted[i] != labels[i]:
                misclassified.append({"response": response[i], "predicted": predicted[i].item(), "label": labels[i].item()})
            else:
                classified.append({"response": response[i], "predicted": predicted[i].item(), "label": labels[i].item()})
            
        
# -

# Test accuracy
test_accuracy = 100 * total_correct / total_samples
print("test accuracy: ", test_accuracy)

# Save misclassified results
classified_df = pd.DataFrame(classified)
classified_df.to_csv("classified_mlp.csv", index = False)
classified_df.head()

# Print some true positives
classified_df = classified_df.sample(frac=1).reset_index(drop=True)
true_positives = classified_df[(classified_df["predicted"] == 1) & (classified_df["label"] == 1)]
true_pos = true_positives["response"]
for i in range(5):
    print(true_pos.iloc[i])
    print("***********")


# Print some true negatives
classified_df = classified_df.sample(frac=1).reset_index(drop=True)
true_neg = classified_df[(classified_df["predicted"] == 0) & (classified_df["label"] == 0)]
true_neg = true_neg["response"]
for i in range(5):
    print(true_neg.iloc[i])
    print("***********")

# Print some false positives
misclassified_df = misclassified_df.sample(frac=1).reset_index(drop=True)
false_positives = misclassified_df[(misclassified_df["predicted"] == 1) & (misclassified_df["label"] == 0)]
false_positive_responses = false_positives["response"]
for i in range(5):
    print(false_positive_responses.iloc[i])
    print("***********")


# Print some false negatives. 
misclassified_df = misclassified_df.sample(frac=1).reset_index(drop=True)
false_negative = misclassified_df[(misclassified_df["predicted"] == 0) & (misclassified_df["label"] == 1)]
false_negative = false_negative["response"]
for i in range(5):
    print(false_negative.iloc[i])
    print("***********")


# Also print precision and recall
precision = len(true_pos) / (len(true_pos) + len(false_positives))
print("Precision: ", precision)
recall = len(true_pos) / (len(true_pos) + len(false_negative))
print("recall: ", recall)


