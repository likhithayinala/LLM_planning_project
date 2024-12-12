# Imports
import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from detection_model import select_det_model
from main_model import select_main_model
from dataloader import dataset

# Initialize device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Utility function to set up configuration
def set_config(config_dict):
    """
    Loads the configuration from a JSON file and updates it with additional settings.
    """
    with open(config_dict['config']) as f:
        config = json.load(f)
    config.update(config_dict)
    if config.get('debug', False):
        config.update({'wandb': False, 'log': False})
    return config

# Configuration settings
config = {
    'config': 'config.json',
    'detection_model': 'MLP',
    'main_model': 'Llama2',
    'dataset': 'CIFAR10',
    'debug': False,
    'wandb': False,
    'log': True,
    'tag': '',
    'layer': 0,
    'token': 0,
    'batch_size': 32,
    'dataset_path': 'path_to_dataset.csv'
}

# Update configuration
config = set_config(config)

# Load the detection model and its weights
detection_model = select_det_model(config['detection_model'], config).to(device)
detection_model.load_state_dict(torch.load('path_to_model_checkpoint.pth'))

# Load test data
data = pd.read_csv(config['dataset_path'])
test_data = data.iloc[int(len(data) * 0.9):].reset_index(drop=True)
test_dataset = dataset(test_data, config)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

# Initialize metrics
total_samples = 0
total_correct = 0
misclassified = []
classified = []

# Evaluation loop
with torch.no_grad():
    for i, data_batch in enumerate(test_dataloader):
        response, safety_class, token_hidden_states, prompt_hidden_states = data_batch

        # Extract the specific hidden state
        state = token_hidden_states[:, config['layer'] - 1, config['token'] - 1, :]
        state, labels = state.to(device), safety_class.to(device).long()

        # Forward pass through the model
        output = detection_model(state)
        _, predicted = torch.max(output.data, 1)

        # Update metrics
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        # Collect misclassified and correctly classified samples
        for j in range(labels.size(0)):
            sample = {
                "response": response[j],
                "predicted": predicted[j].item(),
                "label": labels[j].item()
            }
            if predicted[j] != labels[j]:
                misclassified.append(sample)
            else:
                classified.append(sample)

# Calculate test accuracy
test_accuracy = 100 * total_correct / total_samples
print("Test Accuracy:", test_accuracy)

# Save classified results
classified_df = pd.DataFrame(classified)
classified_df.to_csv("classified_mlp.csv", index=False)

# Save misclassified results
misclassified_df = pd.DataFrame(misclassified)
misclassified_df.to_csv("misclassified_mlp.csv", index=False)

# Shuffle dataframes
classified_df = classified_df.sample(frac=1).reset_index(drop=True)
misclassified_df = misclassified_df.sample(frac=1).reset_index(drop=True)

# Extract true positives
true_positives = classified_df[
    (classified_df["predicted"] == 1) & (classified_df["label"] == 1)
]["response"].reset_index(drop=True)

# Extract true negatives
true_negatives = classified_df[
    (classified_df["predicted"] == 0) & (classified_df["label"] == 0)
]["response"].reset_index(drop=True)

# Extract false positives
false_positives = misclassified_df[
    (misclassified_df["predicted"] == 1) & (misclassified_df["label"] == 0)
]["response"].reset_index(drop=True)

# Extract false negatives
false_negatives = misclassified_df[
    (misclassified_df["predicted"] == 0) & (misclassified_df["label"] == 1)
]["response"].reset_index(drop=True)

# Print sample responses for each category
print("\nTrue Positives:")
for i in range(min(5, len(true_positives))):
    print(true_positives.iloc[i])
    print("***********")

print("\nTrue Negatives:")
for i in range(min(5, len(true_negatives))):
    print(true_negatives.iloc[i])
    print("***********")

print("\nFalse Positives:")
for i in range(min(5, len(false_positives))):
    print(false_positives.iloc[i])
    print("***********")

print("\nFalse Negatives:")
for i in range(min(5, len(false_negatives))):
    print(false_negatives.iloc[i])
    print("***********")

# Calculate precision and recall
precision = len(true_positives) / (len(true_positives) + len(false_positives))
recall = len(true_positives) / (len(true_positives) + len(false_negatives))
print("Precision:", precision)
print("Recall:", recall)