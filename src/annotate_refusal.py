import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import huggingface_hub
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gc
import pickle
import gzip




# Read data
data = pd.read_csv('prompt_output(1).csv')

def extract_user_message(prompt):
    start = prompt.find("<</SYS>>") + len("<</SYS>>")
    end = prompt.find("[/INST]")
    return prompt[start:end].strip()

data['user_message'] = data['prompt'].apply(extract_user_message)




# drop prompt, rename usr_message
data = data.drop(columns=['prompt'])
data = data.rename(columns={'user_message': 'prompt'})


data = data.tolist()





model_name = "protectai/distilroberta-base-rejection-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)



def classify_refusal(output):
    inputs = tokenizer(output, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    return prediction



annotations = [classify_refusal(output) for output in data]

df = pd.DataFrame({'output': data, 'label': annotations})
df.to_csv('final_refusal.csv', index=False)

shuffled_df = df.sample(frac=1).reset_index(drop=True)


shuffled_df.to_csv('final_refusal_shuffled_mistake.csv', index=False)
