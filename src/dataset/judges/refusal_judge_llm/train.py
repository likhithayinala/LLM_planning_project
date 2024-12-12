import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Step 1: Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Load tokenizer and model
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # Replace with the correct Hugging Face model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load pre-trained base model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Step 3: Define PEFT configuration
lora_config = LoraConfig(
    task_type="SEQ_CLS",   # Sequence classification
    r=4,                   # Rank of the LoRA updates
    lora_alpha=32,         # Scaling factor for LoRA updates
    lora_dropout=0.1,      # Dropout for LoRA layers
    bias="none",           # Whether to adapt bias terms ("none", "all", "lora_only")
)

# Apply PEFT to the model
peft_model = get_peft_model(model, lora_config).to(device)

import pandas as pd
train_dataset = pd.read_csv("train.csv")
val_dataset = pd.read_csv("val.csv")
#Convert labels to int
train_dataset['label'] = train_dataset['label'].apply(lambda x: int(x))
# Convert to Hugging Face Dataset format
from datasets import Dataset
dataset = Dataset.from_pandas(train_dataset)
val_dataset = Dataset.from_pandas(val_dataset)
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_val_datasets = val_dataset.map(preprocess_function, batched=True)
# Pass a random sample of the dataset to the model
sample = tokenizer("This is a sample input", return_tensors="pt")
sample = {k: v.to(device) for k, v in sample.items()}
outputs = peft_model(**sample)
print(outputs.loss, outputs.logits.shape)
# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./smollm_peft_refusal_model",
    evaluation_strategy="epoch",
    learning_rate=5e-4,  # Larger LR for PEFT since fewer parameters are being trained
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="none",  # Set to "wandb" or "tensorboard" if integrated
    logging_steps=50,
    fp16=True,  # Enable mixed precision training for faster performance on GPUs
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_val_datasets,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {
        "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
    },
)

# Step 7: Train the model
trainer.train()

# Save the PEFT model
peft_model.save_pretrained("./smollm_peft_refusal_model")
tokenizer.save_pretrained("./smollm_peft_refusal_model")

print("PEFT fine-tuning complete! Model saved.")

# Step 8: Evaluate
results = trainer.evaluate()
print("Evaluation results:", results)
