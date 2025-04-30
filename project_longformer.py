#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Cell 1: Download data if not already present
import os
import requests

file_urls = {
    "data/Gungor_2018_VictorianAuthorAttribution_data-train.csv":
        "https://dataworks.indianapolis.iu.edu/bitstream/handle/11243/23/Gungor_2018_VictorianAuthorAttribution_data-train.csv?sequence=2&isAllowed=y",
    "data/Gungor_2018_VictorianAuthorAttribution_data.csv":
        "https://dataworks.indianapolis.iu.edu/bitstream/handle/11243/23/Gungor_2018_VictorianAuthorAttribution_data.csv?sequence=3&isAllowed=y"
}

os.makedirs("data", exist_ok=True)

for filename, url in file_urls.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...", flush=True)
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Saved to {filename}", flush=True)
    else:
        print(f"File already exists: {filename}", flush=True)

print("Hello world.")


# In[ ]:


# Cell 2: Load datasets
import pandas as pd

train_df = pd.read_csv("data/Gungor_2018_VictorianAuthorAttribution_data-train.csv", encoding="ISO-8859-1")
full_df = pd.read_csv("data/Gungor_2018_VictorianAuthorAttribution_data.csv", encoding="ISO-8859-1")


# In[ ]:


# Cell 3: Prepare labels and split dataset
from sklearn.model_selection import train_test_split
import json

train_df['author_label'] = train_df['author'].astype('category').cat.codes
label_map = dict(enumerate(train_df['author'].astype('category').cat.categories))

with open("data/label_map.json", "w") as f:
    json.dump(label_map, f)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].tolist(),
    train_df['author_label'].tolist(),
    test_size=0.1,
    stratify=train_df['author_label'],
    random_state=42
)


# In[ ]:


# Cell 4: Tokenization with Longformer
from transformers import AutoTokenizer

model_name = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=2048)
val_encodings = tokenizer(val_texts, truncation=True, padding="max_length", max_length=2048)

# Add global attention mask (CLS token = global)
import torch
train_encodings["global_attention_mask"] = [
    [1] + [0] * (len(input_ids) - 1) for input_ids in train_encodings["input_ids"]
]
val_encodings["global_attention_mask"] = [
    [1] + [0] * (len(input_ids) - 1) for input_ids in val_encodings["input_ids"]
]


# In[ ]:


# Cell 5: Dataset wrapper class with global attention mask
import torch

class AuthorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'global_attention_mask': torch.tensor(self.encodings['global_attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

train_dataset = AuthorDataset(train_encodings, train_labels)
val_dataset = AuthorDataset(val_encodings, val_labels)


# In[ ]:


# Cell 6: Load Longformer model for classification
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_map)
)


# In[ ]:


# Cell 7: Training configuration
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    lr_scheduler_type="linear",
    logging_steps=500,
    logging_strategy="steps",
    logging_first_step=True,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


# In[ ]:


# Cell 8: Trainer setup
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class PrintToTerminalCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            total_steps = state.max_steps
            epoch = state.epoch
            log_str = f"[Step {step}/{total_steps}] | Epoch: {epoch:.2f} | " + \
                      " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items())
            print(log_str, flush=True)  # <-- print to terminal/log file

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[PrintToTerminalCallback()]
)



# In[ ]:


# Cell 9: Train the model
trainer.train()


# In[ ]:


# Cell 10: Evaluate the model
trainer.evaluate()


# In[ ]:


# Cell 11: Save and load the Longformer model

# Save model and tokenizer
save_directory = "longformer_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to `{save_directory}`.")

# Example: Load model and tokenizer for future use
from transformers import AutoTokenizer, AutoModelForSequenceClassification

loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_directory)

# You can now use loaded_model and loaded_tokenizer just like before

