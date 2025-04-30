#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import nltk
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


model_path = "longformer_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Loaded Longformer model from {model_path}")


# In[3]:


df = pd.read_csv("data/Gungor_2018_VictorianAuthorAttribution_data-train.csv", encoding="ISO-8859-1")
df['author_label'] = df['author'].astype('category').cat.codes
label_map = dict(enumerate(df['author'].astype('category').cat.categories))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['author_label'].tolist(),
    test_size=0.1,
    stratify=df['author_label'],
    random_state=42
)


# In[4]:


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                synonyms.add(name)
    return list(synonyms)

def synonym_substitution(text, rate=0.05):
    words = word_tokenize(text)
    new_words = words.copy()
    
    candidates = [i for i, w in enumerate(words) if w.isalpha() and w.lower() not in stop_words]
    num_changes = max(1, int(len(candidates) * rate))
    indices_to_replace = random.sample(candidates, min(num_changes, len(candidates)))

    for idx in indices_to_replace:
        word = words[idx]
        synonyms = get_synonyms(word)
        if synonyms:
            new_words[idx] = random.choice(synonyms)

    return ' '.join(new_words)


# In[5]:


def antonym_substitution(text, rate=0.05):
    words = word_tokenize(text)
    new_words = words.copy()
    candidates = [i for i, w in enumerate(words) if w.isalpha() and w.lower() not in stop_words]
    num_changes = max(1, int(len(candidates) * rate))
    indices_to_replace = random.sample(candidates, min(num_changes, len(candidates)))

    for idx in indices_to_replace:
        word = words[idx]
        antonyms = [a.name().replace('_', ' ')
                    for syn in wordnet.synsets(word)
                    for lemma in syn.lemmas()
                    for a in lemma.antonyms()]
        if antonyms:
            new_words[idx] = random.choice(antonyms)

    return ' '.join(new_words)


# In[6]:


def insert_noise_words(text, rate=0.05):
    filler_words = [
        "indeed", "clearly", "furthermore", "basically", "evidently", "thus",
        "notably", "importantly", "meanwhile", "nevertheless", "insofar",
        "consequently", "hence", "nonetheless", "accordingly", "albeit",
        "regardless", "incidentally", "undoubtedly", "specifically"
    ]
    words = word_tokenize(text)
    num_insertions = max(1, int(len(words) * rate))
    positions = sorted(random.sample(range(len(words)), num_insertions))

    for idx in reversed(positions):
        words.insert(idx, random.choice(filler_words))

    return ' '.join(words)


# In[ ]:


substitution_rates = [rate / 100 for rate in range(0, 55, 5)]

attack_results = {
    "synonym": [],
    "antonym": [],
    "insertion": []
}
attack_confidences = {
    "synonym": [],
    "antonym": [],
    "insertion": []
}

attack_funcs = {
    "synonym": synonym_substitution,
    "antonym": antonym_substitution,
    "insertion": insert_noise_words
}

for attack_name in attack_funcs:
    print(f"\nEvaluating attack: {attack_name}")
    for rate in substitution_rates:
        correct = 0
        confidences = []

        for i in tqdm(range(len(val_texts))):
            text = val_texts[i]
            label = val_labels[i]
            adv_text = attack_funcs[attack_name](text, rate) if rate > 0 else text

            inputs = tokenizer(
                adv_text,
                truncation=True,
                padding="max_length",
                max_length=2048,
                return_tensors="pt"
            ).to(device)

            global_attention_mask = torch.zeros_like(inputs['input_ids'])
            global_attention_mask[:, 0] = 1

            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                global_attention_mask=global_attention_mask
            )

            probs = F.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()

            confidences.append(conf)
            if pred == label:
                correct += 1

        acc = correct / len(val_texts)
        attack_results[attack_name].append(1 - acc)  # attack success rate
        attack_confidences[attack_name].append(np.mean(confidences))


# In[ ]:


import pickle

with open("attack_metrics.pkl", "wb") as f:
    pickle.dump({
        "substitution_rates": substitution_rates,
        "attack_results": attack_results,
        "attack_confidences": attack_confidences
    }, f)

print("Saved attack metrics to attack_metrics.pkl")


# In[ ]:


plt.figure(figsize=(7, 3))
rates = [r * 100 for r in substitution_rates]

for name, values in attack_results.items():
    plt.plot(rates, [v * 100 for v in values], label=name.title(), marker='o')

plt.xlabel("Substitution Rate (%)")
plt.ylabel("Attack Success Rate (%)")
plt.title("Adversarial Attack Success Rate (Longformer)")
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(7, 3))

for name, values in attack_confidences.items():
    plt.plot(rates, values, label=name.title(), marker='s')

plt.xlabel("Substitution Rate (%)")
plt.ylabel("Avg Confidence in Predicted Class")
plt.title("Model Confidence vs Perturbation Strength")
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

