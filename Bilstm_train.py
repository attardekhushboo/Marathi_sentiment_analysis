import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load single dataset
df = pd.read_csv("tweets_train.csv")  # Use your full dataset here

# Label Mapping: -1 → 0 (Negative), 0 → 1 (Neutral), 1 → 2 (Positive)
label_map = {-1: 0, 0: 1, 1: 2}
df['label'] = df['label'].map(label_map)

# Split into train, validation, test (80-10-10 split)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Load tokenizer
MODEL_NAME = "l3cube-pune/marathi-sentence-bert-nli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize_fn(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=60)

# Convert to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

# Tokenize
train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)
test_dataset = test_dataset.map(tokenize_fn, batched=True)

# Format for PyTorch
for dataset in [train_dataset, val_dataset, test_dataset]:
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mahasentiment_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Metrics
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), axis=1)
    labels = p.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate on validation set
eval_results = trainer.evaluate()
print("\nValidation Evaluation Results:", eval_results)

# Evaluate on test set
test_results = trainer.predict(test_dataset)
test_preds = torch.argmax(torch.tensor(test_results.predictions), axis=1)
test_labels = test_results.label_ids

# Classification Report
print("\nTest Set Classification Report:\n", classification_report(test_labels, test_preds, target_names=["Negative", "Neutral", "Positive"]))

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Set Confusion Matrix')
plt.show()
