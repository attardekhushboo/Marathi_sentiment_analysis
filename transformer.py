import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_df = pd.read_csv("MahaSent_All_Train.csv")
val_df = pd.read_csv("MahaSent_All_Val.csv")

# Label Mapping: -1 → 0 (Negative), 0 → 1 (Neutral), 1 → 2 (Positive)
label_map = {-1: 0, 0: 1, 1: 2}
train_df['label'] = train_df['label'].map(label_map)
val_df['label'] = val_df['label'].map(label_map)

# Model and tokenizer
MODEL_NAME = "l3cube-pune/marathi-sentence-bert-nli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization
def tokenize_fn(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=60)

# Convert to HuggingFace Dataset format
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

# Apply tokenization
train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

# Format for PyTorch
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load classification model
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

# Evaluate
eval_results = trainer.evaluate()
print("\nEvaluation Results:", eval_results)

# Predict and generate classification report + confusion matrix
predictions = trainer.predict(val_dataset)
preds = torch.argmax(torch.tensor(predictions.predictions), axis=1)
labels = predictions.label_ids

# Classification Report
print("\nClassification Report:\n", classification_report(labels, preds, target_names=["Negative", "Neutral", "Positive"]))

# Confusion Matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
