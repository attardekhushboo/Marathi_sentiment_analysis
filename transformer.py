import torch
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, BatchNormalization, Input, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Load dataset
df = pd.read_csv("tweets_train.csv")
df.dropna(inplace=True)

# Label Mapping: -1 (Negative) -> 0, 0 (Neutral) -> 1, 1 (Positive) -> 2
label_mapping = {-1: 0, 0: 1, 1: 2}
df['label'] = df['label'].map(label_mapping)

# MahaSBERT model and tokenizer
MODEL_NAME = "l3cube-pune/marathi-sentence-bert-nli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
mahabert_model = AutoModel.from_pretrained(MODEL_NAME)

# Function to extract MahaSBERT embeddings (full token-level sequence)
def get_mahabert_embeddings(texts, max_len=60, batch_size=16):
    mahabert_model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, padding='max_length', truncation=True,
                           max_length=max_len, return_tensors='pt')
        with torch.no_grad():
            outputs = mahabert_model(**inputs)
        embeddings = outputs.last_hidden_state  # Shape: (batch_size, max_len, 768)
        all_embeddings.append(embeddings)

    return torch.cat(all_embeddings).numpy()

# Get embeddings
X = get_mahabert_embeddings(df['text'], max_len=60)  # Shape: (samples, max_len, 768)
y = np.array(df['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# BiLSTM Model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),  # (max_len, 768)
    SpatialDropout1D(0.2),

    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    BatchNormalization(),

    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(3, activation='softmax')
])

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)

# Train
history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[lr_scheduler, early_stopping])

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Predict
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
