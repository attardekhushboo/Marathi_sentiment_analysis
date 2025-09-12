# 📝 Marathi Sentiment Analysis with MahaSBERT + BiLSTM  

This project implements a **Marathi Sentiment Analysis** system by combining **MahaSBERT embeddings** with a **BiLSTM model**. The system classifies Marathi comments into three sentiment classes: **Negative, Neutral, and Positive**.  

---

## 📌 Overview  
- **Objective**: Analyze sentiment from Marathi text data.  
- **Dataset**: Labeled tweets (`tweets_train.csv`) with three sentiment classes.  
- **Model**:  
  - Feature extraction using **MahaSBERT (Marathi Sentence-BERT)**.  
  - Classification using a **BiLSTM model** built with Keras/TensorFlow.  
- **Output**: Predicted sentiment class for each comment.  

---

## 📂 Dataset  
- **Format**: CSV file (`tweets_train.csv`) with columns:  
  - `text` → Marathi sentence/comment.  
  - `label` → sentiment (-1 = Negative, 0 = Neutral, 1 = Positive).  

- **Preprocessing**:  
  - Missing values removed.  
  - Labels remapped to 0 (**Negative**), 1 (**Neutral**), 2 (**Positive**).  

---

## 🛠️ Methodology  

### 🔹 Feature Extraction  
- Used **MahaSBERT (`l3cube-pune/marathi-sentence-bert-nli`)** from Hugging Face.  
- Extracted **token-level embeddings** (`last_hidden_state`).  

### 🔹 BiLSTM Model Architecture  
- **Input**: `(max_len=60, 768)` embeddings.  
- **Layers**:  
  - SpatialDropout1D  
  - Bidirectional LSTM (128, then 64 units)  
  - Batch Normalization  
  - Dense (64, ReLU)  
  - Dense (3, Softmax)  
- **Optimizer**: Adam (lr=0.001).  
- **Loss**: Sparse Categorical Crossentropy.  

---

## 🚀 Implementation  
### Steps  
1. Load dataset and preprocess.  
2. Extract **MahaSBERT embeddings**.  
3. Train **BiLSTM classifier** (80-20 train-test split).  
4. Evaluate model with accuracy, classification report, and confusion matrix.  

---

## 📊 Results  

- Achieved **high accuracy** on test data.  
- Example classification report:  

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Negative  | 0.93      | 0.91   | 0.92     |
| Neutral   | 0.87      | 0.89   | 0.88     |
| Positive  | 0.94      | 0.95   | 0.94     |

✅ The **MahaSBERT + BiLSTM** model outperforms traditional methods for Marathi sentiment classification.  

---

## 📌 Applications  
- **Social Media Monitoring** (analyzing Marathi tweets & comments).  
- **Customer Feedback Analysis** (local language surveys).  
- **Opinion Mining** in politics, marketing, and public policy.  

---

## 🔮 Future Work  
- Extend to **multilingual sentiment analysis** (combine Marathi with Hindi, English).  
- Experiment with **transformer fine-tuning** instead of embeddings.  
- Deploy as an interactive **web app** (Streamlit / Flask).  

