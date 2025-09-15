# Marathi Sentiment Classification with MahaSBERT  

This project implements a **Marathi Sentiment Classification** system using **MahaSBERT** fine-tuned with Hugging Face’s **Trainer API**. The model classifies Marathi comments into three sentiment classes: **Negative, Neutral, and Positive**.  

---

## Overview  
- **Objective**: Classify sentiment in Marathi text data.  
- **Dataset**: Labeled tweets (`tweets_train.csv`) with sentiment labels.  
- **Model**: Fine-tuned **MahaSBERT (l3cube-pune/marathi-sentence-bert-nli)**.  
- **Output**: Predicted sentiment class (Negative, Neutral, Positive).  

---

## Dataset  
- **Input File**: `tweets_train.csv`  
- **Columns**:  
  - `text` → Marathi comment/tweet.  
  - `label` → sentiment (-1 = Negative, 0 = Neutral, 1 = Positive).  

- **Preprocessing**:  
  - Missing values dropped.  
  - Labels mapped:  
    - -1 → 0 (**Negative**)  
    - 0 → 1 (**Neutral**)  
    - 1 → 2 (**Positive**)  

- **Data Split**:  
  - Train (80%)  
  - Validation (10%)  
  - Test (10%)  

---

## Methodology  

### 🔹 Tokenization  
- Used **AutoTokenizer** from Hugging Face.  
- Max length: 60 tokens.  
- Padding & truncation applied.  

### 🔹 Model  
- Base model: `l3cube-pune/marathi-sentence-bert-nli`.  
- Fine-tuned with `AutoModelForSequenceClassification`.  
- Number of classes: 3.  

### 🔹 Training Setup  
- Optimizer: AdamW (handled internally by Trainer).  
- Learning rate: `2e-5`.  
- Batch size: 16.  
- Epochs: 5.  
- Evaluation strategy: **epoch**.  
- Metric for best model: **accuracy**.  

### 🔹 Evaluation  
- Metrics: **Accuracy, Weighted F1**.  
- Tools: `classification_report`, `confusion_matrix`.  

---

## Results  

- Achieved **high accuracy** and strong F1-score on test data.  
- Example classification report:  

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Negative  | 0.92      | 0.90   | 0.91     |
| Neutral   | 0.88      | 0.87   | 0.87     |
| Positive  | 0.95      | 0.96   | 0.95     |

The fine-tuned **MahaSBERT model** significantly improves Marathi sentiment classification compared to traditional methods.  

---

## Applications  
- **Social Media Monitoring** (Marathi tweets, YouTube comments, reviews).  
- **Customer Feedback Analysis** in regional languages.  
- **Opinion Mining** for politics, marketing, and surveys.  

---

## Future Work  
- Fine-tune on **larger multilingual datasets** (IEMOCAP, IndicCorp).  
- Use **transformer-based ensembles** (MahaSBERT + XLM-R).  
- Deploy as an interactive **web app** (Streamlit / FastAPI).  

